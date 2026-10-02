"""Multi-resolution TTA, with a *paired* hold-out comparison so one submission can be decided.

The problem this solves
-----------------------
``val_acc`` over the 10116-image hold-out has a 1-sigma of
``sqrt(0.7347 * 0.2653 / 10116) = 0.0044``.  The candidate gains we are choosing
between are ~0.3 points.  Two *separate* training runs are therefore statistically
identical, and picking between them for a single submission slot is a coin flip
dressed up as a measurement.

Everything here is evaluated on **the same 10116 images**, so the comparison is
*paired*.  What decides the winner is not the absolute accuracy but how many images
flip and in which direction: a scheme that wins 60% of 800 disagreements is a
4-sigma paired effect even though its absolute accuracy moved by less than the
unpaired noise floor.  That is what makes the slot decidable.

What it changes, and what it does not
-------------------------------------
The weights are never touched.  The checkpoint is loaded once at its trained
resolution; for each requested resolution the vision tower's positional grid is
resampled from **the grid the checkpoint learned** (not OpenAI's 7x7) using
``train.resize_positional_embedding``.  One model, one set of weights, several
deterministic input sizes, predictions averaged.  That is the organizers' 2026-09
ruling verbatim: changing the input resolution with positional-embedding
interpolation is allowed, and TTA is a permitted special case that uses no test-set
prior.

Note that ``infer.py``'s existing TTA views are *crops*, not resolutions: every view
does ``Resize(img_size * ratio)`` then crops back to ``img_size``, so the tensor
handed to the tower is always ``img_size``.  True resolution diversity -- a tensor
that is genuinely 352 or 448 wide -- has never been tried on this project, and
resolution is the only axis that has paid every time it moved
(224->288 +5.06, 288->320 +0.75, 320->352 +0.89, 352->384 +0.28).

The self-check
--------------
At the checkpoint's own resolution with ``--views plain`` the printed micro accuracy
MUST equal the ``val_acc`` ``train.py`` logged for that epoch.  If it does not, the
split or the forward pass is wrong and nothing else here means anything.  Fix that
first -- the same discipline ``valmetrics.py`` documents.

Memory
------
``TTADataset`` returns ``(B, V, 3, S, S)`` per batch and ``prefetch_factor=1`` keeps
one of those per worker.  At ``infer.py``'s defaults (256 x 8 views x 384px) that is
~9 GB **per worker**, and the container is capped at 60 GiB with the dataloader
workers being what fills it.  The defaults here are deliberately smaller -- 64 x V x
S*S.  Raise ``--batch-size`` only together with ``--workers 0``.

Usage::

    # decidable comparison: which scheme wins on the fixed hold-out, and by how much
    python mres.py --mode val --checkpoint outputs_384pe_local/ep20.pt \
        --res 352 384 416 --views center

    # write the submission with the winning scheme
    python mres.py --mode test --checkpoint outputs_384pe_local/ep20.pt \
        --test /root/autodl-tmp/test --output pred_results_local_mres.csv \
        --res 352 384 416 --views center
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms

import open_clip

from train import (CLIP_MEAN, CLIP_STD, IMG_EXTS, ImageFolderNoisy, Net, check_backbone,
                   resize_positional_embedding)
from infer import (TTA_GROUPS, TTA_VIEWS, TTADataset, _one_thread_per_worker,
                   build_view_transform, resolve_views)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# --------------------------------------------------------------------------- #
# model / resolution plumbing
# --------------------------------------------------------------------------- #
def load_checkpoint(path, device):
    """Build the model at the checkpoint's own resolution and load every weight.

    The order matters and is the whole reason ``infer.py --img-size`` cannot do
    multi-resolution: the positional grid has to be resampled to the *checkpoint's*
    size before ``load_state_dict``, or the saved grid (145 tokens for a 384px run)
    lands on a differently shaped parameter and ``strict=False`` raises a size
    mismatch -- shape errors are never downgraded to "unexpected keys".

    Returns ``(model, ck, ck_size, base_pe, pe_name)`` where ``base_pe`` is a private
    copy of the *trained* grid, used to restore the tower before each resolution.
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ck_args = ck.get('args', {})
    rank = ck.get('lora_rank', ck_args.get('lora_rank', 8))
    target = ck.get('lora_target', ck_args.get('lora_target', 'all'))
    model_name = ck.get('model_name', ck_args.get('model', 'ViT-B-32-quickgelu'))
    check_backbone(model_name)
    img_size = ck.get('img_size', ck_args.get('img_size', 224))
    local_head = bool(ck.get('local_head', ck_args.get('local_head', False)))

    clip_model = open_clip.create_model(model_name, pretrained=ck.get('pretrained', 'openai'))
    if img_size != 224:
        resize_positional_embedding(clip_model.visual, img_size)
    model = Net(clip_model, len(ck['classes']), rank, target, local_head=local_head)
    missing, _ = model.load_state_dict(ck.get('model', ck), strict=False)
    lost = {n for n, p in model.named_parameters() if p.requires_grad} & set(missing)
    assert not lost, f'no trained weights in checkpoint for: {sorted(lost)[:5]}'
    del clip_model

    visual = model.clip.visual
    pe_name = 'positional_embedding' if hasattr(visual, 'positional_embedding') else 'pos_embed'
    assert hasattr(visual, pe_name), 'vision tower exposes no positional embedding'
    base_pe = getattr(visual, pe_name).detach().clone()

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'loaded {path} (epoch {ck.get("epoch", "?")}, {len(ck["classes"])} classes, '
          f'{model_name}, {img_size}px, local_head={local_head})')
    return model, ck, img_size, base_pe, pe_name


def at_resolution(model, base_pe, pe_name, res):
    """Resample the tower onto ``res`` from the checkpoint's trained grid.

    ``resize_positional_embedding`` infers the *source* grid from the tensor it finds
    (``base = sqrt(n_tok - 1)``), so restoring ``base_pe`` first is what makes this
    idempotent and makes every resolution interpolate from the learned 12x12 grid
    rather than from whatever the previous call left behind.
    """
    visual = model.clip.visual
    setattr(visual, pe_name, nn.Parameter(base_pe.clone(), requires_grad=False))
    return resize_positional_embedding(visual, res)


@torch.no_grad()
def predict(model, files, view_names, res, a, device):
    """View-averaged logits for every file, in ``files`` order.

    Averaging happens over the views *within* a resolution (logits, exactly as
    ``infer.py`` does) and over the resolutions *outside* this function (softmax
    probabilities -- logit scales are not comparable across input sizes, because a
    different grid means a different sum over tokens).
    """
    view_tfs = {v: build_view_transform(*TTA_VIEWS[v], res) for v in view_names}
    ds = TTADataset(files, [view_tfs[v] for v in view_names], res)
    kwargs = {'prefetch_factor': 1} if a.workers > 0 else {}
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                        pin_memory=(device.type == 'cuda'),
                        worker_init_fn=_one_thread_per_worker, **kwargs)
    out, names = [], []
    for views, batch_names, _ in loader:
        acc = None
        for v in range(views.shape[1]):
            o = model(views[:, v].to(device)).float().cpu()
            acc = o if acc is None else acc + o
        out.append(acc / views.shape[1])
        names.extend(batch_names)
    return torch.cat(out), names


def combine(per_res):
    """Mean of the per-resolution softmaxes.  See :func:`predict` for why not logits."""
    return torch.stack([lr.softmax(1) for lr in per_res]).mean(0) if len(per_res) > 1 else per_res[0]


def paired(base_pred, new_pred, y):
    """McNemar counts for two schemes scored on the same images."""
    b_ok, n_ok = base_pred == y, new_pred == y
    return (int((b_ok & n_ok).sum()), int((b_ok & ~n_ok).sum()),
            int((~b_ok & n_ok).sum()), int((~b_ok & ~n_ok).sum()))


# --------------------------------------------------------------------------- #
# hold-out: the decidable comparison
# --------------------------------------------------------------------------- #
def evaluate(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck, ck_size, base_pe, pe_name = load_checkpoint(a.checkpoint, device)
    ck_args = ck.get('args', {})
    data = a.data or ck_args.get('data')
    assert data, 'checkpoint has no --data recorded; pass --data explicitly'

    # membership depends only on (val_ratio, seed); the transform is never invoked,
    # because TTADataset does its own -- 'plain' IS the validation transform
    # (Resize(val_resize) + CenterCrop + Normalize), which is what the self-check uses
    va = ImageFolderNoisy(data, transforms.ToTensor(), True,
                          ck_args.get('val_ratio', 0.1), ck_args.get('seed', 3407), 'val')
    assert va.class_to_idx == ck['classes'], 'val split found a different class list'
    files = [Path(p) for p, _ in va.items]
    label = {p.name: y for p, y in va.items}
    wanted = [p.name for p in files]
    n_cls = len(ck['classes'])
    print(f'hold-out: {len(files)} images, {n_cls} classes')

    view_names = resolve_views(list(a.views))
    unknown = sorted(set(view_names) - set(TTA_VIEWS))
    assert not unknown, f'unknown view(s) {unknown}'

    # the checkpoint's own resolution first, so its number is the reference every
    # other block is paired against
    order = [ck_size] + [r for r in a.res if r != ck_size]
    per_res, preds = [], {}
    for res in order:
        grid = at_resolution(model, base_pe, pe_name, res)
        logits, names = predict(model, files, view_names, res, a, device)
        assert names == wanted, 'view order drifted from the hold-out file order'
        y = torch.tensor([label[n] for n in names])
        preds[res] = logits.argmax(1)
        acc = float((preds[res] == y).float().mean())
        rec = torch.zeros(n_cls).index_add_(0, y, (preds[res] == y).float()) \
            / torch.zeros(n_cls).index_add_(0, y, torch.ones_like(y).float()).clamp_min(1)
        tag = '  <- checkpoint resolution' if res == ck_size else ''
        print(f'  {res}px grid {grid:>2}x{grid:<2} acc={acc:.4f}  macro={float(rec.mean()):.4f}{tag}')
        per_res.append(logits)
        if res == ck_size:
            ref_pred, ref_acc = preds[res], acc

    # The self-check needs its own pass: train.py's val_acc is a SINGLE `plain` view,
    # not the tta8 set, so it cannot be read off `preds` (which is the view-averaged
    # prediction and is *supposed* to be higher).  Comparing the two would fail on
    # every checkpoint and cry wolf.
    if a.expect is not None:
        at_resolution(model, base_pe, pe_name, ck_size)
        pl, pn = predict(model, files, ['plain'], ck_size, a, device)
        assert pn == wanted, 'view order drifted from the hold-out file order'
        got = float((pl.argmax(1) == y).float().mean())
        ok = abs(got - a.expect) < 5e-4
        print(f'  SELF-CHECK {"OK " if ok else "FAIL"}: {ck_size}px plain micro {got:.4f} '
              f'vs train.py log {a.expect:.4f}')
        if not ok:
            raise SystemExit('self-check failed -- split or forward pass is wrong; '
                             'no number below can be trusted')

    print('')
    if len(a.res) > 1:
        combo = combine(per_res)
        cp = combo.argmax(1)
        cacc = float((cp == y).float().mean())
        print(f'  COMBINED {sorted(a.res)} acc={cacc:.4f}   ({cacc - ref_acc:+.4f} vs '
              f'{ck_size}px alone)')
        if a.ref_views:
            # what we would actually submit today: the checkpoint's tta8 at its own size
            rv = resolve_views(list(a.ref_views))
            rl, rn = predict(model, files, rv, ck_size, a, device)
            assert rn == wanted
            rp = rl.argmax(1)
            racc = float((rp == y).float().mean())
            cr, rw, wl, cw = paired(rp, cp, y)
            dec = rw + wl
            print(f'  reference: {ck_size}px + {len(rv)} views acc={racc:.4f}')
            print(f'  PAIRED vs reference: both right {cr}, ONLY reference {rw}, '
                  f'ONLY combined {wl}, both wrong {cw}')
            if dec:
                import math
                z = (wl - rw) / math.sqrt(dec) if dec else 0.0
                print(f'  disagreements {dec} ({dec / len(files):.1%} of images), '
                      f'sign test z={z:+.2f}  ->  '
                      f'{"COMBINED is better" if z > 2 else "reference is better" if z < -2 else "no decision (|z|<2)"}')
            print(f'  combined vs reference delta {cacc - racc:+.4f} '
                  f'(unpaired 1-sigma is 0.0044 -- this is why the paired test matters)')
    print('')


# --------------------------------------------------------------------------- #
# test set: the submission
# --------------------------------------------------------------------------- #
def write_submission(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck, ck_size, base_pe, pe_name = load_checkpoint(a.checkpoint, device)
    idx_to_label = {}
    for name, i in ck['classes'].items():
        try:
            idx_to_label[i] = f'{int(name):04d}'
        except ValueError:
            idx_to_label[i] = name

    files = sorted(p for p in Path(a.test).rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS)
    assert files, f'no images found under {a.test}'
    print(f'{len(files)} test images found')
    view_names = resolve_views(list(a.views))
    unknown = sorted(set(view_names) - set(TTA_VIEWS))
    assert not unknown, f'unknown view(s) {unknown}'

    order = [ck_size] + [r for r in a.res if r != ck_size]
    per_res, names = [], None
    for res in order:
        grid = at_resolution(model, base_pe, pe_name, res)
        print(f'  {res}px grid {grid}x{grid}: {len(view_names)} views ...')
        logits, got = predict(model, files, view_names, res, a, device)
        assert got == [f.name for f in files], 'prediction order drifted from the file list'
        names = got
        per_res.append(logits)

    probs = combine(per_res)
    counts = torch.as_tensor(ck.get('class_counts') or [], dtype=torch.float32)
    log_prior = None
    if counts.numel() == len(ck['classes']):
        log_prior = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-12).log()

    logits = probs.log()
    for tau in (a.logit_adjust or [0.0]):
        adj = logits if (not tau or log_prior is None) else logits - tau * log_prior
        rows = [(n, idx_to_label[i]) for n, i in zip(names, adj.argmax(1).tolist())]
        out = a.output if not tau else str(Path(a.output).with_name(
            f'{Path(a.output).stem}_tau{("%.2f" % tau).replace(".", "")}{Path(a.output).suffix}'))
        with open(out, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)
        print(f'wrote {len(rows)} rows to {out} (tau={tau:g})')
    if len(set(names)) != len(names):
        print('WARNING: duplicate file names -- the grader may match by name only')


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='val', choices=['val', 'test'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test', default='/root/autodl-tmp/test')
    p.add_argument('--output', default='pred_results.csv')
    p.add_argument('--data', default='', help='override the training folder '
                                              '(default: as recorded in the checkpoint)')
    p.add_argument('--res', type=int, nargs='+', default=[384],
                   help='input resolutions to average over. Each must be a multiple of 32 '
                        '(the patch size). The checkpoint resolution is always included '
                        'and evaluated first; it is the reference every other block is '
                        'paired against.')
    p.add_argument('--views', nargs='+', default=['center'],
                   help='views within each resolution: names, or the groups '
                        + ', '.join(sorted(TTA_GROUPS)) + '.  `center` is the eight views '
                        'that scored 70.332 at 320px.')
    p.add_argument('--ref-views', nargs='*', default=[],
                   help='if given, also score this view set at the checkpoint resolution '
                        'and run the paired test against the combined multi-resolution '
                        'prediction.  Pass `--ref-views center` to compare against what '
                        'would be submitted today.')
    p.add_argument('--expect', type=float, default=None,
                   help='the val_acc train.py logged for this checkpoint.  Verifies the '
                        'split and the forward pass before reporting anything else.')
    p.add_argument('--logit-adjust', type=float, nargs='*', default=[0.0], dest='logit_adjust',
                   metavar='TAU')
    p.add_argument('--batch-size', type=int, default=64,
                   help='keep this small: TTADataset returns (B, V, 3, S, S) and '
                        'prefetch_factor=1 holds one per worker.  256 x 8 x 384px is ~9 GB '
                        'per worker and the container cap is 60 GiB.')
    p.add_argument('--workers', type=int, default=8)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    (write_submission if args.mode == 'test' else evaluate)(args)
