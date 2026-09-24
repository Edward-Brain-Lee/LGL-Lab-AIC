"""Generate ``pred_results.csv`` for the official test set.

**Single model, single set of trained weights** -- CLIP ViT-B/32 with the LoRA
adapters and the cosine head produced by ``train.py``.

TTA and test-time resolution
----------------------------
The competition permits test-time augmentation and a changed input resolution
as long as the backbone architecture and the trained weights stay the same
(README_AUTODL.md 5).  Both are implemented here, and both stay inside the
"one model" rule:

* every view runs **the same checkpoint**;
* the per-view features are **averaged, not voted**: each view's feature is
  already L2-normalised by ``Net.forward``, the mean is re-normalised once, and
  the head is applied **once**.  There is a single decision path, so this is
  TTA of one model and not an ensemble.  ``--tta-agg logit`` averages the
  logits instead (also one model) -- the feature average is the default because
  normalising before averaging keeps a badly-scaled view from dominating.

Averaging the outputs of several *different* checkpoints (e.g. seed ensembles)
is an ensemble and is **not** permitted.  Do not add it.

There are two independent axes, and they are not equally safe:

* ``--tta-crops`` changes only *which pixels of the one image* are fed to the
  model.  All the policies in ``train.CROP_POLICIES`` stay inside the trained
  patch grid, so no positional embedding is touched.  This axis is free of the
  risk below.  It exists because the default ``center`` policy does not keep the
  whole frame: measured, a 3:4 image keeps 57.5% of it, a 2:3 image 51.0%, and
  even a square one only 76.6% (``Resize(size*256/224)`` + ``CenterCrop(size)``
  removes the outer 1/8 of both axes whatever the shape).  On fine-grained
  images the discriminative part is small and often not centred.
* ``--tta-sizes`` changes the resolution, which means open_clip has to
  interpolate the positional embeddings (``resize_pos_embed``).  That is
  permitted, but it is the axis with real evidence against it -- and the
  distinction that matters is **not** "bigger is worse", it is
  **interpolate-only vs trained-there**:

  - Evaluating at a size the weights never trained at is the case that is
    documented to lose.  DeiT-tiny with interpolated pos-emb, no fine-tuning:
    224: 72.2 -> 384: 71.2 -> 448: 68.8 -> 512: 65.9.  The community report
    (open_clip Discussion #987) of CLIP-B/32 at 320 losing to 224 -- a single,
    unreplicated datapoint with the training configuration unstated -- is most
    likely this case, not a refutation of 320 itself.
  - Training *at* the target size is the opposite: same backbone, same recipe,
    only the fine-tuning resolution changed gives ViT-B/32 +2.1 at 384 and
    +2.5 at 448 (timm/Cherti, ImageNet-1k).

  So a checkpoint produced by ``train.py --image-size 320`` is in-distribution
  at 320 and ``--tta-sizes 320`` adds a legitimate second view.  Pointing
  ``--tta-sizes`` at a size the checkpoint was **not** trained at is the
  configuration that is known to lose.  Measure it (``probe.py --tta``) before
  trusting it, and read ``ck['image_size']`` first -- that is what ``base``
  above comes from.

Each distinct resolution needs its own ``open_clip.create_model`` call because
``force_image_size`` (which resamples the positional embeddings) is applied at
model-creation time; one model is built per size, loaded from the same
checkpoint, used, and freed.

Usage::

    python infer.py --test /root/autodl-tmp/test --checkpoint outputs/best.pt \
                    --output pred_results.csv
    # one checkpoint, 3 crop policies x flip = 6 views, all at the training size:
    python infer.py --test ... --checkpoint outputs/best.pt \
                    --tta-crops center full pad --tta-flip
    # and/or several resolutions (6 views) -- measure that one first:
    python infer.py --test ... --checkpoint outputs/best.pt \
                    --image-size 224 --tta-sizes 288 320 --tta-flip

Every image found under ``--test`` produces exactly one CSV row.  An image that
Pillow cannot decode is replaced by a grey image instead of being skipped, so a
partially truncated file (the competition warns about those) can never make the
submission row count mismatch the test set.

``--logit-adjust`` applies post-hoc logit adjustment (Menon et al., ICLR 2021).
It exists because the two priors genuinely differ here: the brief states the
test set is class-balanced while the training set is not.  The correction is a
pure re-argmax of logits that were computed anyway, so passing several taus
yields several submission files from **one** forward pass and each can be
scored on the leaderboard -- which matters when submissions and GPU hours are
the scarce resources.  On this round's data the effect is small (the training
distribution is nearly flat, see ``datastats.py``), so treat it as a cheap
tie-breaker rather than a headline gain.
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

from train import IMG_EXTS, Net, build_clip, eval_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    classes = ck['classes'] if 'classes' in ck else ck['class_to_idx']
    ck_args = ck.get('args', {})
    rank = a.lora_rank or ck.get('lora_rank', ck_args.get('lora_rank', 8))
    target = ck.get('lora_target', ck_args.get('lora_target', 'all'))
    pretrained = a.pretrained or ck.get('pretrained', 'openai')
    model_name = ck.get('model_name', ck_args.get('model', 'ViT-B-32-quickgelu'))
    # build_clip calls check_backbone, i.e. it refuses to serve a checkpoint
    # built on a tower that is not CLIP ViT-B/32.
    print(f'loaded {a.checkpoint} (epoch {ck.get("epoch", "?")}, '
          f'{len(classes)} classes, {model_name}, lora rank {rank}/{target})')

    def make_model(size):
        """One ``Net`` at ``size``, loaded from the checkpoint, on ``device``."""
        clip_model = build_clip(model_name, pretrained, size)
        net = Net(clip_model, len(classes), rank, target)
        missing, _ = net.load_state_dict(ck.get('model', ck), strict=False)
        trained = {n for n, p in net.named_parameters() if p.requires_grad}
        lost = trained & set(missing)
        assert not lost, f'checkpoint has no trained weights for: {sorted(lost)[:5]}'
        del clip_model
        return net.to(device).eval()

    # folder name -> four-digit submission label
    idx_to_label = {}
    for name, i in classes.items():
        try:
            idx_to_label[i] = f'{int(name):04d}'
        except ValueError:
            idx_to_label[i] = name
            print(f'WARNING: class folder "{name}" is not numeric, passing it through unchanged')

    counts = torch.as_tensor(ck.get('class_counts') or [], dtype=torch.float32)
    log_prior = None
    if counts.numel() == len(classes):
        # clamp the *denominator*: clamp_min after the division would leave a
        # 0/0 = nan untouched
        log_prior = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-12).log()
        print(f'class-count prior loaded (log range {float(log_prior.max() - log_prior.min()):.2f})')
    elif any(a.logit_adjust):
        print('WARNING: checkpoint carries no usable class_counts -- --logit-adjust is ignored')

    # ---- the eval-time views.  `--image-size 0` means "as trained".
    base = a.image_size or ck.get('image_size', ck_args.get('image_size', 224))
    sizes = [base] + [s for s in dict.fromkeys(a.tta_sizes) if s and s != base]
    crops = list(dict.fromkeys(a.tta_crops)) or ['center']
    views = [(s, c, f) for s in sizes for c in crops
             for f in ((False, True) if a.tta_flip else (False,))]

    files = sorted(p for p in Path(a.test).rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS)
    assert files, f'no images found under {a.test}'
    n_img = len(files)
    def vname(s, c, f):
        """``288px`` for the plain default view, more when it is not that."""
        bits = [f'{s}px']
        if c != 'center':
            bits.append(c)
        if f:
            bits.append('flip')
        return ' '.join(bits)

    print(f'{n_img} test images found')
    print(f'{len(views)} view(s): ' + ', '.join(vname(*v) for v in views) +
          f'  (aggregation: {a.tta_agg})')

    # fp32 by default.  Training evaluates in bf16, but a submission is a
    # decision, and bf16's ~3 significant digits can flip a near tie in a
    # 750-way argmax -- so the submission path stays where the scores we are
    # comparing against were measured.  --amp bf16 is there for the multi-view
    # sweeps, where the extra speed matters and the CSVs are only compared.
    use_amp = a.amp == 'bf16' and device.type == 'cuda'
    names, unreadable = [], 0
    head, feat_sum, logit_sum = None, None, None
    for vi, (size, crop, flip) in enumerate(views):
        model = make_model(size)
        tf = eval_transform(size, flip, crop=crop)
        if head is None:
            head = model.head        # same weights in every view; kept for the final decision
        with torch.no_grad():
            for s in range(0, n_img, a.batch_size):
                ims = []
                for p in files[s:s + a.batch_size]:
                    try:
                        img = Image.open(p).convert('RGB')
                    except Exception:                   # truncated / unreadable
                        img = Image.new('RGB', (size, size), (127, 127, 127))
                        if vi == 0:                     # count each image once, not once per view
                            unreadable += 1
                    ims.append(tf(img))
                    if vi == 0:
                        names.append(p.name)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    out, z = model(torch.stack(ims).to(device, non_blocking=True), return_feat=True)
                # `z` is already L2-normalised by Net.forward
                if a.tta_agg == 'feat':
                    zz = z.float().cpu()
                    feat_sum = zz if feat_sum is None else feat_sum + zz
                else:
                    oo = out.float().cpu()
                    logit_sum = oo if logit_sum is None else logit_sum + oo
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'  view {vi + 1}/{len(views)} done ({vname(size, crop, flip)})')

    # one decision from the averaged views -- the head is applied exactly once
    if a.tta_agg == 'feat':
        feats = F.normalize(feat_sum / len(views), dim=-1)
        with torch.no_grad():
            logits = head(feats.to(device)).float().cpu()
    else:
        logits = logit_sum / len(views)

    for tau in (a.logit_adjust or [0.0]):
        adj = logits if (not tau or log_prior is None) else logits - tau * log_prior
        rows = [(n, idx_to_label[i]) for n, i in zip(names, adj.argmax(1).tolist())]
        out = adjusted_path(a.output, tau)
        with open(out, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)
        print(f'wrote {len(rows)} rows to {out} (logit-adjust tau={tau:g})')

    print(f'unreadable images replaced by grey: {unreadable}')
    if len(names) != n_img:
        print('WARNING: row count does not match the number of test images')
    if len(set(names)) != len(names):
        print('WARNING: duplicate file names found -- the grader may match by name only')


def adjusted_path(base, tau):
    """``pred_results.csv`` for tau == 0, ``pred_results_tau050.csv`` otherwise.

    Always returns a ``str``.  The tau == 0 branch hands ``base`` back untouched,
    so returning a ``Path`` from the other branch made the type depend on the
    argument -- and ``Path('x.csv') == 'x.csv'`` is False, which is precisely how
    the self-test flagged it.
    """
    if not tau:
        return base
    p = Path(base)
    return str(p.with_name(f'{p.stem}_tau{("%.2f" % tau).replace(".", "")}{p.suffix}'))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--test', required=True, help='test image folder (searched recursively)')
    p.add_argument('--checkpoint', required=True, help='best.pt / last.pt from train.py')
    p.add_argument('--output', default='pred_results.csv')
    p.add_argument('--pretrained', default='', help='override the CLIP weights (default: as in the checkpoint)')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--logit-adjust', type=float, nargs='*', default=[0.0], dest='logit_adjust',
                   metavar='TAU',
                   help='post-hoc logit adjustment: subtract tau*log(train class frequency) '
                        'before the argmax. The rules state the test set is class-balanced while '
                        'the training set is not, so tau>0 trades head accuracy for tail '
                        'accuracy. Several values give several CSVs from ONE forward pass, e.g. '
                        '--logit-adjust 0 0.25 0.5 1.0')
    p.add_argument('--lora-rank', type=int, default=0, help='override (default: as in the checkpoint)')
    p.add_argument('--image-size', type=int, default=0,
                   help='base (non-augmented) input resolution; 0 = as trained, which is '
                        'recorded in the checkpoint. Anything else resamples the positional '
                        'embeddings -- permitted, see README_AUTODL.md 5.')
    p.add_argument('--tta-sizes', type=int, nargs='*', default=[], dest='tta_sizes',
                   metavar='N',
                   help='extra resolutions to average in, e.g. --tta-sizes 288 320. '
                        'Same checkpoint, same head; features are averaged before a single '
                        'decision, so this is TTA of one model, not an ensemble. Keep them '
                        'multiples of the 32px patch size (288/320/352) -- 336 is a '
                        'patch-14 number and train.build_clip will warn about it.')
    p.add_argument('--tta-flip', action='store_true',
                   help='also average the horizontally flipped image at every size '
                        '(a cheap, well-behaved TTA for natural images)')
    p.add_argument('--tta-crops', nargs='*', default=['center'], dest='tta_crops',
                   metavar='POLICY',
                   help='crop policies to average in, from train.CROP_POLICIES: '
                        'center (default; resize short side to size*256/224 then centre '
                        'crop -- throws away part of the frame), full (squash the whole '
                        'image to size x size), pad (fit the long side and pad the rest: '
                        'keeps both the content and the aspect ratio). All three stay '
                        'inside the trained patch grid, so unlike --tta-sizes they need '
                        'no positional-embedding interpolation. '
                        'e.g. --tta-crops center full pad')
    p.add_argument('--amp', default='none', choices=['none', 'bf16'],
                   help='none (default) keeps the submission path in fp32, which is where the '
                        'scores we compare against were measured. bf16 is ~2x faster and is '
                        'fine for the multi-view sweeps -- but it can flip a near tie in a '
                        '750-way argmax, so do not use it for the final submission without '
                        'checking that it changes nothing.')
    p.add_argument('--tta-agg', default='feat', choices=['feat', 'logit'], dest='tta_agg',
                   help='feat: average the L2-normalised features then apply the head once '
                        '(default). logit: apply the head per view then average the logits. '
                        'Both use one checkpoint and one head.')
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
