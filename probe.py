"""Diagnostics on frozen-CLIP features: how noisy are the labels, and is the
leaderboard gap label noise or domain shift?  Read-only -- trains nothing, writes
no submission.

Why this file exists
--------------------
``HANDOFF.md`` §15.4 concluded that the 6.54-point gap between ``val_acc``
(0.7070) and the leaderboard (64.16) could only be resolved by burning
submission slots.  At 2 submissions/day that makes the submission budget the
project's scarcest resource, well ahead of GPU time -- the whole project stalls
behind one unresolved question.  This script buys back a local signal instead,
from **one frozen-CLIP forward pass** that every measurement below reuses.

What it measures, and why each one decides something
---------------------------------------------------
**rho -- the label-noise rate.**  Nobody has ever measured it on this dataset.
The noise is *keyword* noise: class folders are named by an English search
keyword and the images were crawled by that keyword, so a polysemous keyword
drags in off-concept images.  ``train/0000/`` really does contain a red
"bluebird pure Sialia" water heater sitting next to its bluebirds.  That is the
brief's "weak-correlation annotation", and unlike random noise it is *learnable
and generalises*, which is exactly why hold-out accuracy can come out above the
leaderboard score.  If rho is ~6%+ the noise can carry the gap; if it is ~1%
the noise story is dead and the gap is capacity/resolution instead.  Scored with
**leave-one-out** class means, so a sample never helps build the centroid it is
judged against.

**V* -- a clean proxy for the val split.**  val is drawn from the noisy training
distribution, so ``val_acc`` rewards memorising the noise and ``--select
val_acc`` actively picks the checkpoint that memorised the most of it.  V* keeps
only the val samples whose given label agrees with an *independent* judge --
frozen CLIP's nearest-class-mean, whose centroids come from the disjoint train
split, so no val label influenced them.  ``acc@V*`` is the one local number that
should NOT be inflated by noise memorisation, so it is what ``valmetrics.py``
should rank checkpoints by.  (The QDA self-check: on val, the same judge is
~100% by construction and therefore useless *as a number*; it is V*'s
*membership* that is the artefact here, not its accuracy.)

**shift -- is the test set simply harder?**  train and test are separate crawls:
they share zero source URLs (filenames are md5s of the source URL, and the two
1.5e5/3.7e4 name sets are disjoint).  Test labels are hidden, but *label-free*
statistics are not: if the max cosine to any train centroid, and the prediction
entropy, have the same distribution on test as on val, then there is no
measurable domain shift and the gap must be something training did.

**gate -- does higher resolution help?**  The obvious lever for a patch-32 model
on fine-grained data (224 is only 49 tokens).  But the published gains are
~+1-3, and there is a contrary report of CLIP-B/32 fine-tuned at 320 losing to
224, so it is measured here *before* anyone spends hours on a long run.  Every
size is measured identically -- centroids from the same per-class subsample of
train (never from val itself, where ~18 images/class would let centroid noise
swamp an effect worth only a point or two) -- so the columns compare.  Note the
gate cannot report ``acc@V*``: V* is *defined* by this judge, so the judge would
score ~1.00 on it by construction.  ``acc@V*`` is a number for a trained model,
and ``valmetrics.py`` is what computes it.

**The montage is the ground truth.**  Everything above rests on the assumption
that the flagged images are genuinely off-concept.  That is an assumption about
*images*, and the only honest way to check it is to look.  Three panels, because
they answer different questions: the globally worst offenders, the worst image
of each of the worst classes (this is the one that tests the *keyword* story --
polysemy should concentrate breakage in a few folders, not spread it evenly),
and a random control.  If these are mostly ordinary same-species hard examples,
the noise hypothesis is wrong and everything downstream of it is void.

Outputs (all under ``--out``)
-----------------------------
  feat_{split}_{size}[flip].npz  cached features -- re-runs are seconds, not minutes
  report.txt                the compact report
  noise_montage.png         the most extreme violations      <- look first
  noise_byclass_montage.png worst image of each worst class  <- tests keyword story
  control_montage.png       random train images, same layout
  vstar.npz                 val V* mask + frozen predictions   -> valmetrics.py
  trainscores.npz           per-image train scores              -> train.py

**res / TTA -- ``--tta`` searches inference recipes for free.**  Resolution and
test-time augmentation are inference-only: the same weights are read at another
resolution (open_clip interpolates the position embeddings), on a differently
cropped copy, and/or on a flipped copy, and the features averaged.  No
retraining, so unlike capacity this lever costs no training budget and is
*additive* with the noise axis rather than in tension with it.  The section
scores each candidate recipe on val and reports a ranking, which ``infer.py``
should then follow.

Of the three axes, **crop policy is the one with a mechanism behind it**:

* **crop** -- ``eval_transform``'s default resizes the short side to
  ``size*256/224`` and centre-crops, so it discards a fixed slice of every
  frame.  Measured: a 3:4 portrait keeps **57.5%** of the frame, a 2:3 one
  **51.0%**, and even a *square* image only **76.6%** -- that last one is pure
  protocol, ``Resize(256)`` then ``CenterCrop(224)`` drops the outer 1/8 of both
  axes whatever the aspect ratio.  Fine-grained discriminative parts are small
  and often off-centre, so what is cut may be the part that decides.  ``full``
  and ``pad`` keep the whole image; all three stay inside the trained patch
  grid, so there is no interpolation risk at all.
* **flip** -- cheap and safe, but expect little: CLIP image features are ~0.99
  cosine-similar under a flip, so the two views are nearly redundant and an
  average of them has very little variance to remove.
* **resolution** -- the *risky* one.  It is the only axis here that makes
  open_clip interpolate the position embeddings, and the evidence is against it
  (a community report has CLIP-B/32 at 320 still losing to 224 after
  fine-tuning; ViTs do not transfer cleanly to unseen patch grids).

The recipe search is cached, but the first ``--tta`` run with all three crop
policies is ~36 frozen passes (sizes x crops x 2 flips x 2 splits) -- roughly
half an hour on a 4090, once.  ``--tta-crops center full`` halves it.

Usage::

    python probe.py --selftest
    python probe.py --data /root/autodl-tmp/train --test /root/autodl-tmp/test
    python probe.py --data ... --skip-gate --skip-test    # rho + montage only, fastest
    python probe.py --data ... --tta                      # + the inference recipe search
"""
import argparse
import atexit
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFile, ImageFont
from torch.utils.data import DataLoader, Dataset

from train import (CROP_POLICIES, IMG_EXTS, ImageFolderNoisy, build_clip, check_backbone,
                   eval_transform)

ImageFile.LOAD_TRUNCATED_IMAGES = True

_LINES = []


def say(msg=''):
    """Print and remember, so the report on disk matches what was on screen."""
    print(msg, flush=True)
    _LINES.append(msg)


def hr(title):
    say()
    say(f'{"=" * 4} {title} ' + '=' * max(0, 60 - len(title)))


def _write_report():
    """Also installed via atexit: the feature caches make a crash cheap to
    recover from, but a *lost report* is not, because the forward pass behind it
    took 20+ minutes.  Whatever was printed must survive the traceback."""
    p = globals().get('_REPORT')
    if p is not None and _LINES:
        try:
            Path(p).write_text('\n'.join(_LINES) + '\n', encoding='utf-8')
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class FlatImages(Dataset):
    """Flat ``test/*.jpg`` folder.  Mirrors infer.py's grey fallback: a file
    that Pillow cannot decode must still produce a row, or every statistic
    computed over the set is silently computed over a different set."""

    def __init__(self, root, transform):
        self.paths = sorted(str(p) for p in Path(root).rglob('*')
                            if p.is_file() and p.suffix.lower() in IMG_EXTS)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (127, 127, 127))
        return self.transform(img), i


# The eval transform lives in train.py so that train / infer / valmetrics /
# analyze / probe all describe the same pixels.  A drift here is invisible and
# only shows up as a checkpoint's val_acc mysteriously failing to reproduce.
transform_for = eval_transform


def set_transform(ds, tf):
    """Set the transform on a dataset *or* a Subset wrapping one.

    ``Subset`` has no transform of its own, so ``subset.transform = tf`` would
    silently create an unused attribute and every resolution in the gate would
    quietly be evaluated at 224 -- a bug that produces plausible-looking numbers
    rather than an error.  Hence the explicit unwrap.
    """
    if isinstance(ds, torch.utils.data.Subset):
        set_transform(ds.dataset, tf)
    else:
        ds.transform = tf


def paths_of(ds):
    """Item paths of a dataset or Subset, in dataset order."""
    if isinstance(ds, torch.utils.data.Subset):
        inner = paths_of(ds.dataset)
        return [inner[i] for i in ds.indices]
    if hasattr(ds, 'items'):
        return [p for p, _ in ds.items]
    return list(getattr(ds, 'paths', []))


def batch_for(size, base):
    """Keep the token count (and so the memory) roughly constant across sizes."""
    return max(16, int(round(base * (224 * 224) / float(size * size))))


def build_split_datasets(a):
    """The same stratified split train.py builds -- same seed, same ratio, so a
    val item here is the same val item the model was scored on."""
    dummy = transform_for(224)
    tr = ImageFolderNoisy(a.data, dummy, False, a.val_ratio, a.seed, 'train')
    va = ImageFolderNoisy(a.data, dummy, True, a.val_ratio, a.seed, 'val')
    assert len(tr) and len(va), f'no images under {a.data}'
    return tr, va


# --------------------------------------------------------------------------- #
# frozen-CLIP features
# --------------------------------------------------------------------------- #
@torch.no_grad()
def view_sfx(crop, flip):
    """Cache-name suffix for a view.  Empty for the default view, so feature
    caches written before crop policies existed stay valid."""
    bits = []
    if crop != 'center':
        bits.append(crop)
    if flip:
        bits.append('flip')
    return ''.join(f'.{b}' for b in bits)


def feat_cache(out_dir, tag, size, crop='center', flip=False):
    """Where a view's features live.

    Shared by ``extract`` (which writes it) and the recipe search (which reads
    it back by name).  When those two built the name separately they could drift
    apart, and the symptom would be a *stale* feature file being read as if it
    belonged to another view -- silently, and with a plausible-looking number.
    """
    return Path(out_dir) / f'feat_{tag}_{size}{view_sfx(crop, flip)}.npz'


@torch.no_grad()
def extract(clip_model, ds, size, device, bs, workers, tag, out_dir, limit=None,
            flip=False, crop='center'):
    """One frozen-CLIP pass.  Cached per (tag, size, crop, flip) and written
    immediately, so a crash in the analysis below never costs the pass again."""
    sfx = view_sfx(crop, flip)
    cache = feat_cache(out_dir, tag, size, crop, flip)
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        say(f'[cache] {tag:5s} @{size:<4d}{sfx:>8s} {d["f"].shape[0]:>7d} images')
        return {k: d[k] for k in d.files}

    set_transform(ds, transform_for(size, flip, crop=crop))
    n_all = len(ds)
    n = n_all if limit is None else min(limit, n_all)
    if limit is not None:
        ds = torch.utils.data.Subset(ds, list(range(n)))
    loader = DataLoader(ds, batch_size=batch_for(size, bs), shuffle=False,
                        num_workers=workers, pin_memory=True)

    dim = getattr(clip_model.visual, 'output_dim', 512)   # ViT-B/32 is always 512
    f = np.zeros((n, dim), dtype=np.float16)
    y = np.zeros(n, dtype=np.int64)
    idx_all = np.zeros(n, dtype=np.int64)
    t0, seen = time.time(), 0
    for batch in loader:
        if len(batch) == 3:
            x, lbl, idx = batch
        else:
            x, lbl, idx = batch[0], None, batch[1]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == 'cuda'):
            z = clip_model.encode_image(x.to(device, non_blocking=True))
        z = F.normalize(z.float(), dim=-1)
        rows = idx.numpy()
        f[rows] = z.half().cpu().numpy()
        idx_all[rows] = rows
        if lbl is not None:
            y[rows] = lbl.numpy()
        seen += len(rows)
        if seen % (batch_for(size, bs) * 20) < batch_for(size, bs):
            el = time.time() - t0
            say(f'  [{tag}@{size}] {seen}/{n} imgs  {seen / max(el, 1e-9):.0f}/s  '
                f'eta {(n - seen) / max(seen / max(el, 1e-9), 1e-9) / 60:.1f} min')

    paths = np.asarray(paths_of(ds))
    say(f'[{tag:5s} @{size}] {n} images in {(time.time() - t0) / 60:.1f} min  '
        f'({(time.time() - t0) / max(n, 1) * 1000:.1f} ms/img)')
    np.savez(cache, f=f, y=y, paths=paths)
    return dict(f=f, y=y, paths=paths)


# --------------------------------------------------------------------------- #
# the measurement itself
# --------------------------------------------------------------------------- #
@torch.no_grad()
def class_sums(f, y, nclass):
    """Per-class sums and counts of L2-normalised features -- the ingredients of
    a nearest-class-mean centroid, kept separate so that one split's centroids
    can be applied to a *different* split."""
    f = F.normalize(f.float(), dim=-1)
    S = torch.zeros(nclass, f.shape[1], device=f.device, dtype=f.dtype)
    n = torch.zeros(nclass, device=f.device, dtype=f.dtype)
    S.index_add_(0, y, f)
    n.index_add_(0, y, torch.ones_like(y, dtype=f.dtype))
    return S, n


@torch.no_grad()
def ncc_scores(f, y, nclass, ref=None):
    """Nearest-class-mean scores for every sample.

    ``s_self``   cosine to the centroid of the given class
    ``s_other``  cosine to the best-scoring *other* class
    ``margin``   ``s_self - s_other``.  Clearly negative means an independent
                 semantic judge prefers a different label to the given one.

    ``ref``      centroids as ``(S, n)`` from :func:`class_sums` on a *different*
                 split.  The two modes are not interchangeable:

                 * ``ref=None`` -- centroids are built from ``f`` itself and the
                   given-class one is **leave-one-out**.  Required there: a
                   self-inclusive mean lets the noisiest image in a class score
                   ~1.0 against itself, making it look like the cleanest.
                 * ``ref=(S, n)`` -- centroids come from the reference split, so
                   the sample is not part of its own centroid and no correction
                   is needed.  This cross-fitting is what keeps a val score
                   independent of the val labels -- the reason ``val`` and
                   ``train`` are scored in different modes.
    """
    f = F.normalize(f.float(), dim=-1)
    ar = torch.arange(len(y), device=f.device)
    if ref is None:
        S, n = class_sums(f, y, nclass)
        loo = True
    else:
        S, n = ref
        loo = False

    sim = f @ F.normalize(S, dim=-1).t()                    # (N, C)
    s_self = sim.gather(1, y[:, None]).squeeze(1)
    if loo:
        s_loo = (f * F.normalize(S[y] - f, dim=-1)).sum(-1)
        lone = n[y] <= 1                                    # LOO undefined
        s_self = torch.where(lone, s_self, s_loo)

    sim[ar, y] = float('-inf')
    s_other, other = sim.max(1)
    sim[ar, y] = s_self                                     # corrected vector
    return dict(sim=sim, s_self=s_self, s_other=s_other, other=other,
                pred=sim.argmax(1), margin=s_self - s_other, top1=sim.max(1).values)


def q(x, *ps):
    """Percentiles of a 1-D tensor, as a compact string."""
    x = x.float()
    vals = torch.quantile(x, torch.tensor(ps, device=x.device))
    return ' '.join(f'p{int(p * 100)}={v:.3f}' for p, v in zip(ps, vals.tolist()))


def format_hist(counts, edges, width=44):
    """A tiny horizontal bar chart, for pasting into a chat window."""
    peak = max(counts) or 1
    out = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = '#' * max(1, int(width * c / peak)) if c else ''
        out.append(f'    [{lo:+.2f},{hi:+.2f}) {c:>7d} {bar}')
    return '\n'.join(out)


# --------------------------------------------------------------------------- #
# montage -- the ground-truth check
# --------------------------------------------------------------------------- #
def _font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def montage(paths, labels, preds, margins, out, cols=10, cell=160, title=''):
    """Tile images with `given->ncc-predicted` drawn on each.  The labels are
    the whole point: an off-concept image shows a *large* semantic distance to
    its folder, so the interesting thing to look at is whether these really are
    water heaters and logos, or just ordinary hard examples."""
    rows = (len(paths) + cols - 1) // cols
    bar = 60
    img = Image.new('RGB', (cols * cell, rows * (cell + bar)), (20, 20, 24))
    dr = ImageDraw.Draw(img)
    f1, f2 = _font(max(11, cell // 12)), _font(max(10, cell // 14))
    for k, (p, y, pr, m) in enumerate(zip(paths, labels, preds, margins)):
        r, c = divmod(k, cols)
        x0, y0 = c * cell, r * (cell + bar)
        try:
            im = Image.open(p).convert('RGB')
            im.thumbnail((cell - 4, cell - 4))
            img.paste(im, (x0 + (cell - im.width) // 2, y0 + (cell - im.height) // 2))
        except Exception:
            dr.rectangle([x0 + 2, y0 + 2, x0 + cell - 3, y0 + cell - 3], fill=(90, 30, 30))
        above = int(y) != int(pr)
        dr.text((x0 + 3, y0 + cell + 2), f'{int(y):04d} -> {int(pr):04d}',
                fill=(255, 120, 120) if above else (170, 230, 170), font=f1)
        dr.text((x0 + 3, y0 + cell + 4 + cell // 12), f'm={float(m):+.2f}', fill=(200, 200, 200), font=f2)
    if title:
        dr.text((4, 4), title, fill=(255, 255, 255), font=f1)
    img.save(out)
    say(f'wrote {out}  ({cols}x{rows} tiles, {len(paths)} images)')


# --------------------------------------------------------------------------- #
# selftest -- validates the maths without data, GPU or weights
# --------------------------------------------------------------------------- #
def selftest():
    """Synthetic check.  The scorer is the load-bearing code here and it cannot
    be exercised on the real data without a GPU, so it is exercised on features
    whose answer we know: well-separated clusters, a known fraction of which
    have been replaced by random vectors (the analogue of keyword noise)."""
    say('selftest: synthetic features, no data / GPU / weights needed')
    torch.manual_seed(0)
    nclass, per, dim, noise = 20, 60, 64, 0.10
    cen = F.normalize(torch.randn(nclass, dim), dim=-1)
    y = torch.arange(nclass).repeat_interleave(per)
    # tight clusters: 0.1 of noise leaves the own-class cosine (~0.78) far above
    # the best cross-class cosine (~0.25), so clean data is separable and any
    # error the scorer reports is its own fault rather than the construction's
    clean = F.normalize(cen[y] + 0.1 * torch.randn(len(y), dim), dim=-1)
    # corrupt a random subset spread across ALL classes, not a prefix -- a prefix
    # would wipe out whole classes and destroy the very centroids being tested
    nbad = int(len(y) * noise)
    bad = torch.randperm(len(y))[:nbad]
    f = clean.clone()
    f[bad] = F.normalize(clean[bad] + 1.2 * torch.randn(nbad, dim), dim=-1)

    r = ncc_scores(f, y, nclass)
    acc = float((r['pred'] == y).float().mean())
    rho = float((r['margin'] < 0).float().mean())
    say(f'  injected noise          : {noise:.2f} of {len(y)} samples')
    say(f'  LOO-NCC accuracy        : {acc:.3f} (expect ~{1 - noise:.2f}: the '
        f'corrupted ones should be the only errors)')
    say(f'  flagged (margin < 0)    : {rho:.3f} (expect near {noise:.2f})')

    # The leave-one-out correction is the one thing here that is easy to get
    # wrong *silently*, so it gets a case with a known answer.  Three samples in
    # one class; two of them agree, the third is an outlier.  Without LOO the
    # outlier is part of the centroid it is compared against and scores ~1.0 --
    # i.e. it certifies itself as the cleanest possible member of its class,
    # which is exactly backwards for the purpose of flagging noise.
    lone_y = torch.tensor([0, 0, 0])
    f_lone = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])   # two agree, one is 90 deg off
    rl = ncc_scores(f_lone, lone_y, 1)
    no_loo = float((f_lone[2] * F.normalize(f_lone.mean(0, keepdim=True), dim=-1)[0]).sum())
    loo = float(rl['s_self'][2])
    say(f'  the odd one out, score for its OWN class:')
    say(f'    non-LOO (centroid contains it) {no_loo:.3f}  ->  LOO {loo:.3f}')
    say(f'    a self-inclusive mean credits it with {no_loo:.2f} it has not earned;')
    say(f'    dropping it from its own centroid is what makes the score mean anything')

    ok = (acc > 0.85 and 0.4 * noise < rho < 4 * noise
          and loo < 0.05 and no_loo > 0.4 and loo < no_loo - 0.3)

    # montage plumbing, on real (synthetic) images so the paste path is exercised
    # rather than just the decode-failure fallback
    d = Path('probe_selftest_tmp')
    d.mkdir(exist_ok=True)
    tmp = d / 'montage.png'
    try:
        paths = []
        for i in range(6):
            p = d / f'{i}.png'
            Image.new('RGB', (90 + i, 70 + i), (40 * i % 256, 90, 150)).save(p)
            paths.append(str(p))
        montage(paths, [0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2],
                [0.1, -0.4, 0.2, -0.1, -0.9, 0.3], tmp, cols=3, cell=64, title='selftest')
        ok = ok and tmp.exists() and tmp.stat().st_size > 0
    finally:
        for p in list(d.glob('*')):
            p.unlink()
        if d.exists():
            d.rmdir()
    say(f'selftest: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(a):
    if a.selftest:
        return selftest()

    check_backbone(a.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in a.sizes]
    globals()['_REPORT'] = out / 'report.txt'
    atexit.register(_write_report)

    say(f'probe.py  device={device}  data={a.data}  out={out}')
    if a.limit:
        say(f'!! --limit {a.limit} -- numbers below are NOT meaningful')

    t_start = time.time()
    clip_model = build_clip(a.model, a.pretrained, sizes[0]).to(device).eval()

    tr, va = build_split_datasets(a)
    say(f'split: train={len(tr)} val={len(va)} classes={len(tr.class_to_idx)} '
        f'(same seed/ratio as train.py)')

    hr('FEATURES')
    Ftr = extract(clip_model, tr, sizes[0], device, a.batch_size, a.workers, 'train', out, a.limit)
    Fva = extract(clip_model, va, sizes[0], device, a.batch_size, a.workers, 'val', out, a.limit)

    y_tr = torch.as_tensor(Ftr['y'], device=device)
    y_va = torch.as_tensor(Fva['y'], device=device)
    f_tr = torch.as_tensor(Ftr['f'].astype(np.float32), device=device)
    f_va = torch.as_tensor(Fva['f'].astype(np.float32), device=device)
    nclass = int(max(y_tr.max(), y_va.max())) + 1
    assert nclass >= 2, (f'only {nclass} class(es) present -- raise --limit; '
                         f'nearest-class-mean needs at least two to mean anything')
    say(f'classes present: {nclass}   feature dim: {f_tr.shape[1]}')
    say(f'norm check (should be 1.000): train {f_tr.norm(dim=-1).mean():.4f} '
        f'val {f_va.norm(dim=-1).mean():.4f}')

    cnt = torch.bincount(y_tr, minlength=nclass)
    say(f'train per-class n: min {int(cnt.min())} median {int(cnt.float().median())} '
        f'max {int(cnt.max())}')

    # ---- rho, measured on the train split ---------------------------------- #
    hr('LABEL NOISE ESTIMATE (rho)')
    Str, ntr = class_sums(f_tr, y_tr, nclass)
    rtr = ncc_scores(f_tr, y_tr, nclass)                 # train vs itself -> LOO
    rva = ncc_scores(f_va, y_va, nclass, ref=(Str, ntr))  # val vs TRAIN centroids
    # val is deliberately scored against *train* centroids: no val label ever
    # touches the judge, so the verdict on val is an independent opinion rather
    # than a restatement.  It still cannot *confirm* the noise rate -- val carries
    # the same noise as train -- the level of both columns is the evidence, not
    # their gap.
    say('rho(tau) = fraction of images where an independent judge (frozen-CLIP')
    say('LOO nearest-class-mean) ranks a DIFFERENT class above the given one by')
    say('more than tau.  This is an UPPER BOUND on the true noise rate: ordinary')
    say('fine-grained confusion between two similar species counts toward it too.')
    say()
    say('   tau     rho(train)        n     rho(val)')
    for tau in (0.0, 0.02, 0.05, 0.10):
        say(f'  {tau:+.2f}      {(rtr["margin"] < -tau).float().mean():.3f}   '
            f'{int((rtr["margin"] < -tau).sum()):>8d}       '
            f'{(rva["margin"] < -tau).float().mean():.3f}')
    say()
    say('margin distribution (train).  A clean dataset puts almost everything right')
    say('of 0; keyword noise shows up as the left tail.  Values are clamped to the')
    say('plotted range, so the counts still sum to the training-set size:')
    lo, hi, nb = -0.5, 0.5, 10
    edges = [lo + (hi - lo) * i / nb for i in range(nb + 1)]
    h = torch.histc(rtr['margin'].float().clamp(lo, hi), bins=nb, min=lo, max=hi)
    say(format_hist([int(v) for v in h.tolist()], edges))

    # orphan: unexplained by ANY class.  This is the sharper statistic -- a
    # fine-grained confusion still lands near *some* class, a water heater does not.
    say()
    say('orphans -- max cosine to ANY class centroid below x (train / val):')
    for th in (0.30, 0.40, 0.50, 0.60):
        say(f'  < {th:.2f}:  train {(rtr["top1"] < th).float().mean():.4f}   '
            f'val {(rva["top1"] < th).float().mean():.4f}')

    # per-class concentration: a keyword with a dominant wrong sense shows up as
    # one class carrying most of the flagged samples
    per_cls = torch.zeros(nclass, device=device)
    per_cls.index_add_(0, y_tr, (rtr['margin'] < 0).float())
    rho_cls = per_cls / cnt.clamp_min(1)
    order = rho_cls.argsort(descending=True)
    rev = {v: k for k, v in tr.class_to_idx.items()}
    say()
    say('worst classes by rho (folder, rho, n, n_flagged):')
    for i in order[:10].tolist():
        say(f'    {rev[i]}  rho={float(rho_cls[i]):.2f}  n={int(cnt[i]):>4d}  '
            f'n_flag={int(per_cls[i]):>4d}')
    say('cleanest classes:')
    for i in order[-5:].tolist():
        say(f'    {rev[i]}  rho={float(rho_cls[i]):.2f}  n={int(cnt[i]):>4d}')

    # ---- V* ---------------------------------------------------------------- #
    hr('FROZEN-CLIP NCC REFERENCE AND V*')
    acc_va = float((rva['pred'] == y_va).float().mean())
    say(f'frozen-CLIP LOO-NCC accuracy on val @{sizes[0]}: {acc_va:.4f}')
    say(f'  (for reference: the trained model reached val_acc 0.7070 / test 0.6416)')
    for tau in (0.0, 0.02, 0.05):
        keep = (rva['pred'] == y_va) & (rva['margin'] >= tau)
        say(f'  V*(tau={tau:.2f}) = {int(keep.sum()):>6d} / {len(y_va)} '
            f'({100 * float(keep.float().mean()):.1f}% of val)')
    keep = (rva['pred'] == y_va) & (rva['margin'] >= a.vstar_margin)
    say()
    say(f'-> using tau={a.vstar_margin}. V* is the val subset whose given label an')
    say('   independent judge agrees with. acc@V* is the selection signal to trust.')

    np.savez(out / 'vstar.npz',
             paths=Fva['paths'], y=Fva['y'],
             vstar=keep.cpu().numpy(),
             ncc_pred=rva['pred'].cpu().numpy(),
             margin=rva['margin'].cpu().numpy(),
             top1=rva['top1'].cpu().numpy(),
             size=sizes[0], val_ratio=a.val_ratio, seed=a.seed)

    # ---- domain shift ------------------------------------------------------- #
    hr('DOMAIN SHIFT (val vs test, label-free)')
    if a.skip_test:
        say('skipped (--skip-test)')
    else:
        Fte = extract(clip_model, FlatImages(a.test, transform_for(sizes[0])), sizes[0],
                      device, a.batch_size, a.workers, 'test', out, a.limit)
        f_te = torch.as_tensor(Fte['f'].astype(np.float32), device=device)
        say(f'test images: {f_te.shape[0]}')
        cen = F.normalize(Str, dim=-1)
        for name, ff in (('val', f_va), ('test', f_te)):
            s = F.normalize(ff, dim=-1) @ cen.t()
            mx = s.max(1).values
            ent = -(s.softmax(-1) * s.softmax(-1).clamp_min(1e-12).log()).sum(-1)
            say(f'  {name:4s}  max_cos {q(mx, .1, .5, .9)}   '
                f'entropy {q(ent, .1, .5, .9)}   (n={ff.shape[0]})')
        say('  same distribution => no measurable feature-space shift => the gap is')
        say('  something training did, not "the test set is drawn differently".')
        # predicted class histogram: the test set is balanced, so a wildly
        # non-uniform histogram is the model's bias, not the data's
        pv = (F.normalize(f_va, dim=-1) @ cen.t()).argmax(1)
        pt = (F.normalize(f_te, dim=-1) @ cen.t()).argmax(1)
        hv = torch.bincount(pv, minlength=nclass).float()
        ht = torch.bincount(pt, minlength=nclass).float()
        say(f'  predicted-class histogram (frozen NCC, balanced test set expected):')
        say(f'    val  min {int(hv.min())} max {int(hv.max())} '
            f'(max/mean {float(hv.max() / hv.mean().clamp_min(1)):.1f}x)')
        say(f'    test min {int(ht.min())} max {int(ht.max())} '
            f'(max/mean {float(ht.max() / ht.mean().clamp_min(1)):.1f}x)')

    # ---- resolution gate ---------------------------------------------------- #
    hr('RESOLUTION GATE (does more resolution separate the classes better?)')
    # hoisted: both the gate and the TTA recipe need these centroids, and either
    # section may be requested without the other
    sub = _per_class_subset(tr, a.gate_per_class, nclass, a.seed)
    if a.skip_gate:
        say('skipped (--skip-gate)')
    else:
        say(f'every size is measured identically -- centroids from the same '
            f'{a.gate_per_class}/class subsample of train, scored on the whole val')
        say('split -- so the columns may be compared.  The signals that mean')
        say('something are ncc_acc and mean_margin; a size that raises neither is')
        say('not worth the extra hours.  Note acc@V* is deliberately NOT here: V*')
        say('is *defined* by this judge, so the judge would score ~1.00 on it by')
        say('construction.  acc@V* is a number for a trained model, and it is')
        say('valmetrics.py that computes it.')
        say()
        say('   size   ncc_acc(val)   mean_margin   orphan<0.4      |V*|   kept vs 224')
        base_keep = None
        for size in sizes:
            if size == sizes[0]:
                # the 224 val features are the ones already extracted above, from
                # the identical transform -- reuse them rather than re-run 15k
                cl, f_v, y_v = clip_model, f_va, y_va
            else:
                cl = build_clip(a.model, a.pretrained, size).to(device).eval()
                Fv = extract(cl, va, size, device, a.batch_size, a.workers,
                             f'val{size}', out, None)
                y_v = torch.as_tensor(Fv['y'], device=device)
                f_v = torch.as_tensor(Fv['f'].astype(np.float32), device=device)
            # centroids from a per-class subsample of TRAIN.  Not from val: with
            # ~18 val images per class the centroid noise would swamp the very
            # effect being measured, which is only a point or two.
            Fs = extract(cl, sub, size, device, a.batch_size, a.workers,
                         f'gate{size}', out, None)
            Ss, ns = class_sums(
                torch.as_tensor(Fs['f'].astype(np.float32), device=device),
                torch.as_tensor(Fs['y'], device=device), nclass)
            rr = ncc_scores(f_v, y_v, nclass, ref=(Ss, ns))
            keep_v = (rr['pred'] == y_v) & (rr['margin'] >= a.vstar_margin)
            ov = '' if base_keep is None else (
                f'{100 * float((keep_v & base_keep).sum()) / max(int(keep_v.sum()), 1):.1f}%')
            base_keep = keep_v if base_keep is None else base_keep
            say(f'  {size:>5d}   {float((rr["pred"] == y_v).float().mean()):>9.4f}   '
                f'{float(rr["margin"].mean()):>+10.3f}   '
                f'{float((rr["top1"] < 0.4).float().mean()):>9.4f}   '
                f'{int(keep_v.sum()):>7d}   {ov}')
            if size != sizes[0]:
                del cl
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        say()
        say('  "kept vs 224" = how much of 224\'s V* survives at this size.  High')
        say('  overlap means resolution changes how *confidently* classes separate,')
        say('  not WHICH images look noisy -- i.e. V* is a stable artefact.')

    # ---- TTA / resolution recipe -------------------------------------------- #
    if a.tta:
        hr('TTA AND RESOLUTION RECIPE (what infer.py should actually do)')
        say('Every lever here is inference-only: the same weights are read on a')
        say('flipped copy, on a differently-cropped copy, and/or at a different')
        say('resolution (position embeddings interpolated by open_clip), and the')
        say('resulting features are averaged.  No retraining, so this costs no')
        say('training budget and does not compete with the noise/capacity axes --')
        say('which is exactly why it is worth measuring first.')
        say()
        say('The two axes are NOT equally risky.  The crop policies only change')
        say('which pixels of the one image reach the model and stay inside the')
        say('trained patch grid; a resolution change has to interpolate the')
        say('position embeddings, and the reported evidence for that is poor (a')
        say('community report has CLIP-B/32 at 320 losing to 224 even after')
        say('fine-tuning).  So the crop rows are listed at the base size first,')
        say('separately, and they are the ones to trust if they disagree.')
        say()
        say('Each row averages the L2-normalised features of every listed view for')
        say('BOTH the train centroids and val, so both sides are built identically.')
        say(f'The "{sizes[0]} center (id)" row must reproduce the gate table above --')
        say('if it does not, view handling is inconsistent and no row here can be')
        say('trusted.')
        say()

        crops = list(dict.fromkeys(a.tta_crops)) or ['center']
        for c in crops:
            assert c in CROP_POLICIES, f'--tta-crops {c!r}, expected one of {CROP_POLICIES}'

        def vtag(split, size, crop, flip):
            return (f'{"gate" if split == "tr" else "val"}{size}'
                    f'{view_sfx(crop, flip).replace(".", "")}')

        # warm the cache for every view any recipe needs; cached ones are free
        for size in sizes:
            cl = (clip_model if size == sizes[0]
                  else build_clip(a.model, a.pretrained, size).to(device).eval())
            for split in ('tr', 'va'):
                for crop in crops:
                    for flip in (False, True):
                        extract(cl, sub if split == 'tr' else va, size, device,
                                a.batch_size, a.workers, vtag(split, size, crop, flip),
                                out, None, flip, crop)
            if size != sizes[0]:
                del cl
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        def combo(split, views):
            """Average the L2-normalised features of ``views``.

            Each view is a ``(size, crop, flip)`` triple.  Both sides -- the
            train centroids and val -- are built from the *same* view list, so a
            row here compares like with like.
            """
            acc = None
            for size, crop, flip in views:
                d = np.load(feat_cache(out, vtag(split, size, crop, flip), size, crop, flip),
                            allow_pickle=True)
                f = F.normalize(torch.as_tensor(d['f'].astype(np.float32),
                                                device=device), dim=-1)
                acc = f if acc is None else acc + f
            y = torch.as_tensor(d['y'], device=device)
            return F.normalize(acc, dim=-1), y

        b = sizes[0]
        # The crop axis is measured at the base size FIRST, on its own: it needs
        # no positional-embedding interpolation, so it is the cheap, safe lever,
        # and scoring it alongside the resolution axis would leave it unclear
        # which of the two moved the number.
        combos = [(f'{b} center (id)', ((b, 'center', False),)),
                  (f'{b} center (id+flip)', ((b, 'center', False), (b, 'center', True)))]
        for crop in [c for c in crops if c != 'center']:
            combos += [(f'{b} {crop} (id)', ((b, crop, False),)),
                       (f'{b} {crop} (id+flip)', ((b, crop, False), (b, crop, True))),
                       (f'{b} center+{crop}', ((b, 'center', False), (b, crop, False))),
                       (f'{b} center+{crop} (id+flip)',
                        ((b, 'center', False), (b, 'center', True),
                         (b, crop, False), (b, crop, True)))]
        if len(crops) > 1:
            combos.append((f'{b} ALL crops (id+flip)',
                           tuple((b, c, f) for c in crops for f in (False, True))))
        for s in sizes[1:]:
            combos += [(f'{s} (id)', ((s, 'center', False),)),
                       (f'{b}+{s} (id)', ((b, 'center', False), (s, 'center', False))),
                       (f'{b}+{s} (id+flip)', ((b, 'center', False), (b, 'center', True),
                                               (s, 'center', False), (s, 'center', True)))]
        combos.append(('+'.join(str(s) for s in sizes) + ' (id+flip)',
                       tuple((s, 'center', f) for s in sizes for f in (False, True))))
        say('   recipe                            ncc_acc(val)   mean_margin   flagged')
        best = None
        for name, views in combos:
            fc, yc = combo('tr', views)
            fv, yv = combo('va', views)
            S, n = class_sums(fc, yc, nclass)
            rr = ncc_scores(fv, yv, nclass, ref=(S, n))
            acc = float((rr['pred'] == yv).float().mean())
            mark = ''
            if best is None or acc > best[1]:
                best, mark = (name, acc), '   <- best so far'
            say(f'   {name:<32s} {acc:>9.4f}   {float(rr["margin"].mean()):>+10.3f}   '
                f'{float((rr["margin"] < 0).float().mean()):>7.3f}{mark}')
        say()
        say(f'   best frozen recipe: {best[0]}  ({best[1]:.4f})')
        say('   This is a RECIPE, not a score: the frozen classifier is far weaker')
        say('   than the trained one, so the *ranking* of recipes is the signal and')
        say('   the level is not.  If nothing beats "[base] center (id)" here,')
        say('   infer.py should stay on the plain single view too.')
        say('   Expect flip to move this very little: CLIP image features are about')
        say('   0.99 cosine-similar under a flip, so the two views are nearly')
        say('   redundant and there is hardly any variance for the average to')
        say('   remove.  A crop policy is the one that could plausibly matter,')
        say('   because it can recover image content the centre crop discards.')

    # ---- montages ----------------------------------------------------------- #
    hr('MONTAGE (the ground-truth check -- LOOK AT THESE)')
    yl = y_tr.cpu().tolist()
    mg = rtr['margin'].cpu().tolist()
    pr = rtr['pred'].cpu().tolist()

    def panel(idx, name, title):
        montage([str(Ftr['paths'][i]) for i in idx], [yl[i] for i in idx],
                [pr[i] for i in idx], [mg[i] for i in idx], out / name,
                cols=a.montage_cols, title=title)

    # 1. the most extreme individual violations, wherever they live
    worst = rtr['margin'].argsort()[:a.topk].cpu().tolist()
    panel(worst, 'noise_montage.png',
          f'LOWEST {len(worst)} MARGINS of {len(yl)} (frozen ncc_acc={acc_va:.3f})')

    # 2. one per class, worst class first.  This is the panel that actually tests
    # the claim: if the noise comes from polysemous *keywords*, breakage should be
    # concentrated in a few folders (whose whole vocabulary is off-concept) rather
    # than spread evenly, and each tile should be recognisably not that species.
    seen, per_cls = set(), []
    for i in rtr['margin'].argsort().cpu().tolist():
        c = yl[i]
        if c not in seen:
            seen.add(c)
            per_cls.append(i)
            if len(per_cls) >= a.topk:
                break
    panel(per_cls, 'noise_byclass_montage.png',
          f'WORST IMAGE OF EACH OF {len(per_cls)} CLASSES, worst class first')

    # 3. control, so a wrong idea about "what suspicious looks like" is visible
    g = torch.Generator().manual_seed(a.seed)
    rand = torch.randperm(len(yl), generator=g)[:a.topk].tolist()
    panel(rand, 'control_montage.png', 'RANDOM CONTROL')

    np.savez(out / 'trainscores.npz',
             paths=Ftr['paths'], y=Ftr['y'],
             s_self=rtr['s_self'].cpu().numpy(), s_other=rtr['s_other'].cpu().numpy(),
             margin=rtr['margin'].cpu().numpy(), ncc_pred=rtr['pred'].cpu().numpy(),
             top1=rtr['top1'].cpu().numpy(),
             size=sizes[0], val_ratio=a.val_ratio, seed=a.seed)

    hr('DONE')
    say(f'total {(time.time() - t_start) / 60:.1f} min')
    say('next: download noise_montage.png and LOOK at it, then run valmetrics.py')
    _write_report()
    say(f'report saved to {out / "report.txt"}')
    return 0


def _per_class_subset(ds, per_class, nclass, seed):
    """A fixed-size subsample of the train split, `per_class` per class."""
    by = {}
    for i, (_, y) in enumerate(ds.items):
        by.setdefault(y, []).append(i)
    rng = np.random.RandomState(seed)
    keep = []
    for c in range(nclass):
        idx = by.get(c, [])
        keep.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    return torch.utils.data.Subset(ds, sorted(keep))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', default='/root/autodl-tmp/train', help='training folder (class subdirs)')
    p.add_argument('--test', default='/root/autodl-tmp/test', help='flat test folder')
    p.add_argument('--out', default='./probe_out')
    p.add_argument('--model', default='ViT-B-32-quickgelu')
    p.add_argument('--pretrained', default='openai')
    p.add_argument('--sizes', nargs='+', default=[224, 288, 320],
                   help='first entry is the main resolution; the rest are the gate. '
                        'Keep every entry a multiple of the 32px patch size: 336 is NOT '
                        '(336/32 = 10.5), so it would just measure 320 plus a dead margin. '
                        'train.build_clip warns about such sizes.')
    p.add_argument('--val-ratio', type=float, default=0.1, help='must match train.py')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--batch-size', type=int, default=256, help='at 224; scaled down for larger sizes')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--vstar-margin', type=float, default=0.02)
    p.add_argument('--gate-per-class', type=int, default=40,
                   help='train images per class used to build gate centroids')
    p.add_argument('--topk', type=int, default=60,
                   help='tiles per montage. Kept small enough that the labels stay '
                        'legible in a downscaled view -- the montage is only useful '
                        'if it can actually be read')
    p.add_argument('--montage-cols', type=int, default=6)
    p.add_argument('--limit', type=int, default=0, help='smoke test: only N images per split')
    p.add_argument('--skip-gate', action='store_true')
    p.add_argument('--skip-test', action='store_true')
    p.add_argument('--tta', action='store_true',
                   help='also extract flipped and alternative-crop views, then score '
                        'multi-view recipes. Costs len(--sizes)*len(--tta-crops)*4 '
                        'extra passes over the train subsample and val; everything is '
                        'cached, so re-runs are free')
    p.add_argument('--tta-crops', nargs='*', default=['center', 'full', 'pad'],
                   dest='tta_crops', metavar='POLICY',
                   help='crop policies to score, from train.CROP_POLICIES. Defaults to '
                        'all three so the measurement actually happens: the centre '
                        'crop (the current default) discards a fixed fraction of every '
                        'frame, and on this dataset that is over a third of the height '
                        'of a 3:4 image. All policies stay inside the trained patch '
                        'grid, so unlike --sizes they involve no '
                        'positional-embedding interpolation')
    p.add_argument('--selftest', action='store_true', help='synthetic check, no data/GPU needed')
    a = p.parse_args(argv)
    a.limit = a.limit or None
    return a


if __name__ == '__main__':
    raise SystemExit(main(parse_args()))
