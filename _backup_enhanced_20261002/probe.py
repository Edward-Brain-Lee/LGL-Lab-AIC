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
    python probe.py --data ... --calib --checkpoint ep4.pt ep20.pt   # calibrate the proxy
    python probe.py --data ... --calib --calib-sweep --checkpoint best.pt
"""
import argparse
import hashlib
import atexit
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFile, ImageFont
from torch.utils.data import DataLoader, Dataset

from train import (CROP_POLICIES, IMG_EXTS, TRAINING_VIEW, VIEW_LEN, VIEW_RATIOS,
                   VIEW_SETS, VIEW_SETS_SWEPT, FlatImages, ImageFolderNoisy, Net,
                   build_clip, check_backbone, check_view_tuple, ck_image_size,
                   eval_transform, is_degenerate, resolved_views, transform_size,
                   verify_pos_embed)

#: The view table's older name on this side.  The proxy has called it CALIB_VIEWS
#: since it was the only thing that scored views, and the report, the help text and
#: the self-test all say CALIB_VIEWS.  One table under two names, so a set added
#: for the submission path (`infer.py --tta-views`) is swept here without anyone
#: having to remember to add it in a second place.
CALIB_VIEWS = VIEW_SETS

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
# FlatImages is imported above rather than defined here: infer.py reads the same
# folder through the same grey fallback, and two copies of "sorted rglob +
# substitute grey" would be two places for the submission's row order to drift.


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


def _cache_matches(d, want_paths):
    """Was this cache file written for the pass we are about to run?

    Two things can make a feature cache the wrong one, and only one of them is in
    the file name.  The view is (`feat_val_224.flip.npz`); the **image list** is
    not, and `--limit`, `--val-ratio`, `--seed` and `--data` all change it.  So a
    16-image smoke run left a file that the next full run read back as if it were
    the whole split, and every number after that was computed over 16 images with
    nothing in the report saying so -- the silent wrong number this file spends
    most of its comments warning about.  The paths are already stored next to the
    features, so comparing them costs a list compare and closes the whole class
    rather than the one flag that was noticed first.
    """
    got = d['paths']
    if len(got) != len(want_paths):
        return False
    return bool(np.array_equal(np.asarray(got, dtype=object),
                               np.asarray(want_paths, dtype=object)))


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
    # The image list is the part a cache file name cannot carry, so it is resolved
    # before the cache is consulted: `--limit` truncates here, in the same order
    # the pass below will see.
    n_all = len(ds)
    n = n_all if limit is None else min(limit, n_all)
    if n != n_all:
        ds = torch.utils.data.Subset(ds, list(range(n)))
    want = np.asarray(paths_of(ds))
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if _cache_matches(d, want):
            say(f'[cache] {tag:5s} @{size:<4d}{sfx:>8s} {len(want):>7d} images')
            return {k: d[k] for k in d.files}
        say(f'[cache] {cache.name}: NOT reused -- {len(d["paths"])} image(s) in the '
            f'file, {n} in this pass.  (A different --limit / --val-ratio / --seed / '
            f'--data writes to the same name.)')

    set_transform(ds, transform_for(size, flip, crop=crop))
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

    paths = want
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
# the offline leaderboard proxy (--calib)
# --------------------------------------------------------------------------- #
# The table itself -- CALIB_VIEWS, TRAINING_VIEW, VIEW_LEN, check_view_tuple --
# lives in train.py, because the submission path selects from it too
# (`infer.py --tta-views`) and a second copy here would be a second place for
# "what tta4 means" to drift.  The definitions and the full (size, crop,
# ratio_name, flip) contract are there; this file imports them and keeps the name
# CALIB_VIEWS, which is what the report, the help text and the self-test say.
# `resolved_views` / `is_degenerate` (what a set collapses to at a given trained
# resolution) live there too, next to the table they are about.


#: What the platform actually reported for the sibling repository's submissions,
#: in points, on the 750-class 复赛 round.  Measurements, not estimates -- the
#: only ground truth the project has, which is why the proxy is graded on them
#: before anything is decided with it.
CALIB_TRUTH = {
    'epochs': 'plain ep4 -> ep20       +0.0002',
    'wide':   'plain      -> wide       -2.16',
    'tta4':   'plain      -> tta4       +2.09',
    'valacc': 'val_acc on ep4 -> ep20   +5.02   <- must FAIL',
    # The magnitude is what this one is for, not the attribution: the training
    # resolution and the evaluation resolution move together, so it cannot say
    # which of them paid. It says the proxy can see a ~3-point *model* difference
    # at all -- which is the size of the 320 decision it is meant to inform.
    'cross':  '224+tta4 -> 288+tta4     +2.97   <- CONFOUNDED',
}


@torch.no_grad()
def predict_probs(model_for, ds, views, nclass, size, device, bs, workers, agg='logit',
                  cache=None, limit=None):
    """Softmax probabilities of one checkpoint over one set of eval views.

    Returns ``(prob, y, paths)`` in **dataset order**, which is the order
    ``extract`` writes frozen features in -- so a row here and a row there are
    the same image.  That alignment is load-bearing (the V* mask comes from one
    and the predictions from the other) and is asserted by the caller rather
    than assumed.

    ``views`` are ``(size, crop, ratio_name, flip)``, with ``None`` for the
    trained resolution ``size``.  A view at another size needs a tower built at
    that size, so ``model_for`` is a **callable** ``size -> Net`` rather than a
    model: the caller owns how many towers that costs and when they are freed,
    and this function never has to know.  Every model it gets back comes from the
    same checkpoint, which is what keeps this TTA of one model rather than an
    ensemble of several.

    ``agg`` has to match the code the number will be compared against:
    ``'logit'`` averages the logits and is the default because the leaderboard
    numbers this is graded against were produced that way; ``'feat'`` averages
    the L2-normalised penultimate features and applies the head once, which is
    what ``infer.py`` defaults to.
    """
    views = check_view_tuple('predict_probs', views)
    if limit is not None:
        # truncated the same way and in the same order as `extract`, or the smoke
        # test would compare a full frozen pass against a half model pass and the
        # paths assertion below would fire for a reason that looks like corruption
        n_all = len(ds)
        ds = torch.utils.data.Subset(ds, list(range(min(limit, n_all))))
    n = len(ds)
    want = np.asarray(paths_of(ds))
    if cache is not None and Path(cache).exists():
        d = np.load(cache, allow_pickle=True)
        if _cache_matches(d, want):
            say(f'[cache] {Path(cache).name}  {len(want)} images ({agg})')
            return d['prob'].astype(np.float32), d['y'], d['paths']
        say(f'[cache] {Path(cache).name}: NOT reused -- {len(d["paths"])} image(s) in '
            f'the file, {n} in this pass.  (A different --limit / --val-ratio / '
            f'--seed / --data writes to the same name.)')

    assert agg in ('logit', 'feat'), f'agg={agg!r}'
    acc = None                      # running SUM, over views, of logits or of features
    y_all = np.zeros(n, dtype=np.int64)
    for v_size, crop, ratio_name, flip in views:
        v_size = int(size if v_size is None else v_size)
        model = model_for(v_size)
        # `ratio_name` is None for policies that do not take one (`full`/`pad`
        # squash or letterbox the whole frame, so there is no short side left to
        # choose); omitting it is what those policies already do.
        kw = {} if ratio_name is None else {'ratio': VIEW_RATIOS[ratio_name]}
        set_transform(ds, transform_for(v_size, flip, crop=crop, **kw))
        # one pass per view, accumulating into a fixed (n, C) buffer: memory stays
        # flat in the number of views instead of holding every view at once
        for x, y, idx in DataLoader(ds, batch_size=batch_for(v_size, bs), shuffle=False,
                                    num_workers=workers, pin_memory=True):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                if agg == 'feat':
                    _, z = model(x.to(device, non_blocking=True), return_feat=True)
                    z = z.float()           # already L2-normalised by Net.forward
                else:
                    z = model(x.to(device, non_blocking=True)).float()
            if acc is None:
                acc = torch.zeros(n, z.shape[1], device=device)
            # index_add, not acc[idx] = z: the same index cannot repeat within a
            # batch, but the *buffer* is carried across views, and a plain
            # assignment would silently leave only the last view's value in place
            # for any index a batch happened to cover twice
            acc.index_add_(0, idx.to(device), z)
            y_all[idx.numpy()] = y.numpy()

    if agg == 'feat':
        # average the normalised features, re-normalise, then decide ONCE with the
        # head -- this is infer.py's default and the reason it is not a vote
        logits = model.head(F.normalize(acc, dim=-1))
    else:
        # dividing does not change the argmax, but it does keep the softmax
        # temperatures comparable across view sets -- an undivided sum of four
        # logits is a *sharper* distribution than one, which would flatter any
        # confidence threshold applied to it later
        logits = acc / len(views)
    prob = logits.softmax(1).half().cpu().numpy()
    paths = want
    if cache is not None:
        np.savez(cache, prob=prob, y=y_all, paths=paths)
    return prob.astype(np.float32), y_all, paths


def score_probs(prob, y, vstar, margin=None):
    """The four numbers the proxy reports, from one checkpoint's val probs.

    ``val_acc``    micro, exactly the quantity ``train.py`` logs -- reproducing
                   it is the check that the split and the forward pass agree.
    ``macro``      class-equal recall.  The test set is class-balanced and val
                   is not, so this is closer to what the platform measures.
    ``acc_vstar``  accuracy on V*, the val subset whose given label the frozen
                   judge agrees with.  Should not reward memorising the noise.
    ``acc_vstar_hard``  the same, on the half of V* the judge was *least* sure
                   about.  V* selects for agreement, i.e. for prototypical
                   images, so ``acc_vstar`` can saturate near 1.0 and stop
                   ranking anything; the low-margin half is where the ranking
                   signal has to come from if it exists at all.  The threshold is
                   the median of V*'s own margins, so the subset is half of V* by
                   construction and does not need tuning.
    """
    pred = prob.argmax(1)
    hit = pred == y
    nclass = prob.shape[1]
    per_c = np.bincount(y[hit], minlength=nclass).astype(np.float64)
    per_t = np.bincount(y, minlength=nclass).astype(np.float64)
    seen = per_t > 0
    out = {'val_acc': float(hit.mean()),
           'macro': float((per_c[seen] / per_t[seen]).mean())}
    if vstar is None or not vstar.any():
        out['acc_vstar'] = float('nan')
        out['acc_vstar_hard'] = float('nan')
        return out, None
    out['acc_vstar'] = float(hit[vstar].mean())
    hard = np.zeros_like(vstar)
    if margin is not None:
        hard = vstar & (margin <= np.quantile(margin[vstar], 0.5))
        if not hard.any():              # all margins identical: no low half to take
            hard = vstar
    out['acc_vstar_hard'] = float(hit[hard].mean()) if hard.any() else float('nan')
    return out, hard


def calib_gate(rows, metric='acc_vstar', tol=0.5, view_min=0.5, cross_min=2.0):
    """Grade the proxy against the four facts the leaderboard settled.

    ``rows`` maps ``(run, epoch, viewset)`` to a dict of ``metric -> value``, with
    the values as **fractions** (what ``score_probs`` returns).  Returns
    ``(ok, lines)``.

    ``run`` is the resolution the checkpoint was trained at, and it is part of the
    key for two reasons.  The fourth fact is *between* two runs, so it cannot be
    expressed without it; and two runs both have an epoch 20, so a two-part key
    would let one silently overwrite the other -- the gate would then grade one
    run twice while reporting that it had graded four facts.

    Every delta is converted to **points** before it is compared, because points
    are the unit of the facts this is graded against -- 0.0002, -2.16, +2.09 are
    all platform scores out of 100.  Comparing a fraction against ``tol`` would
    not fail loudly either; it would pass, because 0.0002 <= 0.5 is true for
    entirely the wrong reason, and the gate would be checking nothing.

    The bar is deliberately symmetric and stated up front: the epochs differ by
    **0.0002 points** on the platform, so the proxy has to call them equal to
    within ``tol`` = 0.5; the views differ by about **2 points**, so it has to
    resolve an effect of at least ``view_min`` = 0.5 in the right direction.  A
    metric whose noise floor is above 0.5 points cannot separate those two
    scales, and that is the only thing being tested here.

    The negative control matters as much as the positives: ``val_acc`` moves
    **+5.02 points** across the same two epochs, so a gate that passes ``val_acc``
    too has demonstrated nothing -- it would be a metric that agrees with the
    leaderboard by accident because it disagrees with nothing strongly.  This
    function is pure arithmetic over a plain dict on purpose, so that the control
    can be run through it (``selftest.py`` does exactly that, with a table built
    to fail).

    The fourth fact is the only cross-run one, and the only **confounded** one:
    ``288-trained @288+tta4`` beats ``224-trained @224+tta4`` by 2.97 points, but
    the training resolution and the evaluation resolution move together, so it
    attributes the gain to neither.  It is here for its *size*: the pending
    decision is whether to spend ~2.5 GPU-hours on a 320 run, and a proxy that
    cannot resolve a 3-point model difference cannot inform that decision.  It is
    labelled CONFOUNDED, and when only one run has been measured it is reported as
    **not measured** rather than passed -- on the line that authorises the next
    GPU spend, silence and success must not look alike.
    """
    runs = sorted({r for r, _, _ in rows})
    if not runs:
        return False, ['gate: no rows']
    # The within-run facts were all measured on one run's checkpoints.  Which run
    # is chosen must be deterministic *and* printed: the epoch comparison means
    # nothing if it silently came from a different run than the reader assumes.
    by_run = {r: {(e, v) for (rr, e, v) in rows if rr == r} for r in runs}
    primary = sorted(runs, key=lambda r: (-len(by_run[r]), r))[0]
    epochs = sorted({e for e, _ in by_run[primary]})
    if len(epochs) < 2:
        return False, [f'gate: the run it would be graded on ({primary}px) has '
                       f'{len(epochs)} epoch(s) among the rows; two are needed to '
                       f'compare them']
    eps = {'first': epochs[0], 'last': epochs[-1]}
    lines = [f'-- metric {metric}: run {primary}px, '
             f'ep{eps["first"]} vs ep{eps["last"]} --']

    def d(name, a, b):
        va = rows.get((primary, eps[a[0]], a[1]), {}).get(name)
        vb = rows.get((primary, eps[b[0]], b[1]), {}).get(name)
        if va is None or vb is None:
            return None
        return 100 * (float(vb) - float(va))       # fraction -> points, see above

    def show(label, got, want, ok):
        s = 'n/a' if got is None else f'{got:+.4f}'
        lines.append(f'   {"PASS" if ok else "FAIL"}  {label:<34s} {s:>9s} pt   '
                     f'leaderboard {want:+.4f} pt')
        return ok

    ok = True
    # 1. the epochs are equal on the platform.  The proxy must agree ...
    got = d(metric, ('first', 'plain'), ('last', 'plain'))
    ok &= show(f'{metric} ep diff (want ~0)', got, 0.0002,
               got is not None and abs(got) <= tol)
    # ... and val_acc, the metric this replaces, must NOT.  Not a formality: if
    # both agree the proxy has bought nothing, and this line is what says so.
    # Read the PASS/FAIL on the *check*, not on val_acc's behaviour: the check
    # passes when val_acc moves a lot, i.e. when the two metrics disagree.
    got_va = d('val_acc', ('first', 'plain'), ('last', 'plain'))
    ok &= show('val_acc ep diff (control: want LARGE)', got_va, 5.02,
               got_va is None or abs(got_va) > tol)
    # 2/3. the two view facts.  Sign is the claim; the floor is the resolution.
    for name, want in (('wide', -2.16), ('tta4', +2.09)):
        got = d(metric, ('last', 'plain'), ('last', name))
        good = got is not None and abs(got) >= view_min and (got > 0) == (want > 0)
        ok &= show(f'{metric}  plain -> {name}', got, want, good)

    # 4. the cross-resolution fact.  Whether it can be measured at all depends on
    #    what was passed, so "not measured" is a *reported* outcome rather than a
    #    silent one -- the reader has to be able to tell it from a pass.
    def cross(name):
        """``(lo_run, hi_run), points`` for the outer pair of runs, or None."""
        if len(runs) < 2:
            return None
        lo, hi = runs[0], runs[-1]

        def at(r):
            es = sorted(e for e, _ in by_run[r])
            return rows.get((r, es[-1], 'tta4'), {}).get(name) if es else None

        a_, b_ = at(lo), at(hi)
        if a_ is None or b_ is None:
            return None
        return (lo, hi), 100 * (float(b_) - float(a_))

    cr = cross(metric)
    if cr is None:
        lines.append(f'   --    {metric}  cross-resolution          not measured  '
                     f'(CONFOUNDED) needs tta4 at two training runs; rows cover '
                     f'{runs}')
    else:
        (lo, hi), got = cr
        ok &= show(f'{metric}  {lo}px -> {hi}px (CONFOUNDED)', got, 2.97,
                   got > 0 and abs(got) >= cross_min)
    return bool(ok), lines


def calib(a):
    """Score checkpoints and views **offline**, against the leaderboard facts.

    The project's scarcest resource is not GPU time but submission slots (2/day
    for the whole team), and until now every judgement about a test-time recipe
    cost one.  This section buys a local signal instead: the frozen judge picks
    the val images whose labels are trustworthy (V*), and each checkpoint's
    accuracy *on V** is what gets compared -- no submission, no leaderboard.

    It is only worth using if it reproduces comparisons whose answers are already
    known, so the gate runs first and is what the exit code reflects.  A proxy
    that has not passed the gate is not evidence, and reporting it as though it
    were would be worse than having no proxy at all.
    """
    check_backbone(a.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    globals()['_REPORT'] = out / 'report_calib.txt'
    atexit.register(_write_report)

    hr(f'OFFLINE LEADERBOARD PROXY   device={device}   agg={a.calib_agg}')
    say('Every number below comes from a forward pass over the *hold-out* split')
    say('and costs 0 submission slots.  Checkpoints and view sets are the two')
    say('axes; V* is the val subset whose given label the frozen judge agrees')
    say('with, so accuracy on it should not reward memorising the label noise.')

    stems = [Path(p).stem for p in a.checkpoint]
    assert len(set(stems)) == len(stems), \
        f'two checkpoints share a file stem, so their caches would collide: {stems}'
    cks = []
    for path in a.checkpoint:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        # The gate orders rows by epoch to find "first" and "last", so an
        # unrecorded or duplicated epoch makes it compare the wrong pair -- or
        # raise a bare TypeError inside sorted() *after* the GPU work is done.
        # Both are worth catching here, before any of it runs.
        ep = ck.get('epoch', '?')
        assert isinstance(ep, int), (
            f'{path}: no integer `epoch` recorded (got {ep!r}), so the gate cannot '
            f'tell which checkpoint is which -- pass snapshots saved by train.py')
        # The resolution is taken from the checkpoints, never from --sizes.  A 288
        # checkpoint scored at 224 runs fine and simply scores worse, i.e. the
        # failure this whole compatibility layer exists to make impossible.
        cks.append({'path': path, 'stem': Path(path).stem, 'ck': ck, 'epoch': ep,
                    'run': int(ck_image_size(ck)[0])})

    # Mixed resolutions are allowed and are the point: each checkpoint is scored
    # at the resolution it was trained at, which is the only way the 224 and 288
    # runs can be compared at all.  What is *not* allowed is two checkpoints
    # sharing a (run, epoch) -- they would land on the same row and the gate would
    # grade whichever was measured last, twice.
    runs = sorted({c['run'] for c in cks})
    seen = set()
    for c in cks:
        key = (c['run'], c['epoch'])
        assert key not in seen, (
            f'{(c["run"], c["epoch"])}: two checkpoints are at the same epoch of the '
            f'same training resolution, so their rows collide and the gate would grade '
            f'one of them twice without saying so. Pass at most one per (resolution, '
            f'epoch).')
        seen.add(key)
    say(f'{len(cks)} checkpoint(s) at {len(runs)} training resolution(s): '
        f'{runs}px (read from the checkpoints, not from --sizes)')

    # ---- the frozen judge, and V* ------------------------------------------ #
    tr, va = build_split_datasets(a)
    say(f'split: train={len(tr)} val={len(va)} classes={len(tr.class_to_idx)} '
        f'(same seed/ratio as train.py)')
    hr('FROZEN JUDGE AND V*')
    # The judge is pinned, and 224 is the pin: that is the size OpenAI's CLIP was
    # trained at, so it is the only size where the judge's own grid is
    # un-interpolated.  Running the judge at 288 would first have to resample the
    # judge's positional grid, and the "confident and correct" half of V* would
    # then be partly a product of that resample.
    #
    # The alternative -- a per-size V* -- is not merely different, it is *biased*:
    # the judge is also the selector, so whichever grid the frozen CLIP prefers
    # yields a cleaner subset, and it is the model trained at that very size that
    # then gets scored on it.  Pinning the judge fixes the subset, so a difference
    # between two rows is attributable to the model rather than to the yardstick.
    judge_size = int(a.calib_judge_size)
    say(f'judge pinned at {judge_size}px'
        + ('' if judge_size == 224 else '  <- NOT the native size; V* is then partly '
                                        'an artefact of resampling the judge itself'))
    # `anchor_feat` is not used here on purpose: extract() runs the *plain* CLIP
    # tower, which is checkpoint-independent, so these features are valid for
    # every checkpoint below and the cache is shared across all of them.  The
    # judge tower is never wrapped in a Net -- Net() injects LoRA into whatever
    # tower it is handed -- so it stays exactly the frozen model it claims to be.
    judge = build_clip(a.model, a.pretrained, judge_size).to(device).eval()
    Ftr = extract(judge, tr, judge_size, device, a.batch_size, a.workers, 'train', out,
                  a.limit)
    Fva = extract(judge, va, judge_size, device, a.batch_size, a.workers, 'val', out,
                  a.limit)
    # V* is fixed from here on, so the judge has done its job: free ~350 MB before
    # the model towers are built, which on a two-resolution run is the difference
    # between one tower in memory and three
    del judge
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    y_tr = torch.as_tensor(Ftr['y'], device=device)
    y_va = torch.as_tensor(Fva['y'], device=device)
    f_tr = torch.as_tensor(Ftr['f'].astype(np.float32), device=device)
    f_va = torch.as_tensor(Fva['f'].astype(np.float32), device=device)
    nclass = int(max(y_tr.max(), y_va.max())) + 1
    Str, ntr = class_sums(f_tr, y_tr, nclass)
    rtr = ncc_scores(f_tr, y_tr, nclass)                    # LOO, for rho
    rva = ncc_scores(f_va, y_va, nclass, ref=(Str, ntr))    # val vs TRAIN centroids
    acc_judge = float((rva['pred'] == y_va).float().mean())
    keep = ((rva['pred'] == y_va) & (rva['margin'] >= a.vstar_margin)).cpu().numpy()
    margin = rva['margin'].cpu().numpy()
    say(f'frozen-CLIP NCC accuracy on val @{judge_size}: {acc_judge:.4f}')
    say(f'rho (fraction of train images the judge ranks elsewhere): '
        f'{float((rtr["margin"] < 0).float().mean()):.4f}')
    say(f'|V*| (margin >= {a.vstar_margin:g}) = {int(keep.sum())} / {len(y_va)} '
        f'({100 * float(keep.mean()):.1f}% of val)')
    say('V* is ~100% by construction for the judge itself, so the judge\'s own')
    say('score on it is meaningless -- it is V*\'s *membership* that is the')
    say('artefact, and a trained checkpoint is what gets scored on it.')

    # ---- every checkpoint x every view set --------------------------------- #
    def classes_of(ck):
        return ck['classes'] if 'classes' in ck else ck['class_to_idx']

    # Across runs too, not just within one: the gate's fourth fact compares rows
    # from two different runs, and rows only mean the same thing if their class
    # indices are the same classes.
    classes = classes_of(cks[0]['ck'])
    assert all(classes_of(c['ck']) == classes for c in cks), \
        'the checkpoints disagree about the class list, so their rows are not comparable'
    # Read the LoRA shape exactly the way infer.py does (top level first, then
    # args, then the default).  A wrong rank does not raise: the adapter tensors
    # are simply absent from the load and land in `missing`, where the guard
    # below catches them.
    def lora_of(ck):
        a_ = ck.get('args', {})
        return (ck.get('lora_rank', a_.get('lora_rank', 8)),
                ck.get('lora_target', a_.get('lora_target', 'all')))

    shapes = {lora_of(c['ck']) for c in cks}
    assert len(shapes) == 1, (
        f'the checkpoints were trained with different LoRA shapes {sorted(shapes)}, '
        f'so one Net cannot hold them; score them in separate runs')
    rank, target = shapes.pop()

    # `--calib-sweep` enumerates every single view and every *pair* of them.  The
    # pairs are the point: the one leaderboard-validated recipe wins by
    # averaging views that are individually worse than the plain one, so the
    # question "which two views decorrelate most" is the one worth having an
    # offline answer to, and it needs all of them measured to be answered.
    vsets, order = {}, []
    if a.calib_sweep:
        singles = sorted(v for v in CALIB_VIEWS if len(CALIB_VIEWS[v]) == 1)
        for v in singles:
            vsets[v] = CALIB_VIEWS[v]
            order.append(v)
        for i, x in enumerate(singles):
            for y in singles[i + 1:]:
                name = f'{x}+{y}'
                vsets[name] = tuple(CALIB_VIEWS[x]) + tuple(CALIB_VIEWS[y])
                order.append(name)
        # ... plus the sets a pair of single views cannot build, listed beside the
        # table itself (train.VIEW_SETS_SWEPT) so this sweep cannot quietly stop
        # covering a set the submission path can still be asked for.
        for extra in VIEW_SETS_SWEPT:
            vsets[extra] = CALIB_VIEWS[extra]
            order.append(extra)
    else:
        order = list(a.calib_views) or ['plain', 'wide', 'tta4', 'axis4']
        unknown = sorted(set(order) - set(CALIB_VIEWS))
        assert not unknown, f'unknown --calib-views {unknown}; choose from {sorted(CALIB_VIEWS)}'
        vsets = {v: CALIB_VIEWS[v] for v in order}
    vsets = {v: check_view_tuple(v, vsets[v]) for v in order}

    # The gate's within-run facts are defined on these three, so a run that omits
    # one would report FAIL for a reason that has nothing to do with the proxy.
    # Refuse up front instead: "the gate says the proxy is broken" and "you did not
    # measure what the gate looks at" are very different findings and must not be
    # able to arrive as the same output.  The cross-resolution fact is deliberately
    # *not* here -- it needs two training runs, and requiring that would make the
    # gate unrunnable on the 224 checkpoints alone, which is the common case.
    need = {'plain', 'wide', 'tta4'}
    absent = sorted(need - set(order))
    assert not absent, (
        f'the calibration gate is defined on {sorted(need)}, and --calib-views left '
        f'out {absent}. Without them this run cannot say whether the proxy is '
        f'calibrated, which is the only thing it exists to say.')
    w = max(len(v) for v in order)

    hr('CHECKPOINT x VIEW SET')
    say(f'{len(order)} view set(s), each a deterministic transform of the same '
        f'image under the same weights:')
    say('  ' + ', '.join(f'{v}({len(vsets[v])})' for v in order))
    say(f'aggregation: {a.calib_agg}'
        + ('   <- logit averaging, which is what the leaderboard numbers below were '
           'produced with' if a.calib_agg == 'logit' else
           '   <- infer.py\'s default, so these are the numbers a submission would see'))
    if a.limit:
        say('!! --limit is set -- none of this is meaningful')
    say()

    def guard(net, missing, unexpected, where):
        """Refuse a checkpoint whose weights cannot all be placed in the model.

        Everything the load could not place has to be something the frozen
        backbone rebuilds from the official OpenAI weights.  The old phrasing --
        ``requires_grad ∩ missing`` -- is structurally blind to the positional
        embedding: it is frozen, so it is never in the trainable set, and it is
        also never in a checkpoint, which is exactly why a *wrong* one is silent.
        The two halves here are not symmetric:

        * a **trainable** name that is missing means the checkpoint carries no
          weights for part of what this model trains, so the run being scored is
          not the run that was trained;
        * **unexpected** means the checkpoint holds weights this model will never
          use.  This codebase writes exactly the trainable set, so anything here
          is a naming or LoRA-target mismatch -- and dropping it silently would
          evaluate a partially random model, which is the worst outcome available
          because it still produces a plausible number.
        """
        trainable = {n for n, p in net.named_parameters() if p.requires_grad}
        lost = sorted(trainable & set(missing))
        assert not lost, (
            f'{where}: no trained weights for {lost[:5]}'
            + (f' (and {len(lost) - 5} more)' if len(lost) > 5 else ''))
        assert not unexpected, (
            f'{where}: {len(unexpected)} tensor(s) in this checkpoint are not part of '
            f'the model being built ({sorted(unexpected)[:5]}), so they would be '
            f'dropped without a word and the model scored here would not be the one '
            f'trained. Check --lora-target against the one the run used.')
        return trainable

    rows, hard = {}, None
    for run in runs:
        group = [c for c in cks if c['run'] == run]
        # The grid is *rebuilt* here, not restored: it is frozen, so
        # `trainable_state_dict` drops it and no checkpoint carries it.  That makes
        # this rebuild the thing that decides which model these weights get
        # evaluated as -- and it is silent when wrong, because open_clip's resample
        # and ours agree on shape for 288 and differ in value.  Checkpoints written
        # by this codebase record a fingerprint; the sibling repository's do not,
        # and for those the only protection is that `resize_positional_embedding`
        # reproduces its mechanism exactly.
        # Resolutions the view sets ask for.  Empty unless a size-carrying view was
        # selected, so the default gate run builds exactly one tower.
        extra = sorted({int(s) for v in order for (s, *_) in vsets[v]
                        if s is not None and int(s) != run})
        towers = {run: build_clip(a.model, a.pretrained, run)}
        n_fp = sum(verify_pos_embed(c['ck'], towers[run].visual) is True for c in group)
        if n_fp < len(group):
            say(f'{run}px: {n_fp}/{len(group)} checkpoint(s) record a positional-'
                f'embedding fingerprint; for the rest the grid rebuilt here could not '
                f'be checked against the one they were trained with.  The val_acc '
                f'self-check below is what stands in for it.')
        for s in extra:
            towers[s] = build_clip(a.model, a.pretrained, s)
        # One Net per resolution, built once for the whole run: Net() injects LoRA
        # into the tower it is handed, so building a Net per *checkpoint* would
        # stack a second set of adapters onto the same tower.  Each checkpoint's
        # weights are loaded into these instead -- the frozen backbone is identical
        # across a run's checkpoints, only the adapters and head differ.
        local_head = bool(group[0]['ck'].get('local_head',
                                             group[0]['ck'].get('args', {}).get('local_head', False)))
        nets = {s: Net(t, len(classes), rank, target, local_head=local_head).to(device).eval()
                for s, t in towers.items()}

        def model_for(size, _nets=nets):
            size = int(size)
            if size not in _nets:
                raise SystemExit(
                    f'a view asked for {size}px but towers were built for '
                    f'{sorted(_nets)} -- the view table and the size list disagree, '
                    f'which would otherwise surface as a KeyError after the GPU work')
            return _nets[size]

        for c in group:
            ep, path, stem = c['epoch'], c['path'], c['stem']
            for size, net in nets.items():
                state = dict(c['ck'].get('model', c['ck']))
                trained_size = ck_image_size(c['ck'], verbose=False)[0]
                if c['ck'].get('pos_embed_trained') and int(size) != int(trained_size):
                    # Extra-resolution calibration views use the interpolated
                    # grid at that size; the trained grid has incompatible token
                    # count and must not be loaded into this tower.
                    state.pop('clip.visual.positional_embedding', None)
                    state.pop('clip.visual.pos_embed', None)
                missing, unexpected = net.load_state_dict(state, strict=False)
                guard(net, missing, unexpected, f'{path} @{size}px')
            for v in order:
                # A size view at the size this checkpoint was trained at is the
                # plain view again, so the set is that view counted twice.  It
                # scores, it prints, and it reads like a second measurement of
                # `plain` -- silence is the only honest output for it.
                if is_degenerate(vsets[v], run):
                    # Two views of the set resolve to the same pixels, so the row
                    # would be an average with one input weighted twice.  Saying
                    # *that* rather than "measures nothing" matters: `tta4s320` at
                    # 320px is four distinct views, not none, and a message that
                    # overstates the collapse reads as a bug in the skip rule.
                    say(f'  {run}px {v}: skipped -- its {len(vsets[v])} view(s) '
                        f'resolve to only '
                        f'{len(set(resolved_views(vsets[v], run)))} distinct '
                        f'input(s) at this resolution, so one of them would be '
                        f'counted twice and the row would read as a recipe that '
                        f'was never measured')
                    continue
                vh = hashlib.sha256(
                    repr((vsets[v], run, a.calib_agg)).encode()).hexdigest()[:8]
                prob, y_p, paths = predict_probs(
                    model_for, va, vsets[v], nclass, run, device, a.batch_size,
                    a.workers, a.calib_agg,
                    cache=out / f'calib_{stem}_{v}_{a.calib_agg}_{vh}.npz',
                    limit=a.limit)
                # the V* mask indexes the frozen pass's rows, so the two passes must
                # be the same images in the same order.  If they are not, every
                # acc@V* below is scored against someone else's labels, silently.
                # Both sides validate their own cache against the pass they stand
                # for (see `_cache_matches`) before reaching here, so this compares
                # two image lists that were each checked -- not two files that
                # could be stale in the same way and agree with each other.
                assert np.array_equal(paths, Fva['paths']), (
                    f'{path} / {v}: the model pass saw a different val order than the '
                    f'frozen pass -- acc@V* would be scored against the wrong labels')
                sc, hard = score_probs(prob, y_p, keep, margin)
                rows[(run, ep, v)] = sc
                say(f'  ep{ep:<4} {v:<{w}}  val_acc {sc["val_acc"]:.4f}  macro '
                    f'{sc["macro"]:.4f}  acc@V* {sc["acc_vstar"]:.4f}  '
                    f'acc@V*-hard {sc["acc_vstar_hard"]:.4f}')
                # The checkpoint records the val_acc of exactly these weights, on the
                # same hold-out split through the same transform, so the training
                # view has to reproduce it.  This is the one check here that can
                # notice the model is not the model: a positional grid rebuilt with
                # a different interpolation setting loads without complaint, raises
                # nothing, and moves this number by whole points.
                rec = c['ck'].get('val_acc')
                if vsets[v] == TRAINING_VIEW and rec is not None and a.limit:
                    # A smoke run scores the first `--limit` images, so its
                    # val_acc is computed on a different (and tiny) set than the
                    # recorded one and cannot reproduce it.  Saying that is the
                    # only honest option: firing the assert below would report a
                    # broken model where the actual cause is `--limit`, and
                    # skipping it silently would hide the check the reader
                    # expects to have run.
                    say(f'  self-check ep{ep}: SKIPPED -- --limit {a.limit} scores a '
                        f'truncated val split, so val_acc cannot match the value '
                        f'recorded at training.  This run is a smoke test only.')
                elif vsets[v] == TRAINING_VIEW and rec is not None:
                    drift = 100 * abs(sc['val_acc'] - float(rec))
                    assert drift <= a.calib_valacc_tol, (
                        f'{path}: val_acc {sc["val_acc"]:.4f} here vs {float(rec):.4f} '
                        f'recorded at training -- off by {drift:.2f} points, tolerance '
                        f'{a.calib_valacc_tol}. The split and the forward pass are '
                        f'supposed to be identical, so this is not noise: the model '
                        f'scored here is not the one that was trained. The usual cause '
                        f'is the positional grid (see the {run}px note above). Every '
                        f'number below would describe a different model; pass '
                        f'--calib-valacc-tol <points> to accept that and continue.')
                    say(f'  self-check ep{ep}: val_acc reproduces training '
                        f'({sc["val_acc"]:.4f} vs {float(rec):.4f}, off by '
                        f'{drift:.3f} pt)')

        del nets, towers
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- the gate, which is the only thing here that decides anything ------ #
    hr('CALIBRATION GATE')
    say('The proxy is graded on comparisons the leaderboard already settled')
    say('(750-class 复赛 round, sibling repository):')
    for k, v in CALIB_TRUTH.items():
        say(f'    {k:<7} {v}')
    say()
    say('The bar: the epochs differ by 0.0002 on the platform, so a usable metric')
    say('must call them equal within 0.5; the views differ by ~2, so it must resolve')
    say('at least 0.5 in the right direction.  A metric whose noise floor is above')
    say('0.5 cannot tell those two scales apart.')
    say('The cross-resolution row is graded only when two training resolutions were')
    say('passed. When they were not it reads "not measured", which is not a pass.')
    say()
    gate = {}
    for metric in ('acc_vstar', 'acc_vstar_hard'):
        ok, lines = calib_gate(rows, metric)
        gate[metric] = ok
        for ln in lines:
            say(ln)
        say(f'   => {metric}: {"PASS" if ok else "FAIL"}')
        say()
    if hard is not None:
        say(f'|V*-hard| = {int(hard.sum())} of {int(keep.sum())} in V* '
            f'(the low-margin half, where the ranking signal has to live)')
    # The frozen judge's own val accuracy is the ceiling this whole idea is
    # bounded by, so it belongs next to the verdict rather than in a log line.
    say(f'reference: frozen-CLIP judge itself scored {acc_judge:.4f} on the whole '
        f'val split, so a checkpoint near that on V* has hit the judge\'s ceiling.')
    say()
    passed = [m for m, ok in gate.items() if ok]
    if not passed:
        say('VERDICT: FAIL -- no metric passed.  The proxy is not evidence yet.')
        say('Report this and stop; do not pick a view set with these numbers.')
        return 1
    say(f'VERDICT: PASS for {", ".join(passed)} -- this metric reproduces the')
    say('leaderboard orderings it was graded on, so it can be used to compare view')
    say('sets and checkpoints offline.  It is calibrated for effects of about 2')
    say('points; treat differences well under 0.5 as noise.')
    _write_report()
    say(f'report saved to {out / "report_calib.txt"}')
    return 0


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
    if a.calib:
        assert a.checkpoint, ('--calib needs --checkpoint: the proxy scores trained '
                              'checkpoints, and the gate needs the two whose '
                              'leaderboard scores are known (ep4 and ep20)')
        return calib(a)

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
                        'train.build_clip warns about such sizes. Not used by --calib, '
                        'which takes its resolution from the checkpoints themselves so '
                        'that a 288 checkpoint cannot be silently scored at 224')
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
                        'cached, so re-runs are free. This scores the *frozen* tower, '
                        'so it is the cheap first pass: it cannot report acc@V* (V* is '
                        'defined by this judge, so it would score ~1.00 by '
                        'construction). The trained-model version that can is --calib')
    p.add_argument('--tta-crops', nargs='*', default=['center', 'full', 'pad'],
                   dest='tta_crops', metavar='POLICY',
                   help='crop policies to score, from train.CROP_POLICIES. Defaults to '
                        'all three so the measurement actually happens: the centre '
                        'crop (the current default) discards a fixed fraction of every '
                        'frame, and on this dataset that is over a third of the height '
                        'of a 3:4 image. All policies stay inside the trained patch '
                        'grid, so unlike --sizes they involve no '
                        'positional-embedding interpolation')
    p.add_argument('--calib', action='store_true',
                   help='score trained checkpoints and view sets against V*, i.e. the '
                        'offline leaderboard proxy. Costs zero submission slots, which '
                        'are the project\'s scarcest resource -- every judgement this '
                        'replaces used to cost one. Needs --checkpoint')
    p.add_argument('--checkpoint', nargs='+', default=[], metavar='PT',
                   help='with --calib: the checkpoints to score. Pass the pair whose '
                        'leaderboard scores are already known (ep4 and ep20) the first '
                        'time -- the calibration gate needs two epochs to grade the '
                        'proxy against, and an ungraded proxy is not evidence. '
                        'Checkpoints trained at *different* resolutions may be mixed: '
                        'each is scored at the resolution it records, which is what '
                        'lets the gate grade its cross-resolution fact. Two at the '
                        'same (resolution, epoch) are refused -- their rows would '
                        'collide')
    p.add_argument('--calib-views', nargs='+', default=['plain', 'wide', 'tta4', 'axis4'],
                   dest='calib_views', metavar='NAME',
                   help='with --calib: view sets to score, from '
                        + '/'.join(sorted(CALIB_VIEWS)) + '. Each is deterministic '
                        'transforms of the same image under the same weights, so a set '
                        'is TTA of one model, not an ensemble. plain/wide/tta4 must be '
                        'included -- they are what the gate is defined on. Default adds '
                        'axis4, the crop-axis alternative to tta4')
    p.add_argument('--calib-sweep', action='store_true', dest='calib_sweep',
                   help='with --calib: score every single view AND every pair of them '
                        '(plus ' + ' / '.join(VIEW_SETS_SWEPT) + '). The pairs are the point '
                        '-- the one leaderboard-validated recipe wins by averaging views '
                        'that are individually *worse* than the plain one, so which two '
                        'views decorrelate most is the question, and it needs all of '
                        'them measured. The single views include one per alternative '
                        'resolution, so the sweep covers the crop axis and the size axis '
                        'at once, at that checkpoint\'s training size. Cached per view '
                        'definition AND per resolution, so re-runs are free')
    p.add_argument('--calib-judge-size', type=int, default=224, dest='calib_judge_size',
                   help='with --calib: the resolution the FROZEN JUDGE runs at. Left at '
                        '224 deliberately -- that is the size OpenAI trained CLIP at, so '
                        'the judge\'s own positional grid is un-interpolated there. '
                        'Judging at another size resamples the judge too, which makes V* '
                        'partly a product of that resample, and V* is what every acc@V* '
                        'is measured on. Changing this does not break the gate; it '
                        'changes the yardstick, so numbers from two such runs are not '
                        'comparable')
    p.add_argument('--calib-valacc-tol', type=float, default=0.5, dest='calib_valacc_tol',
                   help='with --calib: how far, in points, a checkpoint\'s val_acc may '
                        'drift from the value the run itself recorded before the proxy '
                        'refuses to continue (default 0.5). The two passes use the same '
                        'split and the same transform, so they should agree to within '
                        'batch-size numerics; a whole-point drift means the model being '
                        'scored is not the model that was trained, and raising this '
                        'accepts that every number below describes a different model')
    p.add_argument('--calib-agg', default='logit', choices=['logit', 'feat'],
                   dest='calib_agg',
                   help='with --calib: how views are combined. logit (default) is what '
                        'the leaderboard numbers were produced with, so it is the one the '
                        'gate has to be graded in; feat is infer.py\'s default and is '
                        'what to switch to once the gate has passed and the number is '
                        'being used to choose the view set a submission will use')
    p.add_argument('--selftest', action='store_true', help='synthetic check, no data/GPU needed')
    a = p.parse_args(argv)
    a.limit = a.limit or None
    return a


if __name__ == '__main__':
    raise SystemExit(main(parse_args()))
