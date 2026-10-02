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
* the per-view outputs are **averaged, not voted**: either the logits are
  averaged directly (``--tta-agg logit``, the default), or each view's feature --
  already L2-normalised by ``Net.forward`` -- is averaged, re-normalised once,
  and the head applied **once** (``--tta-agg feat``).  There is a single
  decision path either way, so this is TTA of one model and not an ensemble.

The default is ``logit`` because it is the only one of the two that has been on
the platform: every leaderboard number this project has (64.16 single view,
66.2456 four views, 69.218 at 288, 69.968 at 320) was measured through it, so a
recipe reproduced here is the recipe that scored.  ``feat`` is the reasoned
alternative -- normalising before averaging keeps a badly-scaled view from
dominating -- but it is a different computation and it has not been scored;
``probe.py --calib-agg`` is what compares them offline.


Averaging the outputs of several *different* checkpoints (e.g. seed ensembles)
is an ensemble and is **not** permitted.  Do not add it.

Two ways to choose the views: ``--tta-views NAME`` (a set from
``train.VIEW_SETS``) or the product flags below.  The product cannot express a
recipe that flips only some of its views, and the one recipe here with a
leaderboard number behind it is exactly that -- so the name is what reproduces
``tta4``.  A set that exists both ways is spelled identically by both paths, and
``selftest.py`` asserts that rather than trusting it.

There are two independent axes, and they are not equally safe:

* ``--tta-crops`` changes only *which pixels of the one image* are fed to the
  model.  All the policies in ``train.CROP_POLICIES`` stay inside the trained
  patch grid, so no positional embedding is touched.  This axis is free of the
  risk below.  It exists because the default ``center`` policy does not keep the
  whole frame: measured, a 3:4 image keeps 57.5% of it, a 2:3 image 51.0%, and
  even a square one only 76.6% (``Resize(size*256/224)`` + ``CenterCrop(size)``
  removes the outer 1/8 of both axes whatever the shape).  On fine-grained
  images the discriminative part is small and often not centred.
* ``--tta-ratios`` is the same axis one notch finer: it changes how much of the
  short side the *centre crop* keeps (``train.VIEW_RATIOS``).  Also inside the
  trained patch grid, so also risk-free, and worth separating from
  ``--tta-crops`` because the sibling repository measured three values of it
  against the leaderboard: ``wide`` **alone** scored 2.16 points *below* the
  default single view, while a four-view average containing it scored 2.09
  *above*.  The lesson is that a view which is worse on its own can still be
  worth averaging -- so neither axis should be pruned by single-view scores.
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
  trusting it, and read the checkpoint's own record of it first -- that is what
  ``base`` above comes from, via ``train.ck_image_size``, which accepts both the
  ``image_size`` this codebase writes and the ``img_size`` the sibling
  repository writes (and says which one it followed).

Each distinct resolution needs its own ``open_clip.create_model`` call because
the positional grid has to be resampled at model-creation time
(``train.resize_positional_embedding``, reached through ``train.build_clip``);
one model is built per size, loaded from the same checkpoint, used, and freed.

The grid is *not* stored in the checkpoint -- it is frozen, so
``trainable_state_dict`` drops it -- which means every one of those builds is
reconstructing part of the model rather than restoring it.  ``build_clip``
therefore takes ``ck=`` and checks what it rebuilt against a fingerprint the
checkpoint records; see its docstring.  Nothing else in the pipeline can notice
if the reconstruction is wrong, because a wrong grid loads without complaint.

Usage::

    python infer.py --test /root/autodl-tmp/test --checkpoint outputs/best.pt \
                    --output pred_results.csv
    # the four-view average that scored 66.2456 against 64.16 for a single view,
    # named.  Use the name: the product flags cannot express it (--tta-flip flips
    # *every* view, so they give its 6-view superset).
    python infer.py --test ... --checkpoint outputs/best.pt --tta-views tta4
    # 8 views -- the same axis swept to saturation -- and 6 on the crop axis
    # instead, which is the question probe.py --calib exists to answer:
    python infer.py --test ... --checkpoint outputs/best.pt --tta-views tta8
    python infer.py --test ... --checkpoint outputs/best.pt --tta-views crops6
    # an unnamed set is still spelled out with the product flags:
    python infer.py --test ... --checkpoint outputs/best.pt \
                    --tta-crops center full pad --tta-flip
    # and/or several resolutions (6 views) -- measure that one first:
    python infer.py --test ... --checkpoint outputs/best.pt \
                    --image-size 224 --tta-sizes 288 320 --tta-flip

Add ``--workers N`` to any of these: the transforms, not the GPU, are what makes
a many-view run take hours.

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
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import ImageFile
from torch.utils.data import DataLoader

from train import (VIEW_RATIOS, VIEW_SETS, FlatImages, Net, build_clip, check_view_tuple,
                   ck_image_size, eval_transform, is_degenerate, pos_embed_fingerprint)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def view_plan(a, base):
    """The views this command line asks for, as ``(views, description)``.

    A view is ``(size, crop, ratio_name, flip)``.  The table writes ``None`` for
    the size to mean "the resolution the checkpoint was trained at" -- that is
    ``base`` here, and every view returned has already had it substituted, so the
    caller can run the list as it stands and never has to know about the
    convention.  Two ways to say it, and exactly one of them is allowed at a
    time:

    ``--tta-views NAME...``
        A named set from ``train.VIEW_SETS``, unioned if several are given.  This
        is the only way to ask for sets the product below cannot express -- a
        recipe that flips *some* of its views, which includes the four-view
        average that scored 66.2456 (``--tta-flip`` would flip every view and
        produce a 6-view superset instead).
    the product flags
        ``--tta-sizes`` x ``--tta-crops`` x ``--tta-ratios`` x ``--tta-flip``.
        The ratio axis only applies to ``center``: ``full`` squashes the whole
        frame and ``pad`` letterboxes it, so both already fix the short side and
        would otherwise multiply the view list by four while producing identical
        tensors.  It is a separate axis rather than a constant because the
        sibling repository scored three of its values on the leaderboard --
        fine-grained objects are small, so how much frame the centre crop keeps
        is a real choice.

    Split out of ``main`` so the self-test can compare the two spellings of one
    set without a forward pass: ``crops6`` and ``--tta-crops center full pad
    --tta-flip`` naming the same views is a property of these two parsings, and a
    CSV comparison would only notice it after an hour of GPU time.
    """
    if a.tta_views:
        # Refusing the combination rather than picking a winner: silently
        # intersecting or ignoring would produce a CSV that looks like the named
        # recipe and is not it, which is the failure this whole table exists to
        # prevent.  The product flags keep `None` defaults so that "was it
        # passed?" is answerable here.
        clash = sorted(k for k, v in (('--tta-sizes', bool(a.tta_sizes)),
                                      ('--tta-crops', a.tta_crops is not None),
                                      ('--tta-ratios', a.tta_ratios is not None),
                                      ('--tta-flip', a.tta_flip)) if v)
        assert not clash, (
            f'--tta-views selects a complete set of views, so it cannot be combined '
            f'with {clash} -- pick one or the other.')
        unknown = sorted(set(a.tta_views) - set(VIEW_SETS))
        assert not unknown, (f'unknown --tta-views {unknown}; choose from '
                             f'{sorted(VIEW_SETS)}')
        # A name that promises more views than it delivers at *this* checkpoint is
        # refused: `tta4s320` against a 320-trained model offers five views of
        # which two are the same pixels (the `None` size and the literal 320 both
        # mean 320 here), so the run would weight one view twice and the CSV would
        # not be the recipe the name denotes.
        degen = [n for n in a.tta_views if is_degenerate(VIEW_SETS[n], base)]
        assert not degen, (
            f'--tta-views {degen}: at {base}px (this checkpoint\'s trained '
            f'resolution) two of that set\'s views are the same input, so it would '
            f'average one view twice and report it as the named recipe. Drop it, or '
            f'pick a size that actually changes the pixels.')
        views, seen = [], set()
        for name in a.tta_views:
            for v in check_view_tuple(name, VIEW_SETS[name]):
                # The table says `None` for the trained size; what comes back is
                # resolved, so the caller can run the list as it stands.  The
                # dedup key is the resolved view, which is what makes the union
                # of two names that overlap come out with each view once.
                key = (base if v[0] is None else int(v[0]),) + tuple(v[1:])
                if key not in seen:        # a union: two names may share views
                    seen.add(key)
                    views.append(key)
        return views, f'--tta-views {" ".join(a.tta_views)}'

    sizes = [base] + [s for s in dict.fromkeys(a.tta_sizes) if s and s != base]
    crops = list(dict.fromkeys(a.tta_crops or [])) or ['center']
    ratios = list(dict.fromkeys(a.tta_ratios or [])) or ['plain']
    unknown = sorted(set(ratios) - set(VIEW_RATIOS))
    assert not unknown, f'unknown --tta-ratios {unknown}; choose from {sorted(VIEW_RATIOS)}'
    views = [(s, c, r, f) for s in sizes for c in crops
             for r in (ratios if c == 'center' else [None])
             for f in ((False, True) if a.tta_flip else (False,))]
    # The view list printed below names every view and its size, so the
    # description only has to say which of the two spellings was used -- that is
    # what a reader needs to reproduce the run.
    return views, 'the product flags'


def _worker_init(worker_id):
    """One CPU thread per loader worker.

    A DataLoader worker is a forked process, and by default it inherits the
    parent's intra-op thread count -- so ``--workers 8`` on a 16-core box would
    run 8 x 16 threads against 16 cores for every tensor op the transform
    pipeline does (``Normalize`` is elementwise and does use the pool), and spend
    the extra time context-switching.  The PIL resizes, which dominate, are
    single-threaded anyway.  This is why the parallelism is per *image*.
    """
    torch.set_num_threads(1)


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
        # `ck=` is passed for *every* size, not just the base one: build_clip
        # compares the size against the one recorded in the checkpoint and only
        # verifies when they agree.  A deliberate extra TTA size therefore gets a
        # printed note instead of a spurious mismatch, and the base size -- the
        # one that must be the model the weights were trained as -- still gets
        # checked.  Doing it this way means the check cannot be skipped by adding
        # a TTA size later.
        clip_model = build_clip(model_name, pretrained, size, ck=ck)
        # Rebuild the same optional single-head adapter used during training.
        # Older checkpoints have no flag and therefore stay on the historical
        # global-only path.
        local_head = bool(ck.get('local_head', ck_args.get('local_head', False)))
        net = Net(clip_model, len(classes), rank, target, local_head=local_head)
        if ck.get('pos_embed_trained') and size == ck_image_size(ck, verbose=False)[0]:
            pe_key = next((k for k in ('clip.visual.positional_embedding', 'clip.visual.pos_embed')
                           if k in ck.get('model', {})), None)
            if pe_key is None:
                raise SystemExit('checkpoint declares a trained positional embedding but the '
                                 'trained tensor is absent; refusing to infer with an interpolated grid')
        state = dict(ck.get('model', ck))
        trained_size = ck_image_size(ck, verbose=False)[0]
        if ck.get('pos_embed_trained') and size != trained_size:
            # An extra-resolution TTA view intentionally uses a freshly
            # interpolated grid.  The trained grid has a different token count,
            # so passing it to load_state_dict would hard-fail on shape mismatch.
            # The base/trained resolution remains strict and is checked above.
            for key in ('clip.visual.positional_embedding', 'clip.visual.pos_embed'):
                state.pop(key, None)
            print(f'note: dropping trained positional grid for extra {size}px view; '
                  f'using interpolated grid for TTA (checkpoint trained at {trained_size}px)')
        missing, _ = net.load_state_dict(state, strict=False)
        trained = {n for n, p in net.named_parameters() if p.requires_grad}
        lost = trained & set(missing)
        assert not lost, f'checkpoint has no trained weights for: {sorted(lost)[:5]}'
        if ck.get('pos_embed_trained') and size == ck_image_size(ck, verbose=False)[0]:
            actual = pos_embed_fingerprint(net.clip.visual)
            expected = ck.get('pos_embed')
            if expected is not None and actual != expected:
                raise SystemExit(f'trained positional embedding fingerprint differs after load: '
                                 f'checkpoint={expected}, loaded={actual}')
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
    base = a.image_size or ck_image_size(ck)[0]
    views, how = view_plan(a, base)

    ds = FlatImages(a.test, eval_transform(base), return_bad=True)
    files = ds.paths
    assert files, f'no images found under {a.test}'
    n_img = len(files)
    def vname(s, c, r, f):
        """``288px`` for the plain default view, more when it is not that."""
        bits = [f'{s}px']
        if c == 'center':
            if r != 'plain':                # `plain` is the default; naming it is noise
                bits.append(r)
        else:
            bits.append(c)
        if f:
            bits.append('flip')
        return ' '.join(bits)

    print(f'{n_img} test images found')
    print(f'{len(views)} view(s) from {how}: ' + ', '.join(vname(*v) for v in views) +
          f'  (aggregation: {a.tta_agg}, {len(views)} forward pass(es) per image)')
    if not a.workers and n_img * len(views) > 20000:
        # Measured on the sibling repository's runs: 8 views at 288 over the test
        # set is CPU-bound to the tune of ~2 h, because every view decodes and
        # resizes every image on one thread while the GPU waits.  Said out loud
        # because the run is long enough that discovering it afterwards costs the
        # GPU hours, and the fix is one flag.
        print(f'note: --workers 0 transforms every view on this thread. '
              f'--workers {min(8, os.cpu_count() or 1)} parallelises them across '
              f'images -- the transforms are deterministic, so the CSV is '
              f'byte-identical -- and on this many forwards it is the slow part.')

    # fp32 by default.  Training evaluates in bf16, but a submission is a
    # decision, and bf16's ~3 significant digits can flip a near tie in a
    # 750-way argmax -- so the submission path stays where the scores we are
    # comparing against were measured.  --amp bf16 is there for the multi-view
    # sweeps, where the extra speed matters and the CSVs are only compared.
    use_amp = a.amp == 'bf16' and device.type == 'cuda'
    names, unreadable = [], 0
    head, feat_sum, logit_sum = None, None, None
    # Workers are created per *view*, not kept: the dataset's transform is
    # re-pointed for each view, and a worker kept alive across views would hold
    # the copy it forked -- i.e. keep transforming with the previous view's
    # geometry.  `persistent_workers` is therefore off, and deliberately not set
    # anywhere.  One fork per view is nothing next to a pass over the test set.
    loader_kw = (dict(num_workers=a.workers, prefetch_factor=1, worker_init_fn=_worker_init)
                 if a.workers else {})
    # One model per *size*, reused across the crop/ratio/flip variants of that
    # size.  Views are ordered size-major, so this keeps exactly one tower
    # resident -- the same peak memory as rebuilding per view -- while a six-view
    # crop sweep at one size stops costing six CLIP loads.
    cur_size, cur_model = None, None
    for vi, (size, crop, ratio, flip) in enumerate(views):
        if size != cur_size:
            del cur_model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            cur_model, cur_size = make_model(size), size
        model = cur_model
        # Assigning the transform also re-resolves the dataset's grey-substitute
        # size (see FlatImages.transform), so an unreadable image is replaced by
        # the square *this* view's transform outputs -- a 288 view must not get a
        # 224-sized grey tile.
        ds.transform = eval_transform(size, flip, crop=crop,
                                      **({} if ratio is None
                                         else {'ratio': VIEW_RATIOS[ratio]}))
        if head is None:
            head = model.head        # same weights in every view; kept for the final decision
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                            pin_memory=device.type == 'cuda', **loader_kw)
        with torch.no_grad():
            for ims, idx, bad in loader:
                if vi == 0:                     # count each image once, not once per view
                    unreadable += int(bad.sum())
                    names.extend(Path(files[j]).name for j in idx.tolist())
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    out, z = model(ims.to(device, non_blocking=True), return_feat=True)
                # `z` is already L2-normalised by Net.forward
                if a.tta_agg == 'feat':
                    zz = z.float().cpu()
                    # Accumulate by *image index*, not batch-by-batch: the last
                    # batch is short unless n_img is a multiple of --batch-size,
                    # and adding the batches element-wise sums different images.
                    if feat_sum is None:
                        feat_sum = torch.zeros(n_img, zz.shape[1])
                    feat_sum.index_add_(0, idx, zz)
                else:
                    oo = out.float().cpu()
                    # Accumulate by *image index*, not batch-by-batch: the last
                    # batch is short unless n_img is a multiple of --batch-size,
                    # and adding the batches element-wise sums different images.
                    if logit_sum is None:
                        logit_sum = torch.zeros(n_img, oo.shape[1])
                    logit_sum.index_add_(0, idx, oo)
        print(f'  view {vi + 1}/{len(views)} done ({vname(size, crop, ratio, flip)})')
    del cur_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

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
    p.add_argument('--image-size', '--img-size', type=int, default=0, dest='image_size',
                   help='base (non-augmented) input resolution; 0 = as trained, which is '
                        'recorded in the checkpoint. Anything else resamples the positional '
                        'embeddings -- permitted, see README_AUTODL.md 5. '
                        '--img-size is the same flag under the sibling repo\'s spelling.')
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
    p.add_argument('--tta-views', nargs='+', default=[], dest='tta_views', metavar='NAME',
                   help='a named view set from train.VIEW_SETS, instead of building one '
                        'out of --tta-sizes/--tta-crops/--tta-ratios/--tta-flip. '
                        'This is how the leaderboard-validated recipes are reproduced: '
                        'tta4 is the four-view average that scored 66.2456 against '
                        '64.16 for a single view, and the product flags cannot express '
                        'it (--tta-flip flips every view, so it produces the 6-view '
                        'superset). Several names give their union. Available: '
                        + ', '.join(sorted(VIEW_SETS)) + '. Cannot be combined with '
                        'the product flags -- that is refused, not resolved, because '
                        'the CSV would then look like the named recipe and not be it.')
    p.add_argument('--tta-crops', nargs='*', default=None, dest='tta_crops',
                   metavar='POLICY',
                   help='crop policies to average in, from train.CROP_POLICIES: '
                        'center (default; resize short side to size*256/224 then centre '
                        'crop -- throws away part of the frame), full (squash the whole '
                        'image to size x size), pad (fit the long side and pad the rest: '
                        'keeps both the content and the aspect ratio). All three stay '
                        'inside the trained patch grid, so unlike --tta-sizes they need '
                        'no positional-embedding interpolation. '
                        'e.g. --tta-crops center full pad')
    p.add_argument('--tta-ratios', nargs='*', default=None, dest='tta_ratios',
                   metavar='NAME',
                   help='short-side ratios for the centre crop, from train.VIEW_RATIOS: '
                        'wide (1.0 -- the short side is the crop, i.e. open_clip\'s own '
                        'preprocessing), plain (1.143, the training default), mid (1.286), '
                        'tight (1.429). Ignored for --tta-crops full and pad, which fix the '
                        'short side themselves. On the leaderboard ONE of these *alone* lost '
                        '2.16 points against plain, while plain+flip+wide+tight averaged won '
                        '2.09 -- so the gain is the averaging, not any single framing. '
                        'e.g. --tta-ratios plain wide tight')
    p.add_argument('--amp', default='none', choices=['none', 'bf16'],
                   help='none (default) keeps the submission path in fp32, which is where the '
                        'scores we compare against were measured. bf16 is ~2x faster and is '
                        'fine for the multi-view sweeps -- but it can flip a near tie in a '
                        '750-way argmax, so do not use it for the final submission without '
                        'checking that it changes nothing.')
    p.add_argument('--tta-agg', default='logit', choices=['feat', 'logit'], dest='tta_agg',
                   help='logit: apply the head per view then average the logits (default). '
                        'feat: average the L2-normalised features then apply the head '
                        'once. Both use one checkpoint and one head. The default is '
                        'logit because every leaderboard number this project has '
                        '(64.16 / 66.2456 / 69.218 / 69.968) was measured that way, so '
                        'a recipe reproduced from infer.py is the recipe that scored; '
                        'the feature average is a reasoned alternative that has not '
                        'been on the platform, and probe.py --calib-agg compares the two.')
    p.add_argument('--workers', type=int, default=0,
                   help='transform worker processes (default 0: everything on this '
                        'thread). The transforms -- decode + resize, per image per view '
                        '-- are what makes a many-view run slow, and they parallelise '
                        'perfectly across images: measured on the sibling repository, 8 '
                        'views at 288 over a 37k-image test set is ~2 h single-threaded. '
                        'The output is byte-identical at any worker count (the '
                        'transforms are deterministic and the loaders yield in order), '
                        'which selftest asserts rather than assumes.')
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
