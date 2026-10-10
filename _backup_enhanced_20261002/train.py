"""Robust fine-tuning of CLIP ViT-B/32 for noisy-label fine-grained recognition.

Single model, single inference path, OpenAI CLIP ViT-B/32 backbone only.

Recipe
------
1. **LoRA** adapters (Hu et al., ICLR 2022) on the frozen CLIP visual tower plus
   a cosine classifier head -- a few hundred thousand trainable parameters, so
   the pretrained prior survives;
2. **class-balanced sampling** (``1/sqrt(freq)``) for the long-tailed stages;
3. a pure-CE **warm-up**, after which an **EMA teacher** drives
   * automatic noise filtering + *mixed* pseudo-labels (DivideMix / co-teaching
     style, see ``noise.py``): the teacher moves only ``--relabel-mix`` of the
     target mass off the given label instead of overwriting it,
   * confidence re-weighting of whatever is left;
4. an **active-passive loss** (NCE + RCE, Ma et al., ICML 2020) that cannot be
   fooled by a memorised wrong sample.  It sees the mixed target but *not* the
   label smoothing (see ``losses.nce``); smoothing only touches the CE term;
5. **EMA class prototypes** + cosine prototype contrastive loss (MoPro /
   Sel-CL style), bootstrapped from the frozen-CLIP class means;
6. **two-view consistency** and a **frozen-CLIP anchor** term to suppress
   representation drift / catastrophic forgetting.

Everything is seedable; ``--cudnn-benchmark`` is off by default so runs are
bit-reproducible on the same GPU.
"""
import argparse
import contextlib
import copy
import hashlib
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import open_clip

from losses import ce as xent, make_robust_loss
from noise import FrozenJudge, LabelTrustTracker, prototype_bootstrap
from diagnostics import append_jsonl, config_fingerprint, epoch_record

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Competition rule 五.1 / 十一(一).1: the backbone *has* to be CLIP ViT-B/32 --
# no other or larger visual model.  These are the only open_clip names that are
# that architecture ('-quickgelu' is the correct one for the official OpenAI
# weights, see the note on --model).  Checked instead of merely documented,
# because `--model ViT-L-14` would otherwise train happily and be disqualified.
ALLOWED_BACKBONES = {'ViT-B-32-quickgelu', 'ViT-B-32'}


def check_backbone(name):
    if name not in ALLOWED_BACKBONES:
        raise SystemExit(
            f'backbone {name!r} is not allowed. The competition requires CLIP ViT-B/32 '
            f'(allowed open_clip names: {sorted(ALLOWED_BACKBONES)}).')


#: short-side ratio of the eval transform.  224 -> 256 is the CLIP convention
#: (``Resize(256), CenterCrop(224)``); keeping the ratio fixed as the crop size
#: grows is what makes 288/320 a fair comparison against 224.
RESIZE_RATIO = 256 / 224

#: The resolution key this codebase writes into ``args`` is ``image_size``.  The
#: sibling repository (the one that produced the leaderboard scores we calibrate
#: against) spells the same thing ``img_size``, and checkpoints travel between
#: the two.  Both are accepted, but see ``ck_image_size`` for why the alias is
#: *announced* rather than silently honoured.
IMAGE_SIZE_KEYS = ('image_size', 'img_size')


def ck_image_size(ck, default=224, verbose=True):
    """Input resolution recorded in a checkpoint: ``(size, key)``.

    ``key`` is which spelling supplied the value -- ``None`` when the checkpoint
    predates the option or carries neither name, in which case ``default`` (224)
    is returned.

    Why this returns the key instead of just the size: reading ``img_size`` is
    reading *another codebase's* field.  It is almost certainly the same
    quantity, but "almost certainly" is exactly the wrong confidence level for a
    number that silently decides whether a 288 checkpoint is evaluated at 288 or
    at 224.  Nothing downstream can tell the two apart -- a 224 forward over
    288-trained weights runs fine and simply scores worse.  So callers that care
    can say which spelling they followed, and ``verbose`` prints it.

    Order matters: ``image_size`` is checked first in both the top level and
    ``args``, so a checkpoint carrying both spellings resolves the way this
    codebase wrote it and the alias cannot override the native field.
    """
    args = ck.get('args') or {}
    for key in IMAGE_SIZE_KEYS:
        for src, where in ((ck, 'top level'), (args, 'args')):
            v = src.get(key)
            if v:
                size = int(v)
                if verbose and key != IMAGE_SIZE_KEYS[0]:
                    print(f'note: checkpoint records its resolution as {key!r} '
                          f'({where}), not {IMAGE_SIZE_KEYS[0]!r} -- reading {size}px. '
                          f'That spelling comes from the sibling repository; if it '
                          f'is not the resolution these weights were trained at, '
                          f'every score below will be quietly wrong.')
                return size, key
    return int(default), None


def patch_grid(clip_model, size):
    """``(patch, n, clean)`` for a ``size``-pixel input.

    ``clean`` is False when the size is not a whole number of patches.  This
    matters for the resolution ladder: ViT-B/32 has **32px** patches, so 224 /
    256 / 288 / 320 / 352 / 384 are clean and **336 is not** (336/32 = 10.5).
    336 is the usual CLIP figure, but it comes from ViT-L/14@336 where
    336/14 = 24 exactly -- copying it onto a patch-32 tower does not mean what
    it means there.

    The consequence is exact, and the interesting part is how it fails.  In
    open_clip (checked against the vendored ``_oc_src/``, 3.x):

    * ``transformer.PatchEmbed.__init__`` sets ``grid_size = image_size // patch``
      (floor division), so 336 would give a 10x10 position-embedding grid;
    * the patch convolution itself can only produce ``(size - patch) // patch + 1``
      tokens, which for 336 is also 10.

    The two floor divisions agree, so 336 is *loadable*: it is the same 10x10
    model as 320, fed a 336px crop whose outer 16px no patch ever reads.  It
    would run, score, and be strictly dominated -- a silent waste rather than an
    error.  That is why it is refused rather than warned about: this function
    reports the geometry, and ``resize_positional_embedding`` raises on any size
    the convolution and the grid disagree about, so the organisers' "336"
    (which belongs to ViT-L/14, patch 14) cannot be adopted by accident here.
    """
    ps = None
    pe = getattr(getattr(clip_model, 'visual', None), 'patch_embed', None)
    for cand in (getattr(pe, 'patch_size', None),
                 getattr(getattr(pe, 'proj', None), 'kernel_size', None)):
        if cand is not None:
            ps = int(cand[0]) if isinstance(cand, (tuple, list)) else int(cand)
            break
    if not ps:                          # unknown tower -- fall back to its stem conv
        for m in clip_model.modules():
            if isinstance(m, nn.Conv2d) and m.stride[0] > 1:
                ps = int(m.kernel_size[0])
                break
    if not ps:
        return None, None, True         # cannot tell; never cry wolf
    size = int(size)
    n = (size - ps) // ps + 1           # what the patch convolution really produces
    return ps, n, (size % ps == 0 and n == size // ps)


def resize_positional_embedding(visual, size, antialias=False, verbose=True):
    """Resample the positional grid so the tower accepts ``size``-pixel input.

    Returns the grid side actually installed, or ``None`` when nothing had to
    change (``size`` matches the grid already there).

    **Why this exists rather than open_clip doing it.**  open_clip resamples the
    positional grid on load when ``force_image_size`` is passed, and *that* is
    what this codebase used to rely on.  It is not what got the score.  The
    sibling repository -- whose 288 run is the one that produced 69.218 on the
    leaderboard -- builds its tower by hand this way, and the two are **not
    interchangeable**: open_clip's ``resize_pos_embed`` interpolates with
    ``antialias=True`` (``_oc_src/model.py:811-817``) while a plain
    ``F.interpolate`` defaults to ``antialias=False``.  Both produce a
    ``(82, 768)`` tensor for 288.  Same shape, different numbers, no error
    anywhere.

    That matters more here than in most codebases because **the positional
    embedding is never stored in a checkpoint**.  ``trainable_state_dict`` keeps
    only parameters with ``requires_grad=True``, the grid is frozen, so every
    load rebuilds it -- which means this function, not the checkpoint, is what
    decides which model you are actually running.  Evaluating their 288 weights
    through a differently-interpolated grid would be scoring a model neither
    repository trained, and nothing would say so.

    ``antialias=False`` is therefore the default: it is the sibling's default and
    it is the one attached to a known leaderboard score.  The parameter is open
    so the difference is nameable and measurable rather than buried in a choice
    of code path.

    Replacing the attribute (``setattr``) rather than copying into it is
    deliberate: the tensor *grows* (50 -> 82 tokens) and ``copy_`` is in-place,
    so it demands the very shape this function exists to change.
    ``requires_grad`` is carried over so the frozen/unfrozen bookkeeping that
    checkpoints depend on does not shift.
    """
    patch = getattr(visual, 'patch_size', None)
    if patch is None:
        pe_ = getattr(getattr(visual, 'patch_embed', None), 'proj', None)
        patch = getattr(pe_, 'kernel_size', None)
    if patch is None:
        raise SystemExit('vision tower exposes no patch size; cannot resize its grid')
    patch = int(patch[0] if isinstance(patch, (tuple, list)) else patch)
    if size % patch:
        raise SystemExit(
            f'image size {size} is not a multiple of the {patch}px patch size. Legal '
            f'sizes are {patch * 7} / {patch * 8} / {patch * 9} / ... -- note 336 is '
            f'not one of them (336/32 = 10.5); that figure belongs to CLIP ViT-L/14, '
            f'whose patch is 14.')
    target = size // patch

    # open_clip spells it `positional_embedding`, shape (1 + g*g, width), CLS
    # first and no batch dimension.  timm spells it `pos_embed` with a leading
    # batch dimension.  Which one appears depends on how the tower was built.
    name = 'positional_embedding'
    pe = getattr(visual, name, None)
    if pe is None:
        name = 'pos_embed'
        pe = getattr(visual, name, None)
    if pe is None:
        raise SystemExit('no positional embedding found on the vision tower')

    flat = pe[0] if pe.dim() == 3 else pe
    n_tok, width = flat.shape
    base = int(round((n_tok - 1) ** 0.5))
    if base * base + 1 != n_tok:
        raise SystemExit(f'{n_tok} positional tokens is not grid^2 + 1; cannot resample')
    if base == target:
        return None

    cls, patches = flat[:1], flat[1:]
    grid = patches.reshape(base, base, width).permute(2, 0, 1).unsqueeze(0).float()
    grid = F.interpolate(grid, size=(target, target), mode='bicubic',
                         align_corners=False, antialias=bool(antialias))
    merged = grid.squeeze(0).permute(1, 2, 0).reshape(target * target, width)
    merged = torch.cat([cls.float(), merged], dim=0).to(pe.dtype).to(pe.device)
    if pe.dim() == 3:
        merged = merged.unsqueeze(0)
    if isinstance(pe, nn.Parameter):
        merged = nn.Parameter(merged, requires_grad=pe.requires_grad)
    setattr(visual, name, merged)

    # keep the tower's own bookkeeping consistent with the grid it now holds
    if hasattr(visual, 'image_size'):
        visual.image_size = (size, size)
    if hasattr(visual, 'grid_size'):
        visual.grid_size = (target, target)
    if verbose:
        print(f'{size}px: positional grid resampled {base}x{base} -> {target}x{target} '
              f'(bicubic, antialias={bool(antialias)})')
    return target


def pos_embed_fingerprint(visual):
    """A cheap, comparable summary of the tower's positional embedding.

    Stored in checkpoints so that a later load can prove it rebuilt the *same*
    grid rather than merely one of the right shape.  Shape is not enough: the
    whole point is that two different interpolation settings give ``(82, 768)``
    and are indistinguishable without looking at the numbers.

    Returns ``None`` when the tower has no positional embedding, and never
    raises -- it runs inside the training loop's save path.
    """
    for name in ('positional_embedding', 'pos_embed'):
        pe = getattr(visual, name, None)
        if pe is not None and hasattr(pe, 'detach'):
            arr = pe.detach().to('cpu', torch.float32).contiguous().numpy()
            return {'shape': list(arr.shape),
                    'sha': hashlib.sha256(arr.tobytes()).hexdigest()[:16]}
    return None


def verify_pos_embed(ck, visual):
    """Check the rebuilt positional grid against the fingerprint in ``ck``.

    Returns ``True`` when it matches, ``None`` when the checkpoint carries no
    fingerprint, and **raises** on a mismatch.

    There is deliberately no size argument.  The grid being checked is the one
    ``visual`` already carries, and the fingerprint records the shape as well as
    the values -- so asking for a size here could only ever restate what the
    comparison already proves, or contradict it.

    ``None`` is not a pass and is reported as such: every checkpoint written
    before this check existed -- including the sibling repository's 288 run that
    scored 69.218 -- has no fingerprint to compare against.  For those the only
    protection is that ``resize_positional_embedding`` reproduces their
    mechanism exactly, which is why the parameters are what they are.  A
    mismatch, by contrast, is exactly the silent failure this exists to catch:
    same shape, different numbers, plausible-looking accuracy on the way out.
    """
    got = pos_embed_fingerprint(visual)
    want = (ck or {}).get('pos_embed')
    if want is None or got is None:
        return None
    mismatch = (
        f'the rebuilt positional embedding does not match the one this checkpoint '
        f'was trained with (built shape {got["shape"]} sha {got["sha"]}, recorded '
        f'shape {want.get("shape")} sha {want.get("sha")}).  Same shape with '
        f'different numbers is the failure this check exists for -- most likely the '
        f'resolution, or the antialias setting, differs from the training run. '
        f'Every number below would be computed on a model that was never trained.')
    # The token count must agree whether or not the grid was trained: it is what
    # the tower's patch grid produced, so a mismatch means the wrong resolution
    # rather than a training choice.
    if got['shape'] != want.get('shape'):
        raise SystemExit(mismatch)
    if got['sha'] != want.get('sha'):
        if (ck or {}).get('pos_embed_trained'):
            # --train-pos-embed: this checkpoint's grid is not the interpolation,
            # so its values are *supposed* to differ from the one build_clip just
            # rebuilt.  They travel in the state dict and are restored a few lines
            # later, which is why this is a note and not the failure above.
            print(f'note: this checkpoint\'s positional grid was trained '
                  f'(--train-pos-embed, {got["shape"][0] - 1} tokens), so it deliberately '
                  f'differs from the interpolation just rebuilt; the trained values are '
                  f'restored with the state dict.')
            return True
        raise SystemExit(mismatch)
    return True


def enable_pos_embed_training(visual):
    """Make the resampled positional grid trainable (``--train-pos-embed``).

    ``resize_positional_embedding`` can only *interpolate* OpenAI's 7x7 grid onto
    the larger grid a bigger input needs, and ``Net.__init__`` then freezes it
    with the rest of the backbone.  The interpolation error is therefore
    permanent, it is largest exactly where the extra resolution was supposed to
    pay, and it grows with the distance from 224 -- this project's own probe put
    the cosine similarity of the resampled grid to the pretrained one at
    0.9917 / 0.9820 / 0.9683 / 0.9581 for 256 / 288 / 320 / 352.  Letting the
    grid learn instead of only interpolating is what turns a larger input from a
    loss into a gain: the same 384px resolution with the same weights scored
    68.5476 with the grid frozen and 71.1382 with it trainable.

    Ordering, all three of which fail *silently* when wrong:

    * **After ``Net(...)``**, whose ``__init__`` freezes every parameter that is
      not a LoRA factor -- including this one.
    * **Before ``param_groups``**, which collects whatever carries
      ``requires_grad`` at that moment.  Enable it later and the grid is simply
      not in the optimiser: nothing raises and nothing learns.
    * **Before the teacher is deep-copied**, so the EMA teacher is the same
      architecture as the student it averages.  (The EMA update follows the
      student's own ``requires_grad`` flags, so the copy is the only thing that
      has to be in order.)

    Returns the parameter's name, for the log line and for the checkpoint's
    ``pos_embed_trained`` marker that ``verify_pos_embed`` reads.  Raises when
    the tower exposes no grid, which cannot happen for the allowed backbones and
    would otherwise be a silent no-op.
    """
    for name in ('positional_embedding', 'pos_embed'):
        pe = getattr(visual, name, None)
        if isinstance(pe, nn.Parameter):
            pe.requires_grad_(True)
            return name
    raise SystemExit(
        'this visual tower exposes no positional embedding to train, so '
        '--train-pos-embed would silently do nothing -- check --model.')


def build_clip(name, pretrained='openai', image_size=224, ck=None):
    """Create the CLIP visual tower at ``image_size``.

    Architecture and weights are unchanged -- resampling the positional grid is
    interpolation, not new information -- and the competition explicitly permits
    changing the input resolution with the backbone and weights held fixed (see
    README_AUTODL.md 5).

    224 is special-cased to *not* resample: its grid already matches, and
    resampling a 7x7 grid onto itself is not guaranteed to be a no-op, so the
    default path stays bit-identical to the runs whose leaderboard scores are
    the calibration.

    Prefer a multiple of the patch size (32 for ViT-B/32) -- ``patch_grid``
    warns, and ``resize_positional_embedding`` refuses outright, because a size
    like 336 is silently *the same model as 320* fed a wider crop no patch ever
    reads.

    ``ck`` is optional and turns on verification: the rebuilt grid is checked
    against the fingerprint recorded in the checkpoint.  Pass it wherever a
    checkpoint is being loaded -- it is the only place the check can be made, and
    a check that has to be remembered is a check that will be forgotten.

    Two situations are *not* a mismatch and are reported as notes instead:
    a checkpoint that records no fingerprint (every one written before this
    check existed, including the sibling repository's), and a caller that has
    deliberately asked for a size other than the one the checkpoint was trained
    at.  The second is a legitimate experiment -- TTA at a second resolution is
    exactly that -- and its grid is *supposed* to differ, so raising there would
    make the flag unusable.
    """
    check_backbone(name)
    size = int(image_size)
    model = open_clip.create_model(name, pretrained=pretrained)
    if size != 224:
        # no warn-then-continue here: resize_positional_embedding refuses any size
        # whose patch grid and token count disagree, and its message explains why
        # (336 is the trap, and it is the number the brief itself names)
        resize_positional_embedding(model.visual, size)
    if ck is not None:
        trained_size, _ = ck_image_size(ck, verbose=False)
        if int(trained_size) != size:
            print(f'note: evaluating a checkpoint trained at {int(trained_size)}px at '
                  f'{size}px, so its positional grid was resampled and the recorded '
                  f'fingerprint deliberately does not apply. This is the configuration '
                  f'the docs call interpolate-only: it loses as a single view and is '
                  f'only worth trying inside an average.')
        elif verify_pos_embed(ck, model.visual) is None:
            print(f'note: this checkpoint carries no positional-embedding fingerprint, so '
                  f'the {size}px grid just built could not be checked against the one it '
                  f'was trained with. Checkpoints written by this codebase record one; '
                  f'the sibling repository\'s do not -- matching its interpolation '
                  f'mechanism is what stands in for the check there.')
    return model


#: the view geometries ``eval_transform(crop=...)`` can produce
CROP_POLICIES = ('center', 'full', 'pad')

#: Named short-side ratios for the ``center`` crop.  The resize is
#: ``size * ratio`` and the crop is ``size``, so on the *short* axis the retained
#: fraction is exactly ``1/ratio``: 1.0 keeps all of it, 1.429 keeps 70%.
#: (The long axis loses more than that -- that is the aspect-ratio term, and it
#: does not depend on ``ratio`` at all.)
#:
#: ``plain`` is the training default.  The others are the sibling repository's
#: names: it measured them on the leaderboard, so they are the only view labels
#: in this project that arrive with evidence attached, and reusing its names is
#: what lets a number measured here be compared against a number measured there.
VIEW_RATIOS = {
    'wide':  1.0,              # short side IS the crop -> open_clip's own preprocessing
    'plain': RESIZE_RATIO,     # 256/224 -> 87.5%, the resize training uses
    'mid':   288 / 224,        # 77.8%
    'tight': 320 / 224,        # 70% -- more pixels per object, less of the frame
}

#: A view tuple is exactly ``(size, crop, ratio_name, flip)`` -- see ``VIEW_SETS``.
VIEW_LEN = 4


def check_view_tuple(name, views):
    """Raise unless every view in ``views`` is a well-formed 4-tuple."""
    for v in views:
        if not isinstance(v, (tuple, list)) or len(v) != VIEW_LEN:
            raise SystemExit(
                f'{name}: a view must be (size, crop, ratio_name, flip) -- got {v!r}. '
                f'Use None for the trained resolution; do not omit the field, or a '
                f'view that meant to change the size silently becomes a duplicate of '
                f'the plain one and the set measures nothing.')
    return tuple(tuple(v) for v in views)


#: Named eval-time view sets, scored by ``probe.py --calib`` and selectable on the
#: submission path with ``infer.py --tta-views NAME``.  Named where possible the
#: way the leaderboard numbers these are graded against were named in the sibling
#: repository.  Each entry is a tuple of ``(size, crop, ratio_name, flip)``:
#:
#: ``size``
#:     ``None`` means **the resolution the checkpoint was trained at**.  So
#:     ``plain`` / ``wide`` / ``tta4`` still mean exactly what they meant before
#:     sizes were expressible here, and the leaderboard facts graded against them
#:     stay comparable.  An int means that size *for this same checkpoint* --
#:     a second view of one model, which is TTA and is permitted, not a second
#:     model, which is not.  Every set below is one checkpoint.
#: ``crop``
#:     resolved through ``CROP_POLICIES``.
#: ``ratio_name``
#:     ``None`` for the policies that take no ratio -- ``full`` squashes and
#:     ``pad`` letterboxes the whole frame, so there is no short side left to
#:     choose from.
#: ``flip``
#:     horizontal flip.
#:
#: A 3-tuple is **not** accepted as a shorthand for "the trained size".  It would
#: be the natural thing to write and it would be wrong in a way nothing else
#: notices: a size-carrying view that lost its size would quietly become a
#: duplicate of the plain one, and a set that measured nothing new would report
#: an average of one view twice.
#:
#: A set that ``infer.py``'s product flags can already produce names the exact
#: flags alongside it, so the two ways of asking for the same views cannot drift.
#: Sets with no such line are precisely the ones the product cannot express (it
#: flips *either every* view or none), and they are why the name table exists.
VIEW_SETS = {
    'plain': ((None, 'center', 'plain', False),),
    'wide':  ((None, 'center', 'wide', False),),
    'mid':   ((None, 'center', 'mid', False),),
    'tight': ((None, 'center', 'tight', False),),
    'flip':  ((None, 'center', 'plain', True),),
    # the four-view average that scored 66.2456 on 2026-09-24, against 64.16 for
    # `plain` alone -- the only test-time recipe with a leaderboard number behind
    # it.  This is the sibling repository's DEFAULT_TTA, in our parameterisation.
    # All four views are zooms of one axis; that is the property the sets below
    # are chosen to contrast with.
    #   infer: --tta-ratios plain wide tight --tta-flip   (6 views -- see there)
    'tta4':  ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
              (None, 'center', 'wide', False), (None, 'center', 'tight', False)),
    # the same idea along the axis that repository did *not* spend on: a different
    # crop policy instead of two more zooms of the same one.  `full` squashes the
    # whole frame in, so it keeps the top and bottom a centre crop throws away --
    # the frames here are fine-grained close-ups, where that is a real difference.
    #   infer: --tta-crops center full --tta-ratios plain --tta-flip
    'axis4': ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
              (None, 'full', None, False), (None, 'full', None, True)),
    'pad4':  ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
              (None, 'pad', None, False), (None, 'pad', None, True)),
    # ---- the same two axes, spent to saturation ------------------------------ #
    # Every ratio of the zoom axis, flipped: the eight-view extrapolation of
    # `tta4`.  If tta4's +2.09 was the average doing the work, this is where it
    # continues; if the axis is saturated it will score the same as tta4 while
    # costing twice the forward passes, which is itself the answer worth having
    # before any of the 6- and 8-view sets below is submitted.
    #   infer: --tta-ratios plain wide mid tight --tta-flip
    'tta8':  ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
              (None, 'center', 'wide', False), (None, 'center', 'wide', True),
              (None, 'center', 'mid', False), (None, 'center', 'mid', True),
              (None, 'center', 'tight', False), (None, 'center', 'tight', True)),
    # Every crop policy, flipped: the crop axis' counterpart of `tta8`, i.e. the
    # same question asked of the axis `tta4` leaves unspent.
    #   infer: --tta-crops center full pad --tta-flip
    'crops6': ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
               (None, 'full', None, False), (None, 'full', None, True),
               (None, 'pad', None, False), (None, 'pad', None, True)),
    # tta4 plus the two crop-axis views, six forwards in total: the cheapest
    # "both axes at once" recipe, and the one to compare `tta8` against, since
    # the two cost the same and differ only in where the extra views are spent.
    # `probe.py --calib` scores them on val; the leaderboard is the tie-break.
    # No product flags can express this one -- flips are all-or-nothing there --
    # so it is reachable only as a name.
    'mix6':  ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
              (None, 'center', 'wide', False), (None, 'center', 'tight', False),
              (None, 'full', None, False), (None, 'full', None, True)),
    # ---- the resolution axis ------------------------------------------------- #
    # A second resolution as a *view* of the same checkpoint.  The prior is
    # negative and stated in the plan rather than discovered later: `RESIZE_RATIO`
    # is fixed, so every size keeps the same fraction of the frame (76.6% of a
    # square image at 224 and at 320 alike), which makes a size view close to a
    # re-encoding of the plain one rather than new evidence -- and a worse,
    # correlated view dilutes an average instead of denoising it.  The exception
    # worth measuring is a *nearby* size: for a 288-trained model, 320 refines the
    # same grid family (82 -> 101 tokens) where 224 would be a bigger jump.
    # For a 320-trained model the nearby size is 288 -- `s256` and `s352` are the
    # jumps, in the losing direction at one end and into upsampling at the other.
    's256':  ((256, 'center', 'plain', False),),
    's288':  ((288, 'center', 'plain', False),),
    's320':  ((320, 'center', 'plain', False),),
    's352':  ((352, 'center', 'plain', False),),
    # the proven four views plus one at 320, which is the only version of the
    # multi-resolution idea that could still be submitted: it is a superset of
    # `tta4`, so it can only be judged against a recipe that already scored.
    'tta4s320': ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
                 (None, 'center', 'wide', False), (None, 'center', 'tight', False),
                 (320, 'center', 'plain', False)),
    # ... and the symmetric version of it: the extra resolution as a real view,
    # flipped like the others, six forwards -- so the comparison against
    # `tta4s320` says whether the fifth view was diluted by being unpaired.
    'tta4s320f': ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
                  (None, 'center', 'wide', False), (None, 'center', 'tight', False),
                  (320, 'center', 'plain', False), (320, 'center', 'plain', True)),
    # The same two recipes for a *320*-trained checkpoint, where the nearby size is
    # 288 and not 320.  This is not a variant for its own sake: at 320 the entries
    # above are skipped as degenerate (`None` and the literal 320 both mean 320px,
    # i.e. a second copy of the plain view), so without these two a 320 run's sweep
    # has *no* multi-resolution multi-view row at all -- and the question the 288
    # run was asked ("does a fifth, nearby-resolution view on top of `tta4` help?")
    # would go unasked on the very checkpoint that scored 69.968.  Registered here
    # before the run rather than after seeing the numbers, which is the only thing
    # that makes a sweep row evidence.
    'tta4s288': ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
                 (None, 'center', 'wide', False), (None, 'center', 'tight', False),
                 (288, 'center', 'plain', False)),
    'tta4s288f': ((None, 'center', 'plain', False), (None, 'center', 'plain', True),
                  (None, 'center', 'wide', False), (None, 'center', 'tight', False),
                  (288, 'center', 'plain', False), (288, 'center', 'plain', True)),
}

#: View sets whose first element is the training view, i.e. the transform
#: ``train.py`` scores its hold-out split with.  Used to decide which row a
#: checkpoint's recorded ``val_acc`` can be checked against; see ``calib``.
TRAINING_VIEW = VIEW_SETS['plain']

#: The multi-view sets ``probe.py --calib-sweep`` measures *in addition* to its
#: singles and pairs -- the ones the pair enumeration cannot reach.  Listed here
#: beside the table so the sweep cannot quietly stop covering a set that the
#: submission path can still be asked for.
VIEW_SETS_SWEPT = ('tta4', 'axis4', 'pad4', 'mix6', 'crops6', 'tta8',
                   'tta4s320', 'tta4s320f', 'tta4s288', 'tta4s288f')


def resolved_views(views, base):
    """``views`` with every ``None`` size replaced by ``base`` -- what runs."""
    return tuple((int(base) if s is None else int(s), c, r, f) for (s, c, r, f) in views)


def is_degenerate(views, base):
    """Would this set, evaluated at ``base``, feed the same input more than once?

    A size view at the size the checkpoint was trained at *is* the plain view --
    ``RESIZE_RATIO`` is fixed, so the same size means the same pixels.  Two views
    collide only if all four fields match, so the flipped twin of a view is not a
    duplicate of it.  The table is written against "the trained resolution", so
    which entries collapse depends on the run: for a 320-trained checkpoint
    ``tta4s320`` offers five views for four distinct inputs (its ``None``-sized
    plain view and its literal 320 one are the same pixels), and for a 288-trained
    one the pair ``s288+plain`` is two views for one.  Scoring such a set is not an
    error, it is an average with one input weighted twice, and it is *reported*
    like any other row -- which is the problem.  Both callers refuse it instead:
    ``probe.py`` skips the row, ``infer.py`` refuses the command, because a
    submission that looks like the named recipe and is not it is worse than no
    submission.

    What this deliberately does *not* flag is a single size view that coincides
    with the trained size (``s320`` on a 320-trained checkpoint).  That set
    declares one view and delivers one; it is the plain view, named by the size it
    uses.  Refusing it would refuse ``plain`` itself, which is a legitimate thing
    to ask for -- it is the 64.16 recipe -- so the redundancy is left visible
    where it does no harm: in a sweep, the two rows print the same number.
    """
    rv = resolved_views(views, base)
    return len(set(rv)) != len(rv)


class PadToSquare:
    """Scale the *long* side to ``size``, then pad the short side to a square.

    ``Resize(size)`` alone would leave a non-square tensor, and the vision tower
    was built for a ``size`` x ``size`` patch grid, so the square has to be
    restored.  Padding with the CLIP mean colour keeps the border away from
    anything the normalisation will amplify.
    """

    def __init__(self, size, fill=CLIP_MEAN):
        self.size = int(size)
        self.fill = tuple(int(round(255 * c)) for c in fill)

    def __call__(self, img):
        w, h = img.size
        s = self.size / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        w, h = img.size
        out = Image.new('RGB', (self.size, self.size), self.fill)
        out.paste(img, ((self.size - w) // 2, (self.size - h) // 2))
        return out


def eval_transform(size=224, flip=False, *, crop='center', ratio=RESIZE_RATIO):
    """Eval-time preprocessing.  One definition, shared by everything.

    Used by ``train.py`` (hold-out val), ``infer.py``, ``valmetrics.py``,
    ``analyze.py`` and ``probe.py``.  A mismatch between any two of them is
    invisible and silently costs accuracy; the only symptom is that a
    checkpoint's logged ``val_acc`` stops reproducing (see the self-check in
    ``valmetrics.py``).

    ``crop`` selects how much of the image survives:

    ``'center'``
        Resize the short side to ``size*ratio``, centre crop.  At the default
        ``ratio`` of 256/224 that is the CLIP and ImageNet convention, and what
        every run up to 2026-09-24 used.  ``ratio`` is a separate knob because
        the sibling repository scored three values of it on the leaderboard --
        see ``VIEW_RATIOS``; the default reproduces the earlier recipe exactly.
    ``'full'``
        Squash the whole image to ``size`` x ``size``: nothing is cropped, but
        the aspect ratio is distorted.
    ``'pad'``
        Scale the long side to ``size`` and pad the rest: nothing is cropped
        *and* the aspect ratio is kept, at the cost of lowering the effective
        resolution along the short side.

    Why this axis exists: the centre crop is not free.  Measured on the
    geometries this dataset actually contains (the resized size is what
    ``Resize(size*256/224)`` actually produces -- 341, not 341.33) ::

        480x640 (3:4)    -> 256x341  keeps 57.5%   87.5% wide, 65.7% high
        480x720 (2:3)    -> 256x384  keeps 51.0%   87.5% wide, 58.3% high
        480x480 (square) -> 256x256  keeps 76.6%   87.5% wide, 87.5% high

    The last row is the point that has nothing to do with aspect ratio:
    ``Resize`` followed by ``CenterCrop`` discards the outer 1/8 of *both* axes
    whatever the shape, so 23.4% of the frame is gone even from a square image.
    For a fine-grained task, where the discriminative part (a bill, a petal, a
    stamen) is small and often off-centre, that is the part that may be
    deciding.  On a 2:3 portrait nearly half the frame never reaches the model.

    All three policies stay inside the trained patch grid, so unlike a
    resolution change this needs no positional-embedding interpolation.  That is
    *not* the same as "no risk", though -- they differ in apparent object size,
    the quantity FixRes (Touvron et al. 2019) says must match between train and
    test.  Relative to ``full`` = 1.000, on a 3:4 image::

        training RRC(scale=(crop_min,1))   [1.000, 1.348], mean 1.148
        'center'                            1.320
        'full'                              1.000
        'pad'                               0.866

    So ``full`` and ``pad`` present objects *smaller* than the training mean --
    exactly the mismatch FixRes warns about -- while ``center`` presents them
    larger.  Neither is neutral, and the two effects (this one, and the frame
    content ``center`` discards) push in opposite directions.  Which wins is a
    measurement, not an argument -- ``probe.py --tta`` scores them on val.

    Within ``center`` the arithmetic is exact: the crop is always ``size`` while
    the resize is ``size*ratio``, so an object's apparent size scales with
    ``ratio`` and nothing else::

        'wide'   ratio 1.000 -> 1.155
        'plain'  ratio 1.143 -> 1.320     <- what training and val use
        'mid'    ratio 1.286 -> 1.485
        'tight'  ratio 1.429 -> 1.650

    ``wide`` sits closest to the training mean and ``tight`` is 44% above it --
    and on the leaderboard ``wide`` alone lost 2.16 points while a four-view
    average containing ``tight`` won 2.09.  So apparent size does **not** rank
    these views, which is the honest reading of the one measurement this project
    actually has.  FixRes is about matching train and test; it is not a licence
    to reorder test-time views by a number computed from training alone.
    """
    if crop not in CROP_POLICIES:
        raise ValueError(f'crop={crop!r}, expected one of {CROP_POLICIES}')
    size = int(size)
    if crop == 'center':
        # ratio == 1.0 makes the resize and the crop the same number, so the
        # short axis survives whole -- that is `wide`, and it is worth allowing
        # as an ordinary value of the knob rather than special-casing.
        ops = [transforms.Resize(max(1, int(round(size * float(ratio))))),
               transforms.CenterCrop(size)]
    elif crop == 'full':
        ops = [transforms.Resize((size, size))]
    else:
        ops = [PadToSquare(size)]
    if flip:                            # test-time augmentation, p=1.0
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
    return transforms.Compose(ops)


def transform_size(tf, default=224):
    """The square side length a transform pipeline outputs, else ``default``.

    ``ImageFolderNoisy`` needs this: when a file cannot be decoded it substitutes
    a grey image, and the grey image should be the size the model expects.  The
    only place that size is knowable from is the transform, which is handed in.

    **Last match wins, not first.**  Every pipeline here ends with its resizing
    op -- ``Resize(256), CenterCrop(224)`` names *two* sizes and only the second
    one is the output resolution -- so scanning forwards and taking the first
    would return 256 for the most common transform in the codebase.  Reading
    the ops in order and keeping the final hit is what makes ``center`` resolve
    to 224 rather than to its resize factor.

    Unrecognised pipelines fall back rather than guessing: a wrong guess here is
    invisible (grey on grey looks the same at any size, see ``_load``).
    """
    size = None
    for op in getattr(tf, 'transforms', [tf]):
        v = getattr(op, 'size', None)
        if isinstance(v, int):
            size = v
        elif isinstance(v, (tuple, list)) and len(v) == 2 \
                and isinstance(v[0], int) and v[0] == v[1]:
            size = int(v[0])
    return int(default) if size is None else int(size)


class FlatImages(Dataset):
    """Flat ``test/*.jpg`` folder, for the submission path and the proxy alike.

    ``infer.py`` and ``probe.py`` both have to read a directory of images that
    carries no labels, in one order, with the same grey fallback: a file Pillow
    cannot decode must still produce a row, or a truncated file (the competition
    warns about those) makes the submission row count mismatch the test set --
    and makes every statistic the proxy computes be computed over a different set
    than the one it is standing in for.  One class, so the two cannot drift.

    ``return_bad`` adds a third element saying whether the grey substitute was
    used, for callers that report how many images were unreadable.  Off by
    default: it is a per-item cost paid to build a diagnostic, and the callers
    that only need features should not pay it.

    Ordering is ``sorted`` over the paths, which is what ``infer.py`` writes its
    CSV in and what ``extract`` assumes when it aligns a frozen pass with a
    model pass -- so a row here and a row there are the same image.

    ``transform`` is a property, not an attribute, because both callers re-point
    it once per view (a size view wants a different square) and the grey
    substitute has to follow: ``img_size`` is resolved out of whatever transform
    is installed *now*, the same way ``ImageFolderNoisy._load`` resolves it once.
    Assigning a transform and leaving a stale ``img_size`` behind would give a
    288 view a 224-sized grey tile -- invisible, because grey on grey looks the
    same, and wrong for the same reason.
    """

    def __init__(self, root, transform, return_bad=False):
        self.paths = sorted(str(p) for p in Path(root).rglob('*')
                            if p.is_file() and p.suffix.lower() in IMG_EXTS)
        self.transform = transform
        self.return_bad = return_bad

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, tf):
        self._transform = tf
        # the grey substitute should be the size the model expects, and the
        # transform is the only thing that knows it
        self.img_size = transform_size(tf)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        bad = False
        try:
            img = Image.open(self.paths[i]).convert('RGB')
        except Exception:                       # truncated / unreadable
            bad = True
            n = self.img_size
            img = Image.new('RGB', (n, n), (127, 127, 127))
        t = self.transform(img)
        return (t, i, bad) if self.return_bad else (t, i)


def smooth_target(t, smooth, nclass):
    """Mix a one-hot-ish target towards uniform.

    Label smoothing is the cheapest defence against the failure this project
    actually measured: on the 500-class round the hold-out accuracy (0.7305)
    came out *above* the leaderboard score (0.6802), which is only possible if
    the model memorised the class-consistent part of the label noise.  Smoothing
    caps how confident a single (possibly wrong) label can make the model, so
    there is less to memorise.

    It is applied to the CE term only -- see the note in ``losses.nce``.
    """
    if smooth <= 0:
        return t
    return (1 - smooth) * t + smooth / nclass


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed=3407, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark picks kernels by timing -> results differ run to run
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id):
    """Seed the python/numpy RNG of every dataloader worker (torch is seeded by
    DataLoader itself from the loader's generator).

    **This is no longer what makes the augmentation reproducible.**  That is done
    per item in ``ImageFolderNoisy.augmented``, which derives its randomness from
    ``(seed, index, view)`` rather than from the worker's RNG -- seeding the
    worker only makes the run repeatable *on one machine with one ``--workers``*.
    Kept because anything else that draws randomness in a worker should still
    start from a defined state.
    """
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# model: LoRA + cosine head + EMA prototypes
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus a trainable low-rank update ``B @ A``.

    ``enabled = False`` collapses the layer back to the original CLIP layer.
    That is how ``Net.anchor_feat`` recovers frozen-CLIP features **without**
    keeping a second copy of the whole visual tower in memory.

    It must stay a drop-in replacement for ``nn.Linear``: ``nn.MultiheadAttention``
    does not call ``out_proj(x)``, it reads ``out_proj.weight`` / ``out_proj.bias``
    and hands them to ``F.multi_head_attention_forward``.  The ``weight`` property
    below therefore returns the *merged* matrix, so the adapter stays effective on
    that path too (the only difference is that LoRA dropout is skipped there,
    which is irrelevant for the frozen/adapted linear algebra).
    """

    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank, self.scale = rank, alpha / rank
        self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.enabled = True

    @property
    def weight(self):
        if not self.enabled:
            return self.base.weight
        return self.base.weight + (self.B @ self.A) * self.scale

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + F.linear(self.drop(x), self.B @ self.A) * self.scale


def add_lora(module, rank=8, alpha=16, dropout=0.05, target='all', prefix=''):
    """Wrap every ``nn.Linear`` of ``module`` (``target='mlp'``: MLP only)."""
    n = 0
    for name, child in list(module.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        is_mlp = name in ('fc1', 'fc2', 'c_fc', 'c_proj', 'mlp')
        if isinstance(child, nn.Linear) and 'head' not in full.lower() and (target == 'all' or is_mlp):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            n += 1
        else:
            n += add_lora(child, rank, alpha, dropout, target, full)
    return n


@contextlib.contextmanager
def lora_disabled(module):
    """Temporarily turn every LoRA layer under ``module`` off (-> frozen CLIP)."""
    mods = [m for m in module.modules() if isinstance(m, LoRALinear)]
    prev = [m.enabled for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, p in zip(mods, prev):
            m.enabled = p


class CosineClassifier(nn.Module):
    """Normalised-weights classifier with a learnable temperature."""

    def __init__(self, dim, nclass, scale=20.):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(nclass, dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(scale)))

    def forward(self, x):
        return self.logit_scale.exp().clamp(1, 100) * F.linear(F.normalize(x), F.normalize(self.weight))

    @torch.no_grad()
    def init_from_centroids(self, means, present):
        """Seed the classifier from frozen-CLIP class centroids (B11).

        The default ``randn * 0.02`` head is the worst-conditioned object in the
        run: on the very first step it is asked to find 750 directions *and* its
        own scale, at the largest LR it will ever see, against labels a sixth of
        which are wrong.  Handing it the frozen tower's own class centroids means
        the first epochs refine a classifier that already separates the classes
        instead of inventing one from noise -- and the epochs before the noise
        tracker has anything to say are exactly the ones this project's own
        measurements say decide the final score (224: ep4 == ep20).

        Absent classes keep their random row: a zero row would be a *neutral*
        competitor at cosine 0, which is the same trap ``FrozenJudge.judge``
        documents.
        """
        # the caller may hold these on either device (the centroid estimate is
        # computed on the CPU from the feature pass, deliberately), so make the
        # device explicit rather than relying on what index_put_ tolerates
        present = torch.as_tensor(present).to(self.weight.device).bool()
        w = F.normalize(means.float(), dim=-1).to(self.weight.device, self.weight.dtype)
        self.weight[present] = w[present]


class LocalPatchHead(nn.Module):
    """A small opt-in patch-token adapter for the frozen CLIP tower.

    The official ViT-B/32 tower still produces the global CLIP embedding.  This
    head only pools the final spatial tokens exposed by open_clip's
    ``forward_intermediates`` API and projects them with a trainable adapter.
    ``gate`` starts at zero so enabling the option is exactly the old global
    path at step zero; training can then learn whether local evidence helps.
    The adapter is deliberately tiny and remains a single final classifier at
    inference time (no ensemble and no second backbone).
    """

    def __init__(self, width, out_dim, base_proj=None):
        super().__init__()
        self.proj = nn.Linear(width, out_dim, bias=False)
        # open_clip represents the visual projection as either a matrix
        # Parameter [width, out_dim] or (in some builds) an nn.Linear/
        # LoRALinear.  Read the frozen base tensor without assuming one form;
        # otherwise --local-head would fail before the first batch on a valid
        # ViT-B/32 checkpoint.
        base_w = None
        if base_proj is not None:
            if hasattr(base_proj, 'base') and hasattr(base_proj.base, 'weight'):
                base_w = base_proj.base.weight.detach()
            elif hasattr(base_proj, 'weight'):
                base_w = base_proj.weight.detach()
            elif torch.is_tensor(base_proj):
                base_w = base_proj.detach()
        with torch.no_grad():
            if base_w is not None and tuple(base_w.shape) == (width, out_dim):
                self.proj.weight.copy_(base_w.t().float())
            elif base_w is not None and tuple(base_w.shape) == (out_dim, width):
                self.proj.weight.copy_(base_w.float())
            else:
                nn.init.normal_(self.proj.weight, std=width ** -0.5)
        # A zero gate preserves the pretrained global representation at the
        # beginning of a run, while still giving the adapter a useful gradient.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, tokens):
        # tokens are final, layer-normalised spatial tokens: [B, N, width]
        pooled = tokens.mean(dim=1)
        return F.normalize(self.proj(pooled).float(), dim=-1)


def param_groups(model, weight_decay):
    """AdamW parameter groups: everything decayed **except the temperature**.

    ``head.logit_scale`` must not be weight-decayed.  Whenever the tracker
    decides anything it compares a *probability* against an absolute threshold
    (``--tau-conf 0.8``), so shrinking the temperature regularises nothing --
    it flattens every posterior and moves the clean/noisy boundary to wherever
    the decay has pushed the scale.  A scalar that is supposed to be learned
    *upwards* is not something to decay towards zero.

    The visual tower's own CLIP ``logit_scale`` is frozen by ``Net.__init__``,
    so at most one scalar ends up in the no-decay group; the assert fires if the
    naming ever changes, because the symptom of getting this wrong is a silent
    accuracy loss rather than an error.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith('logit_scale') else decay).append(p)
    assert no_decay, 'no trainable logit_scale found -- did CosineClassifier get renamed?'
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]


#: LR at the start of the linear warm-up, as a fraction of ``--lr``.
LR_WARMUP_START = 0.1


def lr_factor(step, total, warm, start=LR_WARMUP_START):
    """The LR multiplier of epoch ``step``, as a pure function.

    Split out from the scheduler so the *shape* can be checked without torch
    (``selftest`` checks the wiring; this arithmetic is what a wrong warm-up
    would silently get wrong).

    ``warm == 0`` reproduces torch's ``CosineAnnealingLR`` exactly:
    ``(1 + cos(pi * step / total)) / 2``, which is that class's closed form.

    ``step`` is clamped below at 0.  ``last_epoch`` is ``-1`` until the first
    ``step()``, and the warm-up branch extrapolates linearly -- so an
    unclamped ``-1`` yields ``start - (1-start)/warm``, a **negative** learning
    rate that would move the weights the wrong way.  It does not happen on the
    normal path, where ``_initial_step`` bumps ``last_epoch`` to 0 before
    ``get_lr`` is ever called, which is exactly why it would have been an
    unpleasant thing to discover later.
    """
    step = max(0.0, float(step))
    if warm > 0 and step < warm:
        return start + (1.0 - start) * step / warm
    if total <= warm:
        return 1.0
    # p in [0, 1] across the annealing part; the upper clamp keeps a step past
    # the end (--resume with a changed --epochs) at the floor instead of
    # climbing back up the far side of the cosine
    p = min(1.0, max(0.0, (step - warm) / (total - warm)))
    return 0.5 * (1.0 + math.cos(math.pi * p))


#: torch renamed ``_LRScheduler`` to ``LRScheduler`` in 2.0; both names exist in
#: 2.x, only the old one in 1.x, so take whichever is there.
_LRSchedulerBase = getattr(torch.optim.lr_scheduler, 'LRScheduler',
                           torch.optim.lr_scheduler._LRScheduler)


class WarmupCosineLR(_LRSchedulerBase):
    """Cosine decay reached through a linear warm-up (D6).

    ``CosineAnnealingLR`` hands out the peak LR on the very first epoch, and at
    that moment ``head.weight`` is random: a 750-way cosine head is asked to find
    its scale *and* its 750 directions from a cold start, at the largest step
    size it will ever see, against labels a sixth of which are wrong.  That is
    the worst-conditioned part of the run.

    Written as a plain ``LRScheduler`` subclass rather than ``SequentialLR`` over
    ``LinearLR`` + ``CosineAnnealingLR`` for two reasons: the shape is then our
    arithmetic instead of an interaction between two library schedulers that
    differs between torch versions, and ``state_dict`` round-trips (``SequentialLR``
    nests its children, and ``LambdaLR`` blanks out its lambdas -- either would
    make ``--resume`` subtly wrong or crash).

    ``--lr-warmup-epochs 0`` reproduces the old schedule exactly, so runs started
    before this existed stay reproducible.
    """

    def __init__(self, optimizer, total, warm, start=LR_WARMUP_START, last_epoch=-1):
        self.total, self.warm, self.start = int(total), int(warm), float(start)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        f = lr_factor(self.last_epoch, self.total, self.warm, self.start)
        return [base_lr * f for base_lr in self.base_lrs]


def make_scheduler(opt, a):
    """The LR schedule for a run described by ``a``.

    The warm-up is counted in epochs, to match ``--epochs``; the cosine then
    anneals over whatever is left.  The warm-up is capped at ``epochs - 1`` so
    that a short run still has somewhere to anneal to.
    """
    warm_ep = max(0, min(int(a.lr_warmup_epochs), int(a.epochs) - 1))
    return WarmupCosineLR(opt, a.epochs, warm_ep)


def load_sched(sched, sd):
    """Restore an LR schedule, or say why not.

    Adding the warm-up changed the schedule's structure, so a checkpoint written
    before it cannot be resumed into it.  That is a genuine situation -- a run
    started on the old code and continued on the new one restarts its warm-up --
    so it is reported rather than crashing.
    """
    try:
        sched.load_state_dict(sd)
    except Exception as e:                  # any structure mismatch, of any flavour
        print(f'WARNING: LR schedule not restored ({type(e).__name__}: {e}); it restarts '
              f'from --lr-warmup-epochs. Normal when resuming a checkpoint written before '
              f'the LR warm-up existed.')


class ProtoHead(nn.Module):
    """EMA class prototypes used as a contrastive target (MoPro / Sel-CL).

    Prototypes live in the frozen-CLIP feature space: they are bootstrapped from
    the frozen-CLIP class means and are only ever updated with the features of
    samples the tracker currently trusts, so label noise cannot drag a prototype
    onto the wrong class.
    """

    def __init__(self, dim, nclass, momentum=0.99, temp=0.1):
        super().__init__()
        self.dim, self.nclass = dim, nclass
        self.momentum, self.temp = momentum, temp
        self.register_buffer('proto', torch.zeros(nclass, dim))
        self.register_buffer('filled', torch.zeros(nclass, dtype=torch.bool))

    @torch.no_grad()
    def init_from_means(self, means, present):
        """Seed the prototypes from the frozen-CLIP class means of warm-up."""
        z = F.normalize(means.float(), dim=-1)
        self.proto[present] = z[present]
        self.filled[present] = True

    @torch.no_grad()
    def update(self, feats, targets, mask):
        """EMA update from the (masked) samples of one batch."""
        if not bool(mask.any()):
            return
        f = F.normalize(feats.float(), dim=-1)
        onehot = F.one_hot(targets, self.nclass).float() * mask.float()[:, None]
        counts = onehot.sum(0)
        has = counts > 0
        if not bool(has.any()):
            return
        idx = has.nonzero().squeeze(1)                        # class ids, sorted
        mean = F.normalize((onehot.t() @ f)[idx], dim=-1)
        fresh = idx[~self.filled[idx]]
        if fresh.numel():                                     # first sighting: copy
            self.proto[fresh] = mean[torch.searchsorted(idx, fresh)]
            self.filled[fresh] = True
        upd = idx[self.filled[idx]]
        if upd.numel():                                       # afterwards: EMA
            pos = torch.searchsorted(idx, upd)
            self.proto[upd] = F.normalize(self.momentum * self.proto[upd]
                                          + (1 - self.momentum) * mean[pos], dim=-1)

    def logits(self, feats):
        out = F.normalize(feats.float(), dim=-1) @ F.normalize(self.proto, dim=-1).t() / self.temp
        return out.masked_fill(~self.filled[None, :], -1e4)


class Net(nn.Module):
    def __init__(self, clip_model, nclass, lora_rank=8, lora_target='all',
                 proto_momentum=0.99, proto_temp=0.1, local_head=False):
        super().__init__()
        self.clip = clip_model
        dim = clip_model.visual.output_dim
        self.head = CosineClassifier(dim, nclass)
        self.proto = ProtoHead(dim, nclass, proto_momentum, proto_temp)
        self.n_lora = add_lora(self.clip.visual, lora_rank, 2 * lora_rank, target=lora_target)
        for name, p in self.clip.named_parameters():
            if not ('.A' in name or '.B' in name):
                p.requires_grad = False
        self.local_head = None
        if local_head:
            visual = self.clip.visual
            width = getattr(getattr(visual, 'transformer', None), 'width', None)
            if width is None:
                # ViT-B/32 is the only allowed tower, but keep this diagnostic
                # explicit instead of failing later with an obscure shape error.
                raise SystemExit('local patch head requires a ViT visual transformer with a width')
            self.local_head = LocalPatchHead(int(width), int(dim),
                                             getattr(visual, 'proj', None))
        # Keep an in-memory copy of the official interpolated grid.  When
        # --train-pos-embed later unfreezes the live grid, anchor_feat() must
        # still represent the original frozen CLIP tower rather than silently
        # anchoring the model to its own moving positional embedding.  This is
        # deterministic from the allowed OpenAI weights and image size, so it
        # need not be serialized; a resumed model reconstructs the same copy.
        self._anchor_pos_name = None
        self._anchor_pos_embed = None
        for _name in ('positional_embedding', 'pos_embed'):
            _pe = getattr(self.clip.visual, _name, None)
            if _pe is not None and hasattr(_pe, 'detach'):
                self._anchor_pos_name = _name
                self._anchor_pos_embed = _pe.detach().clone()
                break

    def forward(self, x, return_feat=False):
        if self.local_head is None:
            z = F.normalize(self.clip.encode_image(x).float(), dim=-1)
        else:
            visual = self.clip.visual
            if not hasattr(visual, 'forward_intermediates'):
                raise RuntimeError('this open_clip visual tower has no forward_intermediates; '
                                   'disable --local-head or upgrade the pinned open_clip')
            nblock = len(getattr(getattr(visual, 'transformer', None), 'resblocks', []))
            if not nblock:
                raise RuntimeError('cannot locate ViT transformer blocks for local patch head')
            d = visual.forward_intermediates(
                x, indices=[nblock - 1], stop_early=False,
                normalize_intermediates=True, intermediates_only=False,
                output_fmt='NLC')
            z_global = F.normalize(d['image_features'].float(), dim=-1)
            toks = d['image_intermediates'][-1]
            z_local = self.local_head(toks)
            # tanh keeps the learned residual bounded; gate=0 makes the exact
            # initial model equal to the historical global-only path.
            z = F.normalize(z_global + torch.tanh(self.local_head.gate) * z_local, dim=-1)
        out = self.head(z)
        return (out, z) if return_feat else out

    @torch.no_grad()
    def anchor_feat(self, x):
        """Frozen-CLIP embedding of ``x`` (LoRA off, eval mode, no grad)."""
        visual = self.clip.visual
        was_training = visual.training
        visual.eval()
        old_pe = None
        if self._anchor_pos_name is not None and self._anchor_pos_embed is not None:
            current_pe = getattr(visual, self._anchor_pos_name)
            # Avoid replacing the parameter on the default frozen path.  The
            # replacement is only needed after positional embedding training was
            # enabled and is restored in finally so optimizer references remain
            # intact for the student.
            if current_pe.requires_grad:
                # Keep the Parameter object registered with the module.  Replacing
                # a Parameter with a plain Tensor via setattr raises in PyTorch
                # and would also invalidate the optimizer's parameter reference.
                old_pe = current_pe.detach().clone()
                with torch.no_grad():
                    current_pe.copy_(self._anchor_pos_embed.to(
                        device=current_pe.device, dtype=current_pe.dtype))
        try:
            with lora_disabled(visual):
                z = visual(x)
        finally:
            if old_pe is not None:
                with torch.no_grad():
                    getattr(visual, self._anchor_pos_name).copy_(old_pe)
            visual.train(was_training)
        return F.normalize(z.float(), dim=-1)

    def trainable_state_dict(self):
        """Only what is actually trained -- the frozen backbone is rebuilt from
        the official OpenAI weights, so checkpoints stay a few MB instead of
        ~350 MB of untouched CLIP parameters."""
        frozen = {n for n, p in self.named_parameters() if not p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k not in frozen}


def load_trainable_state(module, state, name='model', required_keys=None):
    """Load a checkpoint containing only trainable tensors.

    Frozen CLIP tensors are intentionally omitted from checkpoints, so they
    must not be treated as missing.  We still reject missing trainable keys,
    unexpected keys, and shape mismatches because each of those means the
    resumed run is using a different architecture or silently losing state.
    """
    if not isinstance(state, dict):
        raise SystemExit(f'{name} checkpoint is not a state-dict')
    current = module.state_dict()
    # Buffers that are updated during training (prototype vectors and masks)
    # are part of the effective trainable state even though they are not
    # Parameters, and trainable_state_dict() already includes them.
    expected = (set(module.trainable_state_dict().keys()) if required_keys is None
                else set(required_keys))
    missing_trainable = sorted(k for k in expected if k not in state)
    unexpected = sorted(k for k in state if k not in current)
    shape_bad = sorted(k for k in state if k in current and tuple(state[k].shape) != tuple(current[k].shape))
    if missing_trainable or unexpected or shape_bad:
        raise SystemExit(f'{name} checkpoint mismatch: missing_trainable={missing_trainable}, '
                         f'unexpected={unexpected}, shape_mismatch={shape_bad}')
    # strict=False is safe after the checks: only frozen backbone parameters
    # and buffers that are deliberately not serialized may remain missing.
    missing, unexpected2 = module.load_state_dict(state, strict=False)
    if unexpected2:
        raise SystemExit(f'{name} checkpoint unexpected keys after load: {unexpected2}')
    return missing


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class ImageFolderNoisy(Dataset):
    """Class-folder dataset with a stratified hold-out split.

    Returns ``(image, target, index)``; ``index`` addresses the *training* item
    list and is what the noise tracker and prototype bootstrap are keyed on.
    """

    def __init__(self, root, transform, val=False, val_ratio=0.1, seed=3407, split='train',
                 stochastic=False):
        self.root, self.transform, self.split = Path(root), transform, split
        self.seed, self.stochastic = int(seed), bool(stochastic)
        #: Mixed into the augmentation seed when ``--epoch-aug`` is on; set by
        #: ``main`` at the top of every epoch.  Forked into the workers, which is
        #: why that flag turns persistent workers off.
        self.epoch = 0
        # what _load builds a grey image at when a file will not decode.  Read
        # off the transform so a 288 run does not reconstruct its broken files
        # at 224: harmless under every transform this file builds (a uniform
        # grey survives resizing unchanged), but "harmless today" is not a
        # reason to leave a hard-coded 224 in the resolution-dependent path.
        self.img_size = transform_size(transform)
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        items = []
        for c in classes:
            for p in sorted((self.root / c).iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    items.append((str(p), self.class_to_idx[c]))
        rng = random.Random(seed)
        rng.shuffle(items)
        by_cls = {i: [] for i in range(len(classes))}
        for it in items:
            by_cls[it[1]].append(it)
        # the hold-out size must not depend on which split is being built,
        # otherwise the training split silently swallows the whole val split
        self.items = []
        for i in range(len(classes)):
            n = max(1, int(len(by_cls[i]) * val_ratio))
            self.items.extend(by_cls[i][:n] if val else by_cls[i][n:])
        self.targets = [y for _, y in self.items]

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        try:
            return Image.open(path).convert('RGB')
        except Exception:                       # truncated / unreadable file
            n = self.img_size
            return Image.new('RGB', (n, n), (127, 127, 127))

    def augmented(self, img, i, view=0):
        """Apply the training transform with randomness derived from ``(seed, i, view)``.

        Why not just let the transform draw from the worker's RNG: because then
        the augmentation of item ``i`` depends on *which worker* happened to draw
        it, and that mapping changes with ``--workers``.  The competition requires
        the submitted code to be able to reproduce the result, so a recipe whose
        output silently depends on a loader flag is a reproducibility hole -- and
        the failure is invisible, because every ``--workers`` setting still runs
        and still converges, just to a different point.

        Seeding per item makes the augmentation a pure function of
        ``(seed, index, view)``, so it is identical for any ``--workers``.
        ``fork_rng`` restores torch's global state on exit; python's ``random``
        module has to be saved by hand.

        With ``--augment-mode index --epoch-aug`` the fourth seed component is
        the epoch, so item ``i`` gets new
        augmentation every epoch instead of the same two pictures for the whole
        run.  Seeding per item *and* per epoch keeps the reproducibility property
        above (the seed is still a pure function of the run's seed, the item, the
        view -- and now the epoch, which is itself a deterministic counter).
        The default ``--augment-mode worker`` bypasses this seed logic and uses
        the same worker RNG stream as the teammate's 71.1382 implementation.

        Cost is a few microseconds per image against a RandAugment that is
        already milliseconds, so it does not show up in throughput.
        """
        if not self.stochastic:
            return self.transform(img)
        # A spread-out mix so that adjacent indices and the two views do not get
        # correlated seeds.  The modulus is 2**63-1 (the int64 max, which is what
        # torch.manual_seed takes): at 2**31-1 a 149k-image set has ~20 birthday
        # collisions between two items' seeds, which is harmless -- the images
        # differ, so only the augmentation *parameters* coincide -- but there is
        # no reason to accept it when the wider space makes it ~0.
        h = ((self.seed * 0x9E3779B1) ^ (int(i) * 0x85EBCA77) ^ (int(view) * 0xC2B2AE3D)
             ^ (int(self.epoch) * 0x27D4EB2F))
        s = (h ^ (h >> 15)) % (2 ** 63 - 1)
        py_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(s)
                random.seed(s)
                return self.transform(img)
        finally:
            random.setstate(py_state)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.augmented(self._load(path), i, 0), y, i


class TwoView(Dataset):
    """Two *independent* augmented views of the same image + its index."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        path, y = self.base.items[i]
        img = self.base._load(path)
        # view 0 and view 1 get different seeds, so the two views are independent
        # *and* each is reproducible on its own (see ImageFolderNoisy.augmented)
        return self.base.augmented(img, i, 0), self.base.augmented(img, i, 1), y, i


def build_datasets(a):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(a.image_size, scale=(a.crop_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(a.randaug_n, a.randaug_m),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    val_tf = eval_transform(a.image_size)
    # stochastic=True only on the training split: the eval transform is
    # deterministic, so forking the RNG per image there would be pure overhead
    tr = ImageFolderNoisy(a.data, train_tf, False, a.val_ratio, a.seed, 'train',
                          stochastic=(a.augment_mode == 'index'))
    va = ImageFolderNoisy(a.data, val_tf, True, a.val_ratio, a.seed, 'val')
    assert len(tr) > 0, f'no images found under {a.data}'
    return tr, va


def build_loaders(a, tr, va):
    if a.sampler == 'balanced':
        counts = Counter(tr.targets)
        weights = torch.as_tensor([1.0 / math.sqrt(counts[y]) for y in tr.targets], dtype=torch.double)
    else:
        weights = torch.ones(len(tr), dtype=torch.double)
    g = torch.Generator()
    g.manual_seed(a.seed)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=g)
    # Persistent workers hold a forked dataset copy.  In index mode with epoch
    # seeds, the main process changes tr.epoch each round, so workers must be
    # restarted.  Worker mode follows the teammate RNG stream and keeps them.
    keep_workers = a.workers > 0 and not (a.augment_mode == 'index' and a.epoch_aug)
    loader = DataLoader(TwoView(tr), batch_size=a.batch_size, sampler=sampler,
                        num_workers=a.workers, pin_memory=True, drop_last=True,
                        persistent_workers=keep_workers, worker_init_fn=seed_worker,
                        generator=g)
    # Keep the sampler RNG accessible so checkpoints can restore the exact
    # replacement sequence on resume.
    loader._sampler_generator = g
    vloader = DataLoader(va, batch_size=a.batch_size * 2, shuffle=False, num_workers=a.workers,
                         pin_memory=True, persistent_workers=keep_workers, worker_init_fn=seed_worker)
    return loader, vloader


# --------------------------------------------------------------------------- #
# frozen-CLIP head initialisation (B11) + full-coverage features (B17)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def frozen_feature_pass(model, loader, dim, n, device, amp_dtype, use_amp, every=200):
    """One deterministic, no-grad forward of the *frozen* tower over a whole split.

    ``loader`` must yield ``(x, y, index)`` with ``index`` addressing the same
    item list the run's tracker is keyed on, so the rows land in the right place.

    Why a dedicated pass instead of reusing the warm-up accumulation the training
    loop already does: the warm-up draws through ``WeightedRandomSampler``, so a
    minority of samples is never drawn at all -- which leaves their judge entry,
    their prototype contribution and (with ``--head-init``) their centroid missing,
    and the code's own note says the sampler has no reason to draw them later
    either.  One pass over every image fixes the coverage rather than sampling
    around it.  The transform is the *eval* one (no augmentation), because these
    features are the ones the classifier will actually see at test time.
    """
    feat = torch.zeros(n, dim, dtype=torch.float32)
    seen = torch.zeros(n, dtype=torch.bool)
    was_training = model.training
    model.eval()
    try:
        for it, (x, _y, idx) in enumerate(loader):
            # Centroids, judge margins and sparse teacher targets are discrete
            # decisions; do this pass in fp32 so bf16/fp16 rounding cannot
            # change which class survives the frozen filter.
            with torch.autocast(device_type='cuda', enabled=False):
                z = model.anchor_feat(x.to(device, non_blocking=True))
            feat[idx] = z.float().cpu()
            seen[idx] = True
            if every and (it + 1) % every == 0:
                print(f'  [frozen] {int(seen.sum())}/{n} features')
    finally:
        model.train(was_training)
    return feat, seen


def robust_centroids(feat, targets, nclass, rounds=2, verbose=True):
    """``(means, present, keep)`` for class centroids that survive label noise.

    Round 1 averages every sample a class folder contains, which is what the
    training labels say and therefore inherits their noise -- on this round about
    a quarter of them are wrong, and a plain mean drags each centroid towards
    whatever it is confused with.  Every later round keeps only the samples whose
    *given* label is the nearest centroid -- a sample the frozen tower can
    independently confirm -- and re-averages on that subset.

    That kept set is the same construction ``probe.py`` calls ``V*``, for the same
    reason: the centroids are built from the training split only, so no label the
    tower has not been asked to predict takes part.  It is automatic (competition
    rule 五.6 forbids a manual cleaning step) and deterministic.
    """
    z = F.normalize(feat.float(), dim=-1)
    y = torch.as_tensor(targets, dtype=torch.long)
    dim = z.shape[1]
    keep = torch.ones(y.numel(), dtype=torch.bool)

    def estimate(mask):
        sums = torch.zeros(nclass, dim)
        cnt = torch.zeros(nclass)
        sel = y[mask]
        sums.index_add_(0, sel, z[mask])
        cnt.index_add_(0, sel, torch.ones(sel.numel(), dtype=torch.float32))
        present = cnt > 0
        means = F.normalize(sums, dim=-1)
        means[~present] = 0.0        # absent classes must not be a *neutral* competitor
        return means, present

    @torch.no_grad()
    def agrees_with(means, present, chunk=8192):
        """``argmax`` of the centroid similarity is the given label, per sample.

        Chunked because the full matrix is ``n x nclass`` -- 446 MB at 148695x750
        -- and there is nothing to gain by materialising it at once.
        """
        out = torch.empty(z.shape[0], dtype=torch.bool)
        for s in range(0, z.shape[0], chunk):
            e = min(s + chunk, z.shape[0])
            sim = z[s:e] @ means.t()
            sim[:, ~present] = -2.0     # absent class: never a neutral competitor
            out[s:e] = sim.argmax(1) == y[s:e]
        return out

    means, present = estimate(keep)
    for r in range(max(0, int(rounds))):
        agree = agrees_with(means, present)
        nxt = keep & agree
        # Never erase an entire class from the classifier/prototype bank.
        for c in range(nclass):
            old_c = keep & (y == c)
            if bool(old_c.any()) and not bool((nxt & (y == c)).any()):
                nxt[old_c] = True
        if verbose:
            print(f'  [frozen] centroid round {r + 1}: {int(present.sum())}/{nclass} classes, '
                  f'agree {int((keep & agree).sum())}/{int(keep.sum())} '
                  f'({(keep & agree).float().sum() / max(int(keep.sum()), 1):.3f})')
        # never let a round empty a class out of the estimates entirely: a class
        # whose samples all disagree is telling us its folder is systematically
        # mislabelled, and the honest thing is to keep its (noisy) mean rather
        # than delete the class from the decision space
        if int(nxt.sum()) == 0:
            print('  [frozen] every sample was rejected; keeping the previous round')
            break
        keep = nxt
        means, present = estimate(keep)
    return means, present, keep


def mean_bank(means, present):
    """The one-prototype-per-class bank: the robust centroids ``robust_centroids``
    returns, in the shape every frozen-teacher consumer expects."""
    nclass = means.shape[0]
    return (F.normalize(means.float(), dim=-1),
            torch.arange(nclass, dtype=torch.long),
            torch.full((nclass,), -1, dtype=torch.long),
            torch.as_tensor(present).bool())


@torch.no_grad()
def spherical_kmeans(z, k, iters=8):
    """``k`` unit-norm centres for unit-norm rows, deterministic.

    Deterministic on purpose: competition rule 五.6 needs the cleaning
    reproducible, and a seeded k-means++ would still make the *bank* depend on
    sampler state.  Farthest-point init from the mean, then Lloyd steps on cosine
    similarity.  An empty cluster keeps its centre rather than being re-seeded --
    re-seeding is what makes a single outlier spawn a prototype that then explains
    only itself.
    """
    ctr = torch.empty(k, z.shape[1])
    ctr[0] = F.normalize(z.mean(0), dim=-1)
    for j in range(1, k):
        # the point the chosen centres explain worst becomes the next centre
        ctr[j] = z[(z @ ctr[:j].t()).max(1).values.argmin()]
    for _ in range(max(1, int(iters))):
        a = (z @ ctr.t()).argmax(1)
        for j in range(k):
            sel = z[a == j]
            if sel.shape[0]:
                ctr[j] = F.normalize(sel.mean(0), dim=-1)
    return ctr


@torch.no_grad()
def proto_bank(feat, targets, kept, nclass, k=2, iters=8, verbose=True):
    """``(protos, proto_class, proto_src, present)``: up to ``k`` medoids per class.

    ``提分路径研究.md`` B2/B15, the part that is not already in the tree.  The
    existing frozen teacher is one mean per class, which averages a class's
    internal *modes* into the middle of the space between them -- on this round's
    folders (English search keywords as class names) a folder that holds two
    visual clusters gets a centroid that is close to neither, and every sample of
    both clusters then looks like an outlier.  ``k`` medoids per class keep the
    modes separate.

    Two properties this design insists on:

    - **Medoids, not centres.**  Each returned prototype is an actual training
      sample (``proto_src``), so a query can exclude itself from its own class
      *exactly* -- a mean cannot do that without recomputing it per query.  That
      is B2's "排除自身及其重复簇" and it is the difference between a teacher and
      a mirror: a nearest-mean teacher agrees with a sample partly because the
      sample is in the mean.
    - **Built from the kept set only**, falling back to the class's full (noisy)
      set when the filter emptied it -- the same rule ``robust_centroids``
      applies, and for the same reason: a class that disappears from the decision
      space is worse than a class with a noisy prototype.

    Only the *target* consumers use this bank.  ``FrozenJudge`` keeps its single
    mean on purpose: its veto is a post-warm-up quantity and is not where the
    score is decided (see HANDOFF §22.2).
    """
    z = F.normalize(feat.float(), dim=-1)
    y = torch.as_tensor(targets, dtype=torch.long)
    kept = torch.as_tensor(kept).bool()
    protos, cls, src = [], [], []
    for c in range(nclass):
        sel = ((y == c) & kept).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            sel = (y == c).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        Z = z[sel]
        kc = max(1, min(int(k), int(sel.numel())))
        ctr = F.normalize(Z.mean(0), dim=-1)[None] if kc == 1 else spherical_kmeans(Z, kc, iters)
        a = (Z @ ctr.t()).argmax(1)
        for j in range(kc):
            sub = (a == j).nonzero(as_tuple=True)[0]
            if sub.numel() == 0:
                continue                      # an empty mode contributes nothing
            best = sub[(Z[sub] @ ctr[j]).argmax()]
            protos.append(Z[best])
            cls.append(c)
            src.append(int(sel[best]))
    if not protos:
        raise SystemExit('proto_bank: no class produced a prototype; the split is empty')
    protos = torch.stack(protos)
    proto_class = torch.tensor(cls, dtype=torch.long)
    proto_src = torch.tensor(src, dtype=torch.long)
    present = torch.zeros(nclass, dtype=torch.bool)
    present[proto_class.unique()] = True
    if verbose:
        per = torch.bincount(proto_class, minlength=nclass)
        print(f'proto bank: {protos.shape[0]} medoids for {int(present.sum())}/{nclass} classes '
              f'(mean {per[present].float().mean():.2f} per class, --frozen-protos {k})')
    return protos, proto_class, proto_src, present


@torch.no_grad()
def frozen_soft_targets(feat, protos, proto_class, proto_src, present, given,
                        temp=0.05, topk=8, chunk=8192):
    """Sparse per-sample targets from the frozen tower's prototype bank (B13/B14).

    One class score per sample: the best of that class's prototypes, with any
    prototype that *is* the sample itself masked out (``proto_src == i``).  Then
    the usual temperature softmax over the top-``topk`` classes.

    The agreement gate is **not** applied here -- ``agree`` is returned and each
    consumer gates for itself, because the two consumers want opposite things:

    - ``--distill-weight`` (B14) must not touch a sample the frozen tower
      disagrees with: its whole safety argument is that it cannot confirm a
      mistake the student makes.
    - ``--frozen-mix-rho`` (B13) very much *does* want those samples.  They are
      where the systematic label errors live (§16.3: 24% of samples have the
      given label below 0.1 posterior), and a *soft* forward correction --
      ``t <- (1-rho) t + rho q_f`` -- is exactly the tool for a class whose folder
      is full of another class's images.

    Returned sparse rather than dense: a full ``n x 750`` float target is 446 MB,
    while top-8 as (int16 index, float16 weight) is 4.6 MB for the whole training
    set, and neither consumer needs the zero mass.  Rows stay on the CPU -- the
    per-batch cost of moving 128x8 entries to the GPU is noise.
    """
    z = F.normalize(feat.float(), dim=-1)
    p = F.normalize(protos.float(), dim=-1)
    present = torch.as_tensor(present).bool()
    proto_class = torch.as_tensor(proto_class).long()
    proto_src = torch.as_tensor(proto_src).long()
    given = torch.as_tensor(given, dtype=torch.long).cpu()
    n, nclass = z.shape[0], present.numel()
    assert nclass < 32000, 'class indices are stored as int16; widen the dtype first'
    valid_classes = int(present.sum())
    k = max(1, min(int(topk), max(valid_classes, 1)))
    idx = torch.zeros(n, k, dtype=torch.int16)
    w = torch.zeros(n, k, dtype=torch.float16)
    agree = torch.zeros(n, dtype=torch.bool)
    loo = bool((proto_src >= 0).any())
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = z[s:e] @ p.t()
        if loo:
            # a medoid must not vote for the sample it *is*
            sim = sim.masked_fill(proto_src[None, :] == torch.arange(s, e)[:, None], -2.0)
        score = torch.full((e - s, nclass), -2.0)
        score.scatter_reduce_(1, proto_class[None, :].expand(e - s, -1), sim,
                              reduce='amax', include_self=True)
        score[:, ~present] = -2.0        # absent class: never a neutral competitor
        vals, ind = score.topk(k, dim=1)
        valid = torch.isin(ind, torch.where(present)[0])
        vals = vals.masked_fill(~valid, -1e9)
        # the target is a *distribution over classes*, so the temperature acts on
        # the cosine similarity -- 0.05 is sharp enough that the argmax carries
        # most of the mass while the fine-grained neighbours keep a visible share,
        # which is the part the one-hot label cannot express
        ww = F.softmax(vals / max(float(temp), 1e-4), dim=1)
        ww = ww * valid.float()
        ww = ww / ww.sum(1, keepdim=True).clamp_min(1e-8)
        w[s:e] = ww.to(torch.float16)
        idx[s:e] = ind.to(torch.int16)
        agree[s:e] = ind[:, 0] == given[s:e]
    return idx, w, agree


def frozen_mix(target, idx, weight, rho):
    """``target <- (1 - rho_i) target + rho_i q_f``, per sample, out of place.

    ``提分路径研究.md`` B13's soft forward correction, at the sample level instead
    of the class level: ``T = (1-rho) I + rho T_hat`` is the same expression with
    the same interpretation -- keep most of the given label, move the rest onto
    what the frozen tower supports.  Doing it per sample avoids having to estimate
    a 750x750 transition matrix, and to decide which of its columns are
    trustworthy; the prototype bank *is* the estimate.

    ``rho_i`` is zeroed where ``weight`` sums to zero, so a row the caller gated
    out keeps its target *and its normalisation* -- scaling by ``1-rho`` without a
    matching place to put the mass is what makes a soft target silently stop
    summing to one.
    """
    w = weight.float()
    r = (float(rho) * (w.sum(1) > 0).to(w.dtype))[:, None]
    # `target * (1 - r)` is out of place, so the in-place add cannot reach a
    # tensor the caller still holds (the warm-up branch aliases mix_t and soft)
    return (target * (1 - r)).scatter_add(1, idx.long(), w * r)


@torch.no_grad()
def frozen_target_report(idx, weight, given, rho):
    """Summarise the *gradient target*, not only its argmax changes.

    A small flip rate can coexist with a large change in every CE gradient.
    Report the teacher's mass on the folder label and the gap to its top class
    before deciding whether a high-rho run is safe.
    """
    i, q = idx.long().cpu(), weight.float().cpu()
    y = torch.as_tensor(given, dtype=torch.long).cpu()
    qy = (q * (i == y[:, None])).sum(1)
    top = q[:, 0]
    disagree = i[:, 0] != y
    gap = top - qy
    def stats(x):
        if x.numel() == 0:
            return 'n=0'
        p = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9]))
        return (f'n={x.numel()} mean={float(x.mean()):.3f} '
                f'p10/p50/p90={float(p[0]):.3f}/{float(p[1]):.3f}/{float(p[2]):.3f}')
    threshold = (1.0 - float(rho)) / max(float(rho), 1e-8)
    flips = disagree & (gap > threshold)
    print(f'frozen target quality: teacher-top mass {stats(top)}; '
          f'folder-label mass {stats(qy)}')
    print(f'frozen target disagreement: {int(disagree.sum())}/{y.numel()} rows; '
          f'top-minus-folder mass {stats(gap[disagree])}; '
          f'rho={rho:g} predicts {int(flips.sum())}/{y.numel()} argmax flips '
          f'(threshold {threshold:.3f}). All rows still receive the full rho '
          f'fraction of the soft target, including those without a flip.')


def sparse_kl(logits, idx, weight):
    """Per-sample ``KL(q || p)`` for the sparse target ``q`` (B14's loss term).

    ``weight`` rows that are entirely zero -- the samples the caller gated out --
    return exactly 0.0 including the gradient, because ``0 * log 0`` is taken as
    0: the absent mass is not "impossible", it is "not part of this term".  That
    is what makes the gate self-enforcing rather than a mask the loss could leak
    through.
    """
    w = weight.float()
    logp = F.log_softmax(logits.float(), dim=1)
    lp = logp.gather(1, idx.long())
    return (w * (torch.log(w.clamp_min(1e-8)) - lp)).sum(1)


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, use_amp):
    """Returns (loss, accuracy, accuracy on high-confidence predictions).

    The hold-out split carries the *same* label noise as the training set, so
    ``acc_hi`` (accuracy restricted to confident predictions) is a useful sanity
    signal on how much of the model's error is label noise rather than model
    error.
    """
    model.eval()
    n = correct = n_hi = correct_hi = 0
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            out = model(x)
        out = out.float()
        total_loss += F.cross_entropy(out, y, reduction='sum').item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        hi = out.softmax(1).max(1).values >= 0.8
        n_hi += int(hi.sum())
        correct_hi += int(((pred == y) & hi).sum())
        n += y.numel()
    return total_loss / n, correct / n, (correct_hi / n_hi if n_hi else 0.0)


# what an inference-only checkpoint needs; everything else (optimiser, RNG,
# tracker) is only read back by `--resume`, which only ever loads `last.pt`
#: ``pos_embed`` rides along because the positional grid is the one thing about the
#: input pipeline that a load cannot infer: unless ``--train-pos-embed`` is on it is
#: frozen, so it never enters ``model`` (``trainable_state_dict`` drops it), which
#: means every load rebuilds it and a wrong rebuild is shape-compatible and silent.
#: ``pos_embed_trained`` records which of those two worlds the checkpoint came from,
#: so ``verify_pos_embed`` can tell "the grid learned, so it is supposed to differ"
#: apart from "the grid was rebuilt wrong".  See ``resize_positional_embedding``.
SNAPSHOT_KEYS = ('model', 'classes', 'class_counts', 'args', 'epoch', 'val_acc',
                 'val_acc_hi', 'lora_rank', 'lora_target', 'pretrained', 'model_name',
                 'local_head', 'pos_embed', 'pos_embed_trained', 'config_fingerprint')


def thin(ck):
    """Inference-only view of a checkpoint.

    The tracker's posterior is ``n_train x n_class`` floats -- 446 MB on this
    round's 148695x750 set -- and the optimiser state is comparable.  Writing
    both into every ``best.pt`` / ``epN.pt`` cost ~3 GB of the data disk per run
    for data that inference never reads.  The snapshots exist to be scored on the
    leaderboard, so they only need the weights.
    """
    return {k: ck[k] for k in SNAPSHOT_KEYS if k in ck}


def main(a):
    check_backbone(a.model)
    if a.pretrained != 'openai':
        # rule 十一(二).3 allows only the official OpenAI ViT-B/32 weights; a local
        # path cannot be verified, so say so rather than silently trusting it
        print(f'WARNING: --pretrained {a.pretrained!r} is not the "openai" tag -- the rules '
              f'allow only the official OpenAI ViT-B/32 weights, make sure that file is exactly those.')
    seed_everything(a.seed, deterministic=not a.cudnn_benchmark)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # resolved this early because --head-init needs them before the first step
    use_amp = a.amp != 'none' and device.type == 'cuda'
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(a.amp)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va = build_datasets(a)
    loader, vloader = build_loaders(a, tr, va)
    nclass = len(tr.class_to_idx)
    # per-class counts, in class-index order -- saved into the checkpoint so that
    # infer.py can offer post-hoc logit adjustment (the test set is balanced, the
    # training set is not, so the two priors differ)
    freq = Counter(tr.targets)
    class_counts = [freq.get(i, 0) for i in range(nclass)]
    print(f'classes={nclass} train={len(tr)} val={len(va)} device={device} amp={a.amp}')
    cfg_fp = config_fingerprint(vars(a))
    print('config: ' + ' '.join(f'{k}={v}' for k, v in sorted(vars(a).items())))
    print(f'config fingerprint: {cfg_fp}')

    clip_model = build_clip(a.model, a.pretrained, a.image_size)
    model = Net(clip_model, nclass, a.lora_rank, a.lora_target,
                a.proto_momentum, a.proto_temp,
                local_head=bool(a.local_head)).to(device)
    # --train-pos-embed, and *where* these lines sit is the whole feature: after
    # Net, whose __init__ froze the grid along with the rest of the backbone;
    # before param_groups, which collects whatever carries requires_grad at that
    # moment; and before the teacher deepcopy.  Anywhere else it is a silent
    # no-op, which is why the ordering is spelled out again in the function.
    pos_embed_trained = bool(a.train_pos_embed)
    if pos_embed_trained:
        pe_name = enable_pos_embed_training(model.clip.visual)
        pe = getattr(model.clip.visual, pe_name)
        print(f'train-pos-embed: {pe_name} is trainable, {pe.numel()} params '
              f'({pe.numel()/1e6:.3f}M), {int(pe.shape[0]) - 1} tokens at {a.image_size}px')
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model={a.model}/{a.pretrained} layers={model.n_lora} trainable params={n_train/1e6:.3f}M')

    # ---- frozen-CLIP class means for prototype bootstrap (filled during warm-up)
    boot_sum = torch.zeros(nclass, model.head.weight.shape[-1], device=device)
    boot_cnt = torch.zeros(nclass, device=device)
    # The judge keeps one frozen feature per training sample (n x 512 fp32, so
    # ~0.3 GB on the 148k round).  Allocated only when it will be used.
    judge = (FrozenJudge(len(tr), model.head.weight.shape[-1], a.judge_margin, device)
             if a.noise_judge else None)

    tracker = LabelTrustTracker(tr.targets, nclass, momentum=a.noise_momentum,
                                tau_conf=a.tau_conf, w_noise=a.w_noise,
                                w_relabel=a.w_relabel, max_noise_frac=a.max_noise_frac,
                                relabel_mix=a.relabel_mix,
                                class_tau_delta=a.class_tau_delta, device=device)

    # ---- B11/B17: seed the head from frozen-CLIP class centroids, before any step.
    #
    # The head ships randomly initialised, so the first epochs spend their budget
    # teaching it a 750-way linear map that frozen CLIP already knows.  This runs
    # one no-grad pass of the *frozen* tower over the whole training split, takes
    # robust class means of those features, and writes them into `head.weight` --
    # so step 0 starts from CLIP's own zero-shot geometry instead of noise.
    #
    # It matters *when* the score is decided.  The one decisive experiment this
    # project has (ep4 == ep20 on the leaderboard to 4 decimals, while val_acc
    # moved 5 points) says the leaderboard-relevant state is fixed within the
    # first few epochs; anything that changes epoch 4 changes the score.  The
    # head's initialisation is inside that window, which is why this is worth a
    # GPU pass and the post-warm-up tracker is not.
    #
    # `robust_centroids` iterates the means: a sample that a *different* class's
    # centroid explains better is dropped and the means are recomputed, which is
    # what keeps a folder full of mislabelled images (the class folders are
    # English search keywords, so `0000/` holds bluebirds and a water heater)
    # from dragging its own centroid off the class.
    # The same frozen pass serves both B11 (seed the head) and B14 (distil the
    # frozen distribution), and it is deliberately one pass: the cost is the
    # forward, not the tensors it produces, and both consumers want the same
    # robust centroids.  They stay separately switchable so each is a
    # single-variable experiment on its own.
    head_init_done = False
    fq = None
    frozen_consumer = (a.distill_weight > 0 or a.frozen_mix_rho > 0)
    if a.frozen_protos > 1 and not frozen_consumer:
        # the bank only exists to feed a *target* consumer.  --head-init uses the
        # single mean, so this combination silently does nothing -- say so, rather
        # than let a run be launched on a flag that was never read.
        print(f'WARNING: --frozen-protos {a.frozen_protos} is ignored without --distill-weight '
              f'or --frozen-mix-rho; --head-init seeds from the single class mean. Continuing '
              f'with the one-mean teacher.')
    if a.head_init == 'frozen' or frozen_consumer:
        if a.resume:
            print('WARNING: --head-init frozen / --distill-weight / --frozen-mix-rho are ignored '
                  'on --resume; the checkpoint carries its own head and this pass would not '
                  'change it')
        else:
            fdim = model.head.weight.shape[-1]
            # Same constructor args as `tr`, minus `stochastic`, so this split is
            # the same *list of items in the same order* -- the features below
            # are indexed by training-item index, and so are the tracker and the
            # judge.  The assertion is the whole safety net for that claim.
            ft_ds = ImageFolderNoisy(a.data, eval_transform(a.image_size), False,
                                     a.val_ratio, a.seed, 'train')
            assert len(ft_ds) == len(tr) and ft_ds.targets == tr.targets, (
                'the head-init split and the training split disagree; feature rows would be '
                'attached to the wrong samples')
            # list the flags that actually switched the pass on, rather than assuming
            # --head-init: the mix-only and distill-only runs are the ones whose logs
            # get read to decide whether the head was seeded, so this line has to name
            # the consumer honestly (it used to print an empty tail for --frozen-mix-rho)
            served = [name for name, on in (('--head-init', a.head_init == 'frozen'),
                                            ('--distill-weight', a.distill_weight > 0),
                                            ('--frozen-mix-rho', a.frozen_mix_rho > 0)) if on]
            print(f'frozen pass: {len(ft_ds)} training images, one no-grad forward of the frozen '
                  f'tower at {a.image_size}px (deterministic transform, no augmentation); '
                  f'serves {" + ".join(served)}')
            ft_loader = DataLoader(ft_ds, batch_size=a.head_init_batch_size, shuffle=False,
                                   num_workers=a.workers, pin_memory=True,
                                   worker_init_fn=seed_worker)
            feat, fseen = frozen_feature_pass(model, ft_loader, fdim, len(tr), device,
                                              amp_dtype, use_amp)
            means, present, kept = robust_centroids(feat, tr.targets, nclass,
                                                    a.head_init_rounds)
            if frozen_consumer:
                # built on the CPU centroids before they move to the GPU: the whole
                # target set is a few MB of indices and weights, and it is read one
                # batch at a time for the rest of the run
                if a.frozen_protos > 1:
                    bank = proto_bank(feat, tr.targets, kept, nclass, a.frozen_protos)
                else:
                    # One medoid per class was too lossy on the real 750-class
                    # split: its argmax agreed with only 32.3% of folder labels
                    # in the 2026-10-01 smoke run.  Use the robust centroid that
                    # seeds the head as the default teacher.  A single sample's
                    # contribution to its class mean is small at ~180/class;
                    # k>1 remains an explicit leave-one-out medoid experiment.
                    bank = mean_bank(means, present)
                qidx, qw, qagree = frozen_soft_targets(feat, *bank, tracker.y,
                                                       a.frozen_temp, a.frozen_topk)
                frozen_target_report(qidx, qw, tr.targets, a.frozen_mix_rho)
                print(f'frozen target: {int(qagree.sum())}/{len(tr)} samples the frozen teacher '
                      f'agrees with ({qagree.float().mean():.3f}), top-{int(qidx.shape[1])} at '
                      f'--frozen-temp {a.frozen_temp:g}, bank='
                      f'{"robust-mean" if a.frozen_protos == 1 else "LOO-medoids"}. '
                      f'This is agreement with noisy folder labels, not clean-label accuracy; '
                      f'compare the holdout report before using it as supervision.')
                # The train agreement can be inflated by class construction or
                # deflated by medoid leave-one-out.  Score the unchanged frozen
                # teacher on the held-out training split before trusting it as
                # supervision.  Validation indices are local to that split, so
                # no training medoid may be masked by an equal integer index.
                # The training validation loader can use persistent workers.
                # A separate non-persistent loader keeps this one-off pass from
                # retaining another worker pool throughout the full run.
                frozen_vloader = DataLoader(
                    va, batch_size=a.head_init_batch_size, shuffle=False,
                    num_workers=a.workers, pin_memory=True,
                    worker_init_fn=seed_worker)
                val_feat, val_seen = frozen_feature_pass(
                    model, frozen_vloader, fdim, len(va), device, amp_dtype,
                    use_amp, every=0)
                if not bool(val_seen.all()):
                    raise RuntimeError('frozen teacher validation pass missed images')
                val_src = torch.full_like(bank[2], -1)
                vqidx, _vqw, vqagree = frozen_soft_targets(
                    val_feat, bank[0], bank[1], val_src, bank[3], va.targets,
                    a.frozen_temp, a.frozen_topk)
                print(f'frozen teacher holdout: {int(vqagree.sum())}/{len(va)} '
                      f'({float(vqagree.float().mean()):.3f}) agree with noisy validation '
                      f'labels. This is not clean-label accuracy.')
                del frozen_vloader, val_feat, val_seen, vqidx, _vqw, vqagree
                fq = (qidx.to(device), qw.to(device), qagree.to(device))
                del qidx, qw, qagree
            # the centroids are estimated on the CPU (the pass writes to CPU); the
            # head, the prototypes and the judge all live on the GPU
            means, present = means.to(device), present.to(device)
            if a.head_init == 'frozen':
                model.head.init_from_centroids(means, present)
                print(f'head-init frozen: head seeded for {int(present.sum())}/{nclass} classes; '
                      f'{int(kept.sum())}/{len(tr)} samples survived the agreement filter '
                      f'({int(kept.sum()) / len(tr):.3f})')
                if a.proto_weight > 0:
                    model.proto.init_from_means(means, present)
                if judge is not None:
                    # Full coverage beats the warm-up trickle: the samples `add` would
                    # have missed are the rare classes, exactly the ones a class-erasing
                    # veto does the most damage to.
                    judge.add_all(feat, fseen)
                    suspect, _top1, _margin = judge.judge(means, tracker.y, present)
                    tracker.set_judge(suspect)
                    print(f'frozen-CLIP judge: full coverage, {int(suspect.sum())}/{len(tr)} '
                          f'samples suspect at --judge-margin {a.judge_margin:g}. The veto only '
                          f'demotes the weight, never the label -- but it does have a '
                          f'false-positive rate, so measure it with probe.py --calib before '
                          f'reading anything into the score.')
                head_init_done = True
            del feat, fseen

    robust = make_robust_loss(a.robust_loss, a.gce_q, a.apl_k, a.apl_b, a.apl_rce)

    # After the seeding, so the teacher starts as a copy of the seeded model --
    # and, on --resume, after the checkpoint is loaded for the same reason.
    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(param_groups(model, a.weight_decay), lr=a.lr)
    sched = make_scheduler(opt, a)
    try:                                    # torch >= 2.3 API, older one as a fallback
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and a.amp == 'fp16')
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and a.amp == 'fp16')

    # -1.0, not 0.0: with --select val_acc_hi the score is 0.0 for exactly as
    # long as no sample clears max(p) >= 0.8, which on 750 classes can hold for
    # the first epochs -- and `score > best` would then write no best.pt at all.
    # Starting below every possible score guarantees the first epoch lands one.
    start_epoch, best = 0, -1.0
    if a.resume and (a.distill_weight > 0 or a.frozen_mix_rho > 0):
        raise SystemExit('--resume with --distill-weight/--frozen-mix-rho is disabled: '
                         'the frozen target bank is not safely reconstructible from the checkpoint')
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        if ck.get('classes') != tr.class_to_idx:
            raise SystemExit('checkpoint class mapping differs from current training data; refusing resume')
        ck_args = ck.get('args') or {}
        for key, current_value in (('model_name', a.model),
                                   ('lora_rank', a.lora_rank),
                                   ('lora_target', a.lora_target),
                                   ('pretrained', a.pretrained),
                                   ('local_head', bool(a.local_head)),
                                   ('train_pos_embed', bool(a.train_pos_embed))):
            fallback = ck_args.get('model') if key == 'model_name' else ck_args.get(key)
            recorded = ck.get(key, fallback)
            if key == 'train_pos_embed' and recorded is None:
                # Checkpoints written by this tree carry the explicit marker;
                # older checkpoints may only have the argparse field.
                recorded = ck.get('pos_embed_trained', False)
            if recorded is not None and str(recorded) != str(current_value):
                raise SystemExit(f'checkpoint/{key} mismatch: checkpoint={recorded!r}, '
                                 f'current={current_value!r}; refusing resume')
        saved_size, size_key = ck_image_size(ck, verbose=False)
        if size_key is not None and int(saved_size) != int(a.image_size):
            raise SystemExit(f'checkpoint image size is {saved_size}px but current '
                             f'--image-size is {a.image_size}px; refusing resume')
        if size_key is None and int(a.image_size) != 224:
            print('WARNING: checkpoint has no recorded image size; allowing resume at '
                  f'{a.image_size}px, but the original training resolution cannot be verified')
        # before the weights, the grid: if this reload rebuilt a different
        # positional embedding than the run recorded, the resumed weights are
        # being continued on top of a different model and nothing else would say so
        verify_pos_embed(ck, model.clip.visual)
        load_trainable_state(model, ck['model'], 'model')
        teacher = copy.deepcopy(model).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        if 'teacher' in ck:
            load_trainable_state(teacher, ck['teacher'], 'teacher',
                                 required_keys=model.trainable_state_dict().keys())
        else:
            print('WARNING: checkpoint has no EMA teacher; resuming with student snapshot as teacher')
        opt.load_state_dict(ck['optim'])
        load_sched(sched, ck['sched'])
        tracker.load_state_dict(ck['tracker'])
        torch.set_rng_state(ck['rng']['torch'])
        if ck['rng'].get('cuda') is not None:
            torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        random.setstate(ck['rng']['python'])
        if ck['rng'].get('sampler') is not None:
            loader._sampler_generator.set_state(ck['rng']['sampler'])
        else:
            print('WARNING: checkpoint has no sampler RNG; resumed sample order is not bit-exact')
        start_epoch = ck['epoch'] + 1
        best = ck.get('val_acc' if a.select == 'val_acc' else 'val_acc_hi', -1.0)
        print(f'resumed from {a.resume} at epoch {start_epoch} (best={best:.4f})')

    for ep in range(start_epoch, a.epochs):
        warm = ep < a.warmup_epochs
        if a.augment_mode == 'index' and a.epoch_aug:
            # The augmentation seed is (seed, index, view) -- fixed for the whole
            # run, so every epoch drew the *same* crop and the same RandAugment
            # for a given image.  Over 20 epochs that is 20 passes over one fixed
            # augmented view of the set, which is not what a long schedule is
            # supposed to be.  Mixing in the epoch restores it.  Main mutates it,
            # the workers read it, hence keep_workers=False in build_loaders.
            tr.epoch = ep
        if not warm:
            st = tracker.refresh()
            print('noise stats:', st)
            # A degenerate tracker is completely silent otherwise.  On the first
            # real run the 40% cap pinned on all 20 epochs and the four counts did
            # not even add up to the training set -- nothing in the log said so,
            # and a full GPU run was spent before it was noticed.  On a brand-new
            # dataset these thresholds have never been calibrated, so say it now.
            if st['clean'] + st['relabel'] + st['noisy'] + st['unseen'] != len(tr):
                print(f'  !! 警告: 四个统计量加起来 {st["clean"] + st["relabel"] + st["noisy"] + st["unseen"]}'
                      f' != 训练集大小 {len(tr)}，划分逻辑坏了，这次训练的结果不可信。')
            if st['capped'] > 0:
                print(f'  !! 警告: 有 {st["capped"]} 个样本撞到了 --max-noise-frac={a.max_noise_frac} 上限。'
                      f'判据对这个数据集不成立，调 --tau-conf 或 --max-noise-frac。')
            if st['relabel'] == 0 and ep >= a.warmup_epochs + 1:
                print(f'  !! 警告: 至今没有任何样本被改标注。{nclass} 类下的 argmax 置信度'
                      f'很难达到 --tau-conf={a.tau_conf}，考虑调低它。')
            if st['mean_weight'] < 0.35:
                print(f'  !! 警告: 平均样本权重跌到 {st["mean_weight"]:.3f}，'
                      f'绝大多数样本几乎不产生梯度，等于只用了很小一部分数据。'
                      f'调 --max-noise-frac 或 --w-noise。')

        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        run_jsd_sum, run_jsd_n, run_jsd_down = 0.0, 0, 0
        run_dl, run_ce, run_g, run_flip = 0.0, 0.0, 0, 0   # B14 term vs CE; B13 target flips
        for it, (x1, x2, y, idx) in enumerate(loader):
            if a.limit_batches and it >= a.limit_batches:
                break
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx_g = idx.to(device, non_blocking=True)
            bsz = y.numel()

            # the warm-up branch only needs the frozen feature while something is
            # still consuming what it collects; --head-init already collected it all
            need_anchor = a.anchor_weight > 0 or (warm and not head_init_done
                                                  and (a.proto_weight > 0 or judge is not None))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                anchor = model.anchor_feat(x1) if need_anchor else None   # frozen CLIP, LoRA off
                out, z = model(x1, True)
                out2, z2 = model(x2, True)
                with torch.no_grad():
                    tout, tz = teacher(x1, True)        # one view is enough for the teacher
                tprob = F.softmax(tout.float(), 1)

            tracker.update(idx_g, tprob)                # free: the teacher already ran

            # the frozen teacher's sparse distribution for this batch, and the two
            # different gates the two consumers want (see frozen_soft_targets)
            if fq is not None:
                bq_i, bq_w, bq_a = fq[0][idx_g].long(), fq[1][idx_g].float(), fq[2][idx_g]
                mix_w = bq_w if not a.frozen_mix_agree_only else bq_w * bq_a[:, None]
                kl_w = bq_w * bq_a[:, None]      # the KL must never see a disagreement
            else:
                bq_i = mix_w = kl_w = None

            if warm:
                # pure supervised warm-up on the raw labels; there is no teacher
                # mixture yet, so both targets coincide
                hard = y
                base = F.one_hot(y, nclass).to(torch.float32)
                # B13 applies from the *first* step, warm-up included: the target
                # is where the systematic label errors live, and the warm-up is
                # where the score is decided (HANDOFF §22.2).  On --frozen-mix-rho
                # 0 this is a no-op and the warm-up is the old pure CE.
                if mix_w is not None and a.frozen_mix_rho > 0:
                    mixed = frozen_mix(base, bq_i, mix_w, a.frozen_mix_rho)
                    run_flip += int((mixed.argmax(1) != base.argmax(1)).sum())
                    base = mixed
                # Keep the robust loss on the unsmoothed target.  Label
                # smoothing is a CE-only regulariser; feeding it to GCE/NCE/
                # RCE/APL weakens their intended resistance to incorrect labels.
                mix_t = base
                soft = smooth_target(base, a.label_smooth, nclass)
                w = torch.ones(bsz, device=device)
            else:
                hard = tracker.label[idx_g]             # possibly corrected label
                # `mix_t` carries the teacher mixture and is what the robust loss
                # sees; `soft` adds label smoothing and is the CE target.  They
                # differ on purpose -- see the note in losses.nce.
                mix_t = tracker.target(idx_g, y)
                if mix_w is not None and a.frozen_mix_rho > 0:
                    # mixed before smooth_target so that the CE target, the robust
                    # loss and the prototype loss all see the same corrected
                    # target -- a forward correction is a property of the target,
                    # not of one of the losses that consume it
                    mixed = frozen_mix(mix_t, bq_i, mix_w, a.frozen_mix_rho)
                    run_flip += int((mixed.argmax(1) != mix_t.argmax(1)).sum())
                    mix_t = mixed
                soft = smooth_target(mix_t, a.label_smooth, nclass)
                w = tracker.weight[idx_g] * tprob.max(1).values.clamp(a.conf_floor, 1.0).pow(a.conf_gamma)
                if a.norm_weights:
                    # By default use the teammate's unlimited batch-mean
                    # normalization.  A positive norm_max_gain explicitly caps
                    # the amplification as a separate experiment.
                    if a.norm_max_gain and a.norm_max_gain > 0:
                        denom = w.mean().clamp_min(1.0 / max(a.norm_max_gain, 1.0))
                        w = (w / denom).clamp_max(max(a.norm_max_gain, 1.0))
                    else:
                        # Unlimited normalization is the teammate reference
                        # path: every batch has mean weight one, with no hidden
                        # cap that can suppress high-trust rows.
                        w = w / w.mean().clamp_min(1e-6)

            # B9: view disagreement as a conservative sample-level trust signal.
            # JSD is bounded and symmetric; beta=0 keeps the historical path
            # exactly unchanged.  Apply it after tracker weights so disagreement
            # can only reduce a sample's influence, never promote a noisy row.
            if a.jsd_weight > 0:
                p1 = F.softmax(out.float(), dim=1)
                p2 = F.softmax(out2.float(), dim=1)
                m12 = 0.5 * (p1 + p2)
                jsd = 0.5 * ((p1 * (p1.clamp_min(1e-8).log() - m12.clamp_min(1e-8).log())).sum(1) +
                              (p2 * (p2.clamp_min(1e-8).log() - m12.clamp_min(1e-8).log())).sum(1))
                w = w * torch.exp(-a.jsd_weight * jsd).clamp_min(a.jsd_floor)
                run_jsd_sum += float(jsd.sum())
                run_jsd_n += int(jsd.numel())
                run_jsd_down += int((jsd > 1e-7).sum())

            # labelled pass on both views (the "passive" term)
            ce1 = xent(out, soft)
            ce2 = xent(out2, soft)
            loss = 0.5 * ((w * ce1).mean() + (w * ce2).mean())
            # B14, and the reason it is here rather than next to the tracker terms:
            # it is *not* gated on `not warm`.  The warm-up is where the run's fate
            # is decided, so a term that only switches on afterwards can at best
            # touch the state the score is probably already fixed by.  The target
            # is the frozen tower's class distribution, which is available at step
            # 0 -- there is nothing about it that needs the teacher to have
            # converged first.  Both views get the same target: it describes the
            # *image*, not a view of it.
            if kl_w is not None and a.distill_weight > 0:
                dl = 0.5 * (sparse_kl(out, bq_i, kl_w) + sparse_kl(out2, bq_i, kl_w))
                # average over the *distilled* samples, so --distill-weight means
                # the same thing whatever the gate ratio turns out to be
                n_g = (kl_w.sum(1) > 0).sum()
                loss = loss + a.distill_weight * dl.sum() / n_g.clamp_min(1)
                # the term's raw size is not knowable off-line, and it is the one
                # thing that decides whether --distill-weight is a nudge or a
                # takeover: the student's own softmax is much sharper than the
                # teacher at --frozen-temp 0.05 (logit_scale ~20 against a
                # temperature on cosines), so most of the value is the student
                # being pulled *softer* onto the frozen neighbours.  Report both
                # sides rather than one number whose scale nobody can calibrate.
                run_dl += float(dl.sum())
                run_ce += float((w * (ce1 + ce2) * 0.5).sum())
                run_g += int(n_g)
            # --warm-robust also applies the robust loss during warm-up.  This is
            # B4-lite (early anchoring) without the extra state: the warm-up is
            # pure CE on the raw labels, and CE on a mislabelled sample is exactly
            # the term that drives memorisation -- during the epoch window the
            # score is decided in.  `w` is all-ones while warm, so this is the same
            # term the post-warm-up branch adds, at the same weight.
            if (a.warm_robust or not warm) and a.robust_weight > 0:
                rob = 0.5 * (robust(out, mix_t) + robust(out2, mix_t))
                loss = loss + a.robust_weight * (w * rob).mean()
            if a.consistency_weight > 0:
                # The verified 71.1382 recipe used a one-way stop-gradient
                # consistency term.  A symmetric term doubles the number of
                # representation paths receiving a gradient and changes the
                # effective LoRA step size.  Keep the higher-risk symmetric
                # variant explicit for ablations instead of silently changing
                # the known-good optimization geometry.
                cons = F.mse_loss(z, z2.detach())
                if a.consistency_symmetric:
                    cons = 0.5 * (cons + F.mse_loss(z2, z.detach()))
                loss = loss + a.consistency_weight * cons
            if anchor is not None and a.anchor_weight > 0:
                loss = loss + a.anchor_weight * (1 - (z * anchor).sum(1)).mean()
            if a.proto_weight > 0 and not warm:
                # contrastive pull towards the trust-aligned class prototype
                trusted = tracker.weight[idx_g] >= a.proto_min_weight
                pl = model.proto.logits(torch.cat([z, z2], 0))
                # The reference objective includes every row with its existing
                # noise weight.  The trusted mask belongs to prototype *updates*;
                # applying it to the loss as well changes the gradient budget.
                pw = w.repeat(2)
                if a.proto_trusted_loss:
                    pw = pw * trusted.repeat(2).float()
                proto_loss = xent(pl, mix_t.repeat(2, 1))
                # Keep the prototype term's scale independent of how many
                # trusted rows a replacement sampler happened to draw.  A
                # batch with few trusted samples should not silently turn the
                # prototype objective off; an empty trusted set remains inert.
                if a.proto_normalize:
                    # Optional trust-set normalization.  This can amplify a
                    # small trusted subset substantially, so it is opt-in.
                    denom = pw.sum().clamp_min(1e-6)
                    ploss = (pw * proto_loss).sum() / denom
                else:
                    # Match the teammate recipe: the prototype objective is a
                    # batch mean.  Its scale naturally falls when fewer rows
                    # are trusted, instead of exploding by 1/trusted_frac.
                    ploss = (pw * proto_loss).mean()
                loss = loss + a.proto_weight * ploss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                # EMA teacher; only trainable tensors actually change
                for (tn, tp), (sn, sp) in zip(teacher.named_parameters(), model.named_parameters()):
                    assert tn == sn
                    if sp.requires_grad:
                        tp.mul_(a.ema).add_(sp, alpha=1 - a.ema)

                if warm and anchor is not None:
                    boot_sum.index_add_(0, y, anchor)       # frozen-CLIP class means
                    boot_cnt.index_add_(0, y, torch.ones(bsz, device=device))
                    # ... but never let these overwrite a full-coverage pass: x1 is
                    # an augmented view, so its feature is a strictly noisier
                    # estimate of the same sample than the deterministic one
                    # --head-init already stored.
                    if judge is not None and not head_init_done:
                        judge.add(idx_g, anchor)
                if a.proto_weight > 0 and not warm:
                    # the prototype EMA still needs a hard assignment: averaging
                    # features under a soft target would let a confused sample
                    # drag two prototypes at once.  When B13 is active, however,
                    # the corrected target is the supervision used by the losses;
                    # updating the EMA with the pre-mix tracker label would partly
                    # undo the correction and make the prototype branch optimize a
                    # different label.  Use the corrected argmax only for that
                    # opt-in path, preserving the verified baseline exactly.
                    proto_hard = (mix_t.argmax(1)
                                  if (mix_w is not None and a.frozen_mix_rho > 0)
                                  else hard)
                    model.proto.update(tz, proto_hard, trusted)

            running += loss.item() * bsz
            seen += bsz

        if warm and ep + 1 == a.warmup_epochs:
            # --head-init frozen already seeded the prototypes and the judge from a
            # full-coverage pass, which strictly dominates what warm-up's partial
            # sampling collected (it covers every sample, rare classes included).
            # Rebuilding from `boot_sum` here would overwrite the better estimate
            # with the worse one.  `tracker.reset()` below still runs either way.
            if (a.proto_weight > 0 or judge is not None) and not head_init_done:
                means, present = prototype_bootstrap(boot_sum, boot_cnt)
                print(f'frozen-CLIP class means built for {int(present.sum())}/{nclass} classes')
                if a.proto_weight > 0:
                    model.proto.init_from_means(means, present)
                if judge is not None:
                    suspect, _jpred, _jmargin = judge.judge(means, tracker.y, present)
                    tracker.set_judge(suspect)
                    n_seen = int(judge.seen.sum())
                    print(f'frozen-CLIP judge: {int(suspect.sum())}/{n_seen} of the samples seen '
                          f'during warm-up are suspect (another class centroid is closer by more '
                          f'than {a.judge_margin:g}). A veto demotes the weight; it never changes '
                          f'the label or the target.')
                    if n_seen < len(tr):
                        print(f'  note: {len(tr) - n_seen} samples were never drawn during '
                              f'warm-up and are not judged (the sampler has no reason to draw '
                              f'them later either, so they keep the given label at weight 1)')
            # The posterior EMA was accumulated against a *randomly initialised*
            # head, and a sample's first observation is stored verbatim rather
            # than averaged away, so that noise survives into every later epoch.
            # Restart it: the next batch's posterior becomes the first
            # observation, from a teacher that has actually been trained.
            if a.reset_tracker_warmup:
                tracker.reset()
                print('tracker posterior reset at the end of warm-up (prob/seen cleared); the next '
                      'epoch runs with every sample unseen, i.e. unfiltered, while it refills')
            else:
                # The leaderboard reference implementation carried the warm-up
                # posterior into the first robust epoch.  Resetting here throws
                # away the only early teacher signal and changes which samples
                # are trusted exactly when the score is decided.
                print('tracker posterior retained at warm-up boundary (reference recipe)')

        sched.step()
        vl, va_acc, va_hi = evaluate(model, vloader, device, amp_dtype, use_amp)
        score = va_acc if a.select == 'val_acc' else va_hi
        t_acc = t_acc_hi = 0.0
        if a.save_teacher:
            # The EMA teacher is an online average of this same model.  Evaluating
            # it costs one validation pass, but gives an early-epoch candidate that
            # is often less affected by label memorisation than the student.
            _, t_acc, t_acc_hi = evaluate(teacher, vloader, device, amp_dtype, use_amp)
        print(f'epoch {ep + 1}/{a.epochs} loss={running / max(seen, 1):.4f} '
              f'val_loss={vl:.4f} val_acc={va_acc:.4f} val_acc_hi={va_hi:.4f} '
              f'lr={opt.param_groups[0]["lr"]:.2e} time={time.time() - t0:.1f}s')
        if a.jsd_weight > 0:
            print(f'  [jsd] mean={run_jsd_sum / max(run_jsd_n, 1):.5f} '
                  f'downweighted={run_jsd_down}/{run_jsd_n} '
                  f'({run_jsd_down / max(run_jsd_n, 1):.3f})')
        if a.save_teacher:
            print(f'  [teacher] val_acc={t_acc:.4f} val_acc_hi={t_acc_hi:.4f} '
                  f'(student {va_acc:.4f} / {va_hi:.4f})')
        if fq is not None and a.distill_weight > 0:
            # per *distilled* sample, next to the CE it sits beside: if this ratio
            # is near or above the CE, the term is steering the run, not anchoring
            # it, and --distill-weight is too high for a first try
            print(f'  [distill] {run_g}/{seen} samples distilled this epoch; KL '
                  f'{run_dl / max(run_g, 1):.4f} vs CE {run_ce / max(seen, 1):.4f} per sample '
                  f'-- effective term {a.distill_weight * run_dl / max(run_g, 1):.4f}')
        if fq is not None and a.frozen_mix_rho > 0:
            # the number that says whether this is a nudge or a relabelling: at
            # rho 0.3 a target only changes its argmax where the frozen mass
            # outweighs 0.7 of the label, so a flip rate near 0 means the mix is
            # doing nothing at all and one near the disagreement rate means the
            # frozen teacher has taken over the target
            print(f'  [mix] rho={a.frozen_mix_rho:g} '
                  f'{"agree-only" if a.frozen_mix_agree_only else "all samples"}: the target '
                  f'changed its argmax on {run_flip}/{seen} samples this epoch '
                  f'({run_flip / max(seen, 1):.3f})')

        # Keep a small append-only sidecar so a finished run can be diagnosed
        # without reopening the very large `last.pt`.  This is deliberately
        # observational: it does not affect gradients, checkpoint selection, or
        # the submitted inference path.
        append_jsonl(out_dir / 'diagnostics.jsonl', epoch_record(
            epoch=ep + 1, loss=running / max(seen, 1), val_loss=vl,
            val_acc=va_acc, val_acc_hi=va_hi, lr=opt.param_groups[0]['lr'],
            noise=(tracker.stats if not warm else None),
            jsd=({'mean': run_jsd_sum / max(run_jsd_n, 1),
                  'downweighted': run_jsd_down,
                  'n': run_jsd_n} if a.jsd_weight > 0 else None),
            distill={'samples': run_g, 'kl_sum': run_dl, 'ce_sum': run_ce}
                     if (fq is not None and a.distill_weight > 0) else None,
            target_flip_rate=(run_flip / max(seen, 1))
                             if (fq is not None and a.frozen_mix_rho > 0) else None,
            teacher_val_acc=t_acc if a.save_teacher else None,
            teacher_val_acc_hi=t_acc_hi if a.save_teacher else None,
            active_flags={
                'train_pos_embed': bool(a.train_pos_embed),
                'head_init': a.head_init,
                'distill_weight': float(a.distill_weight),
                'frozen_mix_rho': float(a.frozen_mix_rho),
                'frozen_protos': int(a.frozen_protos),
                'save_teacher': bool(a.save_teacher),
                'epoch_aug': bool(a.epoch_aug),
                'augment_mode': a.augment_mode,
                'warm_robust': bool(a.warm_robust),
                'consistency_symmetric': bool(a.consistency_symmetric),
                'proto_normalize': bool(a.proto_normalize),
                'proto_trusted_loss': bool(a.proto_trusted_loss),
                'norm_max_gain': float(a.norm_max_gain),
                'reset_tracker_warmup': bool(a.reset_tracker_warmup),
                'local_head': bool(a.local_head),
            },
            config_fingerprint=cfg_fp))

        # ``teacher`` has requires_grad=False on purpose, so calling the
        # student's trainable_state_dict() method on it would omit every LoRA
        # and head parameter.  Select the same key set from the frozen teacher
        # explicitly; otherwise resume silently rebuilds the teacher from the
        # student despite carrying a seemingly valid ``teacher`` field.
        trainable_keys = set(model.trainable_state_dict().keys())
        teacher_state = {k: v.detach().cpu() for k, v in teacher.state_dict().items()
                         if k in trainable_keys}
        ck = {'model': model.trainable_state_dict(),
              'teacher': teacher_state,
              'classes': tr.class_to_idx,
              'class_counts': class_counts,
              'args': vars(a), 'epoch': ep, 'val_acc': va_acc, 'val_acc_hi': va_hi,
              'lora_rank': a.lora_rank, 'lora_target': a.lora_target, 'pretrained': a.pretrained,
              'model_name': a.model, 'local_head': bool(a.local_head),
              'pos_embed': pos_embed_fingerprint(model.clip.visual),
              'pos_embed_trained': pos_embed_trained,
              # Small, CPU-friendly metadata used by diagnostics.py for per-class
              # trust/noise summaries.  The inference-only `thin()` view omits it.
              'targets': torch.as_tensor(tr.targets, dtype=torch.int16).cpu(),
              'config_fingerprint': cfg_fp,
              'optim': opt.state_dict(), 'sched': sched.state_dict(), 'tracker': tracker.state_dict(),
              'rng': {'torch': torch.get_rng_state(),
                      'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                      'python': random.getstate(),
                      'sampler': loader._sampler_generator.get_state()}}
        torch.save(ck, out_dir / 'last.pt')
        if a.save_every and (ep + 1) % a.save_every == 0:
            # The hold-out split carries the same *structured* label noise as the
            # training set, so `val_acc` can be improved by memorising that noise
            # -- which costs accuracy on the clean test set.  Keep snapshots so
            # several epochs can be scored on the real leaderboard and the best
            # one picked, instead of trusting a proxy that rewards the failure.
            torch.save(thin(ck), out_dir / f'ep{ep + 1}.pt')
        if a.save_teacher:
            # The teacher is deliberately not passed through trainable_state_dict:
            # all of its parameters have requires_grad=False.  Select the same key
            # set as the student, which includes a trained positional grid.
            tck = dict(ck)
            tck['model'] = {k: v.detach().cpu() for k, v in teacher.state_dict().items()
                            if k in trainable_keys}
            tck['val_acc'], tck['val_acc_hi'] = t_acc, t_acc_hi
            # The EMA teacher has its own positional grid.  Its fingerprint must
            # travel with the teacher snapshot, otherwise infer.py would compare
            # the loaded teacher grid against the student's hash and reject a
            # perfectly valid snapshot (or, worse, silently score the wrong grid
            # in older loaders).
            tck['pos_embed'] = pos_embed_fingerprint(teacher.clip.visual)
            thin_t = thin(tck)
            if a.save_every and (ep + 1) % a.save_every == 0:
                torch.save(thin_t, out_dir / f'teacher_ep{ep + 1}.pt')
            if ep + 1 == a.epochs:
                torch.save(thin_t, out_dir / 'teacher_last.pt')
        if score > best:
            best = score
            torch.save(thin(ck), out_dir / 'best.pt')
            print(f'  -> new best ({a.select}={best:.4f}) saved to {out_dir / "best.pt"}')

    print(f'best {a.select} = {best:.4f}')


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='folder with one sub-folder per class')
    p.add_argument('--out', default='./outputs')
    p.add_argument('--pretrained', default='openai', help="open_clip tag ('openai') or a local .pt path")
    # the official OpenAI ViT-B/32 weights were trained with QuickGELU; building
    # plain 'ViT-B-32' (GELU) makes open_clip warn "QuickGELU mismatch" and
    # silently changes the activation the weights expect
    p.add_argument('--model', default='ViT-B-32-quickgelu', help='open_clip model name')
    p.add_argument('--resume', default='', help='resume from a last.pt written by this script')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lr-warmup-epochs', type=int, default=0, dest='lr_warmup_epochs',
                   help='linear LR warm-up from --lr * %g up to --lr over this many epochs, '
                        'then cosine decay over the rest. Default 0 matches the verified '
                        'teammate recipe; use 1 only as an explicit ablation.' % LR_WARMUP_START)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--cudnn-benchmark', action='store_true',
                   help='faster but not bit-reproducible (default: off)')
    p.add_argument('--amp', default='bf16', choices=['bf16', 'fp16', 'none'])
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--image-size', '--img-size', type=int, default=224, dest='image_size',
                   help='input resolution (--img-size is the sibling repo\'s spelling of '
                        'the same flag). 224 = the pretrained/CLIP default and '
                        'the only size that reproduces the earlier runs bit-for-bit; '
                        'anything else resamples the positional embeddings '
                        '(competition-permitted, see README_AUTODL.md 5). Recorded '
                        'in the checkpoint, and read back by infer/valmetrics/'
                        'analyze, so a mismatch cannot go unnoticed. Use a multiple '
                        'of the 32px patch size: 256/288/320/352/384 -- NOT 336, '
                        'which is a patch-14 number (336/14=24) with no meaning for '
                        'ViT-B/32.')
    p.add_argument('--limit-batches', type=int, default=0, help='smoke test: stop after N steps')
    p.add_argument('--sampler', default='balanced', choices=['balanced', 'uniform'])
    p.add_argument('--select', default='val_acc', choices=['val_acc', 'val_acc_hi'])
    p.add_argument('--save-every', type=int, default=4, dest='save_every',
                   help='also keep epN.pt every N epochs (0 = only best/last). The hold-out '
                        'split shares the training set\'s label noise, so val_acc can be '
                        'raised by memorising that noise -- snapshots let the real '
                        'leaderboard pick the epoch instead.')
    p.add_argument('--save-teacher', action='store_true', dest='save_teacher',
                   help='evaluate the EMA teacher each epoch and save teacher_epN.pt '
                        '(and teacher_last.pt). This is a weight average of the same '
                        'model, not an ensemble; it adds one validation pass per epoch.')

    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-target', default='all', choices=['all', 'mlp'])
    p.add_argument('--local-head', action='store_true', dest='local_head',
                   help='opt-in final-token local patch adapter. Uses the official CLIP ViT-B/32 '
                        'tower and one cosine classifier; disabled by default.')

    p.add_argument('--crop-min', type=float, default=0.55)
    p.add_argument('--randaug-n', type=int, default=2)
    p.add_argument('--randaug-m', type=int, default=9)
    p.add_argument('--augment-mode', choices=['worker', 'index'], default='worker',
                   help='worker reproduces the teammate DataLoader RNG stream and is the '
                        'default; index uses per-image deterministic seeds for ablations')

    p.add_argument('--robust-loss', default='apl', choices=['ce', 'gce', 'nce', 'rce', 'apl'])
    p.add_argument('--robust-weight', type=float, default=0.5)
    p.add_argument('--gce-q', type=float, default=0.7)
    p.add_argument('--apl-k', type=float, default=0.2)
    p.add_argument('--apl-b', type=float, default=1.0)
    p.add_argument('--apl-rce', type=float, default=1.0)

    p.add_argument('--noise-judge', action='store_true', dest='noise_judge',
                   help='veto the rule "the teacher agrees, so this label is clean" '
                        'with an independent frozen-CLIP NCC judgement. Without it, a '
                        'sample the student has memorised -- including a wrong label -- agrees '
                        'with itself forever and keeps full weight. Costs one frozen feature '
                        'per training sample (~0.3GB at 148k x 512). The veto only demotes the '
                        'weight; it never relabels. Measure its false-positive rate with '
                        'probe.py before trusting it.')
    p.add_argument('--judge-margin', type=float, default=0.02, dest='judge_margin',
                   help='how much closer another class centroid must be, in cosine, before the '
                        'judge calls a sample suspect. 0 vetoes on a bare tie; higher is more '
                        'conservative. probe.py reports the distribution to choose from.')

    p.add_argument('--noise-momentum', type=float, default=0.9)
    p.add_argument('--tau-conf', type=float, default=0.8,
                   help='teacher max(p) needed to overrule the given label')
    p.add_argument('--class-tau-delta', type=float, default=0.0, dest='class_tau_delta',
                   help='optional class-conditional confidence calibration strength; 0 disables')
    p.add_argument('--w-noise', type=float, default=0.1)
    p.add_argument('--w-relabel', type=float, default=0.5)
    p.add_argument('--max-noise-frac', type=float, default=0.4)
    p.add_argument('--relabel-mix', type=float, default=0.5,
                   help='fraction of the target mass that moves onto the teacher\'s pick when a '
                        'sample is confidently relabelled (1.0 = hard overwrite). The brief calls '
                        'the noise "weakly correlated" -- the given label is wrong but related -- '
                        'and in that regime a confident disagreement is often a genuinely '
                        'confusable neighbour rather than a wrong label, so a hard overwrite '
                        'would promote exactly that confusion to ground truth.')
    p.add_argument('--label-smooth', type=float, default=0.05,
                   help='label smoothing on the CE term. Cheapest defence against the failure '
                        'this project actually measured: val_acc (0.7305) came out above the '
                        'leaderboard score (0.6802), which is only possible if the model '
                        'memorised the class-consistent part of the label noise.')
    p.add_argument('--conf-gamma', type=float, default=2.0)
    p.add_argument('--conf-floor', type=float, default=0.1)
    p.add_argument('--jsd-weight', type=float, default=0.0, dest='jsd_weight',
                   help='B9: downweight samples whose two augmented views disagree; 0 disables')
    p.add_argument('--jsd-floor', type=float, default=0.25, dest='jsd_floor',
                   help='minimum multiplicative weight for B9 JSD gating')
    p.add_argument('--norm-weights', type=int, default=1)
    p.add_argument('--norm-max-gain', type=float, default=0.0, dest='norm_max_gain',
                   help='maximum per-batch amplification; 0 means unlimited w/w.mean() '
                        '(the verified teammate rule), positive values enable a cap')
    p.add_argument('--reset-tracker-warmup', action=argparse.BooleanOptionalAction,
                   default=False, dest='reset_tracker_warmup',
                   help='clear the teacher posterior at the warm-up boundary; default '
                   'keeps the posterior as in the verified teammate recipe')

    p.add_argument('--ema', type=float, default=0.995)
    p.add_argument('--anchor-weight', type=float, default=0.1)
    p.add_argument('--consistency-weight', type=float, default=0.05)
    p.add_argument('--consistency-symmetric', action='store_true',
                   help='use symmetric stop-gradient consistency; default is the verified one-way teammate term')
    p.add_argument('--proto-weight', type=float, default=0.5)
    p.add_argument('--proto-temp', type=float, default=0.1)
    p.add_argument('--proto-momentum', type=float, default=0.99)
    p.add_argument('--proto-min-weight', type=float, default=0.5)
    p.add_argument('--proto-normalize', action='store_true',
                   help='normalize prototype loss by trusted weight sum; default uses the teammate batch-mean scale')
    p.add_argument('--proto-trusted-loss', action='store_true',
                   help='restrict prototype loss to trusted rows; default follows the '
                        'teammate rule of weighting all rows while gating prototype updates')

    # ---- B11/B17: frozen-CLIP head initialisation (default off)
    p.add_argument('--head-init', default='none', choices=['none', 'frozen'], dest='head_init',
                   help='"frozen": seed the cosine head (and, if enabled, the prototypes and '
                        'the noise judge) from class centroids of frozen-CLIP features, taken '
                        'over the whole training split in one no-grad pass *before* the first '
                        'step. The head ships randomly initialised, so the early epochs are '
                        'spent relearning a 750-way linear map frozen CLIP already knows -- and '
                        'the early epochs are the ones that decide the score (ep4 == ep20 on '
                        'the leaderboard, to 4 decimals). "none" is bit-identical to every '
                        'run recorded so far.')
    p.add_argument('--head-init-rounds', type=int, default=2, dest='head_init_rounds',
                   help='passes of the drop-and-recompute filter in robust_centroids: a sample '
                        'that a different class centroid explains better is dropped and the '
                        'centroids are recomputed. 1 = plain per-class means; more rounds '
                        'iterate the cleaning. The class folders here are English search '
                        'keywords, so this is what keeps a mislabelled image from dragging its '
                        'own centroid off the class.')
    p.add_argument('--head-init-batch-size', type=int, default=256, dest='head_init_batch_size',
                   help='batch size of the frozen feature pass (no gradients, so this can be '
                        'much larger than --batch-size; it only has to fit activations)')
    p.add_argument('--train-pos-embed', action='store_true', dest='train_pos_embed',
                   help='train the resampled positional grid instead of only interpolating it. '
                        'resize_positional_embedding can only *interpolate* the official 7x7 '
                        'grid onto the grid a larger input needs, and this file then freezes it '
                        'with the rest of the backbone, so the interpolation error is permanent '
                        'and grows with the distance from 224 (the probe\'s own cosine to the '
                        'pretrained grid: 0.9917 / 0.9820 / 0.9683 / 0.9581 at 256 / 288 / 320 / '
                        '352). The same 384px resolution with the same weights scored 68.5476 '
                        'with the grid frozen and 71.1382 with it trainable. Costs '
                        '(1 + (size/32)^2) * 768 params -- 0.11M at 384. Off by default so every '
                        'recorded leaderboard number still reproduces bit-for-bit.')
    p.add_argument('--epoch-aug', action=argparse.BooleanOptionalAction, default=True, dest='epoch_aug',
                   help='in index augmentation mode, mix the epoch into each per-image seed; '
                        'ignored in worker mode. Use --no-epoch-aug for the fixed-view index '
                        'ablation.')
    p.add_argument('--warm-robust', action='store_true', dest='warm_robust',
                   help='B4-lite: apply --robust-loss during the warm-up too, at '
                        '--robust-weight. The warm-up is pure CE on the raw labels, and CE on '
                        'a mislabelled sample is precisely the term that drives memorisation '
                        '-- inside the epoch window the score is decided in.')
    p.add_argument('--distill-weight', type=float, default=0.0, dest='distill_weight',
                   help='B14: weight of ``KL(frozen teacher distribution || student)``, added on '
                        'both views from the *first* step (not gated on warm-up -- see the note '
                        'at the loss). The teacher is the frozen official tower over the robust '
                        'prototype bank (--frozen-protos), i.e. the same construction '
                        '--head-init uses; there is no second trained model and inference is '
                        'unchanged. Only the samples whose frozen argmax is their given label '
                        'are distilled, so the term cannot confirm a mistake the student makes. '
                        '0 disables it and is bit-identical to every run recorded so far. '
                        'Untested on the leaderboard: start at 1.0, and read the [distill] line '
                        'each epoch -- it prints the KL per distilled sample beside the CE, and '
                        'if the two are comparable the term is steering the run rather than '
                        'anchoring it, so halve the weight and try again.')
    # ---- the frozen teacher: one artifact, two consumers (B13/B14), default off
    p.add_argument('--frozen-protos', type=int, default=1, dest='frozen_protos',
                   help='B2/B15: prototypes per class in the frozen teacher bank. 1 = the '
                        'single robust class mean (what --head-init uses). >1 = that many '
                        'spherical-k-means medoids per class, built from the samples the '
                        'agreement filter kept. This is the part of B2/B15 the tree did not '
                        'have: a class folder holding two visual clusters gets a mean that sits '
                        'between them and explains neither, so every sample of both clusters '
                        'looks like an outlier. Medoids are real samples, so a query excludes '
                        'its own source exactly (B2\'s 排除自身).')
    p.add_argument('--frozen-temp', type=float, default=0.05, dest='frozen_temp',
                   help='temperature of the frozen teacher distribution, applied to the cosine '
                        'similarity before the softmax: 0.05 keeps most of the mass on the '
                        'teacher argmax while the fine-grained neighbours (the part the one-hot '
                        'label cannot express) keep a visible share. Lower = sharper. Shared by '
                        '--distill-weight and --frozen-mix-rho: they consume one distribution.')
    p.add_argument('--frozen-topk', type=int, default=8, dest='frozen_topk',
                   help='classes kept per sample in the sparse teacher distribution. The full '
                        'distribution is 750-wide and 446 MB; top-8 as (int16, float16) is '
                        '4.6 MB for the whole training set and carries almost all of the mass '
                        'at --frozen-temp 0.05.')
    p.add_argument('--frozen-mix-rho', type=float, default=0.0, dest='frozen_mix_rho',
                   help='B13: soft forward correction at the sample level, from the *first* '
                   'step (warm-up included). The supervised target becomes '
                   '(1-rho) * label_target + rho * frozen_distribution, so most of the '
                   'given label is kept and the rest moves onto what the frozen tower '
                   'supports. It is not gated on agreement; a bad frozen teacher can '
                   'therefore damage every sample. The 2026-10-01 real-data smoke run '
                   'with one-medoid targets agreed with only 32.3% of noisy folder labels '
                   'and rho=0.7 flipped only 2.6% of target argmax values. A small flip '
                   'rate does not mean a small gradient change: all rows receive rho '
                   'teacher mass. The default target bank now uses robust class means; '
                   'check its holdout and target-quality reports before a full run. '
                   'A flip requires q_top - q_folder > (1-rho)/rho, not merely a '
                   'teacher disagreement. 0 disables the mix. Untested on leaderboard.')
    p.add_argument('--frozen-mix-agree-only', action='store_true', dest='frozen_mix_agree_only',
                   help='restrict --frozen-mix-rho to the samples the frozen teacher agrees '
                        'with, i.e. B14\'s conservative gate applied to B13\'s target. Compare '
                        'against the default: agreeing-only keeps every label the teacher signs '
                        'off on and changes nothing where it disagrees, which is the safe half '
                        'of the correction and none of the part that can fix a systematically '
                        'mislabelled folder.')
    a = p.parse_args(argv)
    # Frozen-target options alter the training objective from the first batch.
    # Reject invalid values here instead of allowing negative targets or silently
    # replacing an invalid temperature with 1e-4 inside frozen_soft_targets().
    if not 0.0 <= a.frozen_mix_rho <= 1.0:
        p.error('--frozen-mix-rho must be between 0 and 1')
    if a.frozen_temp <= 0.0:
        p.error('--frozen-temp must be > 0')
    if a.frozen_topk < 1:
        p.error('--frozen-topk must be >= 1')
    if a.frozen_protos < 1:
        p.error('--frozen-protos must be >= 1')
    return a


if __name__ == '__main__':
    main(parse_args())
