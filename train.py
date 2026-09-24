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


def patch_grid(clip_model, size):
    """``(patch, n, clean)`` for a ``size``-pixel input.

    ``clean`` is False when the size is not a whole number of patches.  This
    matters for the resolution ladder: ViT-B/32 has **32px** patches, so 224 /
    256 / 288 / 320 / 352 / 384 are clean and **336 is not** (336/32 = 10.5).
    336 is the usual CLIP figure, but it comes from ViT-L/14@336 where
    336/14 = 24 exactly -- copying it onto a patch-32 tower does not mean what
    it means there.

    The consequence is exact, and it is worth being precise about because the
    failure is silent rather than an error.  In open_clip (checked against the
    vendored ``_oc_src/``, 3.x):

    * ``transformer.PatchEmbed.__init__`` sets ``grid_size = image_size // patch``
      (floor division), so a forced 336 gives a 10x10 position-embedding grid;
    * ``model.resize_pos_embed`` early-returns when the sequence length already
      matches, else bicubic+antialias interpolates onto that grid;
    * the patch convolution itself can only produce ``(size - patch) // patch + 1``
      tokens, which for 336 is also 10.

    So 336 loads and runs fine -- it is *the same 10x10 model as 320*, fed a
    336px crop whose outer 16px no patch ever reads.  Strictly dominated by 320.
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


def build_clip(name, pretrained='openai', image_size=224):
    """Create the CLIP visual tower at ``image_size``.

    ``force_image_size`` makes open_clip resample the *positional embeddings*
    onto the new patch grid while loading the pretrained weights (bicubic +
    antialias, ``resample_abs_pos_embed``).  ViT-B/32 tolerates this because its
    patch embedding is a stride-32 convolution that does not care about the
    input size.  Architecture and weights are unchanged -- the competition
    explicitly permits this, see README_AUTODL.md 5.

    224 is special-cased to *not* pass the flag, so the default path stays
    bit-identical to the runs whose leaderboard scores we compare against
    (resampling a 7x7 grid onto itself is not guaranteed to be a no-op).
    Passing an unsupported ``force_image_size`` raises rather than being
    ignored: a silently-ignored resolution would mean training at 224 while
    believing 320, and nothing downstream would notice.

    Prefer a multiple of the patch size (32 for ViT-B/32) -- see
    ``patch_grid``; a size like 336 is warned about rather than accepted
    silently.
    """
    check_backbone(name)
    size = int(image_size)
    kw = {} if size == 224 else {'force_image_size': size}
    model = open_clip.create_model(name, pretrained=pretrained, **kw)
    if size != 224:
        ps, n, clean = patch_grid(model, size)
        if not clean:
            print(f'WARNING: --image-size {size} is not a whole number of {ps}px patches '
                  f'({size / ps:g}); the patch convolution produces a {n}x{n} grid, so this '
                  f'behaves like {n * ps}px with a {size - n * ps}px margin that no patch '
                  f'ever sees. Use a multiple of {ps} (224/256/288/320/352/384).')
    return model


#: the view geometries ``eval_transform(crop=...)`` can produce
CROP_POLICIES = ('center', 'full', 'pad')


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


def eval_transform(size=224, flip=False, *, crop='center'):
    """Eval-time preprocessing.  One definition, shared by everything.

    Used by ``train.py`` (hold-out val), ``infer.py``, ``valmetrics.py``,
    ``analyze.py`` and ``probe.py``.  A mismatch between any two of them is
    invisible and silently costs accuracy; the only symptom is that a
    checkpoint's logged ``val_acc`` stops reproducing (see the self-check in
    ``valmetrics.py``).

    ``crop`` selects how much of the image survives:

    ``'center'``
        Resize the short side to ``size*256/224``, centre crop.  The CLIP and
        ImageNet convention, and what every run up to 2026-09-24 used.
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
    """
    if crop not in CROP_POLICIES:
        raise ValueError(f'crop={crop!r}, expected one of {CROP_POLICIES}')
    size = int(size)
    if crop == 'center':
        ops = [transforms.Resize(max(1, int(round(size * RESIZE_RATIO)))),
               transforms.CenterCrop(size)]
    elif crop == 'full':
        ops = [transforms.Resize((size, size))]
    else:
        ops = [PadToSquare(size)]
    if flip:                            # test-time augmentation, p=1.0
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
    return transforms.Compose(ops)


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
        return F.normalize(feats.float(), dim=-1) @ F.normalize(self.proto, dim=-1).t() / self.temp


class Net(nn.Module):
    def __init__(self, clip_model, nclass, lora_rank=8, lora_target='all',
                 proto_momentum=0.99, proto_temp=0.1):
        super().__init__()
        self.clip = clip_model
        dim = clip_model.visual.output_dim
        self.head = CosineClassifier(dim, nclass)
        self.proto = ProtoHead(dim, nclass, proto_momentum, proto_temp)
        self.n_lora = add_lora(self.clip.visual, lora_rank, 2 * lora_rank, target=lora_target)
        for name, p in self.clip.named_parameters():
            if not ('.A' in name or '.B' in name):
                p.requires_grad = False

    def forward(self, x, return_feat=False):
        z = F.normalize(self.clip.encode_image(x).float(), dim=-1)
        out = self.head(z)
        return (out, z) if return_feat else out

    @torch.no_grad()
    def anchor_feat(self, x):
        """Frozen-CLIP embedding of ``x`` (LoRA off, eval mode, no grad)."""
        visual = self.clip.visual
        was_training = visual.training
        visual.eval()
        try:
            with lora_disabled(visual):
                z = visual(x)
        finally:
            visual.train(was_training)
        return F.normalize(z.float(), dim=-1)

    def trainable_state_dict(self):
        """Only what is actually trained -- the frozen backbone is rebuilt from
        the official OpenAI weights, so checkpoints stay a few MB instead of
        ~350 MB of untouched CLIP parameters."""
        frozen = {n for n, p in self.named_parameters() if not p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k not in frozen}


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
            return Image.new('RGB', (224, 224), (127, 127, 127))

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
        h = ((self.seed * 0x9E3779B1) ^ (int(i) * 0x85EBCA77) ^ (int(view) * 0xC2B2AE3D))
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
    tr = ImageFolderNoisy(a.data, train_tf, False, a.val_ratio, a.seed, 'train', stochastic=True)
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
    loader = DataLoader(TwoView(tr), batch_size=a.batch_size, sampler=sampler,
                        num_workers=a.workers, pin_memory=True, drop_last=True,
                        persistent_workers=a.workers > 0, worker_init_fn=seed_worker,
                        generator=g)
    vloader = DataLoader(va, batch_size=a.batch_size * 2, shuffle=False, num_workers=a.workers,
                         pin_memory=True, persistent_workers=a.workers > 0, worker_init_fn=seed_worker)
    return loader, vloader


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
SNAPSHOT_KEYS = ('model', 'classes', 'class_counts', 'args', 'epoch', 'val_acc',
                 'val_acc_hi', 'lora_rank', 'lora_target', 'pretrained', 'model_name')


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
    print('config: ' + ' '.join(f'{k}={v}' for k, v in sorted(vars(a).items())))

    clip_model = build_clip(a.model, a.pretrained, a.image_size)
    model = Net(clip_model, nclass, a.lora_rank, a.lora_target,
                a.proto_momentum, a.proto_temp).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model={a.model}/{a.pretrained} layers={model.n_lora} trainable params={n_train/1e6:.3f}M')

    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

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
                                relabel_mix=a.relabel_mix, device=device)
    robust = make_robust_loss(a.robust_loss, a.gce_q, a.apl_k, a.apl_b, a.apl_rce)

    opt = torch.optim.AdamW(param_groups(model, a.weight_decay), lr=a.lr)
    sched = make_scheduler(opt, a)
    use_amp = a.amp != 'none' and device.type == 'cuda'
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(a.amp)
    try:                                    # torch >= 2.3 API, older one as a fallback
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and a.amp == 'fp16')
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and a.amp == 'fp16')

    # -1.0, not 0.0: with --select val_acc_hi the score is 0.0 for exactly as
    # long as no sample clears max(p) >= 0.8, which on 750 classes can hold for
    # the first epochs -- and `score > best` would then write no best.pt at all.
    # Starting below every possible score guarantees the first epoch lands one.
    start_epoch, best = 0, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        teacher = copy.deepcopy(model).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        opt.load_state_dict(ck['optim'])
        load_sched(sched, ck['sched'])
        tracker.load_state_dict(ck['tracker'])
        torch.set_rng_state(ck['rng']['torch'])
        if ck['rng'].get('cuda') is not None:
            torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        random.setstate(ck['rng']['python'])
        start_epoch = ck['epoch'] + 1
        best = ck.get('val_acc' if a.select == 'val_acc' else 'val_acc_hi', -1.0)
        print(f'resumed from {a.resume} at epoch {start_epoch} (best={best:.4f})')

    probe = next(iter(loader))
    print(f'first batch ok: x1={tuple(probe[0].shape)} x2={tuple(probe[1].shape)} '
          f'labels={probe[2].tolist()[:8]} idx={probe[3].tolist()[:8]}')

    for ep in range(start_epoch, a.epochs):
        warm = ep < a.warmup_epochs
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
        for it, (x1, x2, y, idx) in enumerate(loader):
            if a.limit_batches and it >= a.limit_batches:
                break
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx_g = idx.to(device, non_blocking=True)
            bsz = y.numel()

            need_anchor = a.anchor_weight > 0 or (warm and (a.proto_weight > 0 or judge is not None))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                anchor = model.anchor_feat(x1) if need_anchor else None   # frozen CLIP, LoRA off
                out, z = model(x1, True)
                out2, z2 = model(x2, True)
                with torch.no_grad():
                    tout, tz = teacher(x1, True)        # one view is enough for the teacher
                tprob = F.softmax(tout.float(), 1)

            tracker.update(idx_g, tprob)                # free: the teacher already ran

            if warm:
                # pure supervised warm-up on the raw labels; there is no teacher
                # mixture yet, so both targets coincide
                hard = y
                mix_t = soft = smooth_target(F.one_hot(y, nclass).to(torch.float32),
                                             a.label_smooth, nclass)
                w = torch.ones(bsz, device=device)
            else:
                hard = tracker.label[idx_g]             # possibly corrected label
                # `mix_t` carries the teacher mixture and is what the robust loss
                # sees; `soft` adds label smoothing and is the CE target.  They
                # differ on purpose -- see the note in losses.nce.
                mix_t = tracker.target(idx_g, y)
                soft = smooth_target(mix_t, a.label_smooth, nclass)
                w = tracker.weight[idx_g] * tprob.max(1).values.clamp(a.conf_floor, 1.0).pow(a.conf_gamma)
                if a.norm_weights:
                    # keep the effective step size stable as filtering kicks in
                    w = w / w.mean().clamp_min(1e-6)

            # labelled pass on both views (the "passive" term)
            ce1 = xent(out, soft)
            ce2 = xent(out2, soft)
            loss = 0.5 * ((w * ce1).mean() + (w * ce2).mean())
            if not warm and a.robust_weight > 0:
                rob = 0.5 * (robust(out, mix_t) + robust(out2, mix_t))
                loss = loss + a.robust_weight * (w * rob).mean()
            if a.consistency_weight > 0:
                loss = loss + a.consistency_weight * F.mse_loss(z, z2.detach())
            if anchor is not None and a.anchor_weight > 0:
                loss = loss + a.anchor_weight * (1 - (z * anchor).sum(1)).mean()
            if a.proto_weight > 0 and not warm:
                # contrastive pull towards the trust-aligned class prototype
                trusted = tracker.weight[idx_g] >= a.proto_min_weight
                pl = model.proto.logits(torch.cat([z, z2], 0))
                pw = w.repeat(2)
                loss = loss + a.proto_weight * (pw * xent(pl, mix_t.repeat(2, 1))).mean()

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
                    if judge is not None:
                        judge.add(idx_g, anchor)
                if a.proto_weight > 0 and not warm:
                    # the prototype EMA still needs a hard assignment: averaging
                    # features under a soft target would let a confused sample
                    # drag two prototypes at once
                    model.proto.update(tz, hard, trusted)

            running += loss.item() * bsz
            seen += bsz

        if warm and ep + 1 == a.warmup_epochs:
            if a.proto_weight > 0 or judge is not None:
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
            tracker.reset()
            print('tracker posterior reset at the end of warm-up (prob/seen cleared); the next '
                  'epoch runs with every sample unseen, i.e. unfiltered, while it refills')

        sched.step()
        vl, va_acc, va_hi = evaluate(model, vloader, device, amp_dtype, use_amp)
        score = va_acc if a.select == 'val_acc' else va_hi
        print(f'epoch {ep + 1}/{a.epochs} loss={running / max(seen, 1):.4f} '
              f'val_loss={vl:.4f} val_acc={va_acc:.4f} val_acc_hi={va_hi:.4f} '
              f'lr={opt.param_groups[0]["lr"]:.2e} time={time.time() - t0:.1f}s')

        ck = {'model': model.trainable_state_dict(), 'classes': tr.class_to_idx,
              'class_counts': class_counts,
              'args': vars(a), 'epoch': ep, 'val_acc': va_acc, 'val_acc_hi': va_hi,
              'lora_rank': a.lora_rank, 'lora_target': a.lora_target, 'pretrained': a.pretrained,
              'model_name': a.model,
              'optim': opt.state_dict(), 'sched': sched.state_dict(), 'tracker': tracker.state_dict(),
              'rng': {'torch': torch.get_rng_state(),
                      'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                      'python': random.getstate()}}
        torch.save(ck, out_dir / 'last.pt')
        if a.save_every and (ep + 1) % a.save_every == 0:
            # The hold-out split carries the same *structured* label noise as the
            # training set, so `val_acc` can be improved by memorising that noise
            # -- which costs accuracy on the clean test set.  Keep snapshots so
            # several epochs can be scored on the real leaderboard and the best
            # one picked, instead of trusting a proxy that rewards the failure.
            torch.save(thin(ck), out_dir / f'ep{ep + 1}.pt')
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
    p.add_argument('--lr-warmup-epochs', type=int, default=1, dest='lr_warmup_epochs',
                   help='linear LR warm-up from --lr * %g up to --lr over this many epochs, '
                        'then cosine decay over the rest. Default 1: the cosine head is '
                        'randomly initialised but CosineAnnealingLR starts at peak LR. '
                        '0 restores the old schedule exactly.' % LR_WARMUP_START)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--cudnn-benchmark', action='store_true',
                   help='faster but not bit-reproducible (default: off)')
    p.add_argument('--amp', default='bf16', choices=['bf16', 'fp16', 'none'])
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--image-size', type=int, default=224,
                   help='input resolution. 224 = the pretrained/CLIP default and '
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

    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-target', default='all', choices=['all', 'mlp'])

    p.add_argument('--crop-min', type=float, default=0.55)
    p.add_argument('--randaug-n', type=int, default=2)
    p.add_argument('--randaug-m', type=int, default=9)

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
    p.add_argument('--norm-weights', type=int, default=1)

    p.add_argument('--ema', type=float, default=0.995)
    p.add_argument('--anchor-weight', type=float, default=0.1)
    p.add_argument('--consistency-weight', type=float, default=0.05)
    p.add_argument('--proto-weight', type=float, default=0.5)
    p.add_argument('--proto-temp', type=float, default=0.1)
    p.add_argument('--proto-momentum', type=float, default=0.99)
    p.add_argument('--proto-min-weight', type=float, default=0.5)
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
