"""Fast self-check: run this before spending GPU hours.

    python selftest.py

It needs no dataset, no GPU and no CLIP weights -- ``open_clip`` is replaced by
a tiny stub exposing the same interface.  Covered:

* robust losses: shapes, GCE bound, NCE tangency/continuity at ``p = k``, and
  the soft-target form reducing exactly to the index form on a one-hot target;
* label-trust tracker: agree / pseudo-label / untrusted branches, rejection cap,
  statistics that partition the set, EMA update, the warm-up reset;
* the frozen-CLIP judge: that it demotes without relabelling, that a nonzero
  margin keeps ties out, that unseen samples and classes with no warm-up
  centroid are never judged, and that the veto survives a checkpoint;
* targets: label smoothing, and the tracker's mixed pseudo-label (the given
  label keeps ``1 - relabel_mix`` of the mass -- not a hard overwrite);
* prototype head: bootstrap, EMA update, classes never seen before;
* LoRA: zero-init identity, disable == frozen CLIP, checkpoint filtering;
* ``--image-size``: that a non-224 size is met by resampling the positional grid
  rather than by ``force_image_size`` (the two agree on shape and differ in
  value, and only one of them matches the run that scored 69.218), that the
  resample's grid arithmetic is right and that a non-multiple of 32 is refused,
  that ``antialias`` actually changes the tensor, that the recorded fingerprint
  round-trips and is checked on load, that the eval transform's geometry follows
  the requested size, and that the patch grid is a whole number of 32px patches
  (336 is not -- it is a patch-14 figure);
* optimiser groups: ``head.logit_scale`` is *not* weight-decayed, because the
  tracker's thresholds are absolute probabilities;
* the LR schedule: it ramps linearly to ``--lr`` and only then anneals, and
  ``--lr-warmup-epochs 0`` still reproduces the old peak-first schedule;
* augmentation reproducibility: item ``i``'s two augmented views are a pure
  function of ``(seed, index, view)``, so they are identical under
  ``--workers 0/1/2/3`` -- otherwise the submitted code cannot reproduce the
  submitted score;
* optional trust gates: JSD view disagreement only down-weights samples,
  class-conditional confidence thresholds are opt-in, and batch weight
  normalisation does not turn an all-noisy batch into clean supervision;
* ``--head-init frozen``: that the drop-and-recompute centroid filter beats a
  plain per-class mean on deliberately mislabelled features and that its kept set
  is cleaner than the set it started from, that a class with no samples keeps its
  random row rather than a zero one, that ``--epoch-aug`` reaches the workers
  (it only does because the flag turns persistent workers off) while ``epoch 0``
  still reproduces the old seed formula byte for byte, and that the whole new
  flag set runs the training program end to end with the seeding visible in the
  saved head;
* ``--distill-weight`` (B14): that the sparse frozen target's gate is exactly
  "the frozen argmax is the given label" (asserted against a hand-built
  similarity matrix), that a gated-out row carries no mass at all, that an
  agreeing row is a distribution over the top-k whose modal class is the given
  label, that ``--distill-temp`` acts on the class similarity, and that the KL
  matches the value written out by hand (0.130812 for q=(0.75,0.25), p=(0.5,0.5))
  with zero rows inert in value *and* in gradient -- plus that the flag does not
  seed the head, so B11 and B14 stay separately measurable;
* the proxy's caches: that a 2-image pass is not read back as if it were the
  4-image one (the name carries the view and not the image list, so ``--limit``,
  ``--val-ratio``, ``--seed`` and ``--data`` every one of them used to poison it
  silently), for frozen features and for model probabilities alike;
* a full 4-epoch training run and a full inference run on a throw-away
  4-class image set, including the checkpoint round-trip, the inference-only
  snapshots, the CSV format, the multi-tau logit adjustment and the TTA view
  averaging: that a single view reproduces the plain run exactly, that a named
  view set hands the tower one pass over every image *per view*, and that its
  views are genuinely different inputs from each other.
"""
# NOTE: the count in `main`'s summary line is len(CHECKS), not a number written
# down here -- adding a check cannot make this docstring wrong, which is why it
# does not name one.
import csv
import io
import math
import random
import re
import shutil
import sys
import tempfile
import traceback
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# --------------------------------------------------------------------------- #
# stub out open_clip BEFORE train.py imports it
# --------------------------------------------------------------------------- #
DIM = 32


class _StubBlock(nn.Module):
    """Mimics open_clip's ResidualAttentionBlock *structurally*.

    The point is ``self.attn``: a real ``nn.MultiheadAttention``, which does not
    call ``out_proj(x)`` but reads ``out_proj.weight`` / ``out_proj.bias``
    directly.  A LoRA wrapper that only implements ``forward`` therefore blows up
    here -- which is exactly the crash that a plain-MLP stub used to miss.
    """

    def __init__(self, dim=DIM, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        h = self.norm(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(x)


class _StubVisual(nn.Module):
    """Stand-in for open_clip's ViT: takes images *or* plain feature vectors and
    returns ``(B, dim)`` embeddings, like ``visual(x)`` does for CLIP.

    It carries a positional grid too, because that part is not optional.  The
    grid is frozen, ``trainable_state_dict`` keeps only trainable parameters, so
    the grid never enters a checkpoint and every load *rebuilds* it -- which
    makes the rebuild, not the weights, what decides which model a checkpoint is
    evaluated as.  A stub without a grid would let that whole mechanism, and the
    fingerprint check that guards it, go untested.

    The three geometry attributes mirror the real tower's spellings and types:
    ``patch_size`` is a 2-tuple, ``image_size`` and ``grid_size`` likewise.  The
    patch convolution is called ``stem`` rather than open_clip's ``conv1`` on
    purpose, so ``patch_grid`` has to reach it through its scan fallback -- the
    same path it takes on the real tower, where the grid sitting under ``conv1``
    is not where ``patch_grid`` looks first either.

    The grid's values are seeded rather than random so that two towers built the
    same way are bit-identical, the way two loads of the same pretrained weights
    are.  Without that, the fingerprint round-trip below could only ever fail.
    """

    def __init__(self, dim=DIM, in_dim=12, grid=7, patch=32):
        super().__init__()
        self.output_dim = dim
        self.patch_size = (patch, patch)
        self.image_size = (grid * patch, grid * patch)
        self.grid_size = (grid, grid)
        self.stem = nn.Conv2d(3, in_dim, kernel_size=patch, stride=patch)   # patch embed
        self.proj = nn.Linear(in_dim, dim)                            # 1
        self.block = _StubBlock(dim)                                  # 3 more
        g = torch.Generator().manual_seed(0)
        self.positional_embedding = nn.Parameter(
            torch.randn(grid * grid + 1, dim, generator=g) * 0.02, requires_grad=False)

    def forward(self, x):
        if x.dim() == 4:                    # (B, 3, H, W) -> (B, in_dim)
            x = self.stem(x).mean(dim=(2, 3))
        x = self.proj(x).unsqueeze(1)       # (B, 1, dim): one token
        return self.block(x).squeeze(1)


class _StubCLIP(nn.Module):
    """The parts of ``open_clip``'s ``CLIP`` that this codebase actually touches.

    ``logit_scale`` is here for fidelity, not for show: a real CLIP module has
    one, and ``Net.__init__`` freezes it the same way it freezes every other
    non-LoRA parameter.  Leaving it out of the stub would make
    ``check_param_groups`` test an attribute that does not exist on the object it
    is testing -- which is exactly how it failed the first time it ever ran
    (2026-09-26: the assertion reached for ``net.clip.logit_scale`` and got an
    ``AttributeError``).  With it present, that assertion checks the freeze loop
    instead of the stub's shape.
    """

    def __init__(self, dim=DIM):
        super().__init__()
        self.visual = _StubVisual(dim)
        # open_clip initialises this to log(1/0.07); the value never matters here
        # because every test that looks at it only reads `.requires_grad`.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, x):
        return self.visual(x)


_stub = types.ModuleType('open_clip')
#: Every create_model call as ``(name, pretrained, kwargs, model)``.  The kwargs
#: are there so a test can see what ``train.build_clip`` passed through -- it must
#: *not* be ``force_image_size``, which resamples the positional grid with
#: ``antialias=True`` and is therefore a different model from the one the scored
#: 288 run used (same shape, different numbers, no error).
#:
#: The model is the 4th element rather than a separate list on purpose: it is
#: found by unpacking, so ``_STUB_CALLS.clear()`` clears it too and no test can
#: read a tower left over from an earlier one.  Callers that unpack three values
#: are unaffected.
_STUB_CALLS = []


def _stub_create_model(name, pretrained=None, **kw):
    """A fresh stub per call, but *the same* stub every call.

    Real ``create_model`` loads one file of pretrained weights, so two towers
    built in one process are bit-identical; the stub drew from torch's global RNG
    instead and so differed in every frozen parameter.  That mattered because
    ``trainable_state_dict`` carries only *trainable* parameters, so nothing in a
    checkpoint can restore a frozen weight: the checks that compare the output of
    two runs (``--tta-sizes 224`` against no flag, ``--tta-agg feat``,
    ``--workers 0`` against ``2``) were scoring two unrelated backbones and could
    not pass, whatever the code under test did.  The comparison is the point of
    those checks, so the premise is what gets fixed here.

    Seeding the global stream and restoring it around the construction is what
    makes this safe: the stub keeps torch's normal initialisation (so activation
    scales are the ones every other assertion was written against), and no other
    check sees its own randomness moved.
    """
    state = torch.get_rng_state()
    torch.manual_seed(0)
    try:
        model = _StubCLIP()
    finally:
        torch.set_rng_state(state)
    _STUB_CALLS.append((name, pretrained, kw, model))
    return model


_stub.create_model = _stub_create_model
sys.modules['open_clip'] = _stub

import analyze  # noqa: E402
import infer  # noqa: E402
import losses  # noqa: E402
import noise  # noqa: E402
import probe  # noqa: E402
import train  # noqa: E402


def train_args(argv=()):
    """``train.parse_args`` for a check that only cares about *some* flags.

    ``--data`` is required, so every call has to supply one.  Three call sites
    once passed only the flags under test and all three died on that requirement
    the first time the checks were ever run (2026-09-26) -- which is why the rule
    lives in one helper rather than in four argument lists that each have to
    remember it.  ``.`` is never touched: these checks read the parsed args, not
    the dataset.
    """
    return train.parse_args(['--data', '.'] + list(argv))


def check_losses():
    torch.manual_seed(0)
    logits, y = torch.randn(8, 5), torch.randint(0, 5, (8,))
    for name in ('ce', 'gce', 'nce', 'rce', 'apl'):
        v = losses.make_robust_loss(name)(logits, y)
        assert v.shape == (8,), f'{name}: expected shape (8,), got {tuple(v.shape)}'
        assert torch.isfinite(v).all(), f'{name}: non-finite loss'

    # GCE is bounded by 1/q
    worst = losses.gce(logits * 50, y, q=0.7)
    assert float(worst.max()) <= 1 / 0.7 + 1e-6, 'GCE exceeded its 1/q bound'

    # NCE: at p == k it must equal CE, and the linear branch must be the tangent
    k = 0.2
    lg = torch.zeros(1, 2)
    lg[0, 0] = math.log(k / (1 - k))                 # softmax -> p_0 == k exactly
    tgt = torch.tensor([0])
    assert abs(float(losses.nce(lg, tgt, k)) - float(losses.ce(lg, tgt))) < 1e-5, \
        'NCE is not continuous with CE at p = k'

    lg2 = lg.clone()
    p_target = k * 1.001                             # just inside the linear branch
    lg2[0, 0] = math.log(p_target / (1 - p_target))
    tangent = float(losses.nce(lg2, tgt, k))
    true_val = -math.log(p_target)
    assert abs(tangent - true_val) < 1e-5, \
        f'NCE linear branch is not the tangent line ({tangent} vs {true_val})'

    # RCE is APL's MAE-like term: -log(eps) * (1 - p_y), bounded in [0, 9.21]
    py = F.softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
    assert torch.allclose(losses.rce(logits, y), -math.log(1e-4) * (1 - py), atol=1e-6), \
        'RCE is not -log(eps) * (1 - p_y)'
    worst = losses.rce(logits * 50, y)
    assert 0.0 <= float(worst.min()) and float(worst.max()) <= -math.log(1e-4) + 1e-6, \
        f'RCE left its [0, {(-math.log(1e-4)):.2f}] range'

    # the distribution form must be a pure generalisation: on a one-hot target it
    # has to reproduce the index form exactly, or the mixed pseudo-labels and the
    # label smoothing would silently change the objective that was tuned
    onehot = F.one_hot(y, 5).float()
    for name in ('ce', 'gce', 'nce', 'rce', 'apl'):
        f = losses.make_robust_loss(name)
        assert torch.allclose(f(logits, onehot), f(logits, y), atol=1e-6), \
            f'{name}: one-hot distribution disagrees with the index form'
        assert f(logits, onehot).shape == (8,), f'{name}: soft target changed the output shape'

    # CE is linear in the target, so a 50/50 mixture must be the mean of the two
    y2 = (y + 1) % 5
    half = 0.5 * onehot + 0.5 * F.one_hot(y2, 5).float()
    assert torch.allclose(losses.ce(logits, half),
                          0.5 * (losses.ce(logits, y) + losses.ce(logits, y2)), atol=1e-6), \
        'CE is not linear in the target distribution'
    print('  losses ok')


def check_tracker():
    n, C = 40, 5
    y = torch.arange(n) % C

    def tracker(**kw):
        # momentum 0 -> the first observation is stored verbatim;
        # the rejection cap is disabled unless a test asks for it
        kw = {'momentum': 0.0, 'max_noise_frac': 1.0, **kw}
        return noise.LabelTrustTracker(y.tolist(), C, **kw)

    def peaked(targets, conf=0.95):
        p = torch.full((len(targets), C), (1 - conf) / (C - 1))
        p[torch.arange(len(targets)), targets] = conf
        return p

    # 1) everything agrees with the given label -> all clean, nothing touched
    tk = tracker()
    tk.update(torch.arange(n), peaked(y))
    st = tk.refresh()
    assert st['clean'] == n and st['relabel'] == 0 and st['noisy'] == 0, st
    assert (tk.weight == 1).all() and (tk.label == y).all()

    # 2) the second half is confidently wrong -> pseudo-label correction
    tk.prob = peaked(y).clone()
    tk.prob[n // 2:] = peaked((y + 1) % C)[n // 2:]
    st = tk.refresh()
    assert st['relabel'] == n // 2 and st['clean'] == n // 2, st
    assert (tk.label[n // 2:] == (y[n // 2:] + 1) % C).all(), 'pseudo-labels not applied'
    assert (tk.label[:n // 2] == y[:n // 2]).all(), 'clean labels were modified'
    assert (tk.weight[:n // 2] == 1).all() and (tk.weight[n // 2:] == 0.5).all()

    # 3) a teacher that disagrees and is unsure -> untrusted, only down-weighted.
    #    argmax of a uniform posterior is class 0, so only the y == 0 rows agree
    tk.prob[:] = 1.0 / C
    st = tk.refresh()
    agree = y == 0
    assert st['relabel'] == 0, st
    assert st['clean'] == int(agree.sum()) and st['noisy'] == int((~agree).sum()), st
    assert (tk.weight[agree] == 1.0).all(), 'agreed samples lost their full weight'
    assert (tk.weight[~agree] == 0.1).all(), 'untrusted samples were not down-weighted'
    assert (tk.label == y).all(), 'ambiguous samples must keep their label'

    # 3b) REGRESSION.  The teacher's top-1 pick *is* the given label, it is just
    #     not confident (0.4 << 0.8).  The old `p[y] >= tau_clean` gate dumped
    #     these into the reject pile; on the real 500-class run that was ~83% of
    #     the training set, and the 40% cap then pinned on every single epoch.
    tk = tracker()
    p = torch.full((n, C), 1e-3)
    p[torch.arange(n), y] = 0.4
    tk.update(torch.arange(n), p)
    st = tk.refresh()
    assert st['clean'] == n and st['noisy'] == 0 and st['relabel'] == 0, st
    assert (tk.weight == 1).all(), 'agreement alone must be enough for full weight'

    # 4) the rejection cap holds even if everything looks untrusted, and the
    #    statistics must still partition the set (capped samples go back to
    #    full weight, so they belong to `clean`)
    tk2 = tracker(max_noise_frac=0.25)
    tk2.seen[:] = True
    tk2.prob[:] = 1.0 / C
    st = tk2.refresh()
    assert st['noisy'] <= int(0.25 * n), f'rejection cap ignored: {st}'
    assert st['capped'] > 0, 'the cap never bound, so this test is not testing it'
    assert st['clean'] + st['relabel'] + st['noisy'] + st['unseen'] == n, \
        f'statistics do not partition the set: {st}'

    # 5) samples never drawn keep their label at full weight
    tk3 = tracker()
    tk3.update(torch.arange(n // 2), peaked(y[:n // 2]))
    tk3.prob[n // 2:] = 1.0 / C
    st = tk3.refresh()
    assert st['unseen'] == n // 2 and (tk3.weight[n // 2:] == 1).all(), st

    # 6) EMA maths
    tk4 = noise.LabelTrustTracker(y.tolist(), C, momentum=0.9)
    one = F.one_hot(torch.zeros(1, dtype=torch.long), C).float()
    two = F.one_hot(torch.ones(1, dtype=torch.long), C).float()
    tk4.update(torch.tensor([0]), one)
    tk4.update(torch.tensor([0]), two)
    assert torch.allclose(tk4.prob[0], 0.9 * one[0] + 0.1 * two[0], atol=1e-6), 'EMA is wrong'
    print('  tracker ok')


def check_targets():
    """Label smoothing, and the tracker's *mixed* pseudo-label."""
    t = F.one_hot(torch.tensor([1, 3]), 5).float()
    s = train.smooth_target(t, 0.1, 5)
    assert torch.allclose(s.sum(1), torch.ones(2)), 'a smoothed target must still sum to 1'
    # 0.9 on the target *plus* the 0.1/5 that smoothing puts on every class
    assert torch.allclose(s[:, 1], torch.tensor([0.92, 0.02]), atol=1e-6), s
    assert torch.allclose(s[:, 3], torch.tensor([0.02, 0.92]), atol=1e-6), s
    assert torch.allclose(train.smooth_target(t, 0.0, 5), t), 'smooth=0 must be a no-op'

    # tracker.target(): mix == 0 -> a hard one-hot; mix == m -> m on the teacher's
    # pick and 1 - m still on the given label.  The point is the hedge: under
    # weakly-correlated annotation noise a hard overwrite promotes the teacher's
    # confusion to ground truth.
    C, n = 5, 6
    y = torch.tensor([0, 1, 2, 3, 4, 0])
    tk = noise.LabelTrustTracker(y.tolist(), C, momentum=0.0, max_noise_frac=1.0,
                                 relabel_mix=0.25)
    tk.pred[:] = (y + 1) % C
    tk.mix[:] = 0.0
    tk.mix[[1, 4]] = 0.25
    tg = tk.target(torch.arange(n), y)
    assert tg.shape == (n, C), tg.shape
    assert torch.allclose(tg.sum(1), torch.ones(n)), 'targets must be distributions'
    kept = [0, 2, 3]
    assert torch.allclose(tg[kept], F.one_hot(y[kept], C).float()), \
        'a sample with mix == 0 must stay a hard one-hot'
    assert abs(float(tg[1, 1]) - 0.75) < 1e-6, f'the given label lost too much mass: {tg[1]}'
    assert abs(float(tg[1, 2]) - 0.25) < 1e-6, f'the teacher pick got the wrong mass: {tg[1]}'

    # ... and refresh() is what decides where the mix goes
    tk2 = noise.LabelTrustTracker(y.tolist(), C, momentum=0.0, max_noise_frac=1.0,
                                  relabel_mix=0.4)
    p2 = torch.full((n, C), 0.05)
    p2[torch.arange(n), y] = 0.6                     # first half: the teacher agrees
    p2[n // 2:, :] = 0.02
    # both indices must be arrays of the same length.  With a slice and an array
    # in one index tuple PyTorch does NOT pair them up -- it takes the outer
    # product, writing all nine (row, col) combinations instead of the three
    # intended ones.  That silently built a different p2 and made this assertion
    # fail while looking like a tracker bug.
    rows = torch.arange(n // 2, n)
    p2[rows, (y[rows] + 1) % C] = 0.9                # second half: confident, wrong
    # check the fixture before blaming the tracker: the last time this indexing
    # trap fired, the assertion below failed and the message pointed at
    # LabelTrustTracker, which was innocent
    n_disagree = int((p2.max(1).indices != y).sum())
    assert n_disagree == n // 2, \
        f'the test fixture is malformed: {n_disagree} rows disagree with their label, want {n // 2}'
    tk2.update(torch.arange(n), p2)
    st = tk2.refresh()
    assert st['relabel'] == n // 2, st
    assert abs(float(tk2.mix[0])) < 1e-9, 'an agreed sample must carry no teacher mass'
    assert abs(float(tk2.mix[-1]) - 0.4) < 1e-6, 'relabelled samples must use --relabel-mix'
    assert abs(float(tk2.weight[0]) - 1.0) < 1e-6 and abs(float(tk2.weight[-1]) - 0.5) < 1e-6, \
        'the relabel weight changed'
    print('  targets ok')


def check_proto():
    C, D = 5, DIM
    torch.manual_seed(0)
    means = torch.randn(C, D)
    ph = train.ProtoHead(D, C, momentum=0.9, temp=0.1)
    ph.init_from_means(means, torch.ones(C, dtype=torch.bool))
    assert ph.filled.all()
    lg = ph.logits(F.normalize(means, dim=-1))
    assert lg.shape == (C, C)
    assert (lg.argmax(1) == torch.arange(C)).all(), 'a class mean does not match its own prototype'

    # classes never seen before must be filled on first sight and not crash
    ph2 = train.ProtoHead(D, C)
    ph2.init_from_means(means, torch.tensor([True, True, True, False, False]))
    assert not bool(ph2.filled[3])
    z = F.normalize(torch.randn(6, D), dim=-1)
    tgt = torch.tensor([3, 3, 4, 0, 1, 2])
    before = ph2.proto.clone()
    ph2.update(z, tgt, torch.zeros(6, dtype=torch.bool))     # empty mask -> no-op
    assert torch.allclose(before, ph2.proto), 'update modified prototypes with an empty mask'
    ph2.update(z, tgt, torch.ones(6, dtype=torch.bool))
    assert ph2.filled.all(), 'unseen classes were not filled'

    # EMA moves the prototype towards the incoming class mean, not onto it
    ph3 = train.ProtoHead(D, C, momentum=0.9)
    ph3.init_from_means(means, torch.ones(C, dtype=torch.bool))
    old = ph3.proto[1].clone()
    z1 = torch.randn(4, D)
    ph3.update(z1, torch.full((4,), 1), torch.ones(4, dtype=torch.bool))
    new = ph3.proto[1]
    # the class mean as ProtoHead.update computes it: normalise each feature
    # FIRST, then average, then re-normalise.  F.normalize(z1.mean(0)) is a
    # different direction, and comparing the EMA against it made this assertion
    # a coin flip on a threshold (0.999) it was never measuring.
    target = F.normalize(F.normalize(z1, dim=-1).mean(0), dim=-1)
    assert not torch.allclose(new, old), 'prototype did not move'
    assert not torch.allclose(new, target, atol=1e-6), 'prototype jumped instead of EMA'
    assert float(F.cosine_similarity(new, 0.9 * old + 0.1 * target, dim=0)) > 0.999
    print('  prototype head ok')


def check_lora():
    # competition rule 五.1: only CLIP ViT-B/32 may be built.  A larger backbone
    # would train fine and be disqualified, so it has to be refused up front.
    train.check_backbone('ViT-B-32-quickgelu')
    for bad in ('ViT-L-14', 'ViT-B-16', 'RN50', 'ViT-H-14'):
        try:
            train.check_backbone(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f'{bad} was accepted as a backbone')

    torch.manual_seed(0)
    clip = _StubCLIP()
    net = train.Net(clip, 4, lora_rank=4, lora_target='all')
    assert net.n_lora == 4, f'the stub has 4 Linear layers, wrapped {net.n_lora}'

    # the MHA's out_proj must be adapted too, and the merged ``weight`` must
    # agree with ``forward`` -- otherwise attention silently bypasses LoRA
    mhas = [m for m in net.clip.visual.modules() if isinstance(m, nn.MultiheadAttention)]
    assert mhas, 'stub lost its MultiheadAttention: this regression is no longer covered'
    assert all(isinstance(m.out_proj, train.LoRALinear) for m in mhas), 'out_proj was not adapted'

    x = torch.randn(3, 12)
    with torch.no_grad():
        frozen = F.normalize(clip.visual(x), dim=-1)
        anchored = net.anchor_feat(x)
    assert torch.allclose(frozen, anchored, atol=1e-6), 'anchor_feat is not the frozen CLIP feature'

    # B is zero-initialised, so LoRA starts as an exact no-op
    assert torch.allclose(net(x), net.head(frozen), atol=1e-6), 'LoRA is not identity at init'

    # ... and once it is not, disabling it must recover the frozen features exactly
    with torch.no_grad():
        for m in net.clip.visual.modules():
            if isinstance(m, train.LoRALinear):
                m.A.normal_()
                m.B.normal_()
        with_lora = F.normalize(net.clip.encode_image(x), dim=-1)
        assert not torch.allclose(with_lora, frozen, atol=1e-4), 'LoRA has no effect'
        assert torch.allclose(net.anchor_feat(x), frozen, atol=1e-6), \
            'lora_disabled does not restore the frozen tower'

        # merged weight vs. the module call, with the LoRA dropout switched off
        op = mhas[0].out_proj
        op.eval()
        xs = torch.randn(5, op.in_features)
        assert torch.allclose(op(xs), F.linear(xs, op.weight, op.bias), atol=1e-6), \
            'out_proj.weight does not reproduce forward(): nn.MultiheadAttention would skip LoRA'
        assert op.weight.shape == (op.out_features, op.in_features)
        op.train()
        # ... and disabling must expose the untouched base weight
        op.enabled = False
        assert torch.equal(op.weight, op.base.weight), 'disabled LoRALinear leaks its adapter'
        op.enabled = True

    sd = net.trainable_state_dict()
    assert not any(k.endswith('.base.weight') for k in sd), 'frozen weights leaked into the checkpoint'
    assert any(k.endswith('.A') for k in sd) and any('head.weight' in k for k in sd)
    assert 'proto.proto' in sd and 'proto.filled' in sd, 'prototype buffers are not checkpointed'
    assert len(sd) < len(net.state_dict()), 'nothing was filtered out'

    # a fresh model (as infer.py builds it) must take the checkpoint unchanged
    fresh = train.Net(_StubCLIP(), 4, lora_rank=4, lora_target='all')
    fresh.load_state_dict(sd, strict=False)
    for k, v in sd.items():
        assert torch.equal(fresh.state_dict()[k], v), f'{k} did not survive the round trip'
    print(f'  lora/checkpoint ok ({len(sd)}/{len(net.state_dict())} tensors kept)')


def check_judge():
    """The frozen-CLIP veto, and the warm-up reset, must break self-confirmation."""
    n, C = 6, 5
    y = torch.arange(n) % C
    veto = torch.tensor([False, False, True, False, True, False])

    def tracker(**kw):
        kw = {'momentum': 0.0, 'max_noise_frac': 1.0, **kw}
        return noise.LabelTrustTracker(y.tolist(), C, **kw)

    # baseline: the teacher agreeing with every given label keeps full weight --
    # and that is exactly the branch a memorised wrong label rides
    tk = tracker()
    p = torch.full((n, C), 0.01)
    p[torch.arange(n), y] = 0.95
    tk.update(torch.arange(n), p)
    st = tk.refresh()
    assert st['clean'] == n and (tk.weight == 1).all(), st

    # the veto demotes, and nothing else.  The teacher's pick is still `y`, so
    # the label and the target must be untouched -- only the weight may move.
    tk.set_judge(veto)
    st = tk.refresh()
    assert st['vetoed'] == 2 and st['clean'] == n - 2, st
    # A FrozenJudge veto is a demotion only.  It must not enter the relabel
    # branch (even when the student agrees with the folder label): the given
    # label is retained and the sample receives the ordinary noisy weight.
    assert int(st['relabel']) == 0 and int(st['noisy']) == 2, st
    assert (tk.label == y).all(), 'the veto relabelled a sample; it must only demote'
    tg = tk.target(torch.arange(n), y)
    onehot = F.one_hot(y, C).float()
    assert torch.allclose(tg[[0, 1, 3, 5]], onehot[[0, 1, 3, 5]]), 'the veto changed a target'
    assert torch.allclose(tg[[2, 4]], onehot[[2, 4]]), \
        'the veto changed a target: the teacher pick IS the given label, so mixing it in is a no-op'
    assert (tk.weight[[2, 4]] == 0.1).all() and (tk.weight[[0, 1, 3, 5]] == 1).all(), tk.weight

    # Even if the student actively disagrees with a vetoed sample at high
    # confidence, the independent veto still blocks pseudo-relabeling.
    tk_v = tracker()
    pv = p.clone()
    pv[2] = torch.full((C,), 0.01)
    pv[2, (int(y[2]) + 1) % C] = 0.95
    pv[4] = torch.full((C,), 0.01)
    pv[4, (int(y[4]) + 1) % C] = 0.95
    tk_v.update(torch.arange(n), pv)
    tk_v.set_judge(veto)
    sv = tk_v.refresh()
    assert sv['relabel'] == 0 and sv['noisy'] == 2, sv
    assert (tk_v.label == y).all() and (tk_v.weight[[2, 4]] == 0.1).all(), \
        'vetoed disagreements must be demoted without relabelling'

    # an unconfident teacher sends the vetoed sample to the untrusted weight
    tk2 = tracker()
    p2 = torch.full((n, C), 0.01)
    p2[torch.arange(n), y] = 0.3                 # agrees, but below tau_conf
    tk2.update(torch.arange(n), p2)
    tk2.set_judge(veto)
    tk2.refresh()
    assert (tk2.weight[[2, 4]] == 0.1).all() and (tk2.weight[[0, 1, 3, 5]] == 1).all(), tk2.weight

    # without a judge nothing changes (the flag is off by default, so the
    # previous behaviour must be bit-identical)
    tk3 = tracker()
    tk3.set_judge(None)
    tk3.update(torch.arange(n), p)
    st3 = tk3.refresh()
    assert st3['clean'] == n and st3['vetoed'] == 0, st3

    # the veto must survive the checkpoint round trip, or --resume quietly
    # reverts to the self-confirming rule
    sd = tk.state_dict()
    tk4 = tracker()
    tk4.load_state_dict(sd)
    assert torch.equal(tk4.suspect, veto), 'the veto did not survive state_dict'
    old = dict(sd)
    old.pop('suspect')                            # a pre-judge checkpoint
    tk5 = tracker()
    tk5.load_state_dict(old)
    assert tk5.suspect is None, 'an old checkpoint must load as "no judge"'

    # the warm-up reset must actually clear the contamination
    tk6 = tracker()
    tk6.update(torch.arange(n), torch.full((n, C), 1.0 / C))     # garbage warm-up posterior
    assert bool(tk6.seen.all())
    tk6.reset()
    assert not bool(tk6.seen.any()) and float(tk6.prob.abs().max()) == 0.0, 'reset left state behind'
    tk6.update(torch.arange(n), p)                # the first post-warm-up observation
    assert torch.allclose(tk6.prob, p), \
        'the reset posterior is not the fresh observation (it is still being averaged with warm-up noise)'

    # ---- FrozenJudge.  Orthonormal class means make every cosine exact, so the
    #      assertions below test the logic rather than float noise.
    torch.manual_seed(0)
    D = 8
    n2, C2 = 10, 3
    jy = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    means = torch.eye(C2, D)
    feats = F.normalize(means[jy] + 0.05 * torch.randn(n2, D), dim=-1)
    all_present = torch.ones(C2, dtype=torch.bool)

    j = noise.FrozenJudge(n2, D, margin=0.0)
    j.add(torch.arange(n2), feats)
    assert bool(j.seen.all())
    sus, pred, marg = j.judge(means, jy, all_present)
    assert not bool(sus.any()), f'a sample built from its own class mean was called suspect: {marg}'
    assert (pred == jy).all(), 'the judge did not recover the class it was built from'
    assert (marg < 0).all(), marg

    # one sample's features come from another class -> exactly that one is suspect
    j.feat[3] = means[2].clone()                    # sample 3's given label is 0
    sus, pred, marg = j.judge(means, jy, all_present)
    assert sus.tolist() == [False, False, False, True, False, False, False, False, False, False], \
        f'suspect flags: {sus.tolist()}'
    assert int(pred[3]) == 2, f'the judge should pick class 2 for sample 3, got {int(pred[3])}'
    assert abs(float(marg[3]) - 1.0) < 1e-6, f'margin should be exactly 1 here, got {float(marg[3])}'

    # a sample never drawn during warm-up must not be judged from its zero vector
    j2 = noise.FrozenJudge(n2, D, margin=0.0)
    j2.add(torch.arange(3), feats[:3])
    sus, _, _ = j2.judge(means, jy, all_present)
    assert not bool(sus.any()), 'unseen samples were judged from a zero feature'

    # the margin is a real threshold: an exact tie must not be vetoed
    j3 = noise.FrozenJudge(n2, D, margin=0.05)
    j3.add(torch.arange(n2), feats)
    j3.feat[3] = F.normalize(means[0] + means[2], dim=-1)    # equidistant from both
    sus, _, marg = j3.judge(means, jy, all_present)
    assert abs(float(marg[3])) < 1e-6, f'this fixture is not a tie: {float(marg[3])}'
    assert not bool(sus.any()), 'a tie was vetoed despite a nonzero margin'

    # a class absent from warm-up has a ZERO centroid, whose cosine against
    # anything is 0 -- i.e. it would outrank a genuinely anti-correlated real
    # class.  It must be excluded outright, both as a competitor and as a
    # sample's own class.
    j4 = noise.FrozenJudge(n2, D, margin=0.0)
    j4.add(torch.arange(n2), feats)
    absent = torch.tensor([True, True, False])
    sus, pred, _ = j4.judge(means, jy, absent)
    assert not bool((pred == 2).any()), 'the judge returned a class with no warm-up samples'
    assert not bool(sus.any()), 'a sample whose class was absent from warm-up was judged'
    print('  frozen-CLIP judge ok')


def check_optional_gates():
    """Regression checks for the newly optional trust gates.

    These checks stay independent of a dataset or a training run: they pin the
    JSD gate's monotonicity/identity properties, the class-conditional threshold
    switch, and the batch weight normalisation formula used in ``train.py``.
    """
    # JSD is zero for identical views, and the multiplicative gate can only
    # reduce a sample weight.  This is the exact expression used by train.py.
    p = torch.tensor([[0.8, 0.2], [0.5, 0.5]], dtype=torch.float32)
    q = torch.tensor([[0.8, 0.2], [0.99, 0.01]], dtype=torch.float32)
    m = 0.5 * (p + q)
    jsd = 0.5 * ((p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(1) +
                 (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(1))
    assert abs(float(jsd[0])) < 1e-7, jsd
    gate = torch.exp(-2.0 * jsd).clamp_min(0.25)
    assert torch.all(gate <= 1.0 + 1e-7) and float(gate[1]) < 1.0, gate
    assert torch.allclose(gate[:1], torch.ones(1), atol=1e-6), gate

    # ``class_tau_delta=0`` is the historical path.  A positive delta is an
    # exploratory calibration and should be observable on deliberately unequal
    # class confidence medians.
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    probs = torch.tensor([
        [0.19, 0.81],   # class 0 disagreement, conf .81
        [0.60, 0.40],   # class 0 agreement, median conf .60
        [0.60, 0.40],   # class 0 agreement
        [0.81, 0.19],   # class 1 disagreement, conf .81
        [0.01, 0.99],   # class 1 agreement, high median
        [0.01, 0.99],   # class 1 agreement
    ])
    base = noise.LabelTrustTracker(y.tolist(), 2, momentum=0.0,
                                   tau_conf=0.8, max_noise_frac=1.0,
                                   class_tau_delta=0.0)
    cal = noise.LabelTrustTracker(y.tolist(), 2, momentum=0.0,
                                  tau_conf=0.8, max_noise_frac=1.0,
                                  class_tau_delta=0.2)
    idx = torch.arange(len(y))
    base.update(idx, probs); cal.update(idx, probs)
    sb, sc = base.refresh(), cal.refresh()
    # The delta=0 path must remain unchanged; the unequal medians make the
    # calibrated thresholds cross the .81/.82 disagreements in opposite ways.
    assert sb['relabel'] == 2 and sb['noisy'] == 0, sb
    assert sc['relabel'] != sb['relabel'] or sc['noisy'] != sb['noisy'], (sb, sc)

    # Batch normalisation keeps ordinary batches unchanged and never turns an
    # all-noisy batch into full-weight supervision.  This mirrors the guarded
    # formula in train.py (default gain cap 2x).
    def norm(w, gain=2.0):
        denom = w.mean().clamp_min(1.0 / max(gain, 1.0))
        return (w / denom).clamp_max(max(gain, 1.0))
    one = norm(torch.ones(8))
    assert torch.allclose(one, torch.ones_like(one)), one
    noisy = norm(torch.full((8,), 0.1))
    assert torch.all(noisy < 1.0) and torch.all(noisy >= 0.1), noisy
    mixed = norm(torch.tensor([1., 1., 1., 1., 0.1, 0.1, 0.1, 0.1]))
    assert float(mixed[:4].max()) <= 2.0 and float(mixed[4:].max()) < 1.0, mixed
    print('  optional JSD/class-threshold/weight gates ok')


def check_param_groups():
    """The learnable temperature must not sit in a weight-decayed group.

    ``--tau-conf`` is an absolute probability threshold, so decaying
    ``head.logit_scale`` (as the single-group AdamW used to) lowers every
    posterior and moves the clean/noisy boundary without anything reporting it.
    """
    net = train.Net(_StubCLIP(), 4, lora_rank=4, lora_target='all')
    groups = train.param_groups(net, 0.05)
    assert len(groups) == 2, groups
    decayed = {n for n, p in net.named_parameters() if p.requires_grad}
    got_decay, got_nodecay = set(), set()
    for g in groups:
        ptrs = {id(p) for p in g['params']}
        for n, p in net.named_parameters():
            if id(p) in ptrs:
                (got_decay if g['weight_decay'] else got_nodecay).add(n)
    assert got_nodecay == {'head.logit_scale'}, \
        f'the no-decay group should hold exactly the temperature: {sorted(got_nodecay)}'
    assert got_decay | got_nodecay == decayed, \
        'some trainable parameter landed in neither group'
    assert groups[0]['weight_decay'] == 0.05 and groups[1]['weight_decay'] == 0.0, groups
    # the visual tower's CLIP temperature is frozen, so it must not be trainable
    assert not net.clip.logit_scale.requires_grad, \
        'CLIP logit_scale is trainable -- it would be trained with no signal for it'

    # and the real optimiser must actually be built that way (the run, not just
    # the helper): a wrong grouping here is invisible in the logs
    opt = torch.optim.AdamW(train.param_groups(net, 0.05), lr=1e-4)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[1]['weight_decay'] == 0.0
    print('  param groups ok (temperature not decayed)')


def check_lr_warmup():
    """D6: the schedule must ramp up before it anneals, and reach the peak.

    ``CosineAnnealingLR`` hands out ``--lr`` on the very first epoch, while
    ``head.weight`` is still random -- the worst-conditioned moment of the run
    gets the largest step size.  The assertions below are deliberately loose
    about *where* the peak lands (``SequentialLR``'s hand-over semantics differ
    between torch versions, and pinning that would make this test fail for a
    reason that is not a bug) but strict about the shape: start low, rise to no
    more than the peak, then fall, and actually anneal.
    """
    net = nn.Linear(4, 3)

    def lrs(a):
        opt = torch.optim.AdamW([{'params': list(net.parameters()), 'weight_decay': 0.0}],
                                lr=a.lr)
        s = train.make_scheduler(opt, a)
        out = []
        for _ in range(a.epochs):
            out.append(opt.param_groups[0]['lr'])
            opt.step()                       # keeps torch from warning about step order
            s.step()
        return out

    # ---- the shape, as pure arithmetic ---------------------------------- #
    # warm == 0 must be torch's CosineAnnealingLR closed form, exactly; that is
    # what makes an old run reproducible with the new code
    TOT = 20
    for step in range(TOT):
        want = 0.5 * (1.0 + math.cos(math.pi * step / TOT))
        got = train.lr_factor(step, TOT, 0)
        assert abs(got - want) < 1e-12, \
            f'lr_factor({step}, {TOT}, 0) = {got}, CosineAnnealingLR says {want}'

    WARM = 2
    f = [train.lr_factor(s, TOT, WARM) for s in range(TOT)]
    assert abs(f[0] - train.LR_WARMUP_START) < 1e-12, f'the warm-up must start low: {f[0]}'
    assert f[0] < f[1] < f[2], f'the warm-up must rise: {f[:4]}'
    assert abs(f[WARM] - 1.0) < 1e-12, \
        f'the warm-up must reach the peak at step {WARM}, not {f[WARM]}'
    assert max(f) == 1.0, f'the schedule must never exceed the peak: {max(f)}'
    assert all(x >= y - 1e-12 for x, y in zip(f[WARM:], f[WARM + 1:])), \
        f'it must decay after the peak: {f}'
    assert f[-1] < 0.01, f'the cosine must actually anneal to ~0: {f[-1]}'
    # a warm-up longer than the run must not divide by zero or invert
    f = [train.lr_factor(s, 3, 3) for s in range(3)]
    assert f == sorted(f), f'a degenerate warm-up must stay non-decreasing: {f}'
    # ``last_epoch`` is -1 until the first ``step()``; the warm-up branch
    # extrapolates linearly, so an unclamped -1 is a NEGATIVE learning rate.
    # It cannot happen on the normal path (``_initial_step`` bumps to 0 first),
    # which is exactly why it needs an assertion rather than a comment.
    assert train.lr_factor(-1, TOT, WARM) == train.LR_WARMUP_START, \
        f'lr_factor(-1, ...) must clamp to the warm-up floor, got {train.lr_factor(-1, TOT, WARM)}'
    assert train.lr_factor(-1, TOT, 0) == 1.0, train.lr_factor(-1, TOT, 0)

    # ---- and the wiring: the optimiser must actually see those factors --- #
    off = train_args(['--epochs', str(TOT), '--lr-warmup-epochs', '0'])
    v = lrs(off)
    assert abs(v[0] - off.lr) < 1e-12, \
        f'--lr-warmup-epochs 0 must start at the peak: {v[0]} vs {off.lr}'
    assert all(x >= y - 1e-12 for x, y in zip(v, v[1:])), \
        f'--lr-warmup-epochs 0 must be monotonically decreasing: {v}'

    a = train_args(['--epochs', str(TOT), '--lr-warmup-epochs', str(WARM)])
    v = lrs(a)
    want = [a.lr * train.lr_factor(s, TOT, WARM) for s in range(TOT)]
    for i, (got, w) in enumerate(zip(v, want)):
        assert abs(got - w) < a.lr * 1e-9, \
            f'epoch {i}: the optimiser has lr {got}, the formula says {w}'
    assert abs(v[0] - a.lr * train.LR_WARMUP_START) < 1e-12, \
        f'warm-up must start at {a.lr * train.LR_WARMUP_START}, got {v[0]}'
    assert abs(v[WARM] - a.lr) < 1e-9, f'the peak must land on epoch {WARM}: {v[:4]}'
    assert v[-1] < a.lr * 0.01, f'the run must end annealed: {v[-1]}'

    # the schedule has to survive a --resume round trip, or a resumed run
    # silently restarts its warm-up
    opt2 = torch.optim.AdamW([{'params': list(net.parameters()), 'weight_decay': 0.0}], lr=1.0)
    s2 = train.make_scheduler(opt2, a)
    for _ in range(3):
        s2.step()
    sd = s2.state_dict()
    opt3 = torch.optim.AdamW([{'params': list(net.parameters()), 'weight_decay': 0.0}], lr=1.0)
    s3 = train.make_scheduler(opt3, a)
    train.load_sched(s3, sd)
    assert s3.last_epoch == s2.last_epoch, (s3.last_epoch, s2.last_epoch)
    lr2 = opt2.param_groups[0]['lr']
    s2.step(), s3.step()
    assert abs(opt3.param_groups[0]['lr'] - opt2.param_groups[0]['lr']) < 1e-12, \
        'the resumed schedule diverged from the original one epoch later'
    print(f'  lr warm-up ok (ramps to --lr by epoch {WARM}, then anneals; resume-safe)')


def check_image_size():
    """``--image-size`` must reach the tower, and the eval transform must follow.

    The failure this guards against is silent: if the resample were dropped,
    every script would still run and still print the requested size in its log,
    while actually forwarding 224-sized pixels -- or, worse, reach the forward
    pass with a grid that does not match the patch count and die there, after
    the GPU hours were already spent.
    """
    _STUB_CALLS.clear()
    train.build_clip('ViT-B-32-quickgelu', 'openai', 224)
    train.build_clip('ViT-B-32-quickgelu', 'openai', 320)
    # `force_image_size` must reach open_clip *never* -- not for 224, and not for
    # a size where it would look reasonable.  It is the wrong mechanism, not a
    # redundant one: it resamples the grid with antialias=True, whereas the 288
    # run that scored 69.218 resamples without it, and the two are the same shape
    # with different numbers.  The grid has to be resampled here, by our own
    # function, whose defaults are pinned to the scored configuration.
    for call in _STUB_CALLS:
        assert call[2] == {}, \
            f'build_clip delegated the resample to open_clip: {call[:3]}'
    # ...and it must have actually happened: 7x7 -> 10x10 is 101 tokens.  A tower
    # built at 320 without this keeps its 50-token grid, which for the real ViT
    # is not a smaller model but an immediate shape error on the forward pass.
    assert tuple(_STUB_CALLS[0][3].visual.positional_embedding.shape) == (50, DIM), \
        f'224 must keep the grid it was trained with: {_STUB_CALLS[0][3].visual.positional_embedding.shape}'
    assert tuple(_STUB_CALLS[1][3].visual.positional_embedding.shape) == (101, DIM), \
        ('the 320 tower was built without resampling its grid -- the run would '
         'either crash on the first forward pass or, at a size that happens to '
         'keep the token count, quietly evaluate a different model')

    # ViT-B/32 has 32px patches, so only multiples of 32 land on a whole grid.
    # 336 is the figure people quote from ViT-L/14@336 (336/14 = 24) and it is
    # NOT clean here: the convolution produces a 10x10 grid, i.e. 320 plus a
    # 16px margin no patch ever sees.
    for size, want_n, want_clean in ((224, 7, True), (288, 9, True), (320, 10, True),
                                     (352, 11, True), (384, 12, True), (336, 10, False)):
        ps, n, clean = train.patch_grid(_StubCLIP(), size)
        assert ps == 32, f'the stub patch size came out as {ps}'
        assert (n, clean) == (want_n, want_clean), \
            f'size {size}: grid {n} clean={clean}, want {want_n} clean={want_clean}'

    # the eval transform: resize the short side to size * 256/224, centre-crop size
    for size, short in ((224, 256), (288, 329), (320, 366), (336, 384)):
        tf = train.eval_transform(size)
        rs = [o for o in tf.transforms if isinstance(o, transforms.Resize)][0]
        cc = [o for o in tf.transforms if isinstance(o, transforms.CenterCrop)][0]
        assert rs.size == short, f'resize for {size}: {rs.size}, want {short}'
        assert cc.size == (size, size), f'crop for {size}: {cc.size}'
        assert not [o for o in tf.transforms
                    if isinstance(o, transforms.RandomHorizontalFlip)], \
            'the default (non-TTA) transform must not flip'
    flips = [o for o in train.eval_transform(336, True).transforms
             if isinstance(o, transforms.RandomHorizontalFlip)]
    assert len(flips) == 1 and flips[0].p == 1.0, 'flip=True must flip deterministically'

    # ---- crop policies ------------------------------------------------------ #
    # Every policy must give a size x size tensor, and they must genuinely
    # differ: if one silently fell back to centre crop, the crop measurement in
    # probe.py --tta would compare a view with itself and report "no effect".
    for crop in train.CROP_POLICIES:
        tf = train.eval_transform(224, crop=crop)
        for wh in ((400, 200), (200, 400), (224, 224)):
            t = tf(Image.new('RGB', wh, (10, 120, 200)))
            assert tuple(t.shape) == (3, 224, 224), f'{crop} on {wh}: {tuple(t.shape)}'

    # A 1:2 portrait with a stripe across the top 5% tells the policies apart,
    # and an all-black image of the same shape gives the "stripe is gone"
    # reference.  This is not a formality: the whole reason the crop axis is
    # worth measuring is that `center` throws away the top and bottom of a
    # portrait, and if it does not, the measurement is meaningless.
    black = Image.new('RGB', (200, 400), (0, 0, 0))
    strip = black.copy()
    strip.paste((255, 255, 255), (0, 0, 200, 20))

    def top_row(im, crop):
        return train.eval_transform(224, crop=crop)(im)[:, 0]

    assert torch.allclose(top_row(strip, 'center'), top_row(black, 'center')), \
        'the centre crop kept the top of a 1:2 portrait -- it should have cut it off'
    for crop in ('full', 'pad'):
        assert not torch.allclose(top_row(strip, crop), top_row(black, crop)), \
            f'crop={crop} dropped the top of the frame -- it should keep everything'
    outs = {c: train.eval_transform(224, crop=c)(strip) for c in train.CROP_POLICIES}
    assert not torch.allclose(outs['center'], outs['full']), \
        'crop=full produced the same tensor as crop=center'
    assert not torch.allclose(outs['center'], outs['pad']), \
        'crop=pad produced the same tensor as crop=center'
    # pad keeps the aspect ratio, so a 1:2 portrait must leave the left and
    # right edges at the padding colour.  The fill is an 8-bit pixel, so compare
    # against that pixel pushed through the same normalisation -- comparing
    # against the raw mean would be off by the rounding of round(255 * mean).
    mean = torch.tensor(train.CLIP_MEAN)
    std = torch.tensor(train.CLIP_STD)
    fill = ((255 * mean).round() / 255 - mean) / std
    assert torch.allclose(outs['pad'][:, 112, 0], fill, atol=1e-6), \
        f'crop=pad did not pad the short side: {outs["pad"][:, 112, 0]} vs {fill}'
    # ... and the pasted image must be centred, so the same row in the middle of
    # the frame must NOT be the fill colour
    assert not torch.allclose(outs['pad'][:, 112, 112], fill, atol=1e-6), \
        'crop=pad left the centre of the frame as padding'
    # a misspelled policy must raise rather than quietly mean something else
    try:
        train.eval_transform(224, crop='centre')
    except ValueError:
        pass
    else:
        raise AssertionError('a misspelled crop policy must raise, not fall back')
    # the size must be recorded in the checkpoint, or infer/valmetrics read back 224
    assert 'image_size' in vars(train_args()), \
        '--image-size is missing from the args, so it will not be checkpointed'
    print('  image-size ok')


def check_pos_embed():
    """The positional grid is rebuilt on every load, so rebuild it verifiably.

    None of this is visible in a checkpoint: the grid is frozen,
    ``trainable_state_dict`` keeps only trainable parameters, so it is dropped on
    save and reconstructed on every load.  That makes the *rebuild*, not the
    weights, the thing that decides which model a checkpoint is being evaluated
    as -- and it makes a wrong rebuild silent, because the two interpolation
    settings in play produce the same shape and different numbers.

    The sibling repository's 288 run scored 69.218 through one of those settings.
    Evaluating its weights through the other would be scoring a model neither
    repository trained, and the CSV would look perfectly ordinary.
    """
    v = _StubVisual()
    before = v.positional_embedding.detach().clone()

    # 224 is the grid the pretrained weights already carry: a no-op, bit for bit
    assert train.resize_positional_embedding(v, 224, verbose=False) is None, \
        '224 must not resample -- it is the size OpenAI trained the weights at'
    assert torch.equal(v.positional_embedding, before), '224 changed the grid'

    # 288 -> 9x9 patches -> 81 + 1 tokens, CLS carried over untouched
    assert train.resize_positional_embedding(v, 288, verbose=False) == 9, \
        '288 must install a 9x9 grid'
    assert tuple(v.positional_embedding.shape) == (82, DIM), \
        v.positional_embedding.shape
    assert torch.allclose(v.positional_embedding[0], before[0]), \
        'the CLS token must be carried over rather than resampled with the grid'
    assert tuple(v.grid_size) == (9, 9) and tuple(v.image_size) == (288, 288), \
        f"the tower's bookkeeping disagrees with the grid it holds: {v.grid_size}"
    assert not v.positional_embedding.requires_grad, \
        ('the grid must stay frozen, or it would enter the checkpoints and every '
         'existing one would start meaning something different')

    # 336 is the trap, and it is the number the brief itself names
    for bad, why in ((336, "the brief's own figure, which belongs to patch-14 ViT-L/14"),
                     (300, 'not a multiple of 32'),
                     (225, 'one pixel off a legal size')):
        try:
            train.resize_positional_embedding(_StubVisual(), bad, verbose=False)
        except SystemExit as e:
            assert str(bad) in str(e), f'the refusal for {bad} should name it: {e}'
        else:
            raise AssertionError(
                f'{bad}px must be refused ({why}): the patch convolution and the '
                f'positional grid would disagree about the token count')

    # ---- does `antialias` actually matter at these sizes? ------------------- #
    # This is a question about PyTorch, not about this codebase, and it is the
    # *whole premise* of aligning the mechanism: open_clip resamples the grid
    # with antialias=True, the 69.218 run resampled without it.  If the two agree
    # at 7x7 -> 9x9 then that divergence was imaginary and the alignment above is
    # a refactor rather than a fix.  Asserting a difference would turn an open
    # question into a spurious failure alarm, so this measures it and prints
    # which way it came out.  Either answer is worth having.
    soft, sharp = _StubVisual(), _StubVisual()
    train.resize_positional_embedding(soft, 288, antialias=False, verbose=False)
    train.resize_positional_embedding(sharp, 288, antialias=True, verbose=False)
    assert soft.positional_embedding.shape == sharp.positional_embedding.shape, \
        'the two settings must agree on shape -- that is what makes the trap silent'
    # ...and the default must be the setting the scored run used
    dflt = _StubVisual()
    train.resize_positional_embedding(dflt, 288, verbose=False)
    assert torch.equal(dflt.positional_embedding, soft.positional_embedding), \
        'the default must be antialias=False, the setting the 69.218 run used'
    d = float((soft.positional_embedding - sharp.positional_embedding).abs().max())
    print(f'  antialias True vs False at 288: max |d| = {d:.3e} -- '
          + ('a real difference, so the mechanism alignment is load-bearing'
             if d else 'IDENTICAL at this size, so that divergence was imaginary'))

    # ---- the fingerprint, the only guard on a silent rebuild --------------- #
    fp = train.pos_embed_fingerprint(soft)
    assert fp is not None and len(fp['sha']) == 16 and fp['shape'] == [82, DIM], fp
    assert fp == train.pos_embed_fingerprint(soft), 'the fingerprint is not deterministic'

    ck = {'model': {}, 'pos_embed': fp, 'args': {'image_size': 288}}
    again = _StubVisual()
    train.resize_positional_embedding(again, 288, verbose=False)
    assert train.verify_pos_embed(ck, again) is True, \
        'a rebuild of the same grid must verify against its own fingerprint'

    # Right grid, wrong resolution: the naive failure
    wrong = _StubVisual()
    train.resize_positional_embedding(wrong, 320, verbose=False)
    nudge = _StubVisual()
    train.resize_positional_embedding(nudge, 288, verbose=False)
    with torch.no_grad():
        nudge.positional_embedding[7, 0] += 1e-3
    for name, bad_tower in (('the wrong resolution', wrong), ('nudged values', nudge)):
        try:
            train.verify_pos_embed(ck, bad_tower)
        except SystemExit:
            pass
        else:
            raise AssertionError(
                f'a grid rebuilt with {name} passed verification. "Same shape with '
                f'different numbers" is the exact silent failure this check exists '
                f'to catch, so a check that only compares shapes is not a check.')
    assert train.pos_embed_fingerprint(nudge)['shape'] == fp['shape'], \
        'the nudged grid changed shape, so the case above proved nothing about values'

    assert train.verify_pos_embed({}, again) is None, \
        ('a checkpoint with no fingerprint must report "cannot tell" rather than '
         "pass -- every checkpoint written before this check existed, including the "
         "sibling repository's 69.218 run, is in that state")

    # ---- and build_clip has to use all of it ------------------------------- #
    _STUB_CALLS.clear()
    ck288 = {'args': {'image_size': 288}, 'pos_embed': fp}
    train.build_clip('ViT-B-32-quickgelu', 'openai', 288, ck=ck288)
    assert tuple(_STUB_CALLS[-1][3].visual.positional_embedding.shape) == (82, DIM), \
        'build_clip at 288 did not resample the grid'
    # a *deliberately* different size must not raise: TTA at a second resolution is
    # exactly that, it is permitted, and the mismatch is expected there
    train.build_clip('ViT-B-32-quickgelu', 'openai', 320, ck=ck288)
    assert tuple(_STUB_CALLS[-1][3].visual.positional_embedding.shape) == (101, DIM), \
        'build_clip at a deliberate second size did not resample the grid'
    print('  pos-embed ok (224 is a no-op, 288 -> 9x9, 336 refused, fingerprint enforced)')


def check_ck_image_size():
    """A checkpoint's resolution must be read back from either spelling.

    The sibling repository writes ``args['img_size']``; this one writes
    ``args['image_size']``, and checkpoints travel between the two.  The failure
    when the read misses is the bad kind: a 288-trained checkpoint gets served
    at 224, which runs fine, raises nothing and simply scores worse -- so both
    the alias *and* the note it prints are tested here.  A silent alias would be
    a silent acceptance of a resolution that may be wrong.
    """
    def read(ck, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            size, key = train.ck_image_size(ck, **kw)
        return size, key, buf.getvalue()

    # the native spelling, at the top level and inside args: no note either way
    for ck, want in (({'image_size': 288}, 288), ({'args': {'image_size': 288}}, 288),
                     ({'image_size': 320, 'args': {'image_size': 288}}, 320)):
        size, key, out = read(ck)
        assert (size, key) == (want, 'image_size'), (ck, size, key)
        assert out == '', f'the native spelling must not print a note: {out!r}'

    # the alias: same kind of value, but it must say which spelling it followed
    for ck, want in (({'img_size': 288}, 288), ({'args': {'img_size': 320}}, 320)):
        size, key, out = read(ck)
        assert (size, key) == (want, 'img_size'), (ck, size, key)
        assert 'img_size' in out and str(want) in out, \
            f'following the alias must be announced, got {out!r}'

    # nothing to read -> the default, and no note (there is no alias to blame)
    for ck in ({}, {'args': {}}, {'image_size': 0}, {'args': {'img_size': None}}):
        size, key, out = read(ck)
        assert (size, key) == (224, None), (ck, size, key)
        assert out == '', f'the default must not print a note: {out!r}'
    assert train.ck_image_size({}, default=320, verbose=False) == (320, None)

    # a checkpoint carrying BOTH spellings resolves the way this codebase wrote
    # it.  Not hypothetical: it is what a merged tree or a re-saved checkpoint
    # produces, and letting the foreign key win would silently override the
    # native one -- the exact inversion of what the ordering is for.
    for ck in ({'image_size': 224, 'img_size': 288},
               {'args': {'image_size': 224, 'img_size': 288}},
               {'image_size': 224, 'args': {'img_size': 288}}):
        size, key, out = read(ck)
        assert (size, key) == (224, 'image_size'), (ck, size, key, out)

    # ---- the CLI alias ------------------------------------------------------ #
    # Every script that can be told a resolution accepts both spellings, and the
    # native one still works.  The required arguments differ per script, hence
    # the fixture argv rather than a bare [] -- `train_args` above is the same
    # thing for the two checks that call train.parse_args directly.
    clis = ((train, ['--data', '.']),
            (infer, ['--test', '.', '--checkpoint', 'x.pt']),
            (analyze, ['--data', '.', '--checkpoint', 'x.pt']))
    for mod, base in clis:
        for flag, want in (('--image-size', 288), ('--img-size', 288),
                           ('--image-size', 224), ('--img-size', 320)):
            got = vars(mod.parse_args(base + [flag, str(want)]))
            assert got.get('image_size') == want, \
                f'{mod.__name__} {flag} {want}: image_size came out {got.get("image_size")!r}'

    # ---- the transform -> size reader -------------------------------------- #
    # `center` is the trap: its pipeline names TWO sizes (Resize(256) then
    # CenterCrop(224)) and a naive first-match reader answers 256 for the most
    # common transform in the codebase.  Last match is the output resolution.
    for size, crop in ((224, 'center'), (288, 'center'), (320, 'full'), (288, 'pad')):
        tf = train.eval_transform(size, crop=crop)
        assert train.transform_size(tf) == size, \
            f'transform_size({crop}@{size}) = {train.transform_size(tf)}'
    assert train.transform_size(train.eval_transform(288, True)) == 288, \
        'flip must not change the size reading'
    rrc = transforms.Compose([transforms.RandomResizedCrop(288),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor()])
    assert train.transform_size(rrc) == 288, 'the training transform must read as 288'
    assert train.transform_size(transforms.Compose([transforms.ToTensor()]), default=320) == 320, \
        'an unrecognised pipeline must fall back to the default, not guess'

    # ---- and the reason any of this exists --------------------------------- #
    # The grey substitute for an undecodable file must be the size the model
    # expects.  Cosmetic under every transform built here (uniform grey survives
    # resizing unchanged, which is why this was hard-coded at 224 for so long),
    # but it is a hard-coded resolution in the resolution-dependent path.
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / 'data'
        (root / '0000').mkdir(parents=True)
        # *every* file is a 0-byte one, so whichever split the `max(1, ...)` rule
        # sends each to, the item under test is always the undecodable one.  A
        # single bad file alongside a good one would leave `_load`'s fallback to
        # sort order, and on this split it lands in val -- i.e. the assertion
        # would pass by never running.
        for name in ('a.png', 'b.png', 'c.png'):
            (root / '0000' / name).write_bytes(b'')
        for size in (224, 288):
            for split in ('train', 'val'):
                ds = train.ImageFolderNoisy(root, train.eval_transform(size), split == 'val',
                                            0.5, 3407, split)
                assert ds.img_size == size, f'{size}: dataset stored img_size={ds.img_size}'
                assert len(ds), f'{size}/{split}: the fixture produced no items'
                for i, (p, _) in enumerate(ds.items):
                    t = ds[i][0]
                    assert tuple(t.shape) == (3, size, size), \
                        f'{size}/{split}: {p} -> {tuple(t.shape)}'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('  ck-image-size ok')


def check_calib_gate():
    """The offline proxy's gate must be able to say FAIL.

    A gate that always passes is not a gate, and this one has a specific way of
    degenerating: its checks compare *fractions* (what ``score_probs`` returns)
    against tolerances in **points** (what the leaderboard reports).  Get that
    wrong and 0.0002 <= 0.5 holds for entirely the wrong reason, every row reads
    PASS, and the whole proxy looks calibrated while measuring nothing.  That
    bug was live in this file until this check was written.

    The fixtures are shaped like the real measurement -- the platform moved
    0.0002 points across 16 epochs while ``val_acc`` moved 5.02 -- so the two
    cases below differ only in whether the *metric* behaves like the leaderboard
    or like ``val_acc``.
    """
    def table(ep_delta_vstar, wide_delta, cross=None):
        """Rows keyed ``(run, epoch, viewset)``, in *fractions*.

        The deltas are fractions because that is what ``score_probs`` returns and
        what the gate converts; the numbers were chosen so they land on the
        leaderboard's values once multiplied by 100 -- 0.0002 pt across epochs,
        -2.12 for ``wide``, +2.09 for ``tta4``.

        ``cross``, when given, adds a **second training run** (288px) holding only
        its epoch-20 ``tta4`` row.  That is what the run field is for: both runs
        have an epoch 20, so a two-part key would let one overwrite the other and
        the gate would grade one run twice while reporting four facts.
        """
        rows = {(224, 4, 'plain'): {'val_acc': 0.6568, 'acc_vstar': 0.6400,
                                    'acc_vstar_hard': 0.5100},
                (224, 20, 'plain'): {'val_acc': 0.7070,
                                     'acc_vstar': 0.6400 + ep_delta_vstar,
                                     'acc_vstar_hard': 0.5100 + ep_delta_vstar},
                (224, 20, 'wide'): {'val_acc': 0.6900, 'acc_vstar': 0.6400 + wide_delta,
                                    'acc_vstar_hard': 0.5100 + wide_delta},
                (224, 20, 'tta4'): {'val_acc': 0.7300, 'acc_vstar': 0.6600,
                                    'acc_vstar_hard': 0.5400}}
        if cross is not None:
            v = 0.6600 + cross
            rows[(288, 20, 'tta4')] = {'val_acc': 0.7600, 'acc_vstar': v,
                                       'acc_vstar_hard': v + 0.0100}
        return rows

    # a metric that tracks the leaderboard must pass every fact it can grade
    ok, lines = probe.calib_gate(table(0.0002, -0.0212), 'acc_vstar')
    assert ok, 'the gate failed a table shaped exactly like the leaderboard:\n' + \
        '\n'.join(lines)
    # ... and with only one run it must report the cross fact as NOT MEASURED
    # rather than quietly passing it -- on the line that authorises the next GPU
    # spend, "not measured" and "passed" must not read the same
    assert any('not measured' in ln for ln in lines), \
        'the cross-resolution fact must say so when it could not be graded:\n' + \
        '\n'.join(lines)

    # the cross-resolution fact, in the three shapes that matter
    ok, lines = probe.calib_gate(table(0.0002, -0.0212, cross=0.0297), 'acc_vstar')
    assert ok, 'a 2.97-point cross-resolution gain must PASS:\n' + '\n'.join(lines)
    assert any('CONFOUNDED' in ln for ln in lines), \
        'the cross fact must be labelled CONFOUNDED, or it reads as a clean result:\n' \
        + '\n'.join(lines)
    # below the bar the fact is graded at: a proxy that cannot see 3 points cannot
    # inform a 320 decision, which is the only reason the row is there
    ok, _ = probe.calib_gate(table(0.0002, -0.0212, cross=0.005), 'acc_vstar')
    assert not ok, 'a 0.5-point cross-resolution gain is under the bar and must FAIL'
    # and the sign is the claim: a larger training resolution that scores *worse*
    # is the opposite of the fact this is graded against
    ok, _ = probe.calib_gate(table(0.0002, -0.0212, cross=-0.0297), 'acc_vstar')
    assert not ok, 'a negative cross-resolution effect must FAIL, not pass on |x|'

    # ... and one that tracks val_acc instead must fail.  This is the load-bearing
    # assertion: it is what makes the gate a discriminator rather than a rubber
    # stamp, and it is what the whole exercise is for -- a proxy with no more
    # resolution than val_acc would cost GPU hours and buy nothing.
    ok, lines = probe.calib_gate(table(0.0502, -0.0212), 'acc_vstar')
    assert not ok, 'a metric that moves like val_acc passed the gate:\n' + '\n'.join(lines)

    # the negative control, as a metric in its own right
    ok, lines = probe.calib_gate(table(0.0002, -0.0212), 'val_acc')
    assert not ok, ('val_acc must FAIL the gate -- if it passes, the gate is not '
                    'discriminating:\n' + '\n'.join(lines))

    # an effect below the stated resolution must not be counted as a win
    ok, _ = probe.calib_gate(table(0.0002, -0.001), 'acc_vstar')
    assert not ok, 'a 0.1-point view effect is under the noise floor and must FAIL'

    # a missing row is reported as n/a, never silently read as zero
    bad = table(0.0002, -0.0212)
    del bad[(224, 20, 'tta4')]
    ok, lines = probe.calib_gate(bad, 'acc_vstar')
    assert not ok and any('n/a' in ln for ln in lines), lines

    # one epoch in the graded run cannot support an epoch comparison, and saying
    # FAIL there would blame the proxy for a run that did not measure it
    ok, lines = probe.calib_gate({(224, 20, 'plain'): {'acc_vstar': 0.6},
                                 (288, 20, 'tta4'): {'acc_vstar': 0.7}}, 'acc_vstar')
    assert not ok and any('epoch(s)' in ln for ln in lines), lines

    # ---- score_probs -------------------------------------------------------- #
    # Unequal class counts (3/2/1) on purpose: with 2/2/2 micro and macro are the
    # same number and the test would pass on a macro column that was really micro.
    y = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 0, 1, 2])
    prob = np.zeros((6, 3), dtype=np.float32)
    prob[np.arange(6), pred] = 1.0
    vstar = np.array([True] * 5 + [False])
    margin = np.array([0.5, 0.4, 0.3, 0.2, 0.1, -0.1])
    sc, hard = probe.score_probs(prob, y, vstar, margin)
    assert abs(sc['val_acc'] - 4 / 6) < 1e-9, sc
    assert abs(sc['macro'] - (2 / 3 + 0.5 + 1.0) / 3) < 1e-9, sc
    assert sc['macro'] != sc['val_acc'], 'micro and macro must differ here'
    assert abs(sc['acc_vstar'] - 3 / 5) < 1e-9, sc
    # V*-hard is the low-margin half of V* (median of .5 .4 .3 .2 .1 is .3)
    assert abs(sc['acc_vstar_hard'] - 1 / 3) < 1e-9, sc
    assert hard.tolist() == [False, False, True, True, True, False], hard
    # an empty V* must be nan, not a division by zero or a silent 0.0
    sc2, _ = probe.score_probs(prob, y, np.zeros(6, dtype=bool), margin)
    assert math.isnan(sc2['acc_vstar']), sc2

    # the view sets the gate is defined on must all resolve, and tta4 must be the
    # four views the leaderboard number was measured with
    for name, views in probe.CALIB_VIEWS.items():
        for size, crop, ratio_name, flip in views:
            # a size a view names has to be one the tower can actually be built at,
            # or the set would only fail much later, inside the run
            assert size is None or (int(size) % 32 == 0 and int(size) >= 224), (name, size)
            assert crop in train.CROP_POLICIES, (name, crop)
            assert ratio_name is None or ratio_name in train.VIEW_RATIOS, (name, ratio_name)
            train.eval_transform(int(size or 224), flip, crop=crop,
                                 **({} if ratio_name is None
                                    else {'ratio': train.VIEW_RATIOS[ratio_name]}))
    assert len(probe.CALIB_VIEWS['tta4']) == 4, probe.CALIB_VIEWS['tta4']
    assert probe.CALIB_VIEWS['plain'] == ((None, 'center', 'plain', False),), \
        'plain must be the plain single view at the trained size -- it is what the ' \
        '64.16 was measured on, and what every checkpoint\'s recorded val_acc is ' \
        'checked against'
    assert probe.TRAINING_VIEW == probe.CALIB_VIEWS['plain'], \
        ('the val_acc self-check compares a measured row against the value the run '
         'itself recorded, so the row it recognises has to be the transform training '
         'used -- if these two drift apart the check silently stops running')
    # a view that lost its size field must be refused, not read as "the trained
    # size": that is the natural thing to write and it would turn a size view into
    # a duplicate of the plain one, so a set that measured nothing new would report
    # an average of one view twice
    try:
        probe.check_view_tuple('bad', (('center', 'plain', False),))
    except SystemExit:
        pass
    else:
        raise AssertionError('a 3-tuple view must be refused as a missing size field')
    print('  calib-gate ok')


def check_view_sets():
    """One view table, and the two ways of spelling a set agree.

    `infer.py` and `probe.py` select views from the same table (train.VIEW_SETS)
    now: a second copy is how "what tta4 means" stops being one thing, and the
    proxy's whole job is to predict what the submission path will score.  And a
    set the product flags *can* spell must be spelled identically by both -- that
    is a property of two parsings, so it is far cheaper to assert here than to
    discover from a CSV difference after an hour of GPU time.
    """
    assert probe.CALIB_VIEWS is train.VIEW_SETS, \
        'the proxy and the submission path must read one view table, not two'
    for name in train.VIEW_SETS_SWEPT:
        assert name in train.VIEW_SETS, f'{name} is swept but is not in the table'
    # the sets that make the sweep worth more than its pair enumeration: if one of
    # these silently dropped out, every report would still look complete
    for name in ('tta4', 'axis4', 'mix6', 'crops6', 'tta8', 'tta4s288'):
        assert name in train.VIEW_SETS_SWEPT, f'{name} is no longer swept'
    assert train.TRAINING_VIEW is train.VIEW_SETS['plain'], \
        ('TRAINING_VIEW must be the table\'s own plain entry: the val_acc self-check '
         'recognises a row by comparing against it, so a copy that drifts stops the '
         'check from running at all')

    def plan(*argv, base=224):
        """The view list a command line asks for, as ``main`` would resolve it."""
        return infer.view_plan(
            infer.parse_args(['--test', 'x', '--checkpoint', 'y', *argv]), base)[0]

    def resolved(views, base=224):
        """Sorted, sizes substituted -- so the comparison is order-insensitive."""
        return sorted(train.resolved_views(views, base))

    # (a) a set the product flags can also express: both spellings, one view list.
    # crops6 is `center full pad` x flip, and the comparison is made on the
    # resolved sizes -- the table says `None` (the trained size) where the product
    # has already substituted the base.
    for name, argv in (('crops6', ['--tta-crops', 'center', 'full', 'pad', '--tta-flip']),
                       ('tta8', ['--tta-ratios', 'plain', 'wide', 'mid', 'tight',
                                 '--tta-flip'])):
        assert resolved(plan('--tta-views', name)) == \
            resolved(plan(*argv)) == resolved(train.VIEW_SETS[name]), \
            (f'--tta-views {name} and {" ".join(argv)} no longer describe the same '
             f'views, so one of the two is about to be measured under the other\'s name')
    # (b) the same product *without* the name is not tta4 -- this is why the name
    # exists, and it is the claim infer.py's docstring makes: the product is a
    # 6-view superset, and the four that scored are the four tta4 lists.  The
    # length clause is what makes "the four that scored" a fact about the table
    # rather than about this sentence: a strict-subset test alone still passes if
    # `tta4` quietly loses a view (3 of the 6) or gains a duplicate.
    six = plan('--tta-ratios', 'plain', 'wide', 'tight', '--tta-flip')
    four = resolved(train.VIEW_SETS['tta4'])
    assert len(four) == 4 and len(six) == 6 and set(four) < set(six), \
        (len(four), len(six), set(four) - set(six))
    # (c) a set that promises more views than it delivers at this checkpoint is
    # refused, not scored: `tta4s320` offers five views of which two are the same
    # pixels at 320 (`None` and the literal 320 both mean 320 there), and the row
    # would be reported as a recipe that was never measured
    assert train.is_degenerate(train.VIEW_SETS['tta4s320'], 320)
    assert not train.is_degenerate(train.VIEW_SETS['tta4s320'], 224), \
        ('the 320 view is only the plain view at 320 -- at 224 it is a genuinely '
         'different input, which is the entire reason that entry exists')
    # the 320-trained counterpart: 288 is the nearby size there, so the same set
    # is degenerate on a 288 run and a real five-view average on a 320 one.  The
    # two entries exist because *neither* is usable on both checkpoints.
    assert train.is_degenerate(train.VIEW_SETS['tta4s288'], 288)
    assert not train.is_degenerate(train.VIEW_SETS['tta4s288'], 320)
    # the same collision one resolution down, and in a set the sweep's pair
    # enumeration builds rather than one that is written in the table
    assert train.is_degenerate(train.VIEW_SETS['s288'] + train.VIEW_SETS['plain'], 288)
    assert not train.is_degenerate(train.VIEW_SETS['s288'] + train.VIEW_SETS['plain'], 224)
    # a *single* size view that coincides with the trained size is not degenerate:
    # it declares one view and delivers one, i.e. it is `plain` under another name
    assert not train.is_degenerate(train.VIEW_SETS['s320'], 320)
    assert not train.is_degenerate(train.VIEW_SETS['tta4'], 224), \
        'a flipped view is not a duplicate of its unflipped twin'
    assert train.resolved_views(train.VIEW_SETS['plain'], 288) == \
        ((288, 'center', 'plain', False),)
    for argv, base, why in ((
            ['--tta-views', 'nonsense'], 224, 'an unknown name'),
            (['--tta-views', 'tta4', '--tta-flip'], 224, 'a name plus the product flags'),
            # base is what main derives from the checkpoint (or --image-size), so
            # this is the 320-trained checkpoint's case without needing one
            (['--tta-views', 'tta4s320'], 320, 'a set that collapses at the trained size'),
            (['--tta-ratios', 'nope'], 224, 'an unknown ratio')):
        try:
            plan(*argv, base=base)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'{why} must be refused: {argv}')
    # (d) the default aggregation has to be the one the leaderboard numbers were
    # measured with, or a recipe reproduced from this file is a different recipe
    assert infer.parse_args(['--test', 'x', '--checkpoint', 'y']).tta_agg == 'logit', \
        ('the default --tta-agg must be logit: 64.16 / 66.2456 / 69.218 / 69.968 were '
         'all measured that way, so `feat` by default would silently make every '
         'reproduction of them an unmeasured variant')
    print('  view sets ok')


def check_flat_images():
    """The submission path's image reader: order, grey fallback, size refresh.

    `infer.py` writes one CSV row per image in *this* dataset's order, and a file
    Pillow cannot decode must still produce a row -- the competition warns about
    truncated files, and a short CSV is a failed submission rather than a worse
    score.  The proxy reads the same folder through the same class, so a drift
    here makes every statistic it reports be computed over a different set than
    the one it stands in for.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / 'imgs'
        (root / 'sub').mkdir(parents=True)
        for name, n in (('b.jpg', 40), ('a.jpg', 32)):
            Image.new('RGB', (n, n), (10, 20, 30)).save(root / name)
        # nested on purpose: the folder is searched recursively, and a flat-only
        # glob would drop it from both the CSV and every proxy statistic
        Image.new('RGB', (36, 36), (200, 10, 10)).save(root / 'sub' / 'c.jpg')
        # a .png whose bytes are not a PNG: Pillow refuses it by signature, so
        # this is the unreadable case and not a lucky parse
        (root / 'broken.png').write_bytes(b'this is not a png')

        ds = train.FlatImages(root, train.eval_transform(224), return_bad=True)
        assert ds.paths == sorted(str(p) for p in root.rglob('*')
                                  if p.is_file() and p.suffix.lower() in train.IMG_EXTS), \
            'the path list is no longer the sorted rglob infer.py writes its CSV in'
        assert [Path(p).name for p in ds.paths] == ['a.jpg', 'b.jpg', 'broken.png', 'c.jpg'], \
            [Path(p).name for p in ds.paths]
        got = [ds[i] for i in range(len(ds))]
        assert [i for _, i, _ in got] == list(range(len(ds))), 'index != position'
        assert [bad for _, _, bad in got] == [False, False, True, False], \
            'the unreadable file was not flagged (or a readable one was)'
        assert got[2][0].shape == (3, 224, 224), got[2][0].shape
        # assigning a transform refreshes the grey size: a 288 view must not get a
        # 224-sized grey tile, which is invisible (grey on grey) and wrong
        ds.transform = train.eval_transform(288)
        assert ds.img_size == 288, ds.img_size
        assert ds[2][0].shape == (3, 288, 288), ds[2][0].shape
        # ... and the property is what `set_transform` in probe.py assigns through
        assert train.FlatImages(root, train.eval_transform(224))[2][1] == 2, \
            'return_bad=False must still yield (tensor, index)'

        # Through the loader, at more than one worker count: same order, same
        # tensors.  This is the claim infer.py --workers rests on -- the CSV is
        # the submission, so it may not be a function of how many threads decoded
        # the images.
        ref = None
        for workers in (0, 2):
            ds2 = train.FlatImages(root, train.eval_transform(288))
            tensors, order = [], []
            loader = DataLoader(ds2, batch_size=3, num_workers=workers, shuffle=False,
                                worker_init_fn=infer._worker_init)
            for t, j in loader:
                tensors.extend(t.unbind(0))
                order.extend(int(x) for x in j)
            assert order == list(range(len(ds2))), \
                f'--workers {workers} reordered the images: {order}'
            if ref is None:
                ref = tensors
            else:
                assert len(tensors) == len(ref)
                for k, (x, y) in enumerate(zip(ref, tensors)):
                    assert torch.equal(x, y), (
                        f'--workers {workers} changed image {k}: the submission CSV is '
                        f'a function of --workers, which is not part of the recipe')
        print('  flat-images ok')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_probe_cache_identity():
    """A probe cache file is only reused for the pass it was written for.

    ``feat_val_224.npz`` names the *view* and not the image list, while
    ``--limit``, ``--val-ratio``, ``--seed`` and ``--data`` all change the image
    list without changing the name.  So the 16-image smoke run that
    README_AUTODL.md puts before every measurement wrote a file that the next
    full run read back as if it were the whole val split, and every number after
    it was computed over 16 images with nothing in the report saying so.  The
    paths were already stored next to the features; this is the check that they
    are actually consulted -- both in ``extract`` (frozen features) and in
    ``predict_probs`` (model probabilities), since a fix applied to one of the
    two leaves the other silently wrong.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / 'train'
        random.seed(0)
        for c in range(4):
            d = root / f'{c:04d}'
            d.mkdir(parents=True)
            for i in range(8):
                px = bytes(random.randrange(256) for _ in range(48 * 48 * 3))
                Image.frombytes('RGB', (48, 48), px).save(d / f'img_{c}_{i}.jpg')

        # The dataset ``calib()`` itself scores, not a look-alike.  The shape
        # matters: ``predict_probs`` unpacks ``(x, y, idx)`` and a flat test
        # folder yields ``(x, i)``, so a fixture of the wrong shape dies in the
        # loader before it tests anything -- which is exactly how this check
        # failed on its first real run (2026-09-26, `not enough values to unpack
        # (expected 3, got 2)`).  The cache fix it was written for was already
        # proven by that same log; the check was what was wrong.
        a = train_args(['--data', str(root), '--val-ratio', '0.25', '--seed', '0'])
        _tr, va = probe.build_split_datasets(a)
        n_val = len(va)
        assert n_val >= 3, f'the fixture produced {n_val} val images'

        clip_model = _StubCLIP()
        net = train.Net(clip_model, 4).eval()
        cpu = torch.device('cpu')
        allp = [Path(p).name for p in probe.paths_of(va)]

        # ---- the frozen-feature cache (probe.extract) ------------------------
        small = probe.extract(clip_model, va, 224, cpu, 4, 0, 'val', tmp, limit=2)
        assert small['f'].shape == (2, DIM), small['f'].shape
        assert [Path(p).name for p in small['paths']] == allp[:2], small['paths']

        # the full pass must not read that 2-image file back
        full = probe.extract(clip_model, va, 224, cpu, 4, 0, 'val', tmp)
        assert len(full['paths']) == n_val and full['f'].shape == (n_val, DIM), \
            (f'the 2-image cache was reused for a {n_val}-image pass '
             f'({full["f"].shape}): every number computed from it would be a '
             f'number about 2 images')
        assert [Path(p).name for p in full['paths']] == allp, full['paths']
        # ... while the file that *is* this pass is reused, values and all
        again = probe.extract(clip_model, va, 224, cpu, 4, 0, 'val', tmp)
        for k in ('f', 'y'):
            assert np.array_equal(np.asarray(again[k]), np.asarray(full[k])), k

        # ---- the model-probability cache (probe.predict_probs) ---------------
        cfile = tmp / 'calib_ck_plain_logit.npz'

        def model_for(_size):
            return net

        p2, _y2, pa2 = probe.predict_probs(model_for, va, train.VIEW_SETS['plain'], 4,
                                           224, cpu, 4, 0, cache=cfile, limit=2)
        assert p2.shape == (2, 4) and len(pa2) == 2, (p2.shape, len(pa2))
        p4, y4, pa4 = probe.predict_probs(model_for, va, train.VIEW_SETS['plain'], 4,
                                          224, cpu, 4, 0, cache=cfile)
        assert p4.shape == (n_val, 4) and len(pa4) == n_val, \
            (f'the 2-image probability cache was reused for a {n_val}-image pass '
             f'({p4.shape})')

        # a matching file is reused *without a model*: if this call built one it
        # recomputed, which is what the assertion inside `model_for` means
        def explode(_size):
            raise AssertionError('predict_probs rebuilt a model despite a matching cache')

        p4b, y4b, pa4b = probe.predict_probs(explode, va, train.VIEW_SETS['plain'], 4,
                                             224, cpu, 4, 0, cache=cfile)
        assert np.allclose(p4b, p4, atol=1e-4), 'the reused cache is not what wrote it'
        assert np.array_equal(y4b, y4) and list(pa4b) == list(pa4)
        print('  probe cache identity ok')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_reproducible_augment():
    """The augmentation of item ``i`` must not depend on ``--workers`` (D12).

    Before the fix the transform drew from the *worker's* torch/python RNG, so
    which worker happened to draw a sample decided what that sample looked like,
    and the sample->worker mapping changes with ``--workers``.  Every setting
    still trains and still converges -- just to a different point -- so the
    failure is invisible; it only shows up as "the submitted code does not
    reproduce the submitted score".
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / 'train'
        random.seed(0)
        for c in range(3):
            d = root / f'{c:04d}'
            d.mkdir(parents=True)
            for i in range(6):
                px = bytes(random.randrange(256) for _ in range(48 * 48 * 3))
                Image.frombytes('RGB', (48, 48), px).save(d / f'img_{c}_{i}.jpg')

        tf = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 7),
            transforms.ToTensor()])
        # val_ratio 0.5 on 6 per class -> 3 held out, 3 trained: 9 items
        ds = train.ImageFolderNoisy(root, tf, False, 0.5, 0, 'train', stochastic=True)
        tv = train.TwoView(ds)
        idx = list(range(len(tv)))

        # the property itself: repeated calls give the same tensor
        ref = [tv[i] for i in idx]
        for k, i in enumerate(idx):
            again = tv[i]
            for v in (0, 1):
                assert torch.equal(ref[k][v], again[v]), \
                    f'item {i} view {v} is not a pure function of (seed, index, view)'
            # if the two views were seeded the same, "two-view consistency" in
            # the loss would be comparing a view with itself -- vacuously zero
            assert not torch.equal(ref[k][0], ref[k][1]), \
                f'item {i}: the two views came out identical'

        # ... and it survives the loader, at any worker count
        for workers in (0, 1, 2, 3):
            loader = DataLoader(tv, batch_size=2, num_workers=workers, shuffle=False)
            got = {}
            for v0, v1, _y, ii in loader:
                for k in range(len(ii)):
                    got[int(ii[k])] = (v0[k], v1[k])
            assert len(got) == len(idx), f'--workers {workers}: {len(got)} of {len(idx)} items'
            for i in idx:
                for v in (0, 1):
                    assert torch.equal(got[i][v], ref[i][v]), \
                        f'--workers {workers} changed view {v} of item {i}'
        print('  augmentation is worker-independent ok')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_head_init():
    """B11/B17: seeding the head, the prototypes and the judge from frozen CLIP.

    Three separate claims, each of which can be true while the others are false:

    1. ``robust_centroids`` recovers the class directions better from noisy
       labels than a plain per-class mean does, and its kept set is cleaner than
       the set it started from;
    2. ``init_from_centroids`` writes unit-norm rows for the classes it has and
       leaves the others on their random row -- a *zero* row would be a neutral
       competitor at cosine 0, which the judge's own docstring warns about;
    3. the epoch actually reaches the augmentation through the loader, which is
       only true because ``--epoch-aug`` turns persistent workers off.  That was
       a real bug in the first cut: the flag set ``tr.epoch`` on main's copy and
       the forked workers never saw it, so it was a silent no-op.
    """
    torch.manual_seed(0)
    nclass, dim, per = 4, 16, 40
    dirs = F.normalize(torch.randn(nclass, dim), dim=-1)

    feat, y, y_true = [], [], []
    for c in range(nclass):
        # 0.15 * sqrt(16) = 0.6 of perturbation against a unit signal, so a sample
        # stays nearest its own direction -- the noise makes the *labels* wrong,
        # it does not make the features uninformative
        feat.append(F.normalize(dirs[c] + 0.15 * torch.randn(per, dim), dim=-1))
        for i in range(per):
            # every 4th sample is a class-c image filed under class c+1: exactly
            # the failure mode of keyword-named folders, and 25% of the set
            y.append((c + 1) % nclass if i % 4 == 0 else c)
            y_true.append(c)
    feat, y, y_true = torch.cat(feat), torch.tensor(y), torch.tensor(y_true)

    # the baseline the docstring claims is worse: every sample a folder contains
    # averaged at face value, which is exactly what the training labels say
    cnt = torch.bincount(y, minlength=nclass).float()
    sums = torch.zeros(nclass, dim)
    sums.index_add_(0, y, feat)
    plain = F.normalize(sums / cnt[:, None], dim=-1)
    once, present1, _ = train.robust_centroids(feat, y, nclass, rounds=1, verbose=False)
    kept, present2, keep = train.robust_centroids(feat, y, nclass, rounds=3, verbose=False)
    assert bool(present1.all()) and bool(present2.all()), \
        'every class has 40 samples here, so no class centroid may come out absent'
    cos_plain = float((plain * dirs).sum(1).mean())
    cos_once = float((F.normalize(once, dim=-1) * dirs).sum(1).mean())
    cos_kept = float((F.normalize(kept, dim=-1) * dirs).sum(1).mean())
    assert cos_once > cos_plain and cos_kept > cos_plain, (
        f'the drop-and-recompute rounds did not beat the plain per-class mean: '
        f'{cos_plain:.4f} -> {cos_once:.4f} (1 round) -> {cos_kept:.4f} (3). Either the '
        f'filter is not removing the 10 mislabelled images per class, or it is '
        f'removing the clean ones.')
    assert cos_kept >= cos_once - 0.01, (
        f'more rounds made the centroids worse ({cos_once:.4f} -> {cos_kept:.4f}), so the '
        f'iteration is not monotone')
    base = float((y == y_true).float().mean())
    kept_rate = float((y[keep] == y_true[keep]).float().mean())
    assert kept_rate > base, (
        f'the kept set ({kept_rate:.3f} clean) is no cleaner than the full set '
        f'({base:.3f} clean), so the filter is not selecting on the labels')
    assert 0.3 * y.numel() < int(keep.sum()) < y.numel(), (
        f'kept {int(keep.sum())} of {y.numel()} samples -- a filter that keeps '
        f'everything is inert and one that keeps almost nothing is the class-erasing '
        f'failure this exists to avoid')
    print(f'  robust_centroids ok (cos to truth {cos_plain:.3f} -> {cos_once:.3f} -> '
          f'{cos_kept:.3f}; clean {base:.3f} -> {kept_rate:.3f})')

    # ---- an absent class must keep its random row
    head = train.CosineClassifier(dim, nclass)
    before = head.weight.detach().clone()
    m = torch.randn(nclass, dim)
    m[2] = 0.0                                   # no samples -> zero estimate
    pres = torch.tensor([True, True, False, True])
    head.init_from_centroids(m, pres)
    w = head.weight.detach()
    for c in (0, 1, 3):
        assert torch.allclose(w[c], F.normalize(m[c], dim=-1), atol=1e-6), \
            f'class {c} was not seeded from its centroid'
    assert torch.equal(w[2], before[2]) and float(w[2].abs().sum()) > 0, \
        'an absent class must keep its random row -- zeroing it makes it a neutral ' \
        'competitor at cosine 0'
    print('  init_from_centroids ok (unit rows; absent class untouched)')

    # ---- B14: the sparse frozen target, and the loss that consumes it
    #
    # The gate is the whole safety argument for this term -- "it cannot confirm a
    # mistake the student makes" reduces to "the modal class of the target is the
    # given label, or the row carries nothing" -- so it is asserted directly
    # rather than inferred from a loss going down.
    bank = train.mean_bank(kept, present2)
    t_i, t_w, t_agree = train.frozen_soft_targets(feat, *bank, y, temp=0.05, topk=4)
    assert t_i.shape == (y.numel(), 4) and t_w.shape == (y.numel(), 4), \
        f'sparse target shape {tuple(t_i.shape)}/{tuple(t_w.shape)} is not (n, topk)'
    assert t_i.dtype == torch.int16 and t_w.dtype == torch.float16, \
        'the target is stored sparse precisely to stay at 4.6 MB for 148695 samples'
    by_hand = F.normalize(feat, dim=-1) @ F.normalize(kept, dim=-1).t()
    by_hand[:, ~present2] = -2.0
    assert torch.equal(t_agree, by_hand.argmax(1) == y), \
        'the agreement flag is not "the frozen argmax is the given label"'
    assert bool((t_i[t_agree][:, 0] == y[t_agree]).all()), \
        'on an agreeing sample the modal class of the target must be the given label'
    # the gate lives at the *consumer*, not in the target: B13 wants the
    # disagreeing samples and B14 must never see them, so the function returns one
    # ungated distribution and each site applies its own mask
    rows = t_w.float()
    assert torch.allclose(rows.sum(1), torch.ones(rows.shape[0]), atol=5e-3), \
        ('every row must be a distribution (sum to 1), agreeing or not -- the mix needs the '
         'target to keep its normalisation on the rows it touches. The tolerance is float16 '
         'rounding on topk entries, not slack in the claim')
    assert float(t_w[~t_agree].abs().sum()) > 0, \
        ('disagreeing rows were zeroed inside frozen_soft_targets; the gate must be applied '
         'where it is used, or --frozen-mix-rho has nothing to correct with')
    assert not bool((t_i[~t_agree][:, 0] == y[~t_agree]).any()), \
        'a disagreeing row must not have the given label as its modal class'
    sharp = train.frozen_soft_targets(feat, *bank, y, temp=0.05, topk=4)[1][t_agree].float()
    flat = train.frozen_soft_targets(feat, *bank, y, temp=0.5, topk=4)[1][t_agree].float()
    assert float(sharp[:, 0].mean()) > float(flat[:, 0].mean()), \
        ('--frozen-temp does not act on the class similarity: a lower temperature must put '
         'more mass on the frozen argmax')
    print(f'  frozen_soft_targets ok ({int(t_agree.sum())}/{y.numel()} agreed, top-1 mass '
          f'{float(sharp[:, 0].mean()):.3f} at temp 0.05 vs {float(flat[:, 0].mean()):.3f} '
          f'at 0.5)')

    # ---- B2/B15: the LOO medoid bank
    #
    # A class with two visual modes is the case the single mean cannot represent,
    # so build one: 20 samples near +d, 20 near -d, filed under one label. A plain
    # class mean lands between the modes (cosine ~0 to both); two medoids must land
    # on them. And the exact-LOO claim has to hold: querying a medoid's own source
    # sample must not score that medoid.
    # the two modes are fixed *orthogonal* axes rather than random directions: in
    # 16-d a random pair has cosine with sd ~0.25, which leaves the k=1 contrast
    # below resting on a 3-sigma draw.  Orthogonal makes cos(m_a, m_b) exactly 0,
    # so the numbers this block asserts are arithmetic, not luck.
    m_a, m_b = torch.zeros(dim), torch.zeros(dim)
    m_a[0], m_b[1] = 1.0, 1.0
    bimodal = torch.cat([F.normalize(m_a + 0.05 * torch.randn(20, dim), dim=-1),
                         F.normalize(m_b + 0.05 * torch.randn(20, dim), dim=-1)])
    p, pc, ps, pres = train.proto_bank(bimodal, torch.zeros(40, dtype=torch.long),
                                       torch.ones(40, dtype=torch.bool), 1, k=2, verbose=False)
    assert p.shape[0] == 2 and bool(pres.all()), \
        f'k=2 medoids on a two-mode class gave {p.shape[0]} prototypes, present={pres.tolist()}'
    got = sorted(float((p @ x).max()) for x in (m_a, m_b))
    assert got[0] > 0.9 and got[1] > 0.9, (
        f'the two medoids did not land on the two modes (best cosines {got}); a bank that '
        f'reproduces the class mean would score both near 0')
    # the contrast that makes the bank worth its cost: one prototype cannot cover
    # both modes, so k=1 must leave one of them unexplained.  If it did not, the
    # single mean was already sufficient and --frozen-protos > 1 buys nothing.
    one, _, _, _ = train.proto_bank(bimodal, torch.zeros(40, dtype=torch.long),
                                    torch.ones(40, dtype=torch.bool), 1, k=1, verbose=False)
    got1 = sorted(float((one @ x).max()) for x in (m_a, m_b))
    assert got1[0] < 0.9 < got1[1], (
        f'k=1 covered both modes ({got1}), so the class mean was already enough and the bank '
        f'is not doing the thing it exists to do')
    assert p.shape[0] == 2 and len(set(ps.tolist())) == 2 and bool((pc == 0).all()), \
        f'the medoids must be two distinct source samples of that class (LOO depends on it); ' \
        f'got sources {ps.tolist()}'
    assert bool((ps >= 0).all()), \
        'proto_src must carry the source index -- the LOO exclusion is done from it'
    # bit-identical on a second call: the bank is built once, but a checkpoint can
    # only be reproduced if the run that produced it re-derives the same teacher
    again = train.proto_bank(bimodal, torch.zeros(40, dtype=torch.long),
                             torch.ones(40, dtype=torch.bool), 1, k=2, verbose=False)
    assert torch.equal(again[0], p) and torch.equal(again[2], ps), \
        'proto_bank is not deterministic across calls; the run would not be reproducible'
    # an absent class contributes nothing and must not be reported present
    p3, pc3, ps3, pres3 = train.proto_bank(torch.cat([bimodal, bimodal], 0),
                                           torch.cat([torch.zeros(40), torch.ones(40)]).long(),
                                           torch.ones(80, dtype=torch.bool), 3, k=2, verbose=False)
    assert pres3.tolist() == [True, True, False], \
        f'a class with no samples came out {pres3.tolist()}; it must be absent'
    # exact leave-one-out, as a contrast, because "the query matched class 1" is only
    # evidence of exclusion when it matches class 0 with the same weights and no LOO.
    # Each medoid is a near-copy of exactly one sample, so without LOO the query's own
    # class wins; with LOO that medoid is void and the other class must take over.
    # orthogonal queries for the same reason as the modes above: the contrast is only
    # evidence if the wrong class is clearly farther than the right one
    zq = torch.zeros(2, dim)
    zq[0, 0], zq[1, 1] = 1.0, 1.0
    pr = F.normalize(zq + 0.01 * torch.randn(2, dim), dim=-1)
    pc2, ps2 = torch.tensor([0, 1]), torch.tensor([0, 1])
    pres2, given2 = torch.tensor([True, True]), torch.tensor([0, 1])
    no_loo = train.frozen_soft_targets(zq, pr, pc2, torch.full((2,), -1), pres2, given2,
                                       temp=0.05, topk=1)[0]
    with_loo = train.frozen_soft_targets(zq, pr, pc2, ps2, pres2, given2,
                                         temp=0.05, topk=1)[0]
    assert no_loo[:, 0].tolist() == [0, 1], (
        f'with no source indices each query must match its own near-copy, else the setup is '
        f'not testing anything; got {no_loo[:, 0].tolist()}')
    assert with_loo[:, 0].tolist() == [1, 0], (
        f'a medoid still won its own source sample ({with_loo[:, 0].tolist()}); the exact-LOO '
        f'exclusion in frozen_soft_targets is not firing')
    print(f'  proto_bank ok (2 modes recovered, k=1 covers only {got1[1]:.2f}/{got1[0]:.2f}; '
          f'LOO exclusion verified)')

    # ---- B13: the forward correction, and that it keeps the target normalised
    tgt = torch.full((2, 4), 0.25)
    mix = train.frozen_mix(tgt, torch.tensor([[2, 3], [2, 3]]),
                           torch.tensor([[0.5, 0.5], [0.0, 0.0]]), 0.4)
    assert abs(float(mix[0].sum()) - 1.0) < 1e-6, \
        f'the mixed target sums to {float(mix[0].sum()):.6f}, not 1'
    assert abs(float(mix[0, 2]) - (0.25 * 0.6 + 0.5 * 0.4)) < 1e-6, \
        'the mixed mass is not (1-rho)*target + rho*q at the frozen classes'
    assert torch.allclose(mix[1], tgt[1]), \
        ('a row with no frozen weight was scaled by (1-rho); a gated-out sample must keep its '
         'target *and its normalisation*')
    assert float(tgt[0, 2]) == 0.25, 'frozen_mix mutated its input in place'
    print('  frozen_mix ok (mass preserved; zero rows untouched; input not mutated)')

    # a two-class case whose KL can be written down: q = (0.75, 0.25), p = (0.5, 0.5)
    # gives 0.75 ln 1.5 + 0.25 ln 0.5 = 0.130812
    lg = torch.zeros(2, 2, requires_grad=True)
    qi = torch.tensor([[0, 1], [0, 1]], dtype=torch.int16)
    qw = torch.tensor([[0.75, 0.25], [0.0, 0.0]], dtype=torch.float16)
    kl = train.sparse_kl(lg, qi, qw)
    assert abs(float(kl[0]) - 0.130812) < 1e-5, \
        f'sparse_kl gave {float(kl[0]):.6f} where the KL by hand is 0.130812'
    assert float(kl[1]) == 0.0 and torch.isfinite(kl[1]), \
        ('an all-zero row must contribute exactly 0 -- 0 * log 0 has to be taken as 0, not '
         '-inf, or a gated-out sample poisons the whole loss')
    kl.sum().backward()
    g = lg.grad
    assert bool((g[1].abs().sum() == 0)), \
        'a gated-out sample still produced a gradient; the gate is leaking'
    assert bool((g[0].abs().sum() > 0)), 'the distilled sample produced no gradient at all'
    # the KL is minimised when the student matches the target, so the gradient must
    # push logit 0 up and logit 1 down
    assert float(g[0, 0]) < 0 < float(g[0, 1]), \
        f'the KL gradient points the wrong way: {g[0].tolist()}'
    print('  sparse_kl ok (matches the hand-computed 0.130812; zero rows are inert)')

    # ---- the epoch must reach the augmentation, through the loader
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / 'train'
        random.seed(0)
        for c in range(3):
            d = root / f'{c:04d}'
            d.mkdir(parents=True)
            for i in range(6):
                px = bytes(random.randrange(256) for _ in range(48 * 48 * 3))
                Image.frombytes('RGB', (48, 48), px).save(d / f'img_{c}_{i}.jpg')

        a_off = train_args(['--augment-mode', 'index', '--no-epoch-aug', '--workers', '2'])
        a_on = train_args(['--augment-mode', 'index', '--epoch-aug', '--workers', '2'])
        tr_off, va_off = train.build_datasets(a_off)
        lo, vo = train.build_loaders(a_off, tr_off, va_off)
        tr_on, va_on = train.build_datasets(a_on)
        ln, vn = train.build_loaders(a_on, tr_on, va_on)
        assert lo.persistent_workers and vo.persistent_workers, \
            'the default path must keep its persistent workers (they are the startup cost)'
        assert not ln.persistent_workers and not vn.persistent_workers, \
            ('--epoch-aug must turn persistent workers off on BOTH loaders: main mutates '
             'tr.epoch and a forked worker holds its own copy of the dataset, so the flag '
             'would be a silent no-op')

        ds = tr_on
        ds.epoch = 0
        v0 = ds[0][0]
        ds.epoch = 0
        assert torch.equal(v0, ds[0][0]), 'the same epoch must give the same augmentation'
        ds.epoch = 5
        assert not torch.equal(v0, ds[0][0]), \
            '--epoch-aug did not change the augmentation between epochs'
        # ... and epoch 0 must be the pre-flag behaviour, byte for byte: the epoch
        # enters the seed as an XOR with a constant, and XOR with 0 is the identity.
        # This reproduces the old formula by hand, which is the point -- the check
        # pins the constants, so a silent change to them fails here.
        ds.epoch = 0
        h = (ds.seed * 0x9E3779B1) ^ (0 * 0x85EBCA77) ^ (0 * 0xC2B2AE3D)
        s = (h ^ (h >> 15)) % (2 ** 63 - 1)
        py_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(s)
                random.seed(s)
                ref = ds.augmented(ds._load(ds.items[0][0]), 0, 0)
        finally:
            random.setstate(py_state)
        assert torch.equal(v0, ref), \
            'epoch=0 is not identical to the old (seed, index, view) seed formula'
        print('  epoch augmentation ok (per-epoch; loader reaches it)')

        # ---- every new flag on, through the whole training program
        out = tmp / 'out'
        a = train.parse_args(['--data', str(root), '--out', str(out), '--epochs', '3',
                              '--warmup-epochs', '1', '--batch-size', '4', '--workers', '0',
                              '--val-ratio', '0.25', '--seed', '0', '--amp', 'none',
                              '--head-init', 'frozen', '--head-init-batch-size', '8',
                              '--epoch-aug', '--warm-robust', '--noise-judge'])
        assert a.head_init == 'frozen' and a.head_init_rounds == 2
        assert a.head_init_batch_size == 8 and a.epoch_aug and a.warm_robust
        # Experimental teacher/robustness flags default off.  The data-view and
        # scheduler defaults intentionally follow the verified teammate recipe.
        d = train_args()
        assert d.head_init == 'none' and d.epoch_aug and not d.warm_robust
        assert d.augment_mode == 'worker'
        assert d.lr_warmup_epochs == 0 and d.norm_max_gain == 0
        assert not d.reset_tracker_warmup and not d.consistency_symmetric
        assert not d.proto_normalize
        assert not d.proto_trusted_loss
        assert d.distill_weight == 0.0 and d.frozen_mix_rho == 0.0, \
            ('--distill-weight / --frozen-mix-rho must default off: a non-zero default would '
             'change the loss of every recorded recipe')
        assert d.frozen_temp == 0.05 and d.frozen_topk == 8 and d.frozen_protos == 1, \
            ('the teacher-config knobs must default to the values the recorded recipes were '
             'built at, so that turning only the weight on is a single-variable change')
        assert not d.frozen_mix_agree_only, \
            '--frozen-mix-agree-only is the conservative variant and must be opt-in'
        buf = io.StringIO()
        with redirect_stdout(buf):
            train.main(a)
        log = buf.getvalue()
        assert 'head-init frozen:' in log and 'frozen-CLIP judge: full coverage' in log, \
            f'--head-init did not run its pass:\n{log[-2000:]}'
        assert 'tracker posterior retained at warm-up boundary' in log, \
            'the reference recipe must retain the warm-up posterior with --head-init'
        ck = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        assert 'head.weight' in ck['model'], 'the checkpoint lost the seeded head'
        # the seeding must have moved the head off its random init.  The two
        # populations are far apart and both are known exactly: `randn(750, 512) * 0.02`
        # has row norm 0.02 * sqrt(512) = 0.452 (sd 0.014), and `init_from_centroids`
        # writes unit rows.  0.9 is the midpoint, so neither a few training steps nor
        # the per-class max can move a run across it.
        wn = ck['model']['head.weight'].norm(dim=1)
        assert float(wn.max()) > 0.9, (
            f'the head in the checkpoint still looks randomly initialised (max row norm '
            f'{float(wn.max()):.3f}, random is ~0.45 and seeded is 1.0); --head-init did not '
            f'reach the saved weights')
        print('  head-init + epoch-aug + warm-robust end-to-end ok')

        # ---- B14 alone, with the head *not* seeded: the two flags share one frozen
        # pass but must stay separately switchable, or --distill-weight cannot be
        # measured on its own and neither can --head-init
        out2 = tmp / 'out_distill'
        a2 = train.parse_args(['--data', str(root), '--out', str(out2), '--epochs', '2',
                               '--warmup-epochs', '1', '--batch-size', '4', '--workers', '0',
                               '--val-ratio', '0.25', '--seed', '0', '--amp', 'none',
                               '--distill-weight', '1.0', '--frozen-temp', '0.1',
                               '--frozen-topk', '3'])
        assert a2.distill_weight == 1.0 and a2.head_init == 'none'
        assert a2.frozen_temp == 0.1 and a2.frozen_topk == 3
        buf = io.StringIO()
        with redirect_stdout(buf):
            train.main(a2)
        log = buf.getvalue()
        assert 'frozen target:' in log, \
            f'--distill-weight did not build its targets:\n{log[-2000:]}'
        assert 'head-init frozen: head seeded' not in log, \
            '--distill-weight must not seed the head; that is --head-init\'s job'
        ck = torch.load(out2 / 'best.pt', map_location='cpu', weights_only=False)
        wn = ck['model']['head.weight'].norm(dim=1)
        assert float(wn.max()) < 0.9, (
            f'--distill-weight seeded the head as a side effect (max row norm '
            f'{float(wn.max()):.3f}; random is ~0.45, seeded is 1.0); the two flags are no '
            f'longer separable')
        print('  distill-frozen alone end-to-end ok (no head seeding)')

        # ---- B13 alone: the corrected target must be constructed during warm-up.
        # The default teacher is now the robust class mean.  The old test forced
        # flips by masking the single medoid's own class on synthetic random-noise
        # images; that only tested a leave-one-out artifact.  A true flip depends
        # on teacher quality and is not required on this synthetic fixture.
        out3 = tmp / 'out_mix'
        a3 = train.parse_args(['--data', str(root), '--out', str(out3), '--epochs', '3',
                               '--warmup-epochs', '2', '--batch-size', '4', '--workers', '0',
                               '--val-ratio', '0.25', '--seed', '0', '--amp', 'none',
                               '--frozen-mix-rho', '0.8'])
        assert a3.frozen_mix_rho == 0.8 and a3.frozen_protos == 1
        assert a3.distill_weight == 0.0, 'the mix run must not carry the KL as well'
        buf = io.StringIO()
        with redirect_stdout(buf):
            train.main(a3)
        log = buf.getvalue()
        assert 'frozen target quality:' in log, \
            f'the frozen teacher target was not diagnosed:\n{log[-2000:]}'
        assert '[mix] rho=0.8 all samples' in log, \
            f'--frozen-mix-rho did not report its correction:\n{log[-2000:]}'
        seen = re.findall(r'the target changed its argmax on (\d+)/(\d+) samples', log)
        assert seen, f'the [mix] line has no flip count:\n{log[-2000:]}'
        flips = sum(int(n) for n, _ in seen)
        agree = re.search(r'frozen target: (\d+)/(\d+) samples', log)
        assert agree, f'the frozen teacher never reported its agreement:\n{log[-2000:]}'
        n_agree, n_tot = int(agree.group(1)), int(agree.group(2))
        assert 0 <= n_agree <= n_tot and n_tot > 0
        assert 0 <= flips <= sum(int(n) for _, n in seen)
        assert 'head-init frozen: head seeded' not in log, \
            '--frozen-mix-rho must not seed the head'
        ck = torch.load(out3 / 'best.pt', map_location='cpu', weights_only=False)
        assert float(ck['model']['head.weight'].norm(dim=1).max()) < 0.9, \
            '--frozen-mix-rho seeded the head as a side effect'
        print(f'  frozen-mix alone end-to-end ok ({flips} argmax flips, no head seeding)')

        # ---- the conservative variant must be a different run, not a comment
        out4 = tmp / 'out_mix_agree'
        a4 = train.parse_args(['--data', str(root), '--out', str(out4), '--epochs', '1',
                               '--warmup-epochs', '1', '--batch-size', '4', '--workers', '0',
                               '--val-ratio', '0.25', '--seed', '0', '--amp', 'none',
                               '--frozen-mix-rho', '0.3', '--frozen-mix-agree-only'])
        buf = io.StringIO()
        with redirect_stdout(buf):
            train.main(a4)
        assert '[mix] rho=0.3 agree-only' in buf.getvalue(), \
            '--frozen-mix-agree-only did not change the gate the mix is built with'
        print('  frozen-mix agree-only variant ok')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_end_to_end():
    tmp = Path(tempfile.mkdtemp())
    try:
        root, out = tmp / 'train', tmp / 'out'
        random.seed(0)
        for c in range(4):
            d = root / f'{c:04d}'
            d.mkdir(parents=True)
            for i in range(8):
                px = bytes(random.randrange(256) for _ in range(48 * 48 * 3))
                # unique names: the real test set is flat with distinct file names
                Image.frombytes('RGB', (48, 48), px).save(d / f'img_{c}_{i}.jpg')

        a = train.parse_args(['--data', str(root), '--out', str(out), '--epochs', '4',
                              '--warmup-epochs', '1', '--batch-size', '4', '--workers', '0',
                              '--val-ratio', '0.25', '--seed', '0', '--amp', 'none',
                              '--proto-weight', '0.5', '--anchor-weight', '0.1'])
        # the hold-out split must NOT leak into the training split
        tr_ds, va_ds = train.build_datasets(a)
        tr_paths = {p for p, _ in tr_ds.items}
        va_paths = {p for p, _ in va_ds.items}
        both = tr_paths & va_paths
        assert not both, f'{len(both)} images are in BOTH the train and the val split'
        assert len(tr_paths) + len(va_paths) == 32, (len(tr_paths), len(va_paths))

        train.main(a)
        assert (out / 'best.pt').exists() and (out / 'last.pt').exists(), 'no checkpoint written'
        # --save-every 4 on a 4-epoch run must leave exactly one snapshot
        assert (out / 'ep4.pt').exists(), 'periodic checkpoint was not written'
        assert not (out / 'ep2.pt').exists(), 'snapshots are not on the N-epoch grid'

        ck = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        assert ck['classes'] == {f'{c:04d}': c for c in range(4)}, ck['classes']
        assert ck['epoch'] >= 0
        # snapshots are inference-only.  The tracker posterior alone is
        # n_train x n_class floats -- 446 MB per snapshot on the real 148695x750
        # round -- and inference reads none of it, nor the optimiser or RNG state.
        assert not ({'tracker', 'optim', 'rng'} & set(ck)), \
            f'best.pt carries training state only last.pt needs: {sorted(ck)}'
        assert len(ck['class_counts']) == 4 and sum(ck['class_counts']) == len(tr_paths), \
            f'class_counts must describe the training split: {ck["class_counts"]}'
        # ... while last.pt has to stay resumable
        full = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        for k in ('tracker', 'optim', 'rng'):
            assert k in full, f'last.pt lost {k}, --resume would break'

        csv_path = tmp / 'pred_results.csv'
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(csv_path)]))
        rows = list(csv.reader(csv_path.open()))
        assert len(rows) == 32, f'expected 32 predictions, got {len(rows)}'
        assert all(len(r) == 2 and r[0].endswith('.jpg') and len(r[1]) == 4 and r[1].isdigit()
                   for r in rows), f'bad CSV rows: {rows[:3]}'
        assert len({r[0] for r in rows}) == 32, 'duplicate file names in the CSV'
        print(f'  end-to-end ok ({len(rows)} predictions, sample {rows[0]})')

        # several taus must come out of ONE forward pass, one CSV each
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'p.csv'),
                                     '--logit-adjust', '0', '0.5', '1.0']))
        for f in ('p.csv', 'p_tau050.csv', 'p_tau100.csv'):
            assert (tmp / f).exists(), f'--logit-adjust did not write {f}'
        assert infer.adjusted_path('pred_results.csv', 0) == 'pred_results.csv'
        assert infer.adjusted_path('pred_results.csv', 1.0) == 'pred_results_tau100.csv'
        print('  logit-adjust ok')

        # ---- TTA: one checkpoint, several views (competition-permitted; what
        # is *not* permitted is averaging several checkpoints)
        # (a) one recipe, two spellings, one CSV: "the trained size" written as no
        #     flags and written as the product flags given that size.  This is a
        #     path-equivalence claim, not a double-counting one -- a view averaged
        #     with itself is the same number, so counting one twice cannot move a
        #     byte.  What a difference here would actually mean is that the two
        #     spellings are not the same recipe: a second tower, a transform built
        #     from the flag instead of from the resolved view, or a base size read
        #     from somewhere else.
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'tta_dup.csv'),
                                     '--tta-sizes', '224']))
        assert (tmp / 'tta_dup.csv').read_bytes() == csv_path.read_bytes(), \
            ('--tta-sizes 224 is the trained size, so it must be the same recipe as no '
             'flags at all, and it produced a different CSV')
        # ... and the other aggregation is the same identity for one view: an
        # average over one view is that view, and the head over one feature is that
        # view's logits.  This is the clause that catches `feat` normalising twice
        # or applying the head to a sum rather than a mean.  (It used to be spelled
        # `--tta-agg logit`, which is the default now, so the *feat* one is what
        # still means something.)
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'tta_feat1.csv'),
                                     '--tta-agg', 'feat']))
        assert (tmp / 'tta_feat1.csv').read_bytes() == csv_path.read_bytes(), \
            'single-view feature averaging is not the identity'

        # (b) a genuine multi-size, multi-view run must survive: a model built
        #     per resolution, features averaged, the head applied once
        _STUB_CALLS.clear()
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'tta.csv'),
                                     '--tta-sizes', '288', '--tta-flip']))
        assert not any(kw for _, _, kw, _ in _STUB_CALLS), \
            (f'the extra TTA resolution was delegated to open_clip\'s '
             f'force_image_size, whose resample is not the one the scored 288 run '
             f'used: {[c[2] for c in _STUB_CALLS]}')
        grids = {tuple(c[3].visual.positional_embedding.shape) for c in _STUB_CALLS}
        assert (82, DIM) in grids, \
            f'the extra TTA resolution never resampled its grid: {sorted(grids)}'
        tta_rows = list(csv.reader((tmp / 'tta.csv').open()))
        assert len(tta_rows) == 32, f'TTA changed the row count: {len(tta_rows)}'
        assert [r[0] for r in tta_rows] == [r[0] for r in rows], 'TTA reordered the rows'
        assert all(len(r) == 2 and r[1].isdigit() for r in tta_rows), tta_rows[:3]
        print(f'  tta ok ({len(tta_rows)} rows from 4 views)')

        # (c) the named sets, through a real run.  `tta4` is the four views that
        #     scored 66.2456 and the product flags cannot ask for them, so if this
        #     path breaks, the one leaderboard-validated recipe in the project
        #     becomes unreproducible -- and nothing else here would notice.
        #
        #     What is compared is the pixels the tower was *handed*, not the CSV.
        #     A CSV cannot answer this block's question.  A view average changes an
        #     argmax only where the decision was close, so "the same 32 labels for
        #     one view and for four" is the *normal* case for a confident model --
        #     and on this fixture the decisions do not depend on the input either:
        #     val_acc stayed at 0.25 (= chance for 4 classes) through all four
        #     epochs, the tracker resigns itself to relabelling most samples, and
        #     the tower averages the whole patch grid into one token, which makes
        #     upscaled 48x48 noise close to the same feature under any view.  Either
        #     way a label-difference assertion is a coin flip, it fails on an
        #     *accurate* model too, and it cannot separate "the views were not
        #     applied" from "the views were applied and changed no decision".
        #     Counting forwarded images and comparing their pixels separates those,
        #     and does not depend on what the model learned.
        #
        #     What this still cannot see is whether the *average* decided the row
        #     rather than the first view.  That is not a gap in the assertion but
        #     in the observable: when the model is confident, averaging changes
        #     nothing, so no CSV-level check can distinguish the two.  The one-view
        #     identities above pin the aggregation path, and `probe.py` shares the
        #     view table, not this loop.
        n_img = len(rows)
        for name in ('tta4', 'tta4s320', 'mix6'):
            argv = ['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                    '--output', str(tmp / f'v_{name}.csv'), '--tta-views', name]
            # 224 because the fixture trains without --image-size, and that is
            # what `infer.main` derives as the base for `--tta-views` too
            plan = infer.view_plan(infer.parse_args(argv), 224)[0]
            seen, real_forward = [], _StubVisual.forward

            def spy(self, x, _seen=seen, _real=real_forward):
                _seen.append(x.detach().clone())
                return _real(self, x)

            try:
                _StubVisual.forward = spy
                infer.main(infer.parse_args(argv))
            finally:
                _StubVisual.forward = real_forward

            vr = list(csv.reader((tmp / f'v_{name}.csv').open()))
            assert len(vr) == n_img, f'--tta-views {name} changed the row count: {len(vr)}'
            assert [r[0] for r in vr] == [r[0] for r in rows], f'{name} reordered the rows'
            assert all(len(r) == 2 and r[1].isdigit() for r in vr), vr[:3]
            # every image once per view: the plan's count is what the run *says* it
            # does ("N forward pass(es) per image" in its own log line), and this
            # is what it did
            n_seen = sum(t.shape[0] for t in seen)
            assert n_seen == len(plan) * n_img, (
                f'--tta-views {name} planned {len(plan)} views over {n_img} images, so '
                f'the tower should have been handed {len(plan) * n_img} images; it was '
                f'handed {n_seen}')
            # ... regrouped per view.  `infer.py` walks the dataset once per view
            # ("view 1/4 done", "view 2/4 done", ...), so consecutive batches belong
            # to the same view -- and requiring the blocks to come out as exactly
            # `len(plan)` full passes, with nothing left over, is what asserts that
            # rather than trusting it.
            per_view, cur, got = [], [], 0
            for t in seen:
                cur.append(t)
                got += t.shape[0]
                if got == n_img:
                    per_view.append(torch.cat(cur))
                    cur, got = [], 0
            assert not cur and len(per_view) == len(plan), (len(per_view), len(plan), got)
            # each other view is a different input from the first.  A size or a
            # ratio dropped between the view list and the transform would satisfy
            # every other check in this file -- the view list, the row count, the
            # CSV order -- and show up only here.  Compared against view 1 and not
            # pairwise: the fixture's images are square, which makes `wide` and
            # `full` the same tensor as each other, and a pairwise claim would fail
            # on that property of the data rather than on anything infer.py does.
            for k in range(1, len(per_view)):
                a, b = per_view[0], per_view[k]
                assert a.shape != b.shape or not torch.allclose(a, b), (
                    f'--tta-views {name}: view {k + 1} was handed exactly the pixels of '
                    f'view 1, so the set costs {len(plan)} forward passes per image and '
                    f'buys one input')
        print('  named view sets ok')

        # (d) --workers may not move a single decision.  The CSV is the
        #     submission, and the flag exists only to take the resize work off the
        #     thread that is waiting on the GPU -- so it is not allowed to be part
        #     of the recipe.
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'w0.csv'), '--tta-views', 'mix6']))
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'w2.csv'), '--tta-views', 'mix6',
                                     '--workers', '2']))
        assert (tmp / 'w2.csv').read_bytes() == (tmp / 'w0.csv').read_bytes(), \
            ('--workers changed the CSV: the submission is a function of a flag that '
             'is only about speed')
        print('  tta loader is worker-independent ok')

        # --noise-judge has to survive a real training run: it needs the warm-up
        # feature collection, the frozen class means, the veto and the
        # checkpoint round trip.  Each of those is a separate opportunity to
        # pass the wrong thing to the wrong place, and none of them is
        # exercised by the tracker tests above.
        jout = tmp / 'out_judge'
        train.main(train.parse_args([
            '--data', str(root), '--out', str(jout), '--epochs', '2', '--warmup-epochs', '1',
            '--batch-size', '4', '--workers', '0', '--val-ratio', '0.25', '--seed', '0',
            '--amp', 'none', '--noise-judge', '--judge-margin', '0.0']))
        jck = torch.load(jout / 'last.pt', map_location='cpu', weights_only=False)
        sus = jck['tracker'].get('suspect')
        assert sus is not None, '--noise-judge trained but the checkpoint carries no veto'
        assert sus.shape == (len(tr_paths),), f'veto covers {sus.shape}, want {len(tr_paths)}'
        assert sus.dtype == torch.bool, sus.dtype
        print(f'  noise-judge end-to-end ok ({int(sus.sum())}/{sus.numel()} vetoed)')

        # the error-analysis report must survive a model whose confusion
        # structure is whatever the stub happens to produce (including none)
        adir = tmp / 'analysis'
        analyze.report(analyze.parse_args(['--data', str(root), '--checkpoint', str(out / 'best.pt'),
                                           '--out', str(adir), '--workers', '0', '--split', 'val',
                                           '--val-ratio', '0.25', '--seed', '0']))
        for f in ('per_class.csv', 'confusions.csv', 'per_sample.csv', 'suspected_noisy.csv'):
            assert (adir / f).exists(), f'analyze.py did not write {f}'
        assert len(list(csv.reader((adir / 'per_sample.csv').open()))) == 9, \
            'per_sample.csv should hold a header + the 8 hold-out images'
        print('  analyze ok')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


#: Every check, in order.  Named so the runner can report which one failed.
CHECKS = (
    ('losses', check_losses),
    ('tracker', check_tracker),
    ('judge', check_judge),
    ('optional gates', check_optional_gates),
    ('targets', check_targets),
    ('proto', check_proto),
    ('lora/checkpoint', check_lora),
    ('param groups', check_param_groups),
    ('lr warmup', check_lr_warmup),
    ('image size', check_image_size),
    ('pos embed', check_pos_embed),
    ('ck image size', check_ck_image_size),
    ('calib gate', check_calib_gate),
    ('view sets', check_view_sets),
    ('flat images', check_flat_images),
    ('probe cache identity', check_probe_cache_identity),
    ('reproducible augment', check_reproducible_augment),
    ('head init', check_head_init),
    ('end to end', check_end_to_end),
)


def main():
    """Run every check, report all failures, and only then return.

    The loop does **not** stop at the first failure, which is the whole point:
    this file is the only verification available on a machine without torch, so
    a run that dies on check 7 costs a full round trip to discover what check 8
    would have said.  Reporting every failure in one run is worth the
    caveat that a check which aborted midway leaves the RNG a step ahead of
    where it would otherwise be -- so a failure *after* the first one may be a
    consequence of it rather than a separate bug, and the first failure is the
    one to fix.

    The top-level seed is set exactly once, before the loop, so a check that
    runs after a successful one sees the same input sequence it always has.
    Re-seeding per check would be tidier but would silently change the inputs
    the currently-passing checks are given -- an unverifiable change to code
    that is not broken.
    """
    random.seed(0)
    torch.manual_seed(0)
    print('running self-test...')
    failed = []
    for name, fn in CHECKS:
        try:
            fn()
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit) as e:      # noqa: BLE001
            # SystemExit is caught too: a check that drives a CLI can exit
            # instead of raising, and that must not end the whole run.
            failed.append((name, e))
            print(f'  !! FAILED: {name} -- {type(e).__name__}: {e}')
            traceback.print_exc()
    if failed:
        print(f'\n{len(failed)} of {len(CHECKS)} CHECKS FAILED: '
              + ', '.join(n for n, _ in failed))
        print('(the FIRST one is the one to fix; the rest may be consequences)')
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
