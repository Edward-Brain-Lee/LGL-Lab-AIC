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
* ``--image-size``: that it reaches ``open_clip`` as ``force_image_size``, that
  224 is never passed (so the earlier runs stay bit-identical), that the eval
  transform's geometry follows the requested size, and that the patch grid is a
  whole number of 32px patches (336 is not -- it is a patch-14 figure);
* optimiser groups: ``head.logit_scale`` is *not* weight-decayed, because the
  tracker's thresholds are absolute probabilities;
* the LR schedule: it ramps linearly to ``--lr`` and only then anneals, and
  ``--lr-warmup-epochs 0`` still reproduces the old peak-first schedule;
* augmentation reproducibility: item ``i``'s two augmented views are a pure
  function of ``(seed, index, view)``, so they are identical under
  ``--workers 0/1/2/3`` -- otherwise the submitted code cannot reproduce the
  submitted score;
* a full 4-epoch training run and a full inference run on a throw-away
  4-class image set, including the checkpoint round-trip, the inference-only
  snapshots, the CSV format, the multi-tau logit adjustment and the TTA view
  averaging (including that a single view reproduces the plain run exactly).
"""
import csv
import math
import random
import shutil
import sys
import tempfile
import types
from pathlib import Path

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
    returns ``(B, dim)`` embeddings, like ``visual(x)`` does for CLIP."""

    def __init__(self, dim=DIM, in_dim=12):
        super().__init__()
        self.output_dim = dim
        self.stem = nn.Conv2d(3, in_dim, kernel_size=32, stride=32)   # patch embed
        self.proj = nn.Linear(in_dim, dim)                            # 1
        self.block = _StubBlock(dim)                                  # 3 more

    def forward(self, x):
        if x.dim() == 4:                    # (B, 3, H, W) -> (B, in_dim)
            x = self.stem(x).mean(dim=(2, 3))
        x = self.proj(x).unsqueeze(1)       # (B, 1, dim): one token
        return self.block(x).squeeze(1)


class _StubCLIP(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.visual = _StubVisual(dim)

    def encode_image(self, x):
        return self.visual(x)


_stub = types.ModuleType('open_clip')
#: every create_model call, so tests can check what train.build_clip passed
#: through (notably: force_image_size must reach open_clip, or a run silently
#: trains at 224 while believing it is at 336)
_STUB_CALLS = []


def _stub_create_model(name, pretrained=None, **kw):
    _STUB_CALLS.append((name, pretrained, kw))
    return _StubCLIP()


_stub.create_model = _stub_create_model
sys.modules['open_clip'] = _stub

import analyze  # noqa: E402
import infer  # noqa: E402
import losses  # noqa: E402
import noise  # noqa: E402
import train  # noqa: E402


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
    assert int(st['relabel']) == 2, 'a vetoed sample should land in the disagreement branch'
    assert (tk.label == y).all(), 'the veto relabelled a sample; it must only demote'
    tg = tk.target(torch.arange(n), y)
    onehot = F.one_hot(y, C).float()
    assert torch.allclose(tg[[0, 1, 3, 5]], onehot[[0, 1, 3, 5]]), 'the veto changed a target'
    assert torch.allclose(tg[[2, 4]], onehot[[2, 4]]), \
        'the veto changed a target: the teacher pick IS the given label, so mixing it in is a no-op'
    assert (tk.weight[[2, 4]] == 0.5).all() and (tk.weight[[0, 1, 3, 5]] == 1).all(), tk.weight

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
    off = train.parse_args(['--epochs', str(TOT), '--lr-warmup-epochs', '0'])
    v = lrs(off)
    assert abs(v[0] - off.lr) < 1e-12, \
        f'--lr-warmup-epochs 0 must start at the peak: {v[0]} vs {off.lr}'
    assert all(x >= y - 1e-12 for x, y in zip(v, v[1:])), \
        f'--lr-warmup-epochs 0 must be monotonically decreasing: {v}'

    a = train.parse_args(['--epochs', str(TOT), '--lr-warmup-epochs', str(WARM)])
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
    """``--image-size`` must reach open_clip, and the eval transform must follow.

    The failure this guards against is silent: if ``force_image_size`` were
    dropped, every script would still run and still print 336 in its log, while
    actually forwarding 224-sized pixels.
    """
    _STUB_CALLS.clear()
    train.build_clip('ViT-B-32-quickgelu', 'openai', 224)
    train.build_clip('ViT-B-32-quickgelu', 'openai', 320)
    assert _STUB_CALLS[0][2] == {}, \
        f'224 must not pass force_image_size (earlier runs stay bit-identical): {_STUB_CALLS[0]}'
    assert _STUB_CALLS[1][2] == {'force_image_size': 320}, _STUB_CALLS[1]

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
    assert 'image_size' in vars(train.parse_args([])), \
        '--image-size is missing from the args, so it will not be checkpointed'
    print('  image-size ok')


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
        # (a) a view list that collapses to one view must reproduce the plain
        #     run byte for byte: no double counting, and the averaged feature of
        #     a single view is that view
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'tta_dup.csv'),
                                     '--tta-sizes', '224']))
        assert (tmp / 'tta_dup.csv').read_bytes() == csv_path.read_bytes(), \
            '--tta-sizes 224 (the native size) changed the output: the view was counted twice'
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'tta_logit1.csv'),
                                     '--tta-agg', 'logit']))
        assert (tmp / 'tta_logit1.csv').read_bytes() == csv_path.read_bytes(), \
            'single-view logit averaging is not the identity'

        # (b) a genuine multi-size, multi-view run must survive: a model built
        #     per resolution, features averaged, the head applied once
        _STUB_CALLS.clear()
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'tta.csv'),
                                     '--tta-sizes', '288', '--tta-flip']))
        assert any(kw.get('force_image_size') == 288 for _, _, kw in _STUB_CALLS), \
            'the extra TTA resolution never reached open_clip'
        tta_rows = list(csv.reader((tmp / 'tta.csv').open()))
        assert len(tta_rows) == 32, f'TTA changed the row count: {len(tta_rows)}'
        assert [r[0] for r in tta_rows] == [r[0] for r in rows], 'TTA reordered the rows'
        assert all(len(r) == 2 and r[1].isdigit() for r in tta_rows), tta_rows[:3]
        print(f'  tta ok ({len(tta_rows)} rows from 4 views)')

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


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    print('running self-test...')
    check_losses()
    check_tracker()
    check_judge()
    check_targets()
    check_proto()
    check_lora()
    check_param_groups()
    check_lr_warmup()
    check_image_size()
    check_reproducible_augment()
    check_end_to_end()
    print('ALL CHECKS PASSED')
