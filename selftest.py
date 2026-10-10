"""Fast self-check: run this before spending GPU hours.

    python selftest.py

It needs no dataset, no GPU and no CLIP weights -- ``open_clip`` is replaced by
a tiny stub exposing the same interface.  Covered:

* robust losses: shapes, GCE bound, NCE tangency/continuity at ``p = k``, and
  the soft-target form reducing exactly to the index form on a one-hot target;
* label-trust tracker: agree / pseudo-label / untrusted branches, rejection cap,
  statistics that partition the set, EMA update;
* targets: label smoothing, and the tracker's mixed pseudo-label (the given
  label keeps ``1 - relabel_mix`` of the mass -- not a hard overwrite);
* prototype head: bootstrap, EMA update, classes never seen before;
* LoRA: zero-init identity, disable == frozen CLIP, checkpoint filtering;
* a full 4-epoch training run and a full inference run on a throw-away
  4-class image set, including the checkpoint round-trip, the inference-only
  snapshots, the CSV format and the multi-tau logit adjustment.
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
        # Enough of a positional grid for resize_positional_embedding /
        # --train-pos-embed to be exercised here.  The stub's forward ignores it,
        # exactly as a real tower would ignore a grid whose size it disagrees with
        # -- which is why open_clip raises instead (see probe_resolution.py).
        self.patch_size = (32, 32)
        self.image_size = (224, 224)
        self.grid_size = (7, 7)
        self.positional_embedding = nn.Parameter(torch.randn(1 + 7 * 7, dim),
                                                 requires_grad=False)

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
_stub.create_model = lambda name, pretrained=None: _StubCLIP()
sys.modules['open_clip'] = _stub

import analyze  # noqa: E402
import infer  # noqa: E402
import losses  # noqa: E402
import noise  # noqa: E402
import train  # noqa: E402
import checkpoint_search  # noqa: E402
import json
from semantic_regularization import semantic_cost, expected_semantic_cost
from build_semantic_reference import split_fingerprint


def check_class_regularisation():
    counts = [4, 40, 200, 200]
    weights = train.make_class_loss_weights(counts, .25, .75, 1.5)
    assert .75 <= weights.min() <= weights.max() <= 1.5
    assert weights[0] > weights[-1]
    assert torch.equal(train.make_class_loss_weights(counts), torch.ones(4))
    margins = train.class_margins(counts, .5, .25)
    assert margins[0] == .5 and margins[0] > margins[-1]
    out = torch.tensor([[1., 2., 0.], [3., 0., 1.]], requires_grad=True)
    loss = train.class_margin_loss(out, torch.tensor([0, 0]), torch.tensor([.5, .2]))
    assert torch.allclose(loss, torch.tensor([1.5, 0.]))
    loss.sum().backward()
    # Active margin raises the given logit and lowers its strongest rival.
    assert out.grad[0, 0] < 0 and out.grad[0, 1] > 0
    assert torch.equal(out.grad[1], torch.zeros(3))
    assert torch.equal(train.class_margin_loss(out.detach(), torch.tensor([0, 0])), torch.zeros(2))
    print('  class weights / bounded margin gradients ok')


def check_fixed_anchor_and_consistency():
    class SensitiveVisual(_StubVisual):
        def forward(self, x):
            z = super().forward(x)
            return z + self.positional_embedding[0][None, :]

    clip = _StubCLIP()
    clip.visual = SensitiveVisual()
    net = train.Net(clip, 4, 4, 'all')
    train.enable_pos_embed_training(net)
    train.enable_ln_training(net)
    net.freeze_anchor_reference()
    x = torch.randn(3, 12)
    reference = net.anchor_feat(x)
    with torch.no_grad():
        net.clip.visual.positional_embedding.add_(torch.randn_like(net.clip.visual.positional_embedding))
        for mod in net.clip.visual.modules():
            if isinstance(mod, nn.LayerNorm):
                mod.bias.add_(.3)
            if isinstance(mod, train.LoRALinear):
                mod.B.normal_(std=.1)
    current_pe = net.clip.visual.positional_embedding.detach().clone()
    assert torch.allclose(net.anchor_feat(x), reference, atol=1e-6)
    assert torch.equal(net.clip.visual.positional_embedding, current_pe)
    assert not any(k.startswith('_anchor_reference') for k in net.trainable_state_dict())
    net(x).sum().backward()
    assert net.clip.visual.positional_embedding.grad is not None
    assert net.clip.visual.positional_embedding.grad.norm() > 0
    assert net.clip.visual.training, 'anchor changed student training mode'
    z1, z2 = torch.randn(3, 32, requires_grad=True), torch.randn(3, 32, requires_grad=True)
    u, v = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    cos = train.view_consistency(u, v, 'cosine')
    assert torch.allclose(cos, train.view_consistency(v, u, 'cosine'))
    assert torch.allclose(train.view_consistency(u, v, 'mse') * 16, cos, atol=1e-6)
    cos.backward()
    assert z1.grad.norm() > 0 and z2.grad.norm() > 0
    print('  fixed PE/LayerNorm anchor, restoration and symmetric consistency gradients ok')


def check_semantic_regularisation():
    prototypes = torch.tensor([[1., 0.], [.8, .6], [-1., 0.]])
    cost = semantic_cost(prototypes)
    assert cost[0, 0] == 0 and cost[0, 1] < cost[0, 2]
    close = torch.tensor([[0., 5., 0.]])
    far = torch.tensor([[0., 0., 5.]], requires_grad=True)
    target = torch.tensor([0])
    assert expected_semantic_cost(close, target, cost) < expected_semantic_cost(far, target, cost)
    expected_semantic_cost(far, target, cost).sum().backward()
    assert far.grad[0, 2] > 0 and far.grad[0, 0] < 0
    soft = F.one_hot(target, 3).float()
    assert torch.equal(expected_semantic_cost(close, target, cost), expected_semantic_cost(close, soft, cost))
    print('  semantic costs: nearby/far mistakes, soft targets and gradients ok')


def check_search_optimizer_groups():
    net = train.Net(_StubCLIP(), 4, 4, 'all')
    train.enable_pos_embed_training(net)
    train.enable_ln_training(net)
    groups = train.optimizer_groups(net, .05, 'no_decay_small')
    decay = {id(p) for p in groups[0]['params']}
    no_decay = {id(p) for p in groups[1]['params']}
    assert not decay & no_decay
    assert decay | no_decay == {id(p) for p in net.parameters() if p.requires_grad}
    assert id(net.head.logit_scale) in no_decay
    assert id(net.clip.visual.positional_embedding) in no_decay
    assert id(net.head.weight) in decay
    print('  optimizer groups partition trainables and preserve official frozen parameters ok')


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


def check_lora_qkv_and_ln():
    """``--lora-qkv`` / ``--lora-alpha`` / ``--train-ln``: widening the adapter sweep.

    ``--lora-qkv`` exists because ``nn.MultiheadAttention`` keeps query/key/value in a
    bare ``nn.Parameter`` (``in_proj_weight``), which an ``nn.Linear`` walk cannot see
    -- so the projection that shapes q/k/v was frozen in every run before this flag.
    A flag that "works" by quietly doing nothing costs a 3.5-hour run to discover, so
    what is asserted here is numerics, then the module tree, then the checkpoint.
    """
    torch.manual_seed(0)

    # 1) the replacement must be numerically identical to what it replaces, and the
    #    frozen qkv weight must have been copied bit-exactly -- otherwise the adapted
    #    model is no longer the model the official OpenAI weights describe
    mha = nn.MultiheadAttention(DIM, 4, batch_first=True)
    x = torch.randn(3, 5, DIM)
    with torch.no_grad():
        ref = mha(x, x, x, need_weights=False)[0]
    wrapped = train.LoRAQKVAttention(mha)
    # LoRAQKVAttention only *rebuilds* the fused projection as a plain nn.Linear;
    # it is `add_lora` that wraps it into a LoRALinear (train.py:353).  Without this
    # the assertions below reach for `.base` / `.A` on an nn.Linear and die with an
    # AttributeError that looks like a train.py bug rather than a test bug.
    assert train.add_lora(wrapped, rank=4, alpha=8) == 2, \
        'add_lora must wrap exactly the rebuilt qkv and the existing out_proj'
    wrapped.eval()
    with torch.no_grad():
        got = wrapped(x, x, x, need_weights=False)[0]
    assert torch.allclose(ref, got, atol=1e-6), \
        'LoRAQKVAttention is not identical to the attention it replaces at init'
    with torch.no_grad():
        assert torch.equal(wrapped.qkv.base.weight, mha.in_proj_weight), \
            'the frozen qkv weight was not copied bit-exactly'
        assert torch.equal(wrapped.qkv.base.bias, mha.in_proj_bias)
        # ... and the update must actually reach the qkv path, not merely exist
        wrapped.qkv.A.normal_()
        wrapped.qkv.B.normal_()
        moved = wrapped(x, x, x, need_weights=False)[0]
    assert not torch.allclose(ref, moved, atol=1e-4), \
        'a non-zero qkv adapter changed nothing -- the merged weight is not being used'

    # 2) the tree --lora-qkv builds, and the one it must leave alone
    base = train.Net(_StubCLIP(), 4, 4, 'all')
    assert base.n_lora == 4, f'the default walk changed: {base.n_lora} layers'
    q = train.Net(_StubCLIP(), 4, 4, 'all', lora_qkv=True)
    assert q.n_lora == base.n_lora + 1, \
        f'--lora-qkv wrapped {q.n_lora - base.n_lora} layers, want 1 (the qkv projection)'
    assert not any(isinstance(m, nn.MultiheadAttention) for m in q.clip.visual.modules()), \
        'the attention was not replaced, so qkv is still a bare Parameter'
    assert isinstance(q.clip.visual.block.attn.qkv, train.LoRALinear)
    assert isinstance(q.clip.visual.block.attn.out_proj, train.LoRALinear), \
        '--lora-qkv dropped the out_proj adapter the old walk used to add'
    # zero-init B means the flag must be an exact no-op at step 0: if it is not, every
    # number in the run it is compared against is off by the size of the initial update
    clip_q = _StubCLIP()
    net_q = train.Net(clip_q, 4, 4, 'all', lora_qkv=True)
    xq = torch.randn(3, 12)
    with torch.no_grad():
        frozen_q = F.normalize(clip_q.visual(xq), dim=-1)
        assert torch.allclose(net_q(xq), net_q.head(frozen_q), atol=1e-6), \
            '--lora-qkv is not identity at init'
    # the adapters must reach the file, and the frozen qkv weight must not
    sd = q.trainable_state_dict()
    assert any(k.endswith('attn.qkv.A') for k in sd), 'the qkv adapter is not checkpointed'
    assert any(k.endswith('attn.qkv.B') for k in sd)
    assert not any(k.endswith('qkv.base.weight') for k in sd), \
        'the frozen qkv weight leaked into the checkpoint'

    # 3) alpha sets the effective step size; 0 keeps the historical 2*rank (scale 2),
    #    which is the only reason a rank sweep compares like for like
    assert train.Net(_StubCLIP(), 4, 4, 'all').lora_alpha == 8, \
        'alpha 0 must mean 2*rank'
    n16 = train.Net(_StubCLIP(), 4, 4, 'all', lora_alpha=16)
    assert n16.lora_alpha == 16, '--lora-alpha did not reach the adapter'
    assert n16.clip.visual.block.attn.out_proj.scale == 16 / 4, \
        'alpha/rank was not propagated to the adapter scale'

    # 4) train-ln: the norms become trainable, nothing else does
    net = train.Net(_StubCLIP(), 4, 4, 'all')
    lns = [m for m in net.clip.visual.modules() if isinstance(m, nn.LayerNorm)]
    assert lns, 'the stub lost its LayerNorm: this check is no longer testing anything'
    assert not any(p.requires_grad for m in lns for p in m.parameters()), \
        'the stub LayerNorm should start frozen'
    n_ln = train.enable_ln_training(net)
    assert n_ln == sum(p.numel() for m in lns for p in m.parameters()) > 0
    assert all(p.requires_grad for m in lns for p in m.parameters())
    assert any(k.endswith('block.norm.weight') for k in net.trainable_state_dict()), \
        'the trained norm is not checkpointed'
    leaked = [k for k, p in net.named_parameters()
              if p.requires_grad and not (k.endswith(('.A', '.B'))
                                          or k.startswith(('head.', 'proto.'))
                                          or '.norm' in k or '.ln_' in k)]
    assert not leaked, f'--train-ln unlocked more than the norms: {leaked[:3]}'
    print('  lora-qkv / lora-alpha / train-ln ok')


def check_attn_temp():
    """``--attn-temp``: a per-head softmax temperature that starts as an exact no-op.

    The whole value of this flag is that step 0 is *bit-for-bit* the run without it,
    so "is it a clean single variable" is the first thing asserted.  The second is
    that ``tau`` really is a softmax temperature -- i.e. that scaling a head's query
    rows by ``tau_h`` is equivalent to scaling that head's logits, proved against an
    ``nn.MultiheadAttention`` whose ``in_proj_weight`` was edited by hand.  The third
    is the failure this project cares about most: ``anchor_feat`` means "the frozen
    official tower", so a live temperature there would make the anchor drift silently
    and poison every statistic derived from it.
    """
    torch.manual_seed(0)

    # 1) the flag must be a single variable at step 0: the exact same tensor, not merely
    #    a close one.  `got` and `got_off` go through the identical functional path and
    #    the identical frozen weights, so any difference at all would be tau's doing
    #    (`rho == 0` -> `tau == 1` -> `w * 1.0`, which is exact in floating point).
    mha = nn.MultiheadAttention(DIM, 4, batch_first=True)
    x = torch.randn(3, 5, DIM)
    with torch.no_grad():
        ref = mha(x, x, x, need_weights=False)[0]
    wrapped = train.LoRAQKVAttention(mha, attn_temp=True)
    assert train.add_lora(wrapped, rank=4, alpha=8) == 2, \
        'add_lora must wrap exactly the rebuilt qkv and the existing out_proj'
    wrapped.eval()
    plain = train.LoRAQKVAttention(mha, attn_temp=False)
    train.add_lora(plain, rank=4, alpha=8)
    plain.eval()
    with torch.no_grad():
        got = wrapped(x, x, x, need_weights=False)[0]
        got_off = plain(x, x, x, need_weights=False)[0]
    assert torch.allclose(ref, got, atol=1e-6), \
        'LoRAQKVAttention(attn_temp=True) is not identical to the attention it replaces at init'
    assert torch.equal(got, got_off), \
        '--attn-temp is not bit-for-bit the run without it at step 0'
    assert wrapped.attn_rho.shape == (4,), \
        f'rho must be one scalar per head, got {tuple(wrapped.attn_rho.shape)}'

    # 2) tau must act as a per-head temperature on the logits.  tau_h multiplies head
    #    h's query rows of the fused projection; the functional path then applies one
    #    global 1/sqrt(head_dim), so the logits scale by exactly tau_h.  Checking it
    #    against an MHA with a hand-scaled in_proj_weight is what makes this a test of
    #    the semantics rather than of the plumbing.  (atol 1e-5 not 1e-6: the plain MHA
    #    may take torch's fused SDPA path, which differs from the functional one by
    #    rounding alone -- see the atol=1e-6 on `ref` above for the same reason.)
    taus = torch.tensor([0.5, 1.0, 2.0, 3.0])
    head_dim = DIM // 4
    with torch.no_grad():
        w = wrapped.qkv.weight.clone()
        manual = w.clone()
        manual[:DIM] = (w[:DIM].view(4, head_dim, DIM) * taus[:, None, None]).reshape(DIM, DIM)
        wrapped.attn_rho.copy_(taus.log())
        got_tau = wrapped(x, x, x, need_weights=False)[0]

        ref_mha = nn.MultiheadAttention(DIM, 4, batch_first=True)
        ref_mha.in_proj_weight.copy_(manual)
        ref_mha.in_proj_bias.copy_(mha.in_proj_bias)
        ref_mha.out_proj.weight.copy_(mha.out_proj.weight)
        ref_mha.out_proj.bias.copy_(mha.out_proj.bias)
        want = ref_mha(x, x, x, need_weights=False)[0]
    assert not torch.allclose(ref, got_tau, atol=1e-4), 'a non-unit tau changed nothing'
    assert torch.allclose(got_tau, want, atol=1e-5), \
        'tau is not a per-head scaling of the query rows'

    # 3) the row scale itself: tau only in the query block, exactly 1.0 over k/v
    scale = wrapped._temp_scale(torch.float32)
    assert scale.shape == (3 * DIM, 1)
    assert torch.all(torch.equal(scale[DIM:], torch.ones(2 * DIM, 1))), \
        'the k/v rows must keep a factor of exactly 1.0, they are not touched'
    assert torch.allclose(scale[:DIM, 0], taus.repeat_interleave(head_dim)), \
        'tau was not applied per head across the query rows'

    # 4) --attn-temp needs the rebuilt block even without --lora-qkv, rho must survive
    #    Net's blanket freeze of the backbone, and nothing may appear when it is off
    off = train.Net(_StubCLIP(), 4, 4, 'all')
    assert off.attn_temp is False and off.attn_temperature() is None
    assert not any(isinstance(m, train.LoRAQKVAttention) for m in off.clip.visual.modules())
    assert not any(k.endswith('attn_rho') for k in off.state_dict())

    clip_t = _StubCLIP()
    net_t = train.Net(clip_t, 4, 4, 'all', attn_temp=True)
    assert not any(isinstance(m, nn.MultiheadAttention) for m in net_t.clip.visual.modules()), \
        '--attn-temp did not replace the attention block, so it has nowhere to live'
    assert net_t.n_lora == off.n_lora + 1, \
        f'--attn-temp wrapped {net_t.n_lora - off.n_lora} layers, want 1 (the qkv projection)'
    attn = net_t.clip.visual.block.attn
    assert attn.attn_rho.requires_grad, \
        'Net froze rho along with the backbone -- the flag would silently do nothing'
    assert net_t.attn_temperature().numel() == 4, 'one temperature per head'

    # 5) the anchor is the frozen tower.  A tau only matters when there is more than
    #    one key to reweight -- the stub tower emits a single token, whose softmax is
    #    identically 1 regardless of temperature -- so the live-vs-frozen comparison is
    #    made on the attention module directly, and only the invariance is asserted at
    #    the model level.  If the temperature leaked into anchor_feat, prototype
    #    bootstrap and the junk filter would both read features that drift with rho.
    h = torch.randn(2, 5, DIM)
    xt = torch.randn(3, 12)
    with torch.no_grad():
        attn.attn_rho.zero_()
        eq = attn(h, h, h, need_weights=False)[0]
        base_feat = F.normalize(clip_t.visual(xt), dim=-1)
        attn.attn_rho.fill_(1.0)                      # tau = e for every head
        live = attn(h, h, h, need_weights=False)[0]
        assert not torch.allclose(live, eq, atol=1e-5), \
            'tau = e changed nothing: the temperature never reaches the forward pass'
        assert torch.allclose(net_t.anchor_feat(xt), base_feat, atol=1e-6), \
            'anchor_feat is no longer the frozen tower: the temperature leaked into it'
        with train.lora_disabled(net_t.clip.visual):
            frozen = attn(h, h, h, need_weights=False)[0]
        assert torch.allclose(frozen, eq, atol=1e-6), \
            'lora_disabled did not restore the frozen attention'
        assert attn.temp_enabled is True
        with train.lora_disabled(net_t.clip.visual):
            assert attn.temp_enabled is False, 'lora_disabled left the temperature on'
        assert attn.temp_enabled is True, 'lora_disabled did not restore the temperature'

    # 6) checkpoint round trip, and the *metadata* path the flag also travels on:
    #    --resume reads it from the raw dict, infer.py from the same key, and thin()
    #    silently drops anything missing from SNAPSHOT_KEYS -- which would turn every
    #    --attn-temp checkpoint back into a plain one with nothing in the log.
    assert 'attn_temp' in train.SNAPSHOT_KEYS, \
        'thin() would drop attn_temp: infer.py would rebuild the model without rho'
    assert net_t.attn_temp is True and off.attn_temp is False
    sd = net_t.trainable_state_dict()
    assert any(k.endswith('attn_rho') for k in sd), 'rho is not checkpointed'
    fresh = train.Net(_StubCLIP(), 4, 4, 'all', attn_temp=True)
    fresh.load_state_dict(sd, strict=False)
    assert torch.equal(fresh.clip.visual.block.attn.attn_rho, attn.attn_rho), \
        'rho did not survive the checkpoint round trip'
    print('  attn-temp ok')


def check_junk_filter():
    """``--junk-filter``: frozen-tower screening of 杂图 and near-duplicates.

    Three things have to hold, and none of them is visible from a log line: the
    robust centroids must actually drop the mislabelled samples (otherwise the
    margin rule is comparing against a centroid the noise built), the cap must bound
    the filter rather than decorate it, and the frozen pass must attach row ``i`` to
    training index ``i`` -- a permutation there would reweight the *wrong* images and
    every downstream number would still look perfectly plausible.
    """
    torch.manual_seed(0)

    # 1) robust centroids: a quarter of class 0's samples carry class 1's label
    nclass, per, D = 4, 20, 8
    truth = F.normalize(torch.randn(nclass, D), dim=-1)
    z = truth.repeat_interleave(per, 0) + 0.01 * torch.randn(nclass * per, D)
    y = torch.arange(nclass).repeat_interleave(per)
    y[:4] = 1                                     # 4 of class 0's samples say "class 1"
    means, present, keep = train.robust_centroids(z, y.tolist(), nclass, rounds=2, verbose=False)
    assert bool(present.all()), 'a class with 20 samples came back absent'
    naive = F.normalize(torch.stack([z[y == c].mean(0) for c in range(nclass)]), dim=-1)
    got = float((F.normalize(means, dim=-1) * truth).sum(1).mean())
    assert got > float((naive * truth).sum(1).mean()), \
        'the robust round did not improve the centroids over the plain folder mean'
    assert not bool(keep[:4].any()), 'the mislabelled samples survived the centroid round'
    assert int(keep.sum()) == nclass * per - 4, f'{int(keep.sum())} kept, want {nclass * per - 4}'

    # ... and a class whose samples are *all* rejected must not vanish from the
    # decision space: an absent class has a zero centroid, and a zero centroid is a
    # *neutral* competitor, which beats a genuinely anti-correlated real class
    z_e = z.clone()
    z_e[60:80] = truth[0] + 0.01 * torch.randn(per, D)   # class 3's folder is all class 0
    means_e, present_e, keep_e = train.robust_centroids(z_e, y.tolist(), nclass, rounds=2,
                                                       verbose=False)
    assert bool(present_e[3]), 'a class whose samples were all rejected vanished entirely'
    assert bool(keep_e[60:80].all()), 'the never-empty rule did not keep the rejected class'

    # 2) the margin rule flags exactly the samples a different centroid describes
    clean = truth.repeat_interleave(10, 0)        # sits exactly on its own centroid
    junk = truth[[1, 2, 3]]                       # labelled 0/1/2, is really 1/2/3
    zz = torch.cat([clean, junk])
    yy = torch.arange(nclass).repeat_interleave(10).tolist() + [0, 1, 2]
    present = torch.ones(nclass, dtype=torch.bool)
    w, rep = train.junk_weights(zz, yy, truth, present, mode='margin', max_frac=1.0,
                                floor=0.2, verbose=False)
    assert rep['judged'] == len(yy), 'every class has a centroid, so nothing may be unjudged'
    assert rep['flagged'] == 3 and rep['margin_pos'] == 3, rep
    assert bool((w[-3:] == 0.2).all()) and bool((w[:-3] == 1.0).all()), \
        'the margin rule did not flag exactly the three cross-class images'

    # 3) the cap is the control: with max_frac below the offender count, only the
    #    most extreme are touched and the run says so
    w_cap, rep_cap = train.junk_weights(zz, yy, truth, present, mode='margin',
                                        max_frac=0.05, floor=0.2, verbose=False)
    assert rep_cap['flagged'] == int(0.05 * len(yy)) == 2, rep_cap
    assert rep_cap['cap_bound'] is True, 'the cap bound and did not report it'
    assert int((w_cap < 1).sum()) == 2
    assert float(w_cap.min()) == 0.2, 'a flagged sample did not get --junk-floor'
    assert train.junk_weights(zz, yy, truth, present, mode='margin', max_frac=0.0,
                              floor=0.2, verbose=False)[1]['flagged'] == 0, \
        'max_frac 0 must be a no-op'

    # 4) dedup keeps the first occurrence of a cluster, and only the first
    zd = torch.randn(10, D)
    zd[7] = zd[2]
    wd, rep_d = train.junk_weights(zd, [0] * 10, None, None, mode='dedup',
                                   dedup_tau=0.98, max_frac=1.0, floor=0.2, verbose=False)
    assert rep_d['dup'] == 1, rep_d
    assert float(wd[7]) == 0.2 and float(wd[2]) == 1.0, 'dedup did not keep the first copy'
    assert rep_d['dup_cross'] == 0.0
    # a duplicate outranks a margin flag when the cap binds.  This is arithmetic, not
    # a preference: the margin term is normalised into [0, 1] and a duplicate adds 2,
    # so no margin score can ever outrank one.
    zc = torch.cat([zd, zd[2:3]])                 # index 10 duplicates index 2 as well
    yc = [0] * 10 + [1]
    wc, rep_c = train.junk_weights(zc, yc, truth, present, mode='both', dedup_tau=0.98,
                                   max_frac=0.1, floor=0.2, verbose=False)
    assert rep_c['dup'] == 2, rep_c                 # indices 7 and 10
    assert rep_c['flagged'] == 1 and rep_c['cap_bound'] is True, rep_c
    assert int((wc < 1).nonzero().flatten()[0]) in (7, 10), \
        'the cap preferred a margin flag over a duplicate'

    # 5) the frozen pass must be a full-coverage, index-aligned sweep
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / 'train'
        random.seed(0)
        for c in range(4):
            d = root / f'{c:04d}'
            d.mkdir(parents=True)
            for i in range(5):
                px = bytes(random.randrange(256) for _ in range(48 * 48 * 3))
                Image.frombytes('RGB', (48, 48), px).save(d / f'img_{c}_{i}.jpg')
        a = train.parse_args(['--data', str(root), '--workers', '0', '--amp', 'none'])
        ds = train.ImageFolderNoisy(a.data, train.eval_transform(a), False, a.val_ratio,
                                    a.seed, 'train', a.img_size)
        net = train.Net(_StubCLIP(), 4, 4, 'all')
        feat, seen = train.frozen_feature_pass(net, ds, torch.device('cpu'), None, False,
                                               batch_size=4, workers=0, every=0)
        assert bool(seen.all()), 'the frozen pass left samples unjudged'
        assert feat.shape == (len(ds), net.head.weight.shape[-1]), feat.shape
        with torch.no_grad():
            ref = net.anchor_feat(torch.stack([ds[i][0] for i in range(len(ds))]))
        assert torch.allclose(feat, ref, atol=1e-5), \
            'row i of the frozen pass is not training index i'

        # 6) end to end: the filter runs, travels in the checkpoint, and refuses to be
        #    resumed with a different setting
        out = tmp / 'out'
        run = ['--data', str(root), '--out', str(out), '--epochs', '2', '--warmup-epochs', '1',
               '--batch-size', '4', '--workers', '0', '--val-ratio', '0.25', '--seed', '0',
               '--amp', 'none', '--junk-filter', 'both', '--junk-max-frac', '0.2',
               '--junk-floor', '0.1']
        train.main(train.parse_args(run))
        ck = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        assert ck['junk_w'] is not None, 'the junk weights are not in the checkpoint'
        assert ck['junk_report'] is not None and ck['junk_report']['mode'] == 'both'
        n_tr = len(train.build_datasets(train.parse_args(run))[0])
        assert ck['junk_w'].shape == (n_tr,), (ck['junk_w'].shape, n_tr)
        assert float(ck['junk_w'].min()) == 0.1 and float(ck['junk_w'].max()) == 1.0
        # resuming the filterless way must stop the run rather than silently drop them
        args = train.parse_args(['--data', str(root), '--out', str(tmp / 'out2'),
                                 '--resume', str(out / 'last.pt')])
        try:
            train.main(args)
        except SystemExit:
            pass
        else:
            raise AssertionError('--resume dropped the junk weights without a word')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('  junk-filter ok')


def check_pos_embed_resize():
    """--img-size: the shape maths a two-hour run depends on, checked in a second.

    open_clip does not resample the positional grid itself (``probe_resolution.py``
    shows every non-224 size raising a shape error), so this helper has to get the
    token count, the CLS token and the no-op case exactly right -- a 288 run with a
    broken grid would train to a quietly worse number.
    """
    tower = types.SimpleNamespace(positional_embedding=torch.randn(1 + 7 * 7, 768),
                                  image_size=(224, 224), grid_size=(7, 7),
                                  patch_size=(32, 32))

    assert train.resize_positional_embedding(tower, 224) == 7, '224 must keep the 7x7 grid'
    assert tuple(tower.positional_embedding.shape) == (50, 768), '224 changed the shape'

    assert train.resize_positional_embedding(tower, 288) == 9, '288 needs a 9x9 grid'
    assert tuple(tower.positional_embedding.shape) == (82, 768), \
        f'288 has the wrong token count: {tuple(tower.positional_embedding.shape)}'
    assert tower.image_size == (288, 288), 'image_size bookkeeping not updated'
    assert tower.grid_size == (9, 9), 'grid_size bookkeeping not updated'

    # the CLS token is not part of the patch grid and must come through untouched
    tower.positional_embedding = torch.randn(1 + 7 * 7, 768)
    cls_before = tower.positional_embedding[0].clone()
    train.resize_positional_embedding(tower, 320)
    assert torch.equal(tower.positional_embedding[0], cls_before), \
        'the CLS token was resampled along with the patch grid'

    # a ramp laid out row-major must stay a ramp: this catches a grid that came back
    # transposed or inverted, which a pure shape check would happily accept
    ramp = torch.linspace(-1, 1, 49).view(49, 1).expand(49, 768).clone()
    tower.positional_embedding = torch.cat([torch.zeros(1, 768), ramp]).clone()
    train.resize_positional_embedding(tower, 288)
    got = tower.positional_embedding[1:]
    assert got.abs().max() < 2.0, \
        f'resampled grid left its range: max |v| = {float(got.abs().max()):.2f}'
    assert float(got[0].mean()) < float(got[-1].mean()), \
        'the resampled grid came back inverted'

    # the production tower holds an nn.Parameter, not a plain tensor, and that
    # branch replaces the attribute rather than copying into it -- the shape change
    # is exactly what copy_ refuses.  requires_grad has to survive the swap:
    # checkpoints filter trainable tensors on that flag, so flipping it would leak
    # the 350 MB frozen backbone into every snapshot.
    mod = nn.Module()
    mod.positional_embedding = nn.Parameter(torch.randn(1 + 7 * 7, 768), requires_grad=False)
    mod.patch_size, mod.image_size, mod.grid_size = (32, 32), (224, 224), (7, 7)
    train.resize_positional_embedding(mod, 288)
    assert isinstance(mod.positional_embedding, nn.Parameter), \
        'the parameter was replaced by a plain tensor'
    assert tuple(mod.positional_embedding.shape) == (82, 768), \
        f'parameter swap gave the wrong shape: {tuple(mod.positional_embedding.shape)}'
    assert mod.positional_embedding.requires_grad is False, \
        'requires_grad flipped -- the frozen backbone would leak into checkpoints'

    # a size that is not a multiple of the patch size must be refused, not guessed
    for bad in (336, 250):
        try:
            train.resize_positional_embedding(tower, bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f'--img-size {bad} was accepted; it should be refused')

    assert train.val_resize(224) == 256, f'val_resize(224) = {train.val_resize(224)}, want 256'
    assert train.val_resize(288) == 329, f'val_resize(288) = {train.val_resize(288)}, want 329'
    print('  pos-embed resize ok')


def check_img_size_transforms():
    """The transform layer at a non-default --img-size, with no model involved.

    A 288 run is only 288 if every transform actually emits 288x288.  The old code
    hardcoded 224 in five places, and a missed one would not raise -- it would feed
    224 images to a 9x9 grid and quietly train a worse model.  Pure torchvision, so
    this runs anywhere.
    """
    for size in (224, 288):
        val_tf = transforms.Compose([
            transforms.Resize(train.val_resize(size)), transforms.CenterCrop(size),
            transforms.ToTensor()])
        got = tuple(val_tf(Image.new('RGB', (640, 480))).shape)
        assert got == (3, size, size), f'the val transform emitted {got}, want 3x{size}x{size}'

        for view in sorted(infer.TTA_VIEWS):   # every view, so adding one cannot slip past
            vtf = infer.build_view_transform(*infer.TTA_VIEWS[view], size)
            got = tuple(vtf(Image.new('RGB', (640, 480))).shape)
            assert got == (3, size, size), \
                f'TTA view "{view}" emitted {got} at --img-size {size}, want 3x{size}x{size}'
    print('  img-size transforms ok')


def check_train_pos_embed():
    """--train-pos-embed: the flag must actually reach the optimiser AND the file.

    A flag that silently did nothing would cost a 3-hour run to discover, so assert
    the two things that matter: the grid ends up in `requires_grad` (which is what
    the optimiser filters on) and in `trainable_state_dict()` (which is what the
    checkpoint writes).  The second is the one that would fail quietly -- a grid
    that trains but is never saved would be lost at inference time.
    """
    net = train.Net(_StubCLIP(), 4, 4, 'all')          # nclass 4, lora rank 4
    pe = net.clip.visual.positional_embedding
    assert not pe.requires_grad, 'the stub grid should start frozen, like the real one'

    n = train.enable_pos_embed_training(net)
    assert pe.requires_grad, 'the grid was not made trainable'
    assert n == (1 + 7 * 7) * DIM, f'unexpected grid size {n}'

    key = 'clip.visual.positional_embedding'
    assert key in net.trainable_state_dict(), \
        f'{key} did not reach trainable_state_dict -- it would be lost on save'
    # ... and the flag must not have unlocked anything else in the backbone
    trained = {k for k, v in net.state_dict().items()}
    assert 'clip.visual.positional_embedding' in trained
    frozen_leaked = [k for k, p in net.named_parameters()
                     if p.requires_grad and not (k.endswith('.A') or k.endswith('.B')
                                                 or k == key or k.startswith('head.')
                                                 or k.startswith('proto.'))]
    assert not frozen_leaked, f'--train-pos-embed unlocked more than the grid: {frozen_leaked[:3]}'
    print('  train-pos-embed ok')


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
        split_a = train.parse_args(['--data', str(root), '--seed', '1234', '--split-seed', '0',
                                    '--val-ratio', '.25'])
        _, same_va = train.build_datasets(split_a)
        assert {p for p, _ in same_va.items} == va_paths
        shuffle_a = train.parse_args(['--data', str(root), '--workers', '0', '--sampler', 'shuffle',
                                      '--batch-size', '4'])
        shuffle_loader, _ = train.build_loaders(shuffle_a, tr_ds, va_ds)
        drawn = list(shuffle_loader.sampler)
        assert len(drawn) == len(set(drawn)) == len(tr_ds)

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
                                     '--output', str(csv_path), '--workers', '0']))
        rows = list(csv.reader(csv_path.open()))
        assert len(rows) == 32, f'expected 32 predictions, got {len(rows)}'
        assert all(len(r) == 2 and r[0].endswith('.jpg') and len(r[1]) == 4 and r[1].isdigit()
                   for r in rows), f'bad CSV rows: {rows[:3]}'
        assert len({r[0] for r in rows}) == 32, 'duplicate file names in the CSV'
        print(f'  end-to-end ok ({len(rows)} predictions, sample {rows[0]})')

        # --lora-qkv / --train-ln have to survive the WHOLE train -> save -> infer
        # round trip, not just construction.  The failure this guards against is a
        # checkpoint whose extra adapters never reached the file -- or never came back
        # out of it -- which looks exactly like a normal run right up to the submission.
        out2 = tmp / 'out2'
        semantic_path = tmp / 'semantic.pt'
        torch.save(dict(provenance='official-training-only', backbone='ViT-B-32-quickgelu',
                        pretrained='openai',
                        classes=tr_ds.class_to_idx, train_fingerprint=split_fingerprint(tr_ds),
                        counts=torch.tensor([6, 6, 6, 6]),
                        prototypes=F.normalize(torch.randn(4, DIM), dim=-1)), semantic_path)
        train.main(train.parse_args(['--data', str(root), '--out', str(out2),
                                     '--epochs', '2', '--warmup-epochs', '1',
                                     '--batch-size', '4', '--workers', '0',
                                     '--val-ratio', '0.25', '--seed', '0', '--amp', 'none',
                                     '--lora-rank', '4', '--lora-alpha', '8',
                                     '--lora-qkv', '--train-ln', '--save-teacher',
                                     '--train-pos-embed', '--fixed-anchor', '--consistency-mode', 'cosine',
                                     '--class-weight-beta', '0.25', '--class-margin', '0.5',
                                     '--class-margin-power', '0.25', '--class-margin-conf', '0',
                                     '--semantic-reference', str(semantic_path), '--semantic-weight', '.05',
                                     '--semantic-conf', '0',
                                     '--select', 'val_macro']))
        ck2 = torch.load(out2 / 'best.pt', map_location='cpu', weights_only=False)
        assert ck2['lora_qkv'] is True and ck2['train_ln'] is True, sorted(ck2)
        assert ck2['lora_alpha'] == 8, ck2['lora_alpha']
        assert any(k.endswith('attn.qkv.A') for k in ck2['model']), \
            'the qkv adapter did not survive into the checkpoint'
        assert any('norm.weight' in k for k in ck2['model']), \
            'the trained LayerNorm did not survive into the checkpoint'
        assert (out2 / 'teacher_last.pt').exists()
        assert (out2 / 'val_classes_ep2.csv').exists()
        assert ck2['args']['class_weight_beta'] == .25
        assert ck2['args']['class_margin_power'] == .25
        csv_qkv = tmp / 'pred_qkv.csv'
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out2 / 'best.pt'),
                                     '--output', str(csv_qkv), '--workers', '0']))
        rows_qkv = list(csv.reader(csv_qkv.open()))
        assert len(rows_qkv) == 32, f'--lora-qkv lost rows: {len(rows_qkv)}'
        print('  lora-qkv / train-ln end-to-end ok')

        search_out = tmp / 'search'
        checkpoint_search.main(checkpoint_search.parse_args([
            '--checkpoints', str(out2 / 'best.pt'), '--data', str(root),
            '--recipes', 'plain', '--taus', '0', '.25', '--workers', '0',
            '--batch-size', '4', '--out', str(search_out)]))
        selection = json.loads((search_out / 'selection.json').read_text())
        assert selection['provenance'] == 'official-training-holdout'
        assert len(selection['candidates']) == 2
        reference = Path(selection['winner']['reference'])
        assert len(list(csv.reader(reference.open()))) == len(va_ds)
        assert {r[0] for r in csv.reader(reference.open())} == {Path(p).name for p in va_paths}
        print('  checkpoint search uses the disjoint training hold-out ok')
        original_collect, original_load = checkpoint_search.collect, analyze.load_model
        def forbidden_forward(*args, **kwargs):
            raise AssertionError('cached tau search repeated a model load or forward')
        checkpoint_search.collect = analyze.load_model = forbidden_forward
        try:
            checkpoint_search.main(checkpoint_search.parse_args([
                '--checkpoints', str(out2 / 'best.pt'), '--data', str(root),
                '--recipes', 'plain', '--taus', '0', '.5', '--workers', '0',
                '--batch-size', '4', '--out', str(search_out)]))
        finally:
            checkpoint_search.collect, analyze.load_model = original_collect, original_load
        cache_path = next((search_out / 'logits_cache').glob('*.npz'))
        import numpy as np
        with np.load(cache_path, allow_pickle=False) as cache:
            payload = {k: cache[k] for k in cache.files}
        payload['logits'] = payload['logits'].copy()
        payload['logits'][0, 0] += 1
        np.savez(cache_path, **payload)
        try:
            checkpoint_search.main(checkpoint_search.parse_args([
                '--checkpoints', str(out2 / 'best.pt'), '--data', str(root),
                '--recipes', 'plain', '--taus', '0', '--workers', '0',
                '--batch-size', '4', '--out', str(search_out)]))
        except ValueError as error:
            assert 'payload digest' in str(error)
        else:
            raise AssertionError('corrupted logits cache was reused')
        print('  cached tau search skips inference and rejects corrupt payloads ok')

        # The submission path must fail on extra trained weights, rather than
        # silently dropping a model component and producing a plausible score.
        broken = dict(ck2)
        broken['model'] = dict(ck2['model'], unexpected_trained_weight=torch.ones(1))
        broken_path = tmp / 'broken.pt'
        torch.save(broken, broken_path)
        try:
            infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(broken_path),
                                         '--output', str(tmp / 'broken.csv'), '--workers', '0']))
        except AssertionError as error:
            assert 'cannot hold' in str(error)
        else:
            raise AssertionError('inference silently dropped a trained tensor')

        # several taus must come out of ONE forward pass, one CSV each
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(tmp / 'p.csv'),
                                     '--workers', '0',
                                     '--logit-adjust', '0', '0.5', '1.0']))
        for f in ('p.csv', 'p_tau050.csv', 'p_tau100.csv'):
            assert (tmp / f).exists(), f'--logit-adjust did not write {f}'
        assert infer.adjusted_path('pred_results.csv', 0) == 'pred_results.csv'
        assert infer.adjusted_path('pred_results.csv', 1.0) == 'pred_results_tau100.csv'
        print('  logit-adjust ok')

        # --tta.  The claim that makes the flag safe on a script two people share is
        # that `plain` alone *is* the transform this file used before.  Assert that
        # on the transform, not on the CSV: this self-test stubs open_clip with a
        # freshly random backbone per create_model call, so two infer.main runs here
        # disagree even when nothing changed.  On real OpenAI weights they agree --
        # that is what the md5 comparison against a known submission is for.
        legacy_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(train.CLIP_MEAN, train.CLIP_STD)])
        probe_im = Image.new('RGB', (400, 300))
        px = probe_im.load()
        for y in range(300):                                # deliberately asymmetric
            for x in range(400):
                px[x, y] = (x % 256, y % 256, (x * y) % 256)
        t_plain = infer.build_view_transform(*infer.TTA_VIEWS['plain'], 224)(probe_im)
        t_flip = infer.build_view_transform(*infer.TTA_VIEWS['flip'], 224)(probe_im)
        assert torch.allclose(t_plain, legacy_tf(probe_im)), \
            '--tta plain is no longer the transform infer.py used before'
        assert not torch.allclose(t_plain, t_flip), 'the flip view equals the plain view'
        assert torch.allclose(t_plain, torch.flip(t_flip, dims=[2]), atol=1e-6), \
            'the flip view is not the horizontal mirror of the plain view'

        # the multi-view path must still emit exactly one valid row per test image
        csv_tta = tmp / 'pred_tta.csv'
        infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                     '--output', str(csv_tta), '--tta', '--workers', '0']))
        rows_tta = list(csv.reader(csv_tta.open()))
        assert len(rows_tta) == 32, f'--tta lost rows: {len(rows_tta)}'
        assert len({r[0] for r in rows_tta}) == 32, 'duplicate file names under --tta'
        assert all(len(r) == 2 and len(r[1]) == 4 and r[1].isdigit() for r in rows_tta), \
            f'bad CSV rows under --tta: {rows_tta[:3]}'

        # a typo must fail loudly rather than quietly falling back to one view
        accepted_typo = False
        try:
            infer.main(infer.parse_args(['--test', str(root), '--checkpoint', str(out / 'best.pt'),
                                         '--output', str(tmp / 'never.csv'), '--tta', 'Plane',
                                         '--workers', '0']))
            accepted_typo = True
        except AssertionError:
            pass
        assert not accepted_typo, '--tta accepted an unknown view name instead of failing'
        print('  tta ok')

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
    check_targets()
    check_class_regularisation()
    check_fixed_anchor_and_consistency()
    check_semantic_regularisation()
    check_search_optimizer_groups()
    check_proto()
    check_lora()
    check_lora_qkv_and_ln()
    check_attn_temp()
    check_junk_filter()
    check_pos_embed_resize()
    check_train_pos_embed()
    check_img_size_transforms()
    check_end_to_end()
    print('ALL CHECKS PASSED')
