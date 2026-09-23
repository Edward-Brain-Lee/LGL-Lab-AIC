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
_stub.create_model = lambda name, pretrained=None: _StubCLIP()
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
    check_proto()
    check_lora()
    check_end_to_end()
    print('ALL CHECKS PASSED')
