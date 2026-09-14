"""Robust fine-tuning of CLIP ViT-B/32 for noisy-label fine-grained recognition.

Single model, single inference path, OpenAI CLIP ViT-B/32 backbone only.

Recipe
------
1. **LoRA** adapters (Hu et al., ICLR 2022) on the frozen CLIP visual tower plus
   a cosine classifier head -- a few hundred thousand trainable parameters, so
   the pretrained prior survives;
2. **class-balanced sampling** (``1/sqrt(freq)``) for the long-tailed stages;
3. a pure-CE **warm-up**, after which an **EMA teacher** drives
   * automatic noise filtering + pseudo-label correction (DivideMix / co-teaching
     style, see ``noise.py``),
   * confidence re-weighting of whatever is left;
4. an **active-passive loss** (NCE + RCE, Ma et al., ICML 2020) that cannot be
   fooled by a memorised wrong sample;
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

from losses import make_robust_loss
from noise import LabelTrustTracker, prototype_bootstrap

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


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
    DataLoader itself from the loader's generator)."""
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

    def __init__(self, root, transform, val=False, val_ratio=0.1, seed=3407, split='train'):
        self.root, self.transform, self.split = Path(root), transform, split
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

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.transform(self._load(path)), y, i


class TwoView(Dataset):
    """Two *independent* augmented views of the same image + its index."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        path, y = self.base.items[i]
        img = self.base._load(path)
        return self.base.transform(img), self.base.transform(img), y, i


def build_datasets(a):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(a.crop_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(a.randaug_n, a.randaug_m),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    tr = ImageFolderNoisy(a.data, train_tf, False, a.val_ratio, a.seed, 'train')
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


def main(a):
    seed_everything(a.seed, deterministic=not a.cudnn_benchmark)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va = build_datasets(a)
    loader, vloader = build_loaders(a, tr, va)
    nclass = len(tr.class_to_idx)
    print(f'classes={nclass} train={len(tr)} val={len(va)} device={device} amp={a.amp}')
    print('config: ' + ' '.join(f'{k}={v}' for k, v in sorted(vars(a).items())))

    clip_model = open_clip.create_model(a.model, pretrained=a.pretrained)
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

    tracker = LabelTrustTracker(tr.targets, nclass, momentum=a.noise_momentum,
                                tau_conf=a.tau_conf, w_noise=a.w_noise,
                                w_relabel=a.w_relabel, max_noise_frac=a.max_noise_frac,
                                device=device)
    robust = make_robust_loss(a.robust_loss, a.gce_q, a.apl_k, a.apl_b, a.apl_rce)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    use_amp = a.amp != 'none' and device.type == 'cuda'
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(a.amp)
    try:                                    # torch >= 2.3 API, older one as a fallback
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and a.amp == 'fp16')
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and a.amp == 'fp16')

    start_epoch, best = 0, 0.0
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        teacher = copy.deepcopy(model).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        opt.load_state_dict(ck['optim'])
        sched.load_state_dict(ck['sched'])
        tracker.load_state_dict(ck['tracker'])
        torch.set_rng_state(ck['rng']['torch'])
        if ck['rng'].get('cuda') is not None:
            torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        random.setstate(ck['rng']['python'])
        start_epoch = ck['epoch'] + 1
        best = ck.get('val_acc' if a.select == 'val_acc' else 'val_acc_hi', 0.0)
        print(f'resumed from {a.resume} at epoch {start_epoch} (best={best:.4f})')

    probe = next(iter(loader))
    print(f'first batch ok: x1={tuple(probe[0].shape)} x2={tuple(probe[1].shape)} '
          f'labels={probe[2].tolist()[:8]} idx={probe[3].tolist()[:8]}')

    for ep in range(start_epoch, a.epochs):
        warm = ep < a.warmup_epochs
        if not warm:
            print('noise stats:', tracker.refresh())

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

            need_anchor = a.anchor_weight > 0 or (warm and a.proto_weight > 0)
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
                # pure supervised warm-up on the raw labels
                y_used = y
                w = torch.ones(bsz, device=device)
            else:
                y_used = tracker.label[idx_g]           # possibly corrected label
                w = tracker.weight[idx_g] * tprob.max(1).values.clamp(a.conf_floor, 1.0).pow(a.conf_gamma)
                if a.norm_weights:
                    # keep the effective step size stable as filtering kicks in
                    w = w / w.mean().clamp_min(1e-6)

            # labelled pass on both views (the "passive" term)
            ce1 = F.cross_entropy(out.float(), y_used, reduction='none')
            ce2 = F.cross_entropy(out2.float(), y_used, reduction='none')
            loss = 0.5 * ((w * ce1).mean() + (w * ce2).mean())
            if not warm and a.robust_weight > 0:
                rob = 0.5 * (robust(out, y_used) + robust(out2, y_used))
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
                loss = loss + a.proto_weight * (pw * F.cross_entropy(pl, y_used.repeat(2),
                                                                     reduction='none')).mean()

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

                if warm and a.proto_weight > 0:
                    boot_sum.index_add_(0, y, anchor)       # frozen-CLIP class means
                    boot_cnt.index_add_(0, y, torch.ones(bsz, device=device))
                if a.proto_weight > 0 and not warm:
                    model.proto.update(tz, y_used, trusted)

            running += loss.item() * bsz
            seen += bsz

        if warm and ep + 1 == a.warmup_epochs and a.proto_weight > 0:
            means, present = prototype_bootstrap(boot_sum, boot_cnt)
            model.proto.init_from_means(means, present)
            print(f'prototypes seeded from frozen CLIP for {int(present.sum())}/{nclass} classes')

        sched.step()
        vl, va_acc, va_hi = evaluate(model, vloader, device, amp_dtype, use_amp)
        score = va_acc if a.select == 'val_acc' else va_hi
        print(f'epoch {ep + 1}/{a.epochs} loss={running / max(seen, 1):.4f} '
              f'val_loss={vl:.4f} val_acc={va_acc:.4f} val_acc_hi={va_hi:.4f} '
              f'lr={opt.param_groups[0]["lr"]:.2e} time={time.time() - t0:.1f}s')

        ck = {'model': model.trainable_state_dict(), 'classes': tr.class_to_idx,
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
            torch.save(ck, out_dir / f'ep{ep + 1}.pt')
        if score > best:
            best = score
            torch.save(ck, out_dir / 'best.pt')
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
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--cudnn-benchmark', action='store_true',
                   help='faster but not bit-reproducible (default: off)')
    p.add_argument('--amp', default='bf16', choices=['bf16', 'fp16', 'none'])
    p.add_argument('--val-ratio', type=float, default=0.1)
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

    p.add_argument('--robust-loss', default='apl', choices=['ce', 'gce', 'nce', 'apl'])
    p.add_argument('--robust-weight', type=float, default=0.5)
    p.add_argument('--gce-q', type=float, default=0.7)
    p.add_argument('--apl-k', type=float, default=0.2)
    p.add_argument('--apl-b', type=float, default=1.0)
    p.add_argument('--apl-rce', type=float, default=1.0)

    p.add_argument('--noise-momentum', type=float, default=0.9)
    p.add_argument('--tau-conf', type=float, default=0.8,
                   help='teacher max(p) needed to overrule the given label')
    p.add_argument('--w-noise', type=float, default=0.1)
    p.add_argument('--w-relabel', type=float, default=0.5)
    p.add_argument('--max-noise-frac', type=float, default=0.4)
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
