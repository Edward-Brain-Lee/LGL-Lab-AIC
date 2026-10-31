"""Measure train-set accuracy for every saved checkpoint, to draw the over/under-fitting curve.

    python train_acc_probe.py --data /root/autodl-tmp/train \
        --log outputs_cand.log --out outputs_cand \
        --ckpt outputs_cand/ep2.pt outputs_cand/ep4.pt ... --n 6000

Why this is measured and not read off the log
---------------------------------------------
``train.py`` logs a composite objective (CE + APL + consistency + anchor +
prototype) whose NCE branch is bounded below at ``-1/k + 1 - log k`` (about
-2.39 at k=0.2), so the printed ``loss`` legitimately goes negative and is NOT
comparable with ``val_loss`` (plain CE on the hold-out).  There is therefore no
"train accuracy" anywhere in the training log.  Handoff §15.7-2 already made the
same point about ``val_loss`` across branches; this is the same trap one level
deeper.

So train accuracy is measured here, on the **training split**, with the **same
deterministic eval transform** the hold-out uses -- otherwise the two numbers
would differ by the augmentation, not by the model.

Reading the resulting gap
-------------------------
The training labels are the noisy ones, so the model can and does fit them; a
large gap therefore means **memorising label noise**, not classic overfitting.
The useful question is whether ``val_*`` still rises while the gap grows, and
whether the board agrees.  Handoff §42.3 measured the same thing as a 37x
train/val loss gap at rank 64.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import open_clip

import train as T


def build_train_split(data, img_size, val_ratio, split_seed, n_sample):
    """The training split, eval transform, stratified sample of ``n_sample``."""
    ds = T.ImageFolderNoisy(data, T.eval_transform(argparse.Namespace(img_size=img_size)),
                            False, val_ratio, split_seed, 'train', img_size)
    by_cls = defaultdict(list)
    for i, (_, y) in enumerate(ds.items):
        by_cls[y].append(i)
    nclass = len(ds.class_to_idx)
    k = max(1, -(-n_sample // nclass))
    idx = [i for c in sorted(by_cls) for i in by_cls[c][:k]]
    ds.items = [ds.items[i] for i in idx]          # keep index -> sample mapping trivial
    ds.targets = [y for _, y in ds.items]
    return ds


def load(data, ckpt, device):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    ca = ck.get('args', {})
    name = ck.get('model_name', ca.get('model', 'ViT-B-32-quickgelu'))
    T.check_backbone(name)
    size = int(ck.get('img_size', ca.get('img_size', 224)))
    clip = open_clip.create_model(name, pretrained=ck.get('pretrained', 'openai'))
    if size != 224:
        T.resize_positional_embedding(clip.visual, size)
    kw = dict(local_head=bool(ck.get('local_head', ca.get('local_head', False))),
              lora_qkv=bool(ck.get('lora_qkv', ca.get('lora_qkv', False))),
              lora_alpha=float(ck.get('lora_alpha', ca.get('lora_alpha', 0.0)) or 0.0),
              attn_temp=bool(ck.get('attn_temp', ca.get('attn_temp', False))),
              local_attn=bool(ck.get('local_attn', ca.get('local_attn', False))),
              local_se=bool(ck.get('local_se', ca.get('local_se', False))))
    try:
        net = T.Net(clip, len(ck['classes']), ck.get('lora_rank', ca.get('lora_rank', 8)),
                    ck.get('lora_target', ca.get('lora_target', 'all')),
                    local_gate_init=float(ck.get('local_gate_init', ca.get('local_gate_init', 0.0))),
                    **kw)
    except TypeError:
        # this train.py predates --local-gate-init (the 72.2023 tree); the gate is
        # initialised to zero there, which is exactly what such a checkpoint records
        net = T.Net(clip, len(ck['classes']), ck.get('lora_rank', ca.get('lora_rank', 8)),
                    ck.get('lora_target', ca.get('lora_target', 'all')), **kw)
    missing, unexpected = net.load_state_dict(ck.get('model', ck), strict=False)
    if unexpected:
        raise SystemExit(f'{ckpt}: unexpected tensors {sorted(unexpected)[:5]}')
    lost = {n for n, p in net.named_parameters() if p.requires_grad} & set(missing)
    if lost:
        raise SystemExit(f'{ckpt}: no trained weights for {sorted(lost)[:5]}')
    del clip
    return net.to(device).eval(), ck, size


@torch.no_grad()
def accuracy(net, ds, device, batch, workers):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=False)
    ok = n = 0
    for x, y, _ in loader:
        pred = net(x.to(device, non_blocking=True)).float().argmax(1).cpu()
        ok += int((pred == y).sum())
        n += int(y.numel())
    return ok / max(n, 1), n


def log_curve(path):
    """epoch -> the metrics train.py printed for it (val side only)."""
    out = {}
    if not path or not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.startswith('epoch '):
            continue
        f = dict(kv.split('=', 1) for kv in line.split() if '=' in kv)
        ep = int(line.split()[1].split('/')[0])
        out[ep] = dict(val_acc=float(f.get('val_acc', 'nan')),
                       val_loss=float(f.get('val_loss', 'nan')),
                       val_macro=float(f.get('val_macro', 'nan')),
                       val_tail=float(f.get('val_tail', 'nan')),
                       loss=float(f.get('loss', 'nan')),
                       secs=float(f.get('time', 'nan').rstrip('s')))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='')
    p.add_argument('--log', default='')
    p.add_argument('--ckpt', nargs='+', required=True)
    p.add_argument('--n', type=int, default=6000, help='stratified train images to score')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curve = log_curve(a.log)
    rows = []
    ds = None
    size0 = None
    print(f'{"ckpt":>10} {"ep":>3} {"train_acc":>9} {"val_acc":>8} {"gap":>6} '
          f'{"val_macro":>9} {"val_loss":>8} {"train_loss*":>10}')
    for ckpt in a.ckpt:
        net, ck, size = load(a.data, ckpt, device)
        if ds is None or size != size0:
            ca = ck.get('args', {})
            ds = build_train_split(a.data, size, float(ca.get('val_ratio', 0.1)),
                                   int(ca.get('split_seed') or ca.get('seed', 3407)), a.n)
            size0 = size
            print(f'# train sample: {len(ds)} images, {len(ds.class_to_idx)} classes, {size}px, '
                  f'eval transform (same as the hold-out)')
        tr, n = accuracy(net, ds, device, a.batch_size, a.workers)
        ep = int(ck.get('epoch', -1)) + 1            # checkpoint field is 0-based
        m = curve.get(ep, {})
        va = m.get('val_acc', float('nan'))
        rows.append(dict(checkpoint=Path(ckpt).name, epoch=ep, train_acc=tr,
                         val_acc=va, gap=tr - va,
                         val_macro=m.get('val_macro', float('nan')),
                         val_tail=m.get('val_tail', float('nan')),
                         val_loss=m.get('val_loss', float('nan')),
                         train_loss_obj=m.get('loss', float('nan'))))
        print(f'{Path(ckpt).name:>10} {ep:>3} {tr:>9.4f} {va:>8.4f} {tr - va:>+6.3f} '
              f'{m.get("val_macro", float("nan")):>9.4f} {m.get("val_loss", float("nan")):>8.4f} '
              f'{m.get("loss", float("nan")):>10.4f}')
        del net
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'train_vs_val_curve.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {out / "train_vs_val_curve.csv"}')
    print('* train_loss is the COMPOSITE objective (CE+APL+consistency+anchor+proto); '
          'its NCE branch is bounded below by ~-2.39, so a negative value is normal and '
          'it is NOT comparable with val_loss.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
