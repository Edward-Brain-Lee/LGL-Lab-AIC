"""Per-class (macro) metrics for existing checkpoints -- no training involved.

Why this exists
---------------
``train.py`` reports ``val_acc``: a *micro* average over a hold-out split that is
**stratified per class** (``ImageFolderNoisy``, ``train.py:322``).  Stratifying
preserves the training set's class imbalance, so ``val_acc`` weights every class
by how often it occurs in the *training* distribution -- here the commonest
class has ~56x the images of the rarest (``infer.py`` prints the log range).
The competition's test set is class-balanced, so what the leaderboard reports is
closer to a *macro* average.

The ``val_acc`` -> leaderboard gap therefore has at least three explanations,
and they call for opposite strategies:

  H1 memorisation -- val_acc is inflated because the model fitted the label
                     noise, which the val split shares.  Fix: keep
                     regularising, submit earlier snapshots.
  H2 metric       -- val_acc is head-weighted while the leaderboard is
                     class-equal.  Fix: the model is fine, change what we
                     select on (and this becomes a free local proxy).
  H3 domain shift -- the val split comes from the training distribution and the
                     test set does not.  Not fixable from here.

This script measures H2 for free, on checkpoints that already exist.

**The self-check**: the ``val_acc`` column MUST reproduce the number ``train.py``
logged for that epoch.  If it does, the split and the forward pass match and the
``macro`` column can be trusted.  If it does not, nothing else here means
anything -- fix that first.

Usage::

    python valmetrics.py --checkpoint outputs/ep4.pt outputs/ep8.pt \\
        outputs/ep12.pt outputs/ep16.pt outputs/ep20.pt
    python valmetrics.py --checkpoint outputs_old/ep20.pt
"""
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import open_clip

from train import CLIP_MEAN, CLIP_STD, ImageFolderNoisy, Net, check_backbone


@torch.no_grad()
def run(model, va, device, a):
    """Returns ``(n, micro_acc, per_class_recall, per_class_count)``."""
    per_c = torch.zeros(len(va.class_to_idx), dtype=torch.long)
    per_t = torch.zeros(len(va.class_to_idx), dtype=torch.long)
    n = correct = 0
    loader = DataLoader(va, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=True)
    for x1, y, _ in loader:
        x1 = x1.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=device.type == 'cuda'):
            out = model(x1)
        pred = out.float().argmax(1).cpu()
        n += int(y.numel())
        correct += int((pred == y).sum())
        per_c.index_add_(0, y, (pred == y).long())
        per_t.index_add_(0, y, torch.ones_like(y))
    return n, correct / max(n, 1), per_c.float() / per_t.clamp_min(1).float(), per_t


def report(path, ck, model, device, a):
    ck_args = ck.get('args', {})
    data = a.data or ck_args.get('data')
    assert data, 'checkpoint has no --data recorded; pass --data explicitly'
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    va = ImageFolderNoisy(data, val_tf, True, ck_args.get('val_ratio', 0.1),
                          ck_args.get('seed', 3407), 'val')
    assert va.class_to_idx == ck['classes'], (
        'val split found a different class list than the checkpoint -- '
        'wrong --data?')

    n, acc, rec, per_t = run(model, va, device, a)
    order = rec.argsort()
    half, k = len(rec) // 2, max(1, len(rec) // 10)
    rev = {v: kk for kk, v in va.class_to_idx.items()}

    print(f'\n{path}  (epoch {ck.get("epoch", "?")}, val={n} images)')
    print(f'  val_acc   micro, must match train.py log : {acc:.4f}')
    print(f'  macro     class-equal recall            : {float(rec.mean()):.4f}')
    print(f'  per-class val images : min {int(per_t.min())} / max {int(per_t.max())} '
          f'(ratio {int(per_t.max()) / max(int(per_t.min()), 1):.0f}x)')
    print(f'  recall, rarest 50% of classes : {float(rec[order[:half]].mean()):.4f}')
    print(f'  recall, commonest 10%         : {float(rec[order[-k:]].mean()):.4f}')
    print('  worst 5 classes: ' + ', '.join(
        f'{rev[int(i)]}(n={int(per_t[i])},r={float(rec[i]):.2f})' for i in order[:5]))


def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for path in a.checkpoint:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        ck_args = ck.get('args', {})
        rank = ck.get('lora_rank', ck_args.get('lora_rank', 8))
        target = ck.get('lora_target', ck_args.get('lora_target', 'all'))
        model_name = ck.get('model_name', ck_args.get('model', 'ViT-B-32-quickgelu'))
        check_backbone(model_name)

        clip_model = open_clip.create_model(model_name,
                                            pretrained=ck.get('pretrained', 'openai'))
        model = Net(clip_model, len(ck['classes']), rank, target)
        missing, _ = model.load_state_dict(ck.get('model', ck), strict=False)
        lost = {n for n, p in model.named_parameters() if p.requires_grad} & set(missing)
        assert not lost, f'no trained weights in checkpoint for: {sorted(lost)[:5]}'
        del clip_model
        model.to(device).eval()

        report(path, ck, model, device, a)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', nargs='+', required=True,
                   help='one or more .pt files (ep4.pt / ep20.pt / best.pt)')
    p.add_argument('--data', default='', help='override the training folder '
                                              '(default: as recorded in the checkpoint)')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
