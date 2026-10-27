"""CPU preflight on the real dataset: what the 4090 run will be fed, and how fast.

    python sf_preflight.py --data /root/autodl-tmp/train

Measures, with no GPU and no training:

  * `ImageFolderNoisy` build time (it stats ~90k files on every run);
  * the stratified split the recipe actually produces (train/val sizes);
  * **loader throughput** for the real 384px transform at several `--workers`
    values.  A 4090 doing 20 epochs over 81k images needs ~216 img/s sustained
    (复赛 measured 4.16 ms/img end-to-end); if the loader cannot reach that, the
    GPU idles and `--workers` is the knob -- and the knob only matters if one of
    the measured values actually crosses that line.

The transform is the one `train.py:build_datasets` builds, copied here rather than
imported so the measurement cannot silently drift from the recipe.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train import CLIP_MEAN, CLIP_STD, ImageFolderNoisy

NEED_IMG_S = 216.0          # 81,177 images / 375 s per epoch (复赛's 4.16 ms/img)


def build(root, img_size, val_ratio, split_seed):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(2, 9),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    t0 = time.time()
    tr = ImageFolderNoisy(root, tf, False, val_ratio, split_seed, 'train', img_size)
    va = ImageFolderNoisy(root, tf, True, val_ratio, split_seed, 'val', img_size)
    return tr, va, time.time() - t0


def throughput(ds, workers, batch, batches, img_size):
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                        num_workers=workers, pin_memory=False,
                        persistent_workers=workers > 0)
    it = iter(loader)
    for _ in range(2):                       # warm the workers
        next(it)
    n, t0 = 0, time.time()
    for _ in range(batches):
        x, y, _ = next(it)
        n += x.shape[0]
    dt = time.time() - t0
    del it, loader
    return n / dt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', required=True)
    p.add_argument('--img-size', type=int, default=384)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--split-seed', type=int, default=3407)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--batches', type=int, default=15)
    p.add_argument('--workers', type=int, nargs='+', default=[4, 8, 16])
    p.add_argument('--out', default='')
    a = p.parse_args(argv)

    tr, va, build_s = build(a.data, a.img_size, a.val_ratio, a.split_seed)
    print(f'dataset index built in {build_s:.1f}s')
    print(f'train {len(tr):,}  val {len(va):,}  (val_ratio {a.val_ratio})')
    nclass = len(tr.class_to_idx)
    counts = torch.bincount(torch.as_tensor(tr.targets), minlength=nclass)
    print(f'classes {nclass}  per-class train min {int(counts.min())} '
          f'max {int(counts.max())}  sum {int(counts.sum()):,}')
    steps = len(tr) // a.batch_size
    print(f'steps/epoch at batch {a.batch_size}: {steps}  '
          f'({steps * a.batch_size:,} draws per epoch, with replacement)')

    print(f'\nloader throughput, {a.img_size}px, transform = the training one')
    rows = []
    for w in a.workers:
        rate = throughput(tr, w, a.batch_size, a.batches, a.img_size)
        rows.append((w, rate))
        verdict = 'enough' if rate >= NEED_IMG_S else f'SHORT x{NEED_IMG_S / rate:.2f}'
        print(f'  workers {w:2d}: {rate:7.1f} img/s  '
              f'-> epoch floor {len(tr) / rate / 60:5.1f} min   [{verdict}]')
    best = max(rows, key=lambda kv: kv[1])
    print(f'\n  the GPU recipe needs >= {NEED_IMG_S:.0f} img/s to not be starved.')
    print(f'  fastest measured: workers={best[0]} at {best[1]:.0f} img/s')
    if best[1] < NEED_IMG_S * 1.3:
        print('  WARNING: no margin — raise --workers further before blaming the GPU.')
    if a.out:
        Path(a.out).write_text(json.dumps(
            {'train': len(tr), 'val': len(va), 'throughput': rows}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
