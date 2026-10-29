"""Readout / calibration candidates on the OFFICIAL hold-out -- no test data touched.

    python probe_readout.py --checkpoint outputs_sf_base/best.pt

Three candidates, all on the same checkpoint, same single forward pass:

  head      the delivered cosine head (reference)
  proto     the trained prototype head, which the checkpoint already carries
  head+bias the cosine head plus a per-class bias FITTED ON THE HOLD-OUT with a
            macro objective, under 2-fold cross-validation (fit on one half, score
            on the other) so the number is not an in-sample artefact

Why head+bias is the one with a real prior: the delivered submission over-predicts a
few classes (389 predictions where the balanced target is 100) and never predicts
others.  Fitting a per-class offset on official hold-out labels is the legal analogue
of that correction -- it uses no test image, no test statistic and no batch coupling,
so README 十一(二).1 is untouched and the pipeline stays reproducible (五.5).

Everything here is decision-support: if nothing beats the cosine head on held-out
folds, no submission slot is spent.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import train as T
from infer import TTADataset
from mres import load_checkpoint


def macro_np(pred, y, n_cls):
    rec = np.zeros(n_cls)
    cnt = np.bincount(y, minlength=n_cls)
    hit = np.bincount(y[pred == y], minlength=n_cls)
    seen = cnt > 0
    rec[seen] = hit[seen] / cnt[seen]
    return float(rec[seen].mean())


def fit_bias(L, y, n_cls, grid=(-1.0, -0.5, 0.0, 0.5, 1.0), passes=2):
    """Coordinate ascent on macro accuracy: b_c chosen one class at a time."""
    b = np.zeros(n_cls)
    best = macro_np((L + b).argmax(1), y, n_cls)
    for _ in range(passes):
        improved = False
        for c in range(n_cls):
            keep, keep_v = b[c], best
            for d in grid:
                b[c] = d
                m = macro_np((L + b).argmax(1), y, n_cls)
                if m > best:
                    best, keep_v = m, d
                    improved = True
            b[c] = keep_v
        if not improved:
            break
    return b, best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck, size, base_pe, pe_name = load_checkpoint(a.checkpoint, device)
    model.eval()
    n_cls = len(ck['classes'])
    ck_args = ck.get('args', {})
    data = ck_args.get('data', 'train')

    va = T.ImageFolderNoisy(data, transforms.ToTensor(), True,
                            ck_args.get('val_ratio', 0.1), ck_args.get('seed', 3407), 'val')
    assert va.class_to_idx == ck['classes'], 'val split found a different class list'
    items = [(Path(f), int(y)) for f, y in va.items]
    files = [f for f, _ in items]
    y = torch.tensor([lbl for _, lbl in items])
    print(f'hold-out {len(files)} images, {n_cls} classes, checkpoint res {size}')

    tf = transforms.Compose([
        transforms.Resize(int(round(size * 1.143))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(T.CLIP_MEAN, T.CLIP_STD),
    ])
    ds = TTADataset(files, [tf], size)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=(device.type == 'cuda'),
                        prefetch_factor=1)

    head, proto = [], []
    with torch.no_grad():
        for views, _names, _bad in loader:
            x = views[:, 0].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                out, z = model(x, True)
            head.append(out.float().cpu())
            proto.append(model.proto.logits(z).float().cpu())
    head = torch.cat(head).numpy().astype(np.float64)
    proto = torch.cat(proto).numpy().astype(np.float64)
    yn = y.numpy()

    print(f"\n{'readout':16s} {'micro':>7} {'macro':>7} {'pred cls':>9}")
    for name, L in (('cosine head', head), ('proto head', proto)):
        pred = L.argmax(1)
        print(f'{name:16s} {float((pred == yn).mean()):7.4f} '
              f'{macro_np(pred, yn, n_cls):7.4f} {len(np.unique(pred)):9d}')

    # ---- 2-fold CV so the fitted bias is scored out-of-sample ----
    print('\nper-class bias fitted on the hold-out, scored on the OTHER fold:')
    rng = np.random.default_rng(0)
    order = np.concatenate([np.where(yn == c)[0][rng.permutation((yn == c).sum())]
                            for c in range(n_cls)])
    half = len(order) // 2
    folds = [order[:half], order[half:]]
    gains = []
    for i, fit_ids in enumerate(folds):
        ev_ids = folds[1 - i]
        b, in_fit = fit_bias(head[fit_ids], yn[fit_ids], n_cls)
        base = macro_np(head[ev_ids].argmax(1), yn[ev_ids], n_cls)
        adj = macro_np((head[ev_ids] + b).argmax(1), yn[ev_ids], n_cls)
        gains.append(adj - base)
        print(f'  fold {i + 1}: fit macro {in_fit:.4f} -> held-out macro '
              f'{base:.4f} -> {adj:.4f}  ({adj - base:+.4f})')
    print(f'  mean held-out gain {np.mean(gains):+.4f}   '
          f'micro change {float(((head + b).argmax(1) == yn).mean()) - float((head.argmax(1) == yn).mean()):+.4f}')
    verdict = 'GAIN' if np.mean(gains) >= 0.002 else 'NO GAIN'
    print(f'VERDICT: {verdict} (threshold +0.002 held-out macro)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
