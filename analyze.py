"""Error analysis for a trained checkpoint -- *where* is the model wrong, and why.

    python analyze.py --data /root/autodl-tmp/train --checkpoint outputs2/best.pt \
                      --out analysis/

The hold-out split is rebuilt with exactly the code path ``train.py`` uses
(``ImageFolderNoisy`` + the same seed), so the report describes the same 10116
images the validation curve was computed on.

What it answers, in order of how much it should change your next decision:

1. **Is the label noise structured?**  ``val_acc`` badly over-reports the true
   score (0.7305 vs the platform's 0.6802).  One explanation is that the noise
   is *class-consistent* -- "images of species A are labelled B" -- which a model
   can learn and which therefore transfers to held-out images.  The other is that
   the test set is simply harder.  The confusion report separates them: if a
   handful of class pairs account for most errors, and those pairs are strongly
   *directional* (A->B much more often than B->A), the noise is structured.  If
   errors are spread thinly and symmetrically over many pairs, the classes are
   just visually close and the test set really may be harder.

2. **How much of the "error" is the labels rather than the model?**  Errors the
   model makes *confidently* on the hold-out split are candidates for mislabelled
   samples, not model failures.  That fraction is an upper bound on the noise
   rate and a lower bound on what any further tuning can win.

3. **Which classes are broken?**  A class with near-zero accuracy next to
   classes at 95% is the signature of a systematically mislabelled class.

Writes ``per_class.csv``, ``confusions.csv``, ``per_sample.csv`` (every image)
and ``suspected_noisy.csv`` (confident disagreements, most confident first) into
``--out``.  Open the images in the last one -- that is where the label errors
are, and they are what the whole noise-robust recipe exists to handle.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import open_clip

import train as T


def build_split(a, split, img_size=224):
    """Rebuild the hold-out split exactly as ``train.py`` does.

    Membership depends only on ``(seed, val_ratio, val=...)``; the transform is
    passed here as the deterministic evaluation one so the numbers are stable.
    ``img_size`` must come from the checkpoint -- analysing a 288 run with the 224
    transform would still run, and every number it printed would be wrong.
    """
    val_tf = transforms.Compose([
        transforms.Resize(T.val_resize(img_size)), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(T.CLIP_MEAN, T.CLIP_STD)])
    return T.ImageFolderNoisy(a.data, val_tf, val=(split == 'val'),
                              val_ratio=a.val_ratio, seed=a.seed, split=split)


def load_model(a, device):
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    classes = ck['classes']
    ck_args = ck.get('args', {})
    rank = a.lora_rank or ck.get('lora_rank', ck_args.get('lora_rank', 8))
    target = ck.get('lora_target', ck_args.get('lora_target', 'all'))
    pretrained = a.pretrained or ck.get('pretrained', 'openai')
    model_name = ck.get('model_name', ck_args.get('model', 'ViT-B-32-quickgelu'))
    T.check_backbone(model_name)

    clip_model = open_clip.create_model(model_name, pretrained=pretrained)
    img_size = ck.get('img_size', ck_args.get('img_size', 224))
    if img_size != 224:                 # at the trained size the grid already matches
        T.resize_positional_embedding(clip_model.visual, img_size)
    model = T.Net(clip_model, len(classes), rank, target)
    missing, _ = model.load_state_dict(ck.get('model', ck), strict=False)
    trained = {n for n, p in model.named_parameters() if p.requires_grad}
    lost = trained & set(missing)
    assert not lost, f'checkpoint has no trained weights for: {sorted(lost)[:5]}'
    del clip_model
    model.to(device).eval()
    print(f'loaded {a.checkpoint} (epoch {ck.get("epoch", "?")}, {len(classes)} classes, '
          f'{model_name}, lora rank {rank}/{target}, {img_size}px)')
    return model, classes, img_size


@torch.no_grad()
def collect(model, ds, device, batch_size, workers):
    """One deterministic pass -> per-sample (given, pred, conf, p_given)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    given, pred, conf, p_given, top5 = [], [], [], [], []
    for x, y, _ in loader:
        p = model(x.to(device)).float().softmax(1).cpu()
        c, pr = p.max(1)
        given.append(y)
        pred.append(pr)
        conf.append(c)
        p_given.append(p.gather(1, y[:, None]).squeeze(1))
        # k must be clamped: the self-test runs this on a 4-class stub, and
        # torch.topk(5, dim=1) on a width-4 tensor raises rather than truncating
        top5.append(p.topk(min(5, p.shape[1]), dim=1).indices)
    return (torch.cat(given), torch.cat(pred), torch.cat(conf),
            torch.cat(p_given), torch.cat(top5))


def quantiles(v, ps=(10, 25, 50, 75, 90)):
    q = torch.quantile(v.float(), torch.tensor([p / 100 for p in ps]))
    return ' '.join(f'p{p}={x:.3f}' for p, x in zip(ps, q))


def report(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, classes, img_size = load_model(a, device)
    ds = build_split(a, a.split, img_size)
    names = {i: n for n, i in classes.items()}
    print(f'{a.split} split: {len(ds)} images')

    given, pred, conf, p_given, top5 = collect(model, ds, device, a.batch_size, a.workers)
    n = given.numel()
    correct = pred == given
    acc = float(correct.float().mean())
    acc5 = float((top5 == given[:, None]).any(1).float().mean())
    err = ~correct

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f'\n=== 总体 ({a.split}) ===')
    print(f'N={n}  top1={acc:.4f}  top5={acc5:.4f}  mean_conf={float(conf.mean()):.3f}')
    print(f'错误 {int(err.sum())} 个；其中 conf>=0.8 的占 '
          f'{float((conf[err] >= 0.8).float().mean()) if bool(err.any()) else 0:.3f}，'
          f'conf>=0.5 的占 {float((conf[err] >= 0.5).float().mean()) if bool(err.any()) else 0:.3f}')

    # ---- 1) 标签噪声的估计 ---------------------------------------------- #
    # p(y_given) is the model's belief in the label it was *given*.  A sample the
    # model confidently rejects is either genuinely hard or mislabelled; the tail
    # below 0.2 is the honest upper bound on the clean-set assumption failing.
    print('\n=== 给定标签的可信度 p(y_given)（噪声估计） ===')
    print('  ' + quantiles(p_given))
    for t in (0.5, 0.2, 0.1):
        frac = float((p_given < t).float().mean())
        print(f'  p(y_given) < {t}: {frac:.4f}  ({int(frac * n)} 张)')

    # ---- 2) 类别难度 ---------------------------------------------------- #
    per_cls = {}
    for c in range(len(classes)):
        m = given == c
        k = int(m.sum())
        per_cls[c] = (k, float(correct[m].float().mean()) if k else 0.0,
                      float(conf[m].mean()) if k else 0.0)
    accs = torch.tensor([v[1] for v in per_cls.values() if v[0]])
    print('\n=== 类别难度 ===')
    print(f'  每类准确率: min={float(accs.min()):.3f} 中位={float(accs.median()):.3f} '
          f'max={float(accs.max()):.3f}  <0.3 的类有 {int((accs < 0.3).sum())} 个')
    print(f'  {quantiles(accs, (10, 25, 50, 75, 90))}')
    worst = sorted(per_cls.items(), key=lambda kv: kv[1][1])[:20]
    print('  最差 20 类 (类号, 样本数, 准确率, 平均置信度):')
    for c, (k, ac, cf) in worst:
        print(f'    {names[c]:>5}  n={k:<4} acc={ac:.3f}  conf={cf:.3f}')
    with (out / 'per_class.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'n', 'acc', 'mean_conf'])
        for c in sorted(per_cls):
            w.writerow([names[c], per_cls[c][0], f'{per_cls[c][1]:.4f}', f'{per_cls[c][2]:.4f}'])

    # ---- 3) 混淆结构：结构化噪声 vs 真·细粒度困难 ----------------------- #
    # Direction matters.  "A labelled B" happening far more often than "B
    # labelled A" cannot come from visual similarity (that is symmetric); it is
    # the fingerprint of a systematic annotation error, which a model can learn
    # and which then transfers to held-out images.
    pair = Counter()
    g, p = given[err].tolist(), pred[err].tolist()
    for a_, b_ in zip(g, p):
        pair[(a_, b_)] += 1
    pairs = []
    for (a_, b_), n_ab in pair.items():
        n_ba = pair.get((b_, a_), 0)
        asym = (n_ab - n_ba) / (n_ab + n_ba)
        pairs.append((n_ab + n_ba, a_, b_, n_ab, n_ba, asym))
    pairs.sort(reverse=True)
    tot_err = int(err.sum())
    top20 = sum(x[3] for x in pairs[:20])
    print('\n=== 混淆结构 ===')
    print(f'  总共 {tot_err} 个错误，分布在 {len(pairs)} 个有序类对 / '
          f'{len({frozenset((a_, b_)) for _, a_, b_, _, _, _ in pairs})} 个无序类对')
    print(f'  最集中的 20 个有序对贡献了 {top20} 个错误 = {top20 / max(tot_err, 1):.1%}')
    print('  错误最多的 20 对 (A->B, B->A, 不对称度; |asym|>0.6 说明是单向的):')
    for tot, a_, b_, n_ab, n_ba, asym in pairs[:20]:
        flag = '  <== 单向' if abs(asym) > 0.6 and tot >= 10 else ''
        print(f'    {names[a_]:>5} -> {names[b_]:>5}   A->B={n_ab:<4} B->A={n_ba:<4} '
              f'asym={asym:+.2f}{flag}')
    with (out / 'confusions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['true', 'pred', 'n_true_to_pred', 'n_pred_to_true', 'asymmetry'])
        for tot, a_, b_, n_ab, n_ba, asym in pairs:
            w.writerow([names[a_], names[b_], n_ab, n_ba, f'{asym:.3f}'])

    # ---- 4) 逐样本 + 疑似错标 ------------------------------------------- #
    paths = [pth for pth, _ in ds.items]
    with (out / 'per_sample.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'given', 'pred', 'conf', 'p_given', 'correct'])
        for i in range(n):
            w.writerow([Path(paths[i]).name, names[int(given[i])], names[int(pred[i])],
                        f'{conf[i]:.4f}', f'{p_given[i]:.4f}', int(correct[i])])

    susp = [(float(conf[i]), i) for i in range(n) if err[i] and conf[i] >= a.min_conf]
    susp.sort(reverse=True)
    with (out / 'suspected_noisy.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'given', 'pred', 'conf'])
        for cf, i in susp:
            w.writerow([paths[i], names[int(given[i])], names[int(pred[i])], f'{cf:.4f}'])
    print(f'\n=== 疑似错标样本（conf>={a.min_conf} 且与给定标签不符）===')
    print(f'  {len(susp)} 张，占 {len(susp) / n:.1%}，已写入 {out / "suspected_noisy.csv"}')
    print('  看几张图就能判断：如果它们明显属于预测的类，那就是标签错了。')
    print(f'\n输出目录: {out}/  (per_class.csv, confusions.csv, per_sample.csv, suspected_noisy.csv)')


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='the same training folder train.py used')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', default='analysis')
    p.add_argument('--split', default='val', choices=['val', 'train'])
    p.add_argument('--val-ratio', type=float, default=0.1, help='must match the training run')
    p.add_argument('--seed', type=int, default=3407, help='must match the training run')
    p.add_argument('--min-conf', type=float, default=0.7,
                   help='confidence above which a disagreement is reported as suspected noise')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--pretrained', default='', help='override (default: as in the checkpoint)')
    p.add_argument('--lora-rank', type=int, default=0, help='override (default: as in the checkpoint)')
    return p.parse_args(argv)


if __name__ == '__main__':
    report(parse_args())
