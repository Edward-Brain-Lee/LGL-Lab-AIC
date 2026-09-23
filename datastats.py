"""Class-distribution report for one stage's training set.

    python datastats.py --data /root/autodl-tmp/train          # extracted folder
    python datastats.py --zip D:\\BaiduDisk\\train.zip          # or straight from the zip

Standard library only -- no torch, no numpy, no GPU, so it runs anywhere,
including on the laptop before the 28 GB archive is ever uploaded.

Why this exists: the competition brief says the later stages are long-tailed, but
what actually shipped for this round is nearly flat (mean 198.3 vs median 196,
80% of classes between 159 and 239).  Long-tail machinery -- inverse-frequency
sampling, logit adjustment, class-balanced losses -- is a lot of work for no
return on a flat set, so measure before building it.  This prints the numbers
that decide it, plus the sampling weights ``train.py`` would actually use.

It also reports whether any class-name file exists.  Every 2025 CLIP noise
method that leads the literature (TrustCLIP, NLPrompt, DEFT, Robust-CLIP) needs
text prompts built from real class names; if the folders are bare numbers, that
whole family is unavailable and only visual methods are on the table.
"""
import argparse
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def jpeg_size(data):
    """Width/height straight out of the JPEG header, or None.

    Only the SOF marker is needed and it sits near the start of the file, so
    reading the first 128 KB is enough -- no need to decompress (or even fully
    read) 28 GB of images to find out how big they are.
    """
    if len(data) < 4 or data[:2] != b'\xff\xd8':
        return None
    i, n = 2, len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:                      # fill byte
            i += 1
            continue
        if marker == 0xD8 or 0xD0 <= marker <= 0xD9:
            i += 2                              # standalone, no payload
            continue
        seg = int.from_bytes(data[i + 2:i + 4], 'big')
        # SOF0..SOF15 carry the frame size; C4/C8/CC are DHT/JPG/DAC, not SOF
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return (int.from_bytes(data[i + 7:i + 9], 'big'),
                    int.from_bytes(data[i + 5:i + 7], 'big'))
        i += 2 + seg
    return None


def _spread(items, n_sample):
    step = max(1, len(items) // n_sample)
    return items[::step][:n_sample]


def sizes_from_folder(root, n_sample):
    files = sorted(p for p in Path(root).rglob('*')
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)
    out = []
    for p in _spread(files, n_sample):
        with open(p, 'rb') as fh:
            wh = jpeg_size(fh.read(1 << 17))
        if wh:
            out.append(wh)
    return out


def sizes_from_zip(path, n_sample):
    out = []
    with zipfile.ZipFile(path) as z:
        infos = [i for i in z.infolist()
                 if not i.is_dir() and Path(i.filename).suffix.lower() in IMG_EXTS]
        for info in _spread(infos, n_sample):
            with z.open(info) as fh:
                wh = jpeg_size(fh.read(1 << 17))
            if wh:
                out.append(wh)
    return out


def report_sizes(sizes, label, n_sample):
    if not sizes:
        print(f'\n--- 图像尺寸（{label}）: 解析失败，全部跳过 ---')
        return
    w = sorted(s[0] for s in sizes)
    h = sorted(s[1] for s in sizes)
    short = sorted(min(s) for s in sizes)
    print(f'\n--- 图像尺寸（{label}，均匀抽样 {len(sizes)} 张）---')
    print(f'  宽 中位={w[len(w) // 2]}  p10={w[len(w) // 10]}  p90={w[len(w) * 9 // 10]}')
    print(f'  高 中位={h[len(h) // 2]}  p10={h[len(h) // 10]}  p90={h[len(h) * 9 // 10]}')
    small = sum(1 for s in short if s < 224)
    print(f'  短边中位={short[len(short) // 2]}，短边 <224 的占 {100 * small / len(short):.1f}%')
    if small / len(short) > 0.5:
        print('  => 多数图短边不足 224，**源图细节就是瓶颈**。')
        print('     提高输入分辨率（288/336）拿不到新信息，只会插值放大 —— 不要做。')
    else:
        print('  => 多数图短边 >=224。若想提高分辨率（288），源图尚有细节可挖，值得一试。')


def counts_from_folder(root):
    root = Path(root)
    counts, others = Counter(), []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n = 0
        for p in d.iterdir():
            if p.is_file():
                if p.suffix.lower() in IMG_EXTS:
                    n += 1
                else:
                    others.append(str(p))
        counts[d.name] = n
    return counts, others


def counts_from_zip(path):
    """Count images per top-level sub-folder without extracting anything."""
    counts, others = Counter(), []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            parts = name.split('/')
            # works for both 'train/0000/x.jpg' and '0000/x.jpg': the class is the
            # second-to-last path component
            if len(parts) < 2:
                others.append(name)
                continue
            if Path(name).suffix.lower() in IMG_EXTS:
                counts[parts[-2]] += 1
            else:
                others.append(name)
    return counts, others


def report(counts, others, label):
    if not counts:
        raise SystemExit(f'no class folders found under {label}')
    vals = sorted(counts.values())
    n_cls, total = len(vals), sum(vals)
    mean = total / n_cls
    median = vals[n_cls // 2]

    def q(p):
        return vals[min(n_cls - 1, int(n_cls * p))]

    print(f'=== {label} ===')
    print(f'classes={n_cls}  images={total}  mean={mean:.1f}  median={median}')
    print(f'min={vals[0]}  max={vals[-1]}  max/min={vals[-1] / max(vals[0], 1):.1f}x')
    print(f'p05={q(.05)}  p10={q(.10)}  p25={q(.25)}  p75={q(.75)}  p90={q(.90)}  p95={q(.95)}')

    # the shape test: on a flat set mean and median nearly coincide and the p10-p90
    # band is narrow; on a real long tail both fall apart
    spread = (q(.90) - q(.10)) / mean
    skew = mean / max(median, 1)
    print(f'\n--- 长尾诊断 ---')
    print(f'  (p90-p10)/mean = {spread:.3f}   mean/median = {skew:.3f}')
    if spread < 0.5 and skew < 1.15:
        print('  => 基本均衡。不要上长尾方法（逆频采样 / logit adjustment / 类平衡损失），'
              '投入产出比很低。')
    elif spread < 1.0:
        print('  => 轻度不均衡。保留 --sampler balanced（1/sqrt(freq)）这类温和修正即可。')
    else:
        print('  => 明显长尾。值得考虑更强的采样或 logit adjustment。')

    head = sum(vals[-max(1, n_cls // 10):])
    tail = sum(vals[:n_cls // 2])
    print(f'  头部 10% 的类占 {100 * head / total:.1f}% 的图片（均衡值 10.0%）')
    print(f'  尾部 50% 的类占 {100 * tail / total:.1f}% 的图片（均衡值 50.0%）')
    for t in (10, 20, 50, 100):
        k = sum(1 for v in vals if v < t)
        if k:
            print(f'  少于 {t} 张的类: {k} 个 ({100 * k / n_cls:.1f}%)')

    # what train.py's sampler actually does with this distribution
    print(f'\n--- 采样权重（train.py --sampler）---')
    w_bal = [1.0 / math.sqrt(n) for n in vals if n]
    lo, hi = min(w_bal), max(w_bal)
    # per-sample weight ratio is what the sampler sees; the ratio that matters for
    # fairness is how many times a whole class gets drawn per epoch
    print(f'  balanced  1/sqrt(freq): 单样本权重比 {hi / lo:.2f}x，'
          f'于是每类每轮被抽中的总数大致拉平（这是它的目的）')
    print(f'  uniform   1/freq^0    : 单样本权重全 1，每类被抽中的总数与该类样本数成正比'
          f'（{vals[-1] / max(vals[0], 1):.0f}x 的差距）')

    # tiny classes are the risk case: oversampling repeats their (possibly wrong)
    # labels the most, so list them
    tiny = sorted(((c, n) for c, n in counts.items() if n < 20), key=lambda kv: kv[1])
    if tiny:
        print(f'\n--- 样本数 <20 的类（平衡采样会把它们重复最多次，'
              f'如果标签是错的，放得也最大）---')
        for c, n in tiny[:20]:
            print(f'  {c}: {n}')
    else:
        print('\n没有样本数 <20 的类。')

    print(f'\n--- 非图片文件: {len(others)} 个 ---')
    if others:
        for o in others[:20]:
            print(f'  {o}')
        hits = [o for o in others if re.search(r'(class|name|label|categ|synset)', o, re.I)]
        if hits:
            print('  ⚠️ 上面疑似有类名映射文件 —— 有类名就能用 zero-shot 文本分类做噪声检测，'
                  'TrustCLIP / NLPrompt / DEFT 那一族方法就都打开了。先确认它的内容！')
    else:
        print('  （没有）')
        print('  => 类文件夹只有编号，没有任何语义信息。')
        print('     **所有依赖文本提示的 2025 新方法都用不了**（TrustCLIP 的语义标签验证、')
        print('     NLPrompt 的 PromptOT、DEFT 的正负文本提示、Robust-CLIP 的 prompt 预筛选），')
        print('     因为构造不出文本提示。只能走纯视觉路线。')


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--data', help='extracted folder: training set (class sub-folders) or test set (flat)')
    g.add_argument('--zip', help='the .zip itself (nothing is extracted)')
    p.add_argument('--sample', type=int, default=200,
                   help='how many images to sample for the size report (0 = skip)')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    if args.zip:
        src, c, o = args.zip, *counts_from_zip(args.zip)
        sz = sizes_from_zip(args.zip, args.sample) if args.sample else []
    else:
        src, c, o = args.data, *counts_from_folder(args.data)
        sz = sizes_from_folder(args.data, args.sample) if args.sample else []

    # a test set is flat, so every file lands in one pseudo-class -- say so
    # rather than printing a one-row "distribution"
    if len(c) <= 1:
        print(f'=== {src} ===')
        print(f'顶层没有类别子文件夹（只有 {len(c)} 个），这是扁平结构 —— 跳过类别分布。')
    else:
        report(c, o, src)

    if args.sample:
        report_sizes(sz, src, args.sample)
