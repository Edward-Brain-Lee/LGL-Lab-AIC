"""Semifinal dataset profile, measured on the extracted files (CPU box, stdlib+PIL).

    python sf_data_analysis.py --data /root/autodl-tmp/train --test /root/autodl-tmp/test

Answers the questions that decide whether the recipe should change:

  A. class counts and the long tail (does `--sampler balanced` still make sense?)
  B. image geometry (is 384 a down-sample or an up-sample on this data?)
  C. a full decode pass over every training image: unreadable / truncated files,
     and the real per-image decode cost (which sets the dataloader budget)
  D. exact duplicates inside train and across train/test (labels cannot be trusted
     if the same bytes appear in two classes)
  E. the shape of the hold-out split `train.py` actually builds (per class)
  F. the parameter budget the recipe implies for this class count

Nothing here needs a GPU, and nothing writes to the dataset.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def images(root, recursive):
    root = Path(root)
    it = root.rglob('*') if recursive else root.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


# --------------------------------------------------------------------------- #
# A. counts
# --------------------------------------------------------------------------- #
def section_counts(data):
    counts = {p.name: len(images(p, False)) for p in sorted(Path(data).iterdir())
              if p.is_dir()}
    n = sorted(counts.values())
    total = sum(n)
    print(f'\n== A. class distribution ==')
    print(f'  classes {len(n)}  images {total:,}  '
          f'min {n[0]} / median {statistics.median(n):.0f} / mean {total / len(n):.1f} / max {n[-1]}')
    q = statistics.quantiles([float(x) for x in n], n=10)
    print(f'  deciles {" ".join(f"{int(x)}" for x in q)}')
    print(f'  imbalance max/min {n[-1] / n[0]:.1f}x   (p90-p10)/mean '
          f'{(q[8] - q[0]) / (total / len(n)):.3f}')
    head = sorted(counts.values(), reverse=True)[:max(1, len(counts) // 10)]
    tail = sorted(counts.values())[:max(1, len(counts) // 2)]
    print(f'  head 10% of classes hold {100 * sum(head) / total:.1f}% of images')
    print(f'  tail 50% of classes hold {100 * sum(tail) / total:.1f}% of images')
    for thr in (20, 50, 100):
        k = sum(1 for v in n if v <= thr)
        print(f'  classes with <= {thr:3d} images: {k} ({100 * k / len(n):.1f}%)')
    print('  smallest:', sorted(counts.items(), key=lambda kv: kv[1])[:5])
    return counts


# --------------------------------------------------------------------------- #
# B. geometry
# --------------------------------------------------------------------------- #
def section_geometry(data, test, sample, seed=0):
    rng = random.Random(seed)
    print(f'\n== B. image geometry (sample {sample} train / {sample} test) ==')
    for label, root in (('train', data), ('test', test)):
        files = images(root, recursive=True)
        pick = rng.sample(files, min(sample, len(files)))
        sizes, modes, bad = Counter(), Counter(), 0
        for p in pick:
            try:
                with Image.open(p) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
            except Exception:
                bad += 1
        short = [min(s) for s in sizes.elements()]
        print(f'  {label:5s} distinct sizes {len(sizes)}  top {sizes.most_common(3)}')
        print(f'        modes {dict(modes)}  unreadable {bad}  '
              f'short side p10/median/p90 '
              f'{sorted(short)[len(short) // 10]}/{int(statistics.median(short))}/'
              f'{sorted(short)[int(len(short) * 0.9)]}')
        # what the recipe actually does at 384: eval resize = 1.143 * 384 = 439
        med = statistics.median(short)
        print(f'        at img_size 384 the eval resize is 439 px -> '
              f'{"UP" if med < 439 else "down"}-sample x{439 / med:.2f} of native')


# --------------------------------------------------------------------------- #
# C. full decode pass
# --------------------------------------------------------------------------- #
def _decode_one(path):
    t0 = time.perf_counter()
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
        return (str(path), True, False, (time.perf_counter() - t0) * 1000, w, h)
    except Exception as e:
        return (str(path), False, False, (time.perf_counter() - t0) * 1000, 0, 0)


def section_decode(data, workers, limit=0):
    files = images(data, True)
    if limit:
        files = files[:limit]
    print(f'\n== C. full decode pass over {len(files):,} training images '
          f'({workers} processes) ==')
    t0 = time.time()
    bad, ms = [], []
    with mp.Pool(workers) as pool:
        for i, (p, ok, trunc, dt, w, h) in enumerate(
                pool.imap_unordered(_decode_one, files, chunksize=64), 1):
            ms.append(dt)
            if not ok:
                bad.append(p)
            if i % 20000 == 0:
                print(f'  ... {i:,} decoded ({time.time() - t0:.0f}s)', flush=True)
    wall = time.time() - t0
    ms.sort()
    print(f'  wall {wall:.1f}s  throughput {len(files) / wall:.0f} img/s '
          f'({workers} procs, page cache warm or not)')
    print(f'  decode ms  p50 {ms[len(ms) // 2]:.1f}  p90 {ms[int(len(ms) * 0.9)]:.1f}  '
          f'p99 {ms[int(len(ms) * 0.99)]:.1f}')
    print(f'  unreadable {len(bad)}' + (f' e.g. {bad[:3]}' if bad else ''))
    return dict(unreadable=len(bad), wall=wall, p50=ms[len(ms) // 2])


# --------------------------------------------------------------------------- #
# D. duplicates
# --------------------------------------------------------------------------- #
def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def section_dups(data, test, sample, workers, seed=0):
    rng = random.Random(seed)
    print(f'\n== D. exact duplicates (md5 on {sample} random train + {sample} test) ==')

    def scan(root, recursive, n):
        files = images(root, recursive)
        pick = rng.sample(files, min(n, len(files)))
        with mp.Pool(workers) as pool:
            digests = pool.map(_md5, pick, chunksize=32)
        return len(set(digests)), len(digests)

    a_uniq, a_n = scan(data, True, sample)
    b_uniq, b_n = scan(test, True, sample)
    print(f'  train sample: {a_n} files -> {a_uniq} distinct md5 '
          f'({a_n - a_uniq} duplicate pairs)')
    print(f'  test  sample: {b_n} files -> {b_uniq} distinct md5 '
          f'({b_n - b_uniq} duplicate pairs)')


# --------------------------------------------------------------------------- #
# E. hold-out split
# --------------------------------------------------------------------------- #
def section_split(counts):
    print('\n== E. the hold-out split train.py builds (val_ratio=0.1) ==')
    n_val = {c: max(1, int(v * 0.1)) for c, v in counts.items()}
    n_tr = {c: v - n_val[c] for c, v in counts.items()}
    print(f'  train {sum(n_tr.values()):,}  val {sum(n_val.values()):,}')
    ones = [c for c, v in n_val.items() if v == 1]
    print(f'  classes with a single val image: {len(ones)} ({100 * len(ones) / len(counts):.1f}%) '
          f'-> their per-class recall is 0 or 100, nothing in between')
    thin = sorted(n_tr.items(), key=lambda kv: kv[1])[:5]
    print(f'  thinnest training classes {thin}')
    print(f'  steps/epoch at batch 128: {sum(n_tr.values()) // 128}  '
          f'(the sampler draws len(train) per epoch, with replacement)')


# --------------------------------------------------------------------------- #
# F. budget
# --------------------------------------------------------------------------- #
def section_budget(counts, ms_per_img=None):
    import sf_gate
    e = sf_gate.expected_trainable(len(counts))
    print('\n== F. parameter budget for this class count ==')
    print(f'  adapters {e["adapters"]:,} + head {e["head"]:,} + local {e["local"]:,} '
          f'+ pos {e["pos"]:,} = {e["total"]:,} ({e["total"] / 1e6:.3f}M)')
    n_tr = sum(max(0, v - max(1, int(v * 0.1))) for v in counts.values())
    if ms_per_img:
        print(f'  crude epoch estimate from the measured decode cost: '
              f'{n_tr * ms_per_img / 1000:.0f}s of pure decode per epoch '
              f'(the GPU forward is the other half; 复赛 measured 4.16 ms/img '
              f'end-to-end on a 4090)')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--sample', type=int, default=3000)
    p.add_argument('--workers', type=int, default=min(64, os.cpu_count() or 8))
    p.add_argument('--scan', choices=['full', 'sample', 'none'], default='full')
    p.add_argument('--out', default='')
    a = p.parse_args(argv)

    counts = section_counts(a.data)
    section_geometry(a.data, a.test, a.sample)
    dec = None
    if a.scan == 'full':
        dec = section_decode(a.data, a.workers)
    elif a.scan == 'sample':
        dec = section_decode(a.data, a.workers, limit=a.sample)
    section_dups(a.data, a.test, min(a.sample, 5000), a.workers)
    section_split(counts)
    section_budget(counts, dec['p50'] if dec else None)

    if a.out:
        Path(a.out).write_text(json.dumps(
            {'counts': counts, 'decode': dec}, indent=1))
        print(f'\nwrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
