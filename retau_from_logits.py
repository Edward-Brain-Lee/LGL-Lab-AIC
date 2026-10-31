"""Re-derive any ``--logit-adjust`` submission from logits already on disk.

    python retau_from_logits.py --logits outputs_sf416e20se_test_logits.npz \
        --test test --counts-csv analysis/class_counts.csv --out sub_retau

Why this exists
---------------
``infer.py --save-logits`` writes the ``(N, C)`` averaged logits for one
checkpoint.  Every ``--logit-adjust`` value is a **pure re-argmax** of that same
array, so a whole tau grid costs seconds instead of another ~45-minute 16-view
forward pass.  On the last submission of the stage that is the difference
between testing one tau and testing six.

This only re-argmaxes an existing prediction: it trains nothing, reads no test
label, and uses the same training class prior ``infer.py`` uses.  One
checkpoint, one view set, one tau -- still a single model on a single inference
path (rule 五.4).

The self-check
--------------
``--expect-csv`` is the tau=0 CSV ``infer.py`` wrote.  The tau=0 rows derived
here must match it line for line; if they do not, the logits file does not
describe that submission and nothing else in this run means anything.  That is
the same guard ``infer.py --save-logits`` applies on the way out, applied again
on the way back in.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_logits(path):
    with np.load(path, allow_pickle=False) as z:
        logits = z['logits']
        names = [str(n) for n in z['names']]
        labels = [str(v) for v in z['labels']]
    if logits.dtype != np.float32:
        raise ValueError(f'{path}: logits dtype is {logits.dtype}, expected float32')
    if logits.ndim != 2:
        raise ValueError(f'{path}: logits shape is {logits.shape}, expected (N, C)')
    if len(names) != logits.shape[0]:
        raise ValueError(f'{path}: {len(names)} names for {logits.shape[0]} rows')
    if len(labels) != logits.shape[1]:
        raise ValueError(f'{path}: {len(labels)} labels for {logits.shape[1]} columns')
    if not np.isfinite(logits).all():
        raise ValueError(f'{path}: logits contain non-finite values')
    return logits, names, labels


def load_counts(a, nclass):
    """Training-split class counts in class-index order.

    Preferred source is the checkpoint (exactly what ``infer.py`` reads).  The
    CSV fallback exists so this can run on a box without torch, using the same
    numbers ``analysis/class_counts.csv`` records.
    """
    counts = None
    if a.checkpoint:
        try:
            import torch
        except ImportError:
            if not a.counts_csv:
                raise SystemExit('--checkpoint needs torch; pass --counts-csv instead')
        else:
            ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
            counts = ck.get('class_counts')
            if counts is None:
                raise SystemExit(f'{a.checkpoint} carries no class_counts; pass --counts-csv')
            counts = np.asarray(list(counts), dtype=np.float64)
    elif a.counts_csv:
        rows = []
        with open(a.counts_csv, newline='', encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if len(row) < 2 or not row[1].strip().lstrip('-').isdigit():
                    continue                      # header or junk
                rows.append((int(row[0]), int(row[1])))
        if not rows:
            raise SystemExit(f'{a.counts_csv}: no "class,count" rows found')
        rows.sort()
        counts = np.asarray([c for _, c in rows], dtype=np.float64)
    else:
        raise SystemExit('pass --checkpoint or --counts-csv (tau>0 needs the class prior)')

    if counts.size != nclass:
        raise SystemExit(f'class counts have {counts.size} entries, logits have {nclass} columns')
    if (counts <= 0).any():
        raise SystemExit('class counts contain zeros; a zero prior is -inf and would win every argmax')
    if not np.isfinite(counts).all():
        raise SystemExit('class counts contain non-finite values')
    return counts


def violation(pred, nclass):
    """``0.5 * sum_c |k_c - N/C|``: the lowest attainable error count under a
    class-balanced test set (HANDOFF §52.8).  Label-free."""
    counts = np.bincount(pred, minlength=nclass)
    n = len(pred)
    target = n / nclass
    v = float(np.abs(counts - target).sum() / 2)
    return dict(n=n, target=target, violation=v, frac=100 * v / n,
                ceiling=100 * (1 - v / n), predicted_min=int(counts.min()),
                predicted_max=int(counts.max()), predicted_classes=int((counts > 0).sum()))


def write_rows(path, names, pred, labels):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows((n, labels[i]) for n, i in zip(names, pred.tolist()))


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return [(r[0], r[1]) for r in csv.reader(f) if len(r) == 2]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--logits', required=True, help='the .npz written by infer.py --save-logits')
    p.add_argument('--test', required=True, help='official test dir; only the name set is read')
    p.add_argument('--out', default='sub_retau')
    p.add_argument('--taus', nargs='+', type=float, default=[0.0, 0.75, 1.0, 1.25, 1.5, 2.0])
    p.add_argument('--checkpoint', default='', help='preferred source of class_counts')
    p.add_argument('--counts-csv', default='', help='fallback "class,count" CSV (no torch needed)')
    p.add_argument('--expect-csv', default='', help='tau=0 CSV from infer.py, for the self-check')
    a = p.parse_args(argv)

    logits, names, labels = load_logits(a.logits)
    n, nclass = logits.shape
    print(f'logits {a.logits}: {n} rows x {nclass} classes')

    # Every test image exactly once, in the same name set the submission must cover.
    on_disk = sorted(p.name for p in Path(a.test).rglob('*')
                     if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if len(on_disk) != n:
        raise SystemExit(f'{len(on_disk)} test images but {n} logits rows -- '
                         f'this file does not describe this test set')
    if set(on_disk) != set(names):
        raise SystemExit('the logits file and the test directory have different file names')
    if len(set(names)) != len(names):
        raise SystemExit('duplicate file names in the logits file')

    counts = load_counts(a, nclass) if any(t for t in a.taus) else None
    # clamp the *denominator* before the ratio (a 0/0 would stay nan), then the ratio
    prior = None if counts is None else np.log(np.clip(counts / counts.sum(), 1e-12, None))
    if prior is not None:
        print(f'class prior from training counts: log range '
              f'{float(prior.max() - prior.min()):.2f}')

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tau in a.taus:
        adj = logits if (not tau or prior is None) else logits - tau * prior[None, :]
        pred = adj.argmax(1)
        path = out / ('pred_results.csv' if not tau else
                      f'pred_results_tau{("%.2f" % tau).replace(".", "")}.csv')
        write_rows(path, names, pred, labels)
        v = violation(pred, nclass)
        rows.append((tau, path, v))
        print(f'  tau={tau:<5g} rows={v["n"]:>6,} classes={v["predicted_classes"]:>4} '
              f'min/max={v["predicted_min"]:>4}/{v["predicted_max"]:<4} '
              f'violation={v["frac"]:6.2f}%  ceiling={v["ceiling"]:6.2f}%  -> {path.name}')

    # Self-check: tau=0 must reproduce what infer.py wrote, byte for byte in content.
    if a.expect_csv:
        expect = read_csv_rows(a.expect_csv)
        got = read_csv_rows(out / 'pred_results.csv')
        if expect != got:
            diff = sum(1 for x, y in zip(expect, got) if x != y)
            raise SystemExit(f'SELF-CHECK FAILED: tau=0 differs from {a.expect_csv} '
                             f'in {diff} row(s); these logits do not describe that submission')
        print(f'self-check: tau=0 reproduces {a.expect_csv} exactly')

    best = min(rows, key=lambda r: r[2]['violation'])
    print(f'\nlowest marginal violation: tau={best[0]:g} ({best[2]["frac"]:.2f}% '
          f'-> ceiling {best[2]["ceiling"]:.2f}%)  [{best[1].name}]')
    print('cross-check with the hold-out (validate_tta.py / mres.py) before submitting; '
          'if the two disagree, the violation is the structural one and macro is the noisy one.')
    (out / 'tau_summary.json').write_text(json.dumps(
        [dict(tau=t, file=str(f), **v) for t, f, v in rows], indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
