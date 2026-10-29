"""Pick the logit-adjust tau from label-free + hold-out evidence.

    python select_tau.py --test /root/autodl-tmp/test \
        --csv sub_a/pred_results.csv sub_a/pred_results_tau100.csv ...

Two independent judgements, neither of which uses a test label:

1. **Marginal violation** (label-free, structural).  The rules state the test set is
   class-balanced, and 49,857 = 500*100 - 143, so the target marginal is 99.714/class.
   For any prediction set the number of errors is at least

       L = 0.5 * sum_c |k_c - 99.714|

   because every misclassification moves one count up and one down.  Lower L means a
   higher achievable ceiling, so it ranks taus without touching a label.

2. **Hold-out macro** (from `mres.py`, run separately on the official training
   hold-out).  The test is balanced, so board accuracy ~ mean per-class recall =
   macro, not micro -- HANDOFF §51.4.  micro rewards head classes, which is exactly
   what the prior correction trades away, so micro picks the wrong tau.

The script also prints how many rows each tau changes relative to tau=0, and the
class-count spread, so a tau that "fixes" the marginal by collapsing classes is
visible immediately.
"""
import argparse
import csv
from pathlib import Path


def load(path):
    with open(path, newline='', encoding='utf-8') as f:
        return [(r[0], r[1]) for r in csv.reader(f) if len(r) == 2]


def violation(labels, classes=500):
    counts = {c: 0 for c in range(classes)}
    for _, l in labels:
        counts[int(l)] += 1
    n = len(labels)
    target = n / classes
    v = sum(abs(c - target) for c in counts.values()) / 2
    return dict(n=n, target=target, violation=v,
                frac=100 * v / n, ceiling=100 * (1 - v / n),
                pred_min=min(counts.values()), pred_max=max(counts.values()),
                distinct=sum(1 for c in counts.values() if c > 0),
                unused=sum(1 for c in counts.values() if c == 0))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--test', required=True, help='official test dir (for the name set)')
    p.add_argument('--csv', nargs='+', required=True)
    p.add_argument('--classes', type=int, default=500)
    a = p.parse_args(argv)

    names_on_disk = {f.name for f in Path(a.test).rglob('*') if f.is_file()}
    base = None
    rows = []
    for c in a.csv:
        lab = load(c)
        v = violation(lab, a.classes)
        if len(lab) != len(names_on_disk):
            print(f'!! {Path(c).name}: {len(lab)} rows vs {len(names_on_disk)} test files')
            continue
        if base is None:
            base = lab
            changed = 0
        else:
            changed = sum(1 for x, y in zip(lab, base) if x[1] != y[1])
        rows.append((Path(c).name, v, changed))

    print(f"{'file':34s} {'rows':>7} {'viol%':>7} {'ceiling':>8} "
          f"{'min/max':>10} {'classes':>8} {'unused':>7} {'changed':>8}")
    for name, v, changed in rows:
        print(f'{name:34s} {v["n"]:>7,} {v["frac"]:>6.2f}% {v["ceiling"]:>7.2f}% '
              f'{v["pred_min"]:>4}/{v["pred_max"]:<5} {v["distinct"]:>8} '
              f'{v["unused"]:>7} {changed:>8,}')

    best = min(rows, key=lambda r: r[1]['violation'])
    print(f'\nlowest marginal violation: {best[0]} '
          f'({best[1]["frac"]:.2f}% -> ceiling {best[1]["ceiling"]:.2f}%)')
    print('cross-check with the hold-out macro from mres.py before submitting; '
          'if the two disagree, trust macro (the test is balanced, the hold-out is not).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
