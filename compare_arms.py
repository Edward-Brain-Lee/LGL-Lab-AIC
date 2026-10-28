"""Judge an attention arm against the baseline the way this project says to.

HANDOFF 45.8 is explicit that the decision must NOT be made on global val_acc
(resolution ~1 point, while these effects are 0.3-0.8) and that the judges are:
the recall of the 78 "enough data but can't learn" classes, and the paired
win/loss decomposition.

    python compare_arms.py --base attn_audit_outputs_sf_base.csv \
                           --arm  attn_audit_outputs_sf_se.csv \
                           [--classes attn_audit_outputs_sf_base.classes.json]

Input rows come from attn_audit.py (idx, label, pred, correct, ...).  Rows are
joined on `idx`, the hold-out row index, which is stable across runs because the
split is deterministic -- so this is a genuine paired comparison, not two
independent accuracies.

The report carries the project's own self-check (HANDOFF 43.4): the net exchange
must equal (new - old) x n.  If it does not, the join or the counting is wrong and
nothing else in the table can be trusted.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict


def load(path):
    rows = {}
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows[int(r['idx'])] = dict(label=int(r['label']), pred=int(r['pred']),
                                       correct=int(r['correct']))
    if not rows:
        raise SystemExit(f'{path} has no rows')
    return rows


def sign_test(only_new, only_old):
    """Exact two-sided sign test on the rows where the two arms disagree."""
    n = only_new + only_old
    if n == 0:
        return 1.0
    k = min(only_new, only_old)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def compare(base, arm, train_counts=None, weak_min=100, weak_recall=0.6):
    shared = sorted(set(base) & set(arm))
    if not shared:
        raise SystemExit('the two files share no idx - are they from the same hold-out?')
    if len(shared) != len(base) or len(shared) != len(arm):
        print(f'WARNING: {len(base)} base rows, {len(arm)} arm rows, {len(shared)} shared; '
              f'comparing only the shared ones', file=sys.stderr)

    n = len(shared)
    n_a = sum(base[i]['correct'] for i in shared)
    n_b = sum(arm[i]['correct'] for i in shared)
    only_old = sum(1 for i in shared if base[i]['correct'] and not arm[i]['correct'])
    only_new = sum(1 for i in shared if arm[i]['correct'] and not base[i]['correct'])
    both_wrong = sum(1 for i in shared if not base[i]['correct'] and not arm[i]['correct'])

    per_cls_n, per_cls_a, per_cls_b = defaultdict(int), defaultdict(int), defaultdict(int)
    for i in shared:
        c = base[i]['label']
        per_cls_n[c] += 1
        per_cls_a[c] += base[i]['correct']
        per_cls_b[c] += arm[i]['correct']
    rec_a = {c: per_cls_a[c] / per_cls_n[c] for c in per_cls_n}
    rec_b = {c: per_cls_b[c] / per_cls_n[c] for c in per_cls_n}

    if train_counts is None:
        weak = {c for c in per_cls_n if rec_a[c] < weak_recall}
        weak_rule = f'recall<{weak_recall} (no training counts supplied)'
    else:
        weak = {c for c in per_cls_n
                if train_counts.get(c, 0) >= weak_min and rec_a[c] < weak_recall}
        weak_rule = f'train>={weak_min} and recall<{weak_recall}'

    return dict(
        n=n, acc_a=n_a / n, acc_b=n_b / n, delta=(n_b - n_a) / n,
        only_old=only_old, only_new=only_new, both_wrong=both_wrong,
        net=only_new - only_old, p=sign_test(only_new, only_old),
        discriminating=only_new + only_old,
        weak=sorted(weak), weak_rule=weak_rule, n_weak=len(weak),
        weak_recall_a=(sum(per_cls_a[c] for c in weak) / sum(per_cls_n[c] for c in weak)
                       if weak else float('nan')),
        weak_recall_b=(sum(per_cls_b[c] for c in weak) / sum(per_cls_n[c] for c in weak)
                       if weak else float('nan')),
        per_class={c: (rec_a[c], rec_b[c], per_cls_n[c]) for c in per_cls_n},
    )


def report(s, noise_floor=0.03, resolution=0.07):
    print(f'paired rows      : {s["n"]}')
    print(f'accuracy         : base {s["acc_a"]:.6f}  arm {s["acc_b"]:.6f}  '
          f'delta {s["delta"]:+.6f}')
    print()
    print('paired win/loss (rows the two arms disagree on)')
    print(f'  arm wins       : {s["only_new"]}')
    print(f'  base wins      : {s["only_old"]}')
    print(f'  net exchange   : {s["net"]:+d}  = {s["net"] / s["n"] * 100:+.4f} pp')
    print(f'  discriminating : {s["discriminating"]} rows, both wrong on {s["both_wrong"]}')
    print(f'  sign test p    : {s["p"]:.4f}'
          f'{"" if s["p"] < 0.05 else "   (not significant)"}')
    print()
    print(f'weak classes     : {s["n_weak"]} under "{s["weak_rule"]}"')
    print(f'  recall base    : {s["weak_recall_a"]:.4f}')
    print(f'  recall arm     : {s["weak_recall_b"]:.4f}')
    print(f'  delta          : {s["weak_recall_b"] - s["weak_recall_a"]:+.4f}')
    print()
    moved = sorted(((b - a, c, a, b, n) for c, (a, b, n) in s['per_class'].items()),
                   key=lambda t: t[0])
    # split by sign so the same class can never be listed as both worse and better
    worse = [t for t in moved if t[0] < 0][:3]
    better = [t for t in moved if t[0] > 0][-3:][::-1]
    if worse or better:
        print('per-class recalls that moved most (arm - base)')
        for tag, group in (('worse', worse), ('better', better)):
            for d, c, a, b, n in group:
                print(f'  {tag:6s} class {c:4d}  {a:.3f} -> {b:.3f}  ({d:+.3f}, n={n})')
        print()
    # the project's own invariant (HANDOFF 43.4)
    identity = s['net'] / s['n']
    if abs(identity - s['delta']) > 1e-9:
        print(f'SELF-CHECK FAILED: net/n = {identity:.9f} but delta = {s["delta"]:.9f}. '
              f'The join or the counting is wrong; none of the above can be trusted.')
        return False
    print(f'self-check       : net/n == delta ({identity:.9f}) OK')
    if abs(s['delta']) < noise_floor:
        print(f'VERDICT          : inside the 0.03 seed-noise floor at this hold-out size - '
              f'do not spend a submission slot on it.')
    elif s['p'] >= 0.05:
        print('VERDICT          : outside the noise floor but NOT significant '
              '(p>=0.05) - treat as inconclusive.')
    elif s['delta'] < 0:
        print('VERDICT          : the arm is worse; keep the incumbent.')
    else:
        print(f'VERDICT          : arm better by {s["delta"]:+.4f} with p={s["p"]:.4f}. '
              f'Check it clears the local proxy resolution (~{resolution}) before '
              f'committing a submission slot.')
    return True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--base', required=True, help='audit CSV of the incumbent')
    p.add_argument('--arm', required=True, help='audit CSV of the attention arm')
    p.add_argument('--classes', default=None,
                   help='classes.json from the baseline audit (training counts + weak set)')
    p.add_argument('--noise-floor', type=float, default=0.03)
    p.add_argument('--resolution', type=float, default=0.07)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    base, arm = load(a.base), load(a.arm)
    counts = None
    if a.classes:
        with open(a.classes, encoding='utf-8') as fh:
            blob = json.load(fh)
        counts = {int(k): v for k, v in blob['train_counts'].items()}
    s = compare(base, arm, counts)
    return 0 if report(s, a.noise_floor, a.resolution) else 2


if __name__ == '__main__':
    raise SystemExit(main())
