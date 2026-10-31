"""Pick the checkpoint to submit from several validate_tta.py reports.

The rule this replaces
----------------------
The training script writes ``best.pt`` by ``--select val_acc`` -- a single-view
micro accuracy over a hold-out that shares the training set's label noise.  That
metric has now inverted the ranking twice on this project:

  * HANDOFF §51.4: micro preferred tau=0.3 while the board preferred tau=1.0;
  * HANDOFF §53.4: 416+SE had LOWER single-view val_acc than 384+SE (0.7877 vs
    0.7889) but HIGHER 16-view TTA micro (0.7948 vs 0.7898) and +0.1832 on the
    board.

So ``best.pt`` is not evidence of the best checkpoint, and a stage-2 snapshot
that improves the 16-view read while nudging single-view val_acc down would be
discarded silently.  This picks by the metric with the 2-for-2 record: the
16-view ``scales`` TTA micro computed on the same fixed hold-out split.

Usage (stdout is machine-readable):
    python pick_checkpoint.py --out DIR --holdout 'best.pt=best.json' ... [--classes 500]
prints exactly one line: ``CHOSEN <path> <micro>`` (or exits non-zero).
"""
import argparse
import json
import sys
from pathlib import Path


def load(path):
    with open(path, encoding='utf-8') as f:
        pay = json.load(f)
    metrics = pay.get('metrics') or {}
    # rank by tau=1.0 when it exists (that is the read we submit), else tau=0
    key = '1.0' if '1.0' in metrics else (sorted(metrics)[-1] if metrics else None)
    if key is None:
        return None
    m = metrics[key]
    return dict(tau=key, micro=float(m['top1_micro']), macro=float(m['top1_macro']),
                n=int(m.get('n', pay.get('n', 0))), views=pay.get('views'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True, help='training output dir (holds clean_stage_report.json)')
    p.add_argument('--holdout', nargs='+', required=True, metavar='CKPT=JSON')
    p.add_argument('--classes', type=int, default=500)
    p.add_argument('--min-clean-frac', type=float, default=0.30,
                   help='below this the stage-2 selection is treated as collapsed')
    p.add_argument('--allow-collapsed', action='store_true',
                   help='keep stage-2 snapshots even if the clean set collapsed')
    a = p.parse_args(argv)

    report = {}
    rp = Path(a.out) / 'clean_stage_report.json'
    collapsed, why = False, ''
    if rp.is_file():
        with rp.open(encoding='utf-8') as f:
            report = json.load(f)
        frac = float(report.get('clean_fraction', 0.0))
        ncls = int(report.get('classes_with_clean', 0))
        if frac < a.min_clean_frac:
            collapsed, why = True, f'clean_fraction={frac:.3%} < {a.min_clean_frac:.0%}'
        elif ncls and ncls < a.classes:
            collapsed, why = True, f'classes_with_clean={ncls} < {a.classes}'
    else:
        print('# no clean_stage_report.json -- stage 2 did not run or did not report',
              file=sys.stderr)

    rows = []
    for spec in a.holdout:
        if '=' not in spec:
            raise SystemExit(f'--holdout expects CKPT=JSON, got {spec!r}')
        ck, js = spec.rsplit('=', 1)
        name = Path(ck).name
        if not Path(ck).is_file():
            print(f'# skip {name}: checkpoint missing', file=sys.stderr)
            continue
        if not Path(js).is_file():
            print(f'# skip {name}: report missing', file=sys.stderr)
            continue
        if collapsed and not a.allow_collapsed and name.startswith('clean_ep'):
            print(f'# skip {name}: stage-2 selection collapsed ({why})', file=sys.stderr)
            continue
        r = load(js)
        if r is None:
            print(f'# skip {name}: no metrics in {js}', file=sys.stderr)
            continue
        rows.append((r['micro'], ck, r, name))
        print(f'# {name:>16s} tau={r["tau"]:>4s} micro={r["micro"]:.4f} '
              f'macro={r["macro"]:.4f} n={r["n"]}', file=sys.stderr)

    if not rows:
        raise SystemExit('no candidate had a usable hold-out report')

    rows.sort(key=lambda t: (-t[0], t[1]))
    best = rows[0]
    if len(rows) > 1:
        gap = (rows[0][0] - rows[1][0]) * 100
        print(f'# margin over runner-up: {gap:+.3f} pp '
              f'(hold-out resolution ~0.07 pp; below it this is a coin flip)', file=sys.stderr)
    if collapsed:
        print(f'# WARNING stage-2 collapsed: {why}', file=sys.stderr)
    print(f'CHOSEN {best[1]} {best[0]:.6f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
