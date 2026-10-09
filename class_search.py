"""Read-only per-class comparison of candidate CSVs. Standard library only.

The reference can be training hold-out labels or a reconstructed proxy. This
tool never exports training weights, per-image patches, or a fused prediction.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


def load_predictions(path):
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as z:
            members = [n for n in z.namelist() if Path(n).name == 'pred_results.csv']
            if len(members) != 1:
                raise ValueError(f'{path}: expected exactly one pred_results.csv')
            lines = z.read(members[0]).decode('utf-8-sig').splitlines()
    else:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    result = {}
    for row in csv.reader(lines):
        if (len(row) != 2 or not row[0] or row[0] in result or
                len(row[1]) != 4 or not row[1].isascii() or not row[1].isdigit()):
            raise ValueError(f'{path}: invalid/duplicate row {row}')
        result[row[0]] = row[1]
    if not result:
        raise ValueError(f'{path}: empty predictions')
    return result


def wilson(k, n):
    if not n:
        return 0.0, 1.0
    z = 1.96
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def measure(pred, ref, base, excluded=frozenset(), counts=None):
    if pred.keys() != ref.keys() or base.keys() != ref.keys():
        raise ValueError('candidate/reference/baseline filename sets differ')
    totals, hits, oldhits, predicted = Counter(ref.values()), Counter(), Counter(), Counter(pred.values())
    pairs, gains, losses = Counter(), Counter(), Counter()
    newwin = oldwin = changes = excluded_changes = 0
    good = oldgood = good_n = 0
    for name, label in ref.items():
        hit, old = pred[name] == label, base[name] == label
        hits[label] += hit
        oldhits[label] += old
        gains[label] += hit and not old
        losses[label] += old and not hit
        newwin += hit and not old
        oldwin += old and not hit
        changes += pred[name] != base[name]
        if not hit:
            pairs[label, pred[name]] += 1
        if label not in excluded:
            good += hit
            oldgood += old
            good_n += 1
        else:
            excluded_changes += pred[name] != base[name]
    rows = []
    for label, n in sorted(totals.items()):
        lo, hi = wilson(hits[label], n)
        rows.append(dict(label=label, n=n, correct=hits[label], recall=hits[label] / n,
                         precision=hits[label] / max(predicted[label], 1), predicted=predicted[label],
                         baseline_recall=oldhits[label] / n, gain=gains[label], loss=losses[label],
                         net=gains[label] - losses[label], wilson_low=lo, wilson_high=hi,
                         uncertain_proxy=label in excluded,
                         train_count=(counts or {}).get(label, '')))
    total, current, previous = len(ref), sum(hits.values()), sum(oldhits.values())
    assert current - previous == newwin - oldwin
    stats = dict(n=total, accuracy=current / total, baseline_accuracy=previous / total,
                 delta_pp=100 * (current - previous) / total,
                 macro=sum(r['recall'] for r in rows) / len(rows), new_win=newwin, old_win=oldwin,
                 changed=changes, paired_z=(newwin - oldwin) / math.sqrt(max(newwin + oldwin, 1)),
                 reliable_n=good_n, reliable_accuracy=good / max(good_n, 1),
                 reliable_delta_pp=100 * (good - oldgood) / max(good_n, 1),
                 uncertain_changed=excluded_changes,
                 # Worst case if every changed uncertain row favours one arm.
                 delta_lower_pp=100 * (good - oldgood - excluded_changes) / total,
                 delta_upper_pp=100 * (good - oldgood + excluded_changes) / total)
    if counts:
        ordered = sorted((c for c in totals if c in counts), key=lambda c: (counts[c], c))
        tail = ordered[:max(1, math.ceil(len(ordered) * .2))]
        stats['tail20_macro'] = sum(hits[c] / totals[c] for c in tail) / max(len(tail), 1)
        tiny = [c for c in ordered if counts[c] < 50]
        stats['tiny_classes'] = len(tiny)
        stats['tiny_remaining_pp'] = 100 * sum(totals[c] - hits[c] for c in tiny) / total
    return stats, rows, pairs


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(a):
    ref, base = load_predictions(a.reference), load_predictions(a.baseline)
    excluded = set()
    if a.uncertain_blocks:
        with open(a.uncertain_blocks, encoding='utf-8-sig', newline='') as f:
            excluded = {r['assigned'] for r in csv.DictReader(f)}
    counts = None
    if a.train_counts:
        counts = json.loads(Path(a.train_counts).read_text(encoding='utf-8'))
        if not isinstance(counts, dict) or any(not isinstance(v, int) or v < 0 for v in counts.values()):
            raise ValueError('train-counts must map four-digit class names to nonnegative integers')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ranking, summaries = [], []
    for index, path in enumerate(a.candidates):
        pred = load_predictions(path)
        stats, rows, pairs = measure(pred, ref, base, excluded, counts)
        tag = f'{index:02d}_{Path(path).stem}'
        write_csv(out / f'{tag}_classes.csv', rows, list(rows[0]))
        confusions = [dict(given=c, predicted=p, n=n) for (c, p), n in pairs.most_common()]
        write_csv(out / f'{tag}_confusions.csv', confusions, ['given', 'predicted', 'n'])
        ranking.append(dict(candidate=str(Path(path).resolve()), **stats))
        reliable = [r for r in rows if not r['uncertain_proxy']]
        worst = sorted(reliable, key=lambda r: (r['recall'], r['label']))[:10]
        best = sorted(reliable, key=lambda r: (-r['recall'], r['label']))[:10]
        summaries.append(f"{path}: {100*stats['accuracy']:.4f}% delta {stats['delta_pp']:+.4f} pp; "
                         f"wins/losses {stats['new_win']}/{stats['old_win']}\n"
                         '  worst: ' + ', '.join(f"{r['label']}={r['recall']:.2%}" for r in worst) + '\n'
                         '  best: ' + ', '.join(f"{r['label']}={r['recall']:.2%}" for r in best))
    ranking.sort(key=lambda r: r['accuracy'], reverse=True)
    write_csv(out / 'ranking.csv', ranking, list(ranking[0]))
    manifest = dict(reference=str(Path(a.reference).resolve()),
                    reference_sha256=hashlib.sha256(Path(a.reference).read_bytes()).hexdigest(),
                    reference_kind=a.reference_kind, baseline=str(Path(a.baseline).resolve()),
                    excluded_classes=sorted(excluded), ranking=ranking,
                    candidate_sha256={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                      for p in a.candidates})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    note = ('Reconstructed-test agreement is diagnostic, not measured label accuracy. '
            'Do not fit class weights, offsets or per-image choices to this reference.'
            if a.reference_kind == 'reconstructed-test' else
            'Hold-out score measures noisy official training labels. Confirm generalisation independently.')
    (out / 'report.txt').write_text(note + '\n\n' + '\n\n'.join(summaries) + '\n', encoding='utf-8')
    print(note)
    print('\n\n'.join(summaries))
    print(f'Reports saved to {out.resolve()}')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--reference-kind', required=True, choices=['training-holdout', 'reconstructed-test'])
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidates', nargs='+', required=True)
    p.add_argument('--uncertain-blocks', default='')
    p.add_argument('--train-counts', default='', help='optional JSON from official training-folder counts')
    p.add_argument('--out', default='analysis_class_search')
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
