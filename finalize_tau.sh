#!/usr/bin/env bash
# Gate, compare and pack every tau CSV produced by one inference pass.
#
#   bash finalize_tau.sh sub_tau
#
# For each pred_results*.csv it runs the official-shape gate (row count, class range,
# every test name exactly once), then prints the label-free marginal violation for all
# of them side by side, then zips each into its own pred_results.zip so exactly one can
# be handed to the platform without touching the others.
set -u
cd "$(dirname "$0")"
SUB="${1:-sub_tau}"
PY="${PY:-/root/miniconda3/bin/python}"
export PATH=/root/miniconda3/bin:$PATH

shopt -s nullglob
csvs=("$SUB"/pred_results*.csv)
if [ ${#csvs[@]} -eq 0 ]; then echo "no CSVs in $SUB yet"; exit 1; fi

echo "=== 1. shape gate, one file at a time ==="
for f in "${csvs[@]}"; do
    printf '%-40s ' "$f"
    "$PY" sf_gate.py csv --csv "$f" --test test 2>&1 | tail -1
done

echo
echo "=== 2. marginal violation (label-free structural metric) ==="
"$PY" select_tau.py --test test --csv "${csvs[@]}" 2>&1

echo
echo "=== 3. pack each one separately ==="
for f in "${csvs[@]}"; do
    "$PY" - "$f" <<'EOF'
import hashlib, pathlib, sys, zipfile
src = pathlib.Path(sys.argv[1])
dst = src.with_name('pred_results.zip')
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(src, arcname='pred_results.csv')
print(f'{src.name:34s} -> {dst}  {dst.stat().st_size:>9,} B  '
      f'md5 {hashlib.md5(src.read_bytes()).hexdigest()}')
EOF
done

echo
echo "NOTE: all zips are named pred_results.zip in their own subdir; copy the one you"
echo "      chose to a distinct path before uploading, e.g."
echo "      cp $SUB/pred_results.zip /root/autodl-tmp/submit_tau100.zip"
