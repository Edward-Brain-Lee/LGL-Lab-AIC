#!/usr/bin/env bash
# Semifinal 416 arm: 20 epochs at batch 112 -- the recipe behind HANDOFF §48
# (best val_acc 0.7892; 2 h 25 m train + ~1 h of 16-view TTA on a 4090D 24 GB).
#
# Why batch 112 and not 128: 128 OOMs at 416 on 24 GB (HANDOFF §47.2 -- 22.97 GiB
# allocated, 0.08 GiB free at the first allocation).  112 peaks at 21,604 MiB and
# holds there for all 20 epochs, so the ceiling on this card is 112-127.
#
# Why 20 epochs and not 8: HANDOFF §47.4.  "ep4 == ep20" was measured on the
# earlier stages and does NOT carry over here -- on semifinal data the curve is
# still climbing at ep19 (0.7418 at ep8 -> 0.7834 at ep19 on 384).  A short cosine
# schedule just stops wherever the model was when the LR reached zero, which is
# why 416 + 8 epochs scored 72.8765 (-0.74) and 416 + 20 epochs does not.
#
#   nohup bash run_sf416e20.sh > run_sf416e20.log 2>&1 &
#
# Stages are chained so no wall-clock is lost between them; every stage echoes its
# exit code, and the CSV is gated by sf_gate.py before it is zipped.
set -u
cd "$(dirname "$0")"

# On AutoDL the conda interpreter is not on the non-interactive PATH, so
# `ssh host 'python ...'` dies on line one.  Resolve it explicitly.
PY="${PY:-/root/miniconda3/bin/python}"
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT=outputs_sf416e20
SUB=sub_sf416e20_scales

rm -rf "$OUT" "$SUB" "$SUB.zip"
mkdir -p "$SUB"

echo "=== TRAIN START ==="; date
"$PY" -u train.py --data train --out "$OUT" \
  --epochs 20 --warmup-epochs 3 --batch-size 112 --workers 8 \
  --img-size 416 --train-pos-embed --local-head --save-every 2 \
  --lora-rank 64 --lora-qkv > train_sf416e20.log 2>&1
echo "=== TRAIN EXIT $? ==="; date
grep -E "^epoch |best val_acc|out of memory" train_sf416e20.log

# OMP_NUM_THREADS=1 on inference only: the dataloader workers do the file reads,
# and letting OpenMP fan out too steals CPU from them without speeding up the GPU.
echo "=== INFER START ==="; date
OMP_NUM_THREADS=1 "$PY" -u infer.py --test test \
  --checkpoint "$OUT/best.pt" --output "$SUB/pred_results.csv" \
  --workers 4 --batch-size 64 --tta scales > infer_sf416e20.log 2>&1
echo "=== INFER EXIT $? ==="; date
tail -6 infer_sf416e20.log

echo "=== GATE ==="
"$PY" sf_gate.py csv --csv "$SUB/pred_results.csv" --test test
echo "=== GATE EXIT $? ==="

# The platform wants pred_results.zip containing exactly pred_results.csv.
"$PY" - "$SUB" <<'EOF'
import hashlib, pathlib, sys, zipfile

sub = pathlib.Path(sys.argv[1])
src = sub / 'pred_results.csv'
dst = sub.with_suffix('.zip')
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(src, arcname='pred_results.csv')
print(f'csv md5 {hashlib.md5(src.read_bytes()).hexdigest()}')
print(f'packed {dst} ({dst.stat().st_size} B) holding {src.name} ({src.stat().st_size} B)')
EOF

date
echo "=== ALL DONE ==="
