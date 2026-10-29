#!/usr/bin/env bash
# 384 + --local-se, 20 epochs, batch 128 -- the semifinal recipe that RUNS on this box.
#
# Why 384 and not 416: on this host (torch 2.3.0) every 416 configuration produced
# NaN within two batches -- bare, with SE, in fp32, and with the math-only SDPA
# backend.  The teammates' 416 run (val_acc 0.7892) was on torch 2.8.  So on THIS
# box the only measured-positive combination that actually trains is 384 + SE.
#
# Every switch below has single-variable evidence on semifinal data:
#   384 + rank64 + qkv + PE + local-head  -> 73.6191   (our own run)
#   + --local-se                          -> 73.8599   (teammates, same 384 recipe)
# Nothing else is enabled: attn-temp / local-attn / train-ln / junk / margin /
# SupCon all lack single-variable evidence here, and this run must stay attributable.
#
# The post-hoc evaluation (per-class bias fitted on the official hold-out) is NOT in
# this script -- it runs afterwards on the saved logits, and only if it is positive.
set -u
cd "$(dirname "$0")"

PY="${PY:-/root/miniconda3/bin/python}"
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT=outputs_sf384se
SUB=sub_sf384se_scales
LOG=train_sf384se.log

rm -rf "$OUT" "$SUB" "$SUB.zip"
mkdir -p "$SUB"

echo "=== TRAIN START ==="; date
"$PY" -u train.py --data train --out "$OUT" \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --img-size 384 --train-pos-embed --local-head --local-se --save-every 2 \
  --lora-rank 64 --lora-qkv > "$LOG" 2>&1
rc=$?
echo "=== TRAIN EXIT $rc ==="; date
grep -E "^epoch |best val_acc|out of memory" "$LOG"
[ $rc -eq 0 ] || { echo "TRAIN FAILED"; exit $rc; }

echo "=== GATE: config + no NaN ==="
"$PY" - "$LOG" <<'EOF' || exit 1
import re, sys
t = open(sys.argv[1], encoding='utf-8', errors='replace').read()
need = [('trainable params=10.272M', r'trainable params=10\.272M'),
        ('local_se=True',  r'local_se=True(?=\s|$)'),
        ('img_size=384',   r'img_size=384(?=\s|$)'),
        ('lora_rank=64',   r'lora_rank=64(?=\s|$)'),
        ('lora_qkv=True',  r'lora_qkv=True(?=\s|$)'),
        ('layers=48',      r'layers=48(?=\s|$)'),
        ('classes=500',    r'classes=500(?=\s|$)'),
        ('best val_acc printed', r'best val_acc =')]
bad = [n for n, p in need if not re.search(p, t)]
if bad:
    print('GATE FAIL, missing:', bad); sys.exit(1)
if re.search(r'loss=nan', t):
    print('GATE FAIL: loss=nan in the log'); sys.exit(1)
if 'Traceback' in t:
    print('GATE FAIL: traceback'); sys.exit(1)
print('GATE OK:', ', '.join(n for n, _ in need))
EOF

echo "=== INFER START (16 views + logits) ==="; date
OMP_NUM_THREADS=1 "$PY" -u infer.py --test test \
  --checkpoint "$OUT/best.pt" --output "$SUB/pred_results.csv" \
  --workers 4 --batch-size 64 --tta scales \
  --save-logits "/root/autodl-tmp/${OUT}_test_logits.npz" > infer_sf384se.log 2>&1
echo "=== INFER EXIT $? ==="; date
tail -6 infer_sf384se.log

echo "=== CSV GATE ==="
"$PY" sf_gate.py csv --csv "$SUB/pred_results.csv" --test test || exit 1

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
