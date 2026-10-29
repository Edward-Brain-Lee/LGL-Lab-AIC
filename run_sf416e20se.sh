#!/usr/bin/env bash
# Semifinal COMBINED arm: 416 px + --local-se, 20 epochs at batch 112.
#
# Why this arm exists (HANDOFF §48 + §49): 416 and SE are the only two levers that
# have a POSITIVE single-variable measurement on semifinal data, and they were
# measured separately --
#
#   416 px, no SE   (§48)  val_acc 0.7892  (384 same-epoch 0.7820, +0.72)
#      --local-se   (§49)  val_acc 0.7889  (384 same-epoch 0.7836, +0.54) -> board 73.8599
#
# The rules forbid ensembling, so "both" can only exist inside ONE training run --
# the same argument that produced rank64+qkv in 复赛 (§43.2).  This is that run.
#
# Two things this arm is NOT allowed to change:
#   * batch 112 stays (128 OOMs at 416 on 24 GB, §47.2);
#   * epochs stay 20 (a short cosine just stops where it was, §47.4).
# Everything else is the 73.6191/73.8599 recipe verbatim.
#
# Expected trainable params: 10,291,507 = 10.292M
#   adapters 9,437,184 + head 256,001 + local 393,217 + SE 74,545 + pos 130,560 (170 tokens)
# The gate below pins that number, plus local_se=True and img_size=416, because a
# silently-absent SE tensor is exactly the failure mode a 3-hour run must not hide.
#
#   nohup bash run_sf416e20se.sh > run_sf416e20se.log 2>&1 &
set -u
cd "$(dirname "$0")"

PY="${PY:-/root/miniconda3/bin/python}"
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# TF32 OFF is load-bearing on this host.  With TF32 on, 416 px produced loss=nan from
# the first batch in every configuration (bare, +SE, fp32, math-only SDPA); with
# NVIDIA_TF32_OVERRIDE=0 the same run is finite (loss 6.7093 at step 1).  The tower
# itself is fine at 416 in fp32, so the NaN is in the TF32 GEMM path of the training
# step.  Cost: small -- autocast bf16 already does the heavy matmuls.
export NVIDIA_TF32_OVERRIDE=0

# Batch 72, not 112: on THIS host the 24 GB card returns NaN (not OOM) once the batch
# reaches 80 at 416 px -- the training loss stays finite but the hold-out forward returns
# NaN.  72 is the largest value probed that keeps BOTH the train and val loops finite.
# This is a silent-corruption boundary, not a numerical-stability one: bs64 and bs72 are
# clean, bs80/96/112 are not.  Smaller batch also means more steps per epoch (~1131).

OUT=outputs_sf416e20se
SUB=sub_sf416e20se_scales
LOG=train_sf416e20se.log

rm -rf "$OUT" "$SUB" "$SUB.zip"
mkdir -p "$SUB"

echo "=== TRAIN START ==="; date
"$PY" -u train.py --data train --out "$OUT" \
  --epochs 20 --warmup-epochs 3 --batch-size 72 --workers 8 \
  --img-size 416 --train-pos-embed --local-head --local-se --save-every 2 \
  --lora-rank 64 --lora-qkv > "$LOG" 2>&1
rc=$?
echo "=== TRAIN EXIT $rc ==="; date
grep -E "^epoch |best val_acc|out of memory" "$LOG"
[ $rc -eq 0 ] || { echo "TRAIN FAILED, stopping"; exit $rc; }

echo "=== GATE: config actually used ==="
"$PY" - "$LOG" <<'EOF' || exit 1
import re, sys
t = open(sys.argv[1], encoding='utf-8', errors='replace').read()
need = [('trainable params=10.292M', r'trainable params=10\.292M'),
        ('lora_rank=64',            r'lora_rank=64(?=\s|$)'),
        ('lora_qkv=True',           r'lora_qkv=True(?=\s|$)'),
        ('local_head=True',         r'local_head=True(?=\s|$)'),
        ('local_se=True',           r'local_se=True(?=\s|$)'),
        ('img_size=416',            r'img_size=416(?=\s|$)'),
        ('train_pos_embed=True',    r'train_pos_embed=True(?=\s|$)'),
        ('layers=48',               r'layers=48(?=\s|$)'),
        ('classes=500',             r'classes=500(?=\s|$)'),
        ('positional grid 13x13',   r'7x7 -> 13x13'),
        ('best val_acc printed',    r'best val_acc =')]
bad = [n for n, p in need if not re.search(p, t)]
if bad:
    print('GATE FAIL, missing:', bad); sys.exit(1)
if 'Traceback' in t:
    print('GATE FAIL: traceback in log'); sys.exit(1)
print('GATE OK:', ', '.join(n for n, _ in need))
EOF

echo "=== INFER START ==="; date
OMP_NUM_THREADS=1 "$PY" -u infer.py --test test \
  --checkpoint "$OUT/best.pt" --output "$SUB/pred_results.csv" \
  --workers 4 --batch-size 64 --tta scales `
  --save-logits /root/autodl-tmp/outputs_sf416e20se_test_logits.npz > infer_sf416e20se.log 2>&1
echo "=== INFER EXIT $? ==="; date
tail -6 infer_sf416e20se.log

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
