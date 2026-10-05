#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# QKV arm / matched-capacity control -- PRODUCTION CHAIN ONLY.
#
#   bash smoke_qkv.sh qkv        # 发车前闸门, separate script, read its output
#   nohup bash run_qkv.sh qkv  > run_qkv.log  2>&1 &
#   nohup bash run_qkv.sh ctrl > run_ctrl.log 2>&1 &
#
#   qkv  : --lora-rank 32 --lora-qkv   -> 5.607M params / 48 adapter modules
#   ctrl : --lora-rank 43              -> 5.644M params / 36 adapter modules
#
# ctrl is the capacity control: plain LoRA at the closest matching parameter
# count (+36,864 vs qkv's 5,607,170 = +0.66%).  Without it a qkv win cannot be
# separated from "it is just +1.18M more parameters" (提分路径研究.md §新排序 3).
#
# Verification lives in smoke_qkv.sh (selftest, the real tower, 20 real batches),
# deliberately NOT here: this script is the 3.5-hour job, and a gate buried in
# front of it is easy to scroll past.  Stage 0 refuses to start unless that gate
# has passed against the same train.py.  Everything left here is fail-loud too --
# the run stops at the first stage that does not match, so a wrong tree / wrong
# flag / silently-absent adapter cannot reach a submission.
# ---------------------------------------------------------------------------
set -euo pipefail

ARM="${1:-}"
case "$ARM" in
  qkv)  RANK=32; QKV_FLAG="--lora-qkv"; WANT_QKV="True";  WANT_LAYERS=48; WANT_PARAMS="5.607M"; OUT=outputs_384pe_lr32qkv; SUB=sub_lr32qkv ;;
  ctrl) RANK=43; QKV_FLAG="";            WANT_QKV="False"; WANT_LAYERS=36; WANT_PARAMS="5.644M"; OUT=outputs_384pe_lr43;    SUB=sub_lr43 ;;
  *) echo "usage: bash run_qkv.sh {qkv|ctrl}" >&2; exit 2 ;;
esac

PY="${PY:-python}"
DATA="${DATA:-/root/autodl-tmp/train}"
TEST="${TEST:-/root/autodl-tmp/test}"
RUNLOG="${RUNLOG:-${OUT}.log}"
N_ROWS=37444
N_BYTES=1610092

unset OMP_NUM_THREADS || true
mkdir -p "$SUB"

echo "=== [$ARM] $(date) ==="
echo "tree   : $PWD"
sha256sum train.py infer.py | cut -c1-16
"$PY" -c "import torch, open_clip; print('torch', torch.__version__, '| open_clip', open_clip.__version__)"

# ---- stage 0: preflight -------------------------------------------------------
# The smoke handshake first: refuse to spend 3.5 hours on a tree whose gate has
# not passed.  The marker records the train.py hash it was certified against, so
# editing the code after the smoke invalidates it instead of silently carrying
# the old certification over to a different tree.
marker="smoke_$ARM/OK"
want="$(sha256sum train.py | cut -c1-16)"
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  echo "WARNING: SKIP_SMOKE=1 -- starting with no smoke gate (HANDOFF §36.9)"
elif [ ! -f "$marker" ]; then
  echo "FATAL: no smoke gate for '$ARM' -- run this first:" >&2
  echo "         bash smoke_qkv.sh $ARM" >&2; exit 1
elif [ "$(cut -d' ' -f1 "$marker")" != "$want" ]; then
  echo "FATAL: the smoke gate is STALE -- train.py changed since it passed" >&2
  echo "       (marker $(cut -d' ' -f1 "$marker") vs now $want).  Re-run:" >&2
  echo "         bash smoke_qkv.sh $ARM" >&2; exit 1
else
  echo "smoke gate : PASSED for train.py $want ($(cut -d' ' -f3 "$marker"))"
fi

# nothing else may be on this GPU (HANDOFF §33.3 / §36.2: two 384px jobs do not
# fit, and the failure mode is a confusing cuDNN error, not an OOM)
if pgrep -af 'train\.py' | grep -v grep; then
  echo "FATAL: a train.py is already running on this box" >&2; exit 1
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current
df -h "$PWD" | tail -1

# ---- stage 1: train (single variable vs the rank-32 recipe) -------------------
echo "--- train -> $OUT ---"
"$PY" -u train.py --data "$DATA" \
  --out "$OUT" \
  --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
  --img-size 384 --train-pos-embed --local-head --save-every 4 \
  --lora-rank "$RANK" $QKV_FLAG 2>&1 | tee "$RUNLOG"

"$PY" - <<PYEOF
import re, sys
t = open(r"$RUNLOG", encoding="utf-8", errors="replace").read()
g = lambda p: re.search(p, t)
checks = [
    (f"trainable params=$WANT_PARAMS", g(rf"trainable params=$WANT_PARAMS")),
    (f"layers=$WANT_LAYERS",           g(rf"layers=$WANT_LAYERS ")),
    (f"lora_qkv=$WANT_QKV",            g(rf"lora_qkv=$WANT_QKV ")),
    (f"lora_rank=$RANK",               g(rf"lora_rank=$RANK ")),
    ("img_size=384",                   g(r"img_size=384")),
    ("local_head=True",                g(r"local_head=True")),
    ("train_pos_embed=True",           g(r"train_pos_embed=True")),
    ("pos grid trainable 111.4k",      g(r"positional grid is now trainable \(111\.4k")),
]
bad = [c for c, m in checks if not m]
if bad:
    print("GATE FAIL, missing from the log:", bad); sys.exit(1)
print("GATE OK:", ", ".join(c for c, _ in checks))
PYEOF

# ---- stage 2: checkpoint sanity -----------------------------------------------
"$PY" - <<PYEOF
import torch
ck = torch.load(r"$OUT/best.pt", map_location="cpu", weights_only=False)
assert ck["lora_rank"] == $RANK, ck["lora_rank"]
assert ck["lora_qkv"] is $WANT_QKV, ck["lora_qkv"]
n = sum(1 for k in ck["model"] if k.endswith(("attn.qkv.A", "attn.qkv.B")))
assert n == (24 if $WANT_QKV else 0), f"qkv adapters in checkpoint: {n}"
print(f"ckpt ok: epoch {ck['epoch']} val_acc {ck['val_acc']:.4f} "
      f"val_acc_hi {ck['val_acc_hi']:.4f} qkv tensors {n}")
PYEOF

# ---- stage 3: inference --------------------------------------------------------
# --workers 4 --batch-size 64 is the §35.3 value; the default 256/8 OOMs a 62 GB
# container on 8-view TTA @384.  Never pass --lora-rank/--lora-qkv here: infer.py
# reads rank / qkv / alpha back out of the checkpoint (probe.py:820's trap).
echo "--- infer -> $SUB/pred_results.csv ---"
"$PY" -u infer.py --test "$TEST" \
  --checkpoint "$OUT/best.pt" \
  --output "$SUB/pred_results.csv" \
  --tta --workers 4 --batch-size 64

# ---- stage 4: submission-file verification ------------------------------------
# infer.py writes no header; the file is CRLF, 43 B/row (HANDOFF §36.5).
sz=$(stat -c %s "$SUB/pred_results.csv")
rows=$(sed 's/\r$//' "$SUB/pred_results.csv" | awk 'END{print NR}')
uniq=$(sed 's/\r$//' "$SUB/pred_results.csv" | cut -d, -f1 | sort -u | wc -l)
badfmt=$(sed 's/\r$//' "$SUB/pred_results.csv" | grep -cvE '^[^,]+\.jpg,[0-9]{4}$' || true)
echo "size=$sz rows=$rows unique_names=$uniq malformed=$badfmt"
[ "$sz"   = "$N_BYTES" ] || { echo "FATAL: size != $N_BYTES" >&2; exit 1; }
[ "$rows" = "$N_ROWS"  ] || { echo "FATAL: rows != $N_ROWS" >&2; exit 1; }
[ "$uniq" = "$N_ROWS"  ] || { echo "FATAL: duplicate file names" >&2; exit 1; }
[ "$badfmt" = "0" ]      || { echo "FATAL: malformed rows" >&2; exit 1; }

(cd "$SUB" && zip -q "../${SUB}_tta8.zip" pred_results.csv)
echo "=== [$ARM] done: ${SUB}_tta8.zip $(stat -c %s "${SUB}_tta8.zip") bytes ==="
echo "next: pull $SUB/pred_results.csv to the local box and read it with"
echo "      D:\\BaiduDisk\\proxy_score_share\\score.py  -- do not spend a"
echo "      submission slot before that (rank-32 anchor reads 0.7298)."
