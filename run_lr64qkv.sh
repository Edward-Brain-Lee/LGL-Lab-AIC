#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# COMBINED arm: --lora-rank 64 --lora-qkv -- PRODUCTION CHAIN ONLY.
#
#   bash smoke_lr64qkv.sh                  # 发车前闸门, separate script, read its output
#   nohup bash run_lr64qkv.sh > run_lr64qkv.log 2>&1 &
#
#   lr64qkv : --lora-rank 64 --lora-qkv  -> 10.325M or 10.326M, 48 adapter modules
#
# This is P0 of HANDOFF §39.5.  BOTH axes are measured now, and they DISAGREE
# with each other -- read these two lines together before interpreting the run:
#
#   placement  rank 32 -> +--lora-qkv  val_acc +0.42 -> 73.9077  board, ours (§39.1)
#   capacity   rank 32 -> rank 64      val_acc +0.68 -> no board score (teammate §42)
#
# The val_acc column ranks capacity ABOVE placement; the score column ranks it
# BELOW.  Do not pick between arms by val_acc.  §42.3 gives the mechanism: at
# ep20 train loss 0.0374 against val_loss 1.3691, a 37x gap, so the wider
# adapters buy label memorisation and the noisy hold-out rewards it.
#
#   rank 64 plain  7.966M trainable, val_acc 0.7639 (ep20).  This arm has NO
#                  board score as of 2026-10-06 (§42.1) -- it was never
#                  submitted.  That is the point of the two rows above: the axis
#                  `val_acc` likes most (+0.68) is the one with no board
#                  evidence behind it, while the axis it likes less (+0.42) is
#                  the one that actually moved the board (§39.2).
#
# The input axis was REOPENED 2026-10-06 ("只要能够带来最大增益，就可以用"), having
# been closed since 09-28.  So this chain now emits BOTH view sets off the one
# best.pt and the better of the two goes to the board.  They are two readings of
# one model -- never average them into a third answer.
#
# The two levers are orthogonal by construction -- one adds 12 modules to a new
# projection, the other widens the 36 modules that already exist -- so the
# question this run answers is whether their rates compound, add, or collide.
# It is also the only legal "fusion" of the three teammate arms: the rules
# forbid multi-model ensembling / averaging checkpoints, so combining has to
# happen inside one training run.
#
# ANSWERED 2026-10-07 (board): `sub_lr64qkv_scales` opened at 74.249, +0.3413
# over the incumbent's 73.9077 -- the largest single gain since placement.
#
# What the board can and cannot say:
#   * The submitted file is rank 64 + qkv + the 16-view `--tta scales` set in
#     ONE recipe.  There is no board score for rank 64 + qkv at 8 views, so the
#     compound / add / collide question this header poses is not separable from
#     this run alone.
#   * The ranking inversion this header warns about still stands and is the
#     headline: placement moved the board +0.9188 for a +0.42 val_acc move
#     (2.19 per val_acc point), while capacity moved val_acc +0.68 with no board
#     evidence behind it.  Row 15's 2.19 was the one-time PLACEMENT move, not a
#     property of the qkv path -- spending more parameters there buys hold-out
#     accuracy, not board accuracy.
#   * Submit the scales file, alone.  The 8v file is the same checkpoint under a
#     different view set; averaging the two would be multi-model fusion, which
#     the rules forbid.  The full write-up is HANDOFF §43.
#
# There is no control arm here on purpose.  The control already exists on the
# board: rank 32 + qkv (73.9077, measured) at the same 48-module placement.  The
# ctrl in run_qkv.sh exists to separate "qkv placement" from "just +1.18M more
# parameters" -- that question is orthogonal to this one and is not re-asked.
#
# Verification lives in smoke_lr64qkv.sh (selftest, the real tower, 20 real
# batches), deliberately NOT here: this script is the 3.4-hour job, and a gate
# buried in front of it is easy to scroll past.  Stage 0 refuses to start unless
# that gate has passed against the same train.py.  Everything left here is
# fail-loud too -- the run stops at the first stage that does not match, so a
# wrong tree / wrong flag / silently-absent adapter cannot reach a submission.
#
# Budget note: the board keeps the maximum over all submissions, so a loss here
# costs a slot and no points.
# ---------------------------------------------------------------------------
set -euo pipefail

ARM="${1:-lr64qkv}"
if [ "$ARM" != "lr64qkv" ]; then
  echo "usage: bash run_lr64qkv.sh   (no arm argument)" >&2; exit 2
fi

RANK=64
QKV_FLAG="--lora-qkv"
WANT_QKV="True"
WANT_LAYERS=48
# The base (classifier + --local-head + the 111.4k pos grid) is never printed on
# its own; it has to be backed out of a total.  Five independent 3-decimal prints
# bound it -- rank 16 2.658M, rank 32 4.428M, rank 64 7.966M (§42.1, whose
# adapter part is 110,592 x rank), plus our rank 32+qkv 5.607M and ctrl 5.644M --
# and their intersection is base in [888,556, 888,612).  That puts this run's
# total in [10,325,740, 10,325,796), i.e. 10.326M.  CONFIRMED by the 2026-10-06
# run, which printed exactly that.
#
# An earlier draft of this file claimed "it must print 10.325M".  That was an
# off-by-one at the rounding boundary: f'{n/1e6:.3f}' flips from 10.325 to 10.326
# at 10,325,500, not at 10,325,999.  The alternation below is the only reason
# that mistake did not false-fail the gate after 3.4 hours -- keep it as the
# guard it turned out to be, and keep it honest: the adapter count is separately
# and exactly pinned by smoke_lr64qkv.sh gate 1 (9,437,184).
WANT_PARAMS_RE='trainable params=(?:10\.325|10\.326)M'
OUT=outputs_384pe_lr64qkv
SUBBASE=sub_lr64qkv

PY="${PY:-python}"
DATA="${DATA:-/root/autodl-tmp/train}"
TEST="${TEST:-/root/autodl-tmp/test}"
RUNLOG="${RUNLOG:-${OUT}.log}"
N_ROWS=37444
N_BYTES=1610092

unset OMP_NUM_THREADS || true
# fail-fast write check before the 3.4 hours, not after
mkdir -p "${SUBBASE}_8v"

echo "=== [$ARM] $(date) ==="
echo "tree   : $PWD"
sha256sum train.py infer.py | cut -c1-16
"$PY" -c "import torch, open_clip; print('torch', torch.__version__, '| open_clip', open_clip.__version__)"

# ---- stage 0: preflight -------------------------------------------------------
# The smoke handshake first: refuse to spend 3.4 hours on a tree whose gate has
# not passed.  The marker records the train.py hash it was certified against, so
# editing the code after the smoke invalidates it instead of silently carrying
# the old certification over to a different tree.
marker="smoke_$ARM/OK"
want="$(sha256sum train.py | cut -c1-16)"
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  echo "WARNING: SKIP_SMOKE=1 -- starting with no smoke gate (HANDOFF §36.9)"
elif [ ! -f "$marker" ]; then
  echo "FATAL: no smoke gate for '$ARM' -- run this first:" >&2
  echo "         bash smoke_lr64qkv.sh" >&2; exit 1
elif [ "$(cut -d' ' -f1 "$marker")" != "$want" ]; then
  echo "FATAL: the smoke gate is STALE -- train.py changed since it passed" >&2
  echo "       (marker $(cut -d' ' -f1 "$marker") vs now $want).  Re-run:" >&2
  echo "         bash smoke_lr64qkv.sh" >&2; exit 1
else
  echo "smoke gate : PASSED for train.py $want ($(cut -d' ' -f3 "$marker"))"
fi

# bash has no compile step and these scripts cannot be syntax-checked on the
# Windows box they are edited on, so re-check both here -- a syntax error in the
# smoke is not caught by the smoke.
bash -n smoke_lr64qkv.sh && echo "bash -n smoke_lr64qkv.sh: ok"
bash -n run_lr64qkv.sh   && echo "bash -n run_lr64qkv.sh: ok"

# nothing else may be on this GPU (HANDOFF §33.3 / §36.2: two 384px jobs do not
# fit, and the failure mode is a confusing cuDNN error, not an OOM).  Rank 64
# doubles the trainable parameter count but not the activation memory, so the
# §39.1 rank-32-qkv footprint is still the right expectation here.
if pgrep -af 'train\.py' | grep -v grep; then
  echo "FATAL: a train.py is already running on this box" >&2; exit 1
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current
df -h "$PWD" | tail -1

# ---- stage 1: train (both axes at once) ---------------------------------------
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
    ("trainable params=10.325M or 10.326M", g(r"$WANT_PARAMS_RE")),
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
m = g(r"trainable params=([\d.]+)M")
print(f"trainable params as printed: {m.group(1)}M")
PYEOF

# ---- stage 2: checkpoint sanity -----------------------------------------------
"$PY" - <<PYEOF
import torch
ck = torch.load(r"$OUT/best.pt", map_location="cpu", weights_only=False)
assert ck["lora_rank"] == $RANK, ck["lora_rank"]
assert ck["lora_qkv"] is $WANT_QKV, ck["lora_qkv"]
n = sum(1 for k in ck["model"] if k.endswith(("attn.qkv.A", "attn.qkv.B")))
assert n == 24, f"qkv adapters in checkpoint: {n}"
print(f"ckpt ok: epoch {ck['epoch']} val_acc {ck['val_acc']:.4f} "
      f"val_acc_hi {ck['val_acc_hi']:.4f} qkv tensors {n}")
PYEOF

# ---- stage 3+4: inference + verification, once per view set -------------------
# Two submission files out of ONE best.pt.  This is not an ensemble: same
# checkpoint, same weights, same pipeline, only the view set over the same test
# images changes (infer.py:359 says the same of the TTA path itself), so it stays
# inside the rules' ban on multi-model fusion -- which is also why the two
# variants must never be averaged together into a third answer.
#
#   8v      bare --tta  = DEFAULT_TTA, 8 centre crops, the safe/known path.
#           infer.py:270-273 is `list(a.tta) or list(DEFAULT_TTA)`, so the empty
#           list a bare flag produces falls through to that group.
#   scales  --tta scales = DEFAULT_TTA + 8 finer centre crops (94/82/64/60%).
#           16 views, still all centre.  Corners are excluded on purpose
#           (infer.py:86 measured them at -0.278) and §42.8 blacklists --tta all,
#           so `all` is not run here either.
#
# 8v is emitted and PACKED before scales starts: scales doubles the view count,
# and if the box dies mid-pass the safe file is already sealed -- exactly what
# teammate §42.9.1 lost by queuing both packages to the end.
#
# --workers 4 --batch-size 64 is the §35.3 value; the default 256/8 OOMs a 62 GB
# container on 8-view TTA @384, and scales doubles that again.  Never pass
# --lora-rank/--lora-qkv here: infer.py reads rank / qkv / alpha back out of the
# checkpoint, and a flag that disagrees with the checkpoint is not an error --
# it is silently ignored along with the checkpoint's own value.
emit() {
  local tag="$1" views="$2"
  local dir="${SUBBASE}_${tag}"
  local csv="$dir/pred_results.csv"
  local zip="${SUBBASE}_${tag}.zip"
  mkdir -p "$dir"

  # $views is UNQUOTED on purpose: word splitting turns "" into zero words
  # (bare --tta -> the 8-view group) and "scales" into one.  `"$@"` would do the
  # same but only under `set -u` on bash >= 4.4, and this needs no such promise.
  echo "--- infer [$tag] -> $csv (views: ${views:-bare}) ---"
  "$PY" -u infer.py --test "$TEST" \
    --checkpoint "$OUT/best.pt" \
    --output "$csv" \
    --tta $views --workers 4 --batch-size 64

  # infer.py writes no header; the file is CRLF, 43 B/row (HANDOFF §36.5).
  # sed strips the \r so the awk NR count is per-line, not per-chunk.
  local sz rows uniq badfmt
  sz=$(stat -c %s "$csv")
  rows=$(sed 's/\r$//' "$csv" | awk 'END{print NR}')
  uniq=$(sed 's/\r$//' "$csv" | cut -d, -f1 | sort -u | wc -l)
  badfmt=$(sed 's/\r$//' "$csv" | grep -cvE '^[^,]+\.jpg,[0-9]{4}$' || true)
  echo "  [$tag] size=$sz rows=$rows unique_names=$uniq malformed=$badfmt"
  [ "$sz"     = "$N_BYTES" ] || { echo "FATAL [$tag]: size != $N_BYTES" >&2; exit 1; }
  [ "$rows"   = "$N_ROWS"  ] || { echo "FATAL [$tag]: rows != $N_ROWS" >&2; exit 1; }
  [ "$uniq"   = "$N_ROWS"  ] || { echo "FATAL [$tag]: duplicate file names" >&2; exit 1; }
  [ "$badfmt" = "0" ]        || { echo "FATAL [$tag]: malformed rows" >&2; exit 1; }

  (cd "$dir" && zip -q "../$zip" pred_results.csv)
  echo "  [$tag] packed: $zip $(stat -c %s "$zip") bytes"
}

echo "--- stage 3: inference, 2 view sets ---"
emit 8v ""
emit scales scales

echo "=== [$ARM] done $(date) ==="
echo "produced: ${SUBBASE}_8v.zip and ${SUBBASE}_scales.zip"
echo "next: score this checkpoint on the OFFICIAL training hold-out before"
echo "      spending a slot -- checkpoint_search.py runs that itself, no test set:"
echo "        py -3 checkpoint_search.py --data <official train dir> \\"
echo "            --checkpoints $OUT/best.pt --out analysis_$ARM"
echo "      Both files are the SAME checkpoint under two view sets -- they are"
echo "      two readings of one model, and averaging them into a third answer"
echo "      would be multi-model fusion, which the rules forbid.  Pick one."
echo "      The same-configuration noise floor is 0.03 (§41.2), so a hold-out"
echo "      gap under that does not by itself justify a submission slot."
