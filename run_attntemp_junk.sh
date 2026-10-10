#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# TWO single-variable arms off the SAME recipe (384 + --train-pos-embed +
# --local-head + --lora-rank 64 --lora-qkv), run back to back:
#
#   attntemp : + --attn-temp                 per-head learnable softmax temperature
#   junk     : + --junk-filter both          frozen-tower 杂图 / 重复图 screening
#
#   bash smoke_attntemp_junk.sh                     # 发车前闸门, separate script
#   nohup bash run_attntemp_junk.sh > run_attntemp_junk.log 2>&1 &
#
# Each arm is gated independently and stage 0 refuses to start an arm whose smoke
# has not passed against the same train.py.  `set -e` means a gate failure in the
# first arm stops the second -- deliberate: a broken tree should not be allowed to
# burn a second 3.4 hours.
#
# WHY THESE TWO.  The incumbent's placement is fixed and its capacity axis is
# closed, so the only remaining moves are NEW placement positions.  These are the
# two cheapest honest ones (HANDOFF §42.4/§43.5, 注意力机制决策与候选池.md §2):
#
#   --attn-temp  144 scalars, bit-exact identity at step 0 (rho = 0 -> tau = 1),
#                and it acts on the softmax logits rather than on another low-rank
#                block of a weight matrix, so it is a genuinely new placement.
#                Precedent for "the attention of a transferred ViT is wrong":
#                Zou et al. NeurIPS 2024; our own --local-head (+0.0641) is the
#                in-house evidence that CLS-only attention is a real weakness.
#   --junk-filter  the existing noise machinery is entirely LABEL-side (it only
#                asks whether the teacher's top-1 equals the given label), so a
#                picture that belongs to no class at all passes it untouched.
#                This screens the image side, on frozen-CLIP features, demotions
#                only -- never deletes, never relabels, never reorders, because
#                WeightedRandomSampler and every tracker index are positional.
#
# EXPECTATION, stated before the fact so the result cannot be retro-fitted:
# the temperature paper's own dissenting evidence (UT Austin, NormSoftmax) found
# learnable temperatures gains 不明显, and §41.4 says the proxy resolves only
# +-0.07 with a 0.03 floor.  Both arms are therefore expected to land in the
# +-0.1 band.  A null result here is a real result -- it closes a placement --
# and neither arm is allowed to blend with the other (that would be fusion).
#
# Budget: 2 x ~3.4h train + 4 inference passes.  The board keeps the maximum over
# all submissions, so a loss costs a slot and no points.
# ---------------------------------------------------------------------------
set -euo pipefail

if [ $# -gt 0 ]; then
  echo "usage: bash run_attntemp_junk.sh   (no arm argument)" >&2; exit 2
fi

RANK=64
WANT_LAYERS=48
WANT_PARAMS_RE='trainable params=(?:10\.325|10\.326)M'
# train.py's own defaults, passed explicitly so the log's config line shows them.
# §21.3: the strength of centroid filtering was never calibrated, so this arm
# moves the mechanism only and does not stack a second unmeasured choice on it.
JUNK_FLOOR=0.2
JUNK_MAX_FRAC=0.05

PY="${PY:-python}"
DATA="${DATA:-/root/autodl-tmp/train}"
TEST="${TEST:-/root/autodl-tmp/test}"
N_ROWS=37444
N_BYTES=1610092

unset OMP_NUM_THREADS || true

echo "=== [attntemp+junk] $(date) ==="
echo "tree   : $PWD"
sha256sum train.py infer.py | cut -c1-16
"$PY" -c "import torch, open_clip; print('torch', torch.__version__, '| open_clip', open_clip.__version__)"

# ---- stage 3+4: inference + verification, once per view set -------------------
# Two files out of ONE best.pt per arm.  Not an ensemble: same checkpoint, same
# weights, only the view set moves -- which is also why the two variants must
# never be averaged into a third answer, and why arms are never averaged either.
#
#   8v      bare --tta   = DEFAULT_TTA, 8 centre crops (infer.py:270-273 turns the
#                          empty list back into the default group).
#   scales  --tta scales = DEFAULT_TTA + 8 finer centre crops, 16 views.
#
# Corners are excluded (infer.py:86: -0.278) and §42.8 blacklists --tta all, so
# `all` is not run.  --workers 4 --batch-size 64 is the §35.3 value: the default
# 256/8 OOMs a 62 GB container on 8-view TTA @384 and scales doubles that again.
emit() {
  local out="$1" subbase="$2" tag="$3" views="$4"
  local dir="${subbase}_${tag}"
  local csv="$dir/pred_results.csv"
  local zip="${subbase}_${tag}.zip"
  local sz rows uniq badfmt
  mkdir -p "$dir"

  # $views is UNQUOTED on purpose: word splitting turns "" into zero words (bare
  # --tta -> the 8-view group) and "scales" into one.
  echo "--- infer [$subbase/$tag] -> $csv (views: ${views:-bare}) ---"
  "$PY" -u infer.py --test "$TEST" \
    --checkpoint "$out/best.pt" \
    --output "$csv" \
    --tta $views --workers 4 --batch-size 64

  # infer.py writes no header; the file is CRLF, 43 B/row (HANDOFF §36.5).
  # sed strips the \r so the awk NR count is per-line, not per-chunk.
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

run_arm() {
  local ARM="$1" EXTRA="$2" WANT_EXTRA_RE="$3"
  local OUT="${4}" SUBBASE="${5}" RUNLOG="${6}"

  echo "=== arm [$ARM] $(date) ==="

  # ---- stage 0: preflight -----------------------------------------------------
  # The smoke handshake first: the marker records the train.py hash it was
  # certified against, so editing the code after the smoke invalidates it instead
  # of silently carrying the old certification onto a different tree.
  local marker="smoke_$ARM/OK" want
  want="$(sha256sum train.py | cut -c1-16)"
  if [ "${SKIP_SMOKE:-0}" = "1" ]; then
    echo "WARNING: SKIP_SMOKE=1 -- starting with no smoke gate (HANDOFF §36.9)"
  elif [ ! -f "$marker" ]; then
    echo "FATAL: no smoke gate for '$ARM' -- run this first:" >&2
    echo "         bash smoke_attntemp_junk.sh" >&2; exit 1
  elif [ "$(cut -d' ' -f1 "$marker")" != "$want" ]; then
    echo "FATAL: the smoke gate for '$ARM' is STALE -- train.py changed since it passed" >&2
    echo "       (marker $(cut -d' ' -f1 "$marker") vs now $want).  Re-run:" >&2
    echo "         bash smoke_attntemp_junk.sh" >&2; exit 1
  else
    echo "smoke gate : PASSED for train.py $want ($(cut -d' ' -f3 "$marker"))"
  fi

  # bash has no compile step, and these scripts are edited on a Windows box that
  # has no bash at all (WSL is not installed there), so neither file has been
  # syntax-checked by anything before this line.  A syntax error in the smoke is
  # not caught by the smoke, and would show up here as a missing marker.
  bash -n smoke_attntemp_junk.sh && echo "bash -n smoke_attntemp_junk.sh: ok"
  bash -n run_attntemp_junk.sh   && echo "bash -n run_attntemp_junk.sh: ok"

  # fail-fast write check before the 3.4 hours, not after
  mkdir -p "${SUBBASE}_8v"

  # nothing else may be on this GPU (HANDOFF §33.3 / §36.2: two 384px jobs do not
  # fit and the failure mode is a confusing cuDNN error, not an OOM).
  if pgrep -af 'train\.py' | grep -v grep; then
    echo "FATAL: a train.py is already running on this box" >&2; exit 1
  fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv
  cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current
  df -h "$PWD" | tail -1

  # ---- stage 1: train ---------------------------------------------------------
  echo "--- train [$ARM] -> $OUT ---"
  # shellcheck disable=SC2086  # $EXTRA is a deliberate word-split flag list
  "$PY" -u train.py --data "$DATA" \
    --out "$OUT" \
    --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 \
    --img-size 384 --train-pos-embed --local-head --save-every 4 \
    --lora-rank "$RANK" --lora-qkv $EXTRA 2>&1 | tee "$RUNLOG"

  "$PY" - <<PYEOF
import re, sys
t = open(r"$RUNLOG", encoding="utf-8", errors="replace").read()
g = lambda p: re.search(p, t)
checks = [
    ("trainable params=10.325M or 10.326M", g(r"$WANT_PARAMS_RE")),
    ("layers=$WANT_LAYERS",          g(r"layers=$WANT_LAYERS\b")),
    ("lora_qkv=True",                g(r"lora_qkv=True\b")),
    ("lora_rank=$RANK",              g(r"lora_rank=$RANK\b")),
    ("img_size=384",                 g(r"img_size=384")),
    ("local_head=True",              g(r"local_head=True\b")),
    ("train_pos_embed=True",         g(r"train_pos_embed=True\b")),
    ("pos grid trainable 111.4k",    g(r"positional grid is now trainable \(111\.4k")),
    ("arm flag: $WANT_EXTRA_RE",     g(r"$WANT_EXTRA_RE")),
    ("20 epochs reached",            g(r"best val_acc = ")),
]
bad = [c for c, m in checks if not m]
if bad:
    print("GATE FAIL, missing from the log:", bad); sys.exit(1)
if "Traceback" in t:
    print("GATE FAIL: traceback in the log"); sys.exit(1)
print("GATE OK [arm=$ARM]:", ", ".join(c for c, _ in checks))
m = g(r"trainable params=([\d.]+)M")
print(f"  trainable params as printed: {m.group(1)}M")
PYEOF

  # ---- stage 2: checkpoint sanity ---------------------------------------------
  "$PY" - <<PYEOF
import torch
ck = torch.load(r"$OUT/best.pt", map_location="cpu", weights_only=False)
assert ck["lora_rank"] == $RANK, ck["lora_rank"]
assert ck["lora_qkv"] is True, ck["lora_qkv"]
n = sum(1 for k in ck["model"] if k.endswith(("attn.qkv.A", "attn.qkv.B")))
assert n == 24, f"qkv adapters in checkpoint: {n}"
print(f"  ckpt ok: epoch {ck['epoch']} val_acc {ck['val_acc']:.4f} "
      f"val_acc_hi {ck['val_acc_hi']:.4f} qkv tensors {n}")
if "$ARM" == "attntemp":
    assert ck["attn_temp"] is True, ck["attn_temp"]
    # 12 tensors = one per block; 144 scalars = num_heads per block.  Both, so a
    # silently-wrong head count is a failure and not a curiosity.
    nr = sum(1 for k in ck["model"] if k.endswith("attn_rho"))
    nrs = sum(v.numel() for k, v in ck["model"].items() if k.endswith("attn_rho"))
    assert nr == 12 and nrs == 144, f"attn_rho: {nr} tensors / {nrs} scalars, want 12/144"
    print(f"  attn_rho: {nr} x {nrs // nr} = {nrs} scalars")
elif "$ARM" == "junk":
    jw = ck["junk_w"]
    assert jw is not None, "junk_w is not in best.pt"
    n_tr = sum(ck["class_counts"])
    assert jw.shape == (n_tr,), f"junk_w {tuple(jw.shape)} vs training split {n_tr}"
    hit = int((jw < 1).sum())
    assert 0 < hit <= int($JUNK_MAX_FRAC * n_tr) + 1, f"{hit} flagged, cap {int($JUNK_MAX_FRAC*n_tr)}"
    assert abs(float(jw.min()) - $JUNK_FLOOR) < 1e-6, f"min {float(jw.min())!r} != $JUNK_FLOOR"
    assert float(jw.max()) == 1.0, "an unflagged sample is not at weight 1.0"
    assert ck["junk_report"]["mode"] == "both", ck["junk_report"]["mode"]
    print(f"  junk_w: {hit}/{n_tr} = {hit/n_tr:.2%} at {float(jw.min()):.3f}, "
          f"mean {float(jw.mean()):.4f}")
    print(f"  junk_report: {ck['junk_report']}")
else:
    raise SystemExit(f"unknown arm '$ARM'")
PYEOF

  # The checkpoint alone cannot show whether the temperature actually moved --
  # rho could be at 1e-9 and tau would still be "not exactly 1".  The per-epoch
  # line is the only place the direction is visible, so read the trajectory.
  if [ "$ARM" = "attntemp" ]; then
    "$PY" - <<PYEOF
import re, sys
t = open(r"$RUNLOG", encoding="utf-8", errors="replace").read()
rows = re.findall(r"\[attn-temp\] tau min=([\d.]+) mean=([\d.]+) max=([\d.]+)", t)
if not rows:
    print("GATE FAIL: no [attn-temp] tau line anywhere in the log"); sys.exit(1)
print(f"  tau trajectory: {len(rows)} epochs, first {rows[0]}, last {rows[-1]}")
drift = max(max(abs(float(a) - 1), abs(float(c) - 1)) for a, _, c in rows)
if drift < 1e-7:
    print("GATE FAIL: tau is 1.0 to 1e-7 in EVERY epoch -- the temperature "
          "never left its init, so this arm is the incumbent plus 144 dead scalars"); sys.exit(1)
if drift < 1e-4:
    print(f"  WARNING: tau moved by only {drift:.2e}.  Real but tiny -- read the "
          "final val_acc against the incumbent before spending a slot.")
sharper = float(rows[-1][2]) > 1.0
print(f"  final tau moved {drift:.3e} from 1.0, i.e. the attention ended up "
      f"{'SHARPER' if sharper else 'SOFTER'} than the frozen tower "
      f"(Zou NeurIPS'24 expected softer cross-domain; same-domain fine-grained is "
      f"where the other sign is plausible -- this is the measurement, not a bug)")
PYEOF
  fi

  # ---- stage 3: inference, 2 view sets ----------------------------------------
  echo "--- stage 3 [$ARM]: inference, 2 view sets ---"
  emit "$OUT" "$SUBBASE" 8v ""
  emit "$OUT" "$SUBBASE" scales scales

  echo "=== arm [$ARM] done $(date): ${SUBBASE}_8v.zip, ${SUBBASE}_scales.zip ==="
}

run_arm attntemp "--attn-temp" 'attn_temp=True' \
        outputs_384pe_lr64qkv_attntemp sub_lr64qkv_attntemp \
        outputs_384pe_lr64qkv_attntemp.log
run_arm junk "--junk-filter both --junk-floor $JUNK_FLOOR --junk-max-frac $JUNK_MAX_FRAC" \
        'junk_filter=both' \
        outputs_384pe_lr64qkv_junk sub_lr64qkv_junk \
        outputs_384pe_lr64qkv_junk.log

echo "=== [attntemp+junk] ALL DONE $(date) ==="
echo "produced (4 submission candidates, 2 arms x 2 view sets):"
echo "  sub_lr64qkv_attntemp_8v.zip / sub_lr64qkv_attntemp_scales.zip"
echo "  sub_lr64qkv_junk_8v.zip / sub_lr64qkv_junk_scales.zip"
echo "next: pull the csvs to the local box and score ALL of them, plus the"
echo "      incumbent, on the SAME ruler -- never a proxy number beside a board number:"
echo "        py -3 score.py sub_lr64qkv_attntemp_scales\\pred_results.csv"
echo "        py -3 score.py sub_lr64qkv_junk_scales\\pred_results.csv"
echo "        py -3 score.py <the 74.249 scales CSV>          <- incumbent"
echo "      The proxy resolves only +-0.07 (§41.4) and the floor is 0.03 (§41.2):"
echo "      a gap under ~0.07 does not justify a slot.  Never average two view"
echo "      sets, two arms, or two checkpoints into a third answer -- that is fusion."
