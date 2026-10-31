#!/usr/bin/env bash
# =============================================================================
# 416 + SE + LocalAttention + gate-init 0.02 + colour/erasing + two-stage,
# with the two *evidence-refuted* judgment rules replaced.
#
# What was REMOVED (both are measured failures of this project, not preferences)
# -----------------------------------------------------------------------------
#  1. tau chosen by the hold-out **micro** accuracy (the packaged two-stage
#     script did this).  On the SE weights micro is tau=0:0.7901 vs
#     tau=1.0:0.7898, so that rule always submits tau=0 and throws away the
#     +0.1701 the board actually paid for it (HANDOFF §51.3/§51.5).  §52.8 then
#     showed hold-out macro also misses (+/-0.0000 while the board moved
#     +0.1009).  Replacement: `select_tau.py`, the label-free marginal-violation
#     metric that got it right.
#  2. checkpoint chosen by `--select val_acc` (i.e. trusting `best.pt`).  That
#     metric inverted the ranking twice: §51.4 (micro vs macro) and §53.4
#     (416+SE: single-view val_acc DOWN 0.0012, 16-view TTA UP 0.0050, board
#     +0.1832).  Replacement: 16-view `scales` micro on the same fixed hold-out,
#     over `best.pt` and every `clean_ep*.pt`, so a stage-2 snapshot that helps
#     the 16-view read can no longer be discarded silently.
#
# What was KEPT (per instruction: adopt everything not refuted)
# ------------------------------------------------------------
#  --local-attn, --local-gate-init 0.02, --augment-extra clip_color_erasing,
#  --clean-finetune-epochs 2 (two-stage), 416px, SE, local-head, rank64+qkv,
#  trainable positional grid.  None of these has been measured negative; they
#  are unproven, and they are what this run exists to measure.
#
# Deliberately NOT introduced (all measured negative elsewhere in this project):
#  --lora-rank 32 (-0.70pp, p=0.0058), --tta all (-0.278), prototype readout
#  (-0.95pp), per-class hold-out bias (0.0000), --tau-conf 0.3 (-0.13).
#
#   nohup bash run_cand.sh > run_cand.log 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
unset OMP_NUM_THREADS || true

PY="${PY:-/root/miniconda3/bin/python}"
export PATH=/root/miniconda3/bin:$PATH
DATA="${DATA:-/root/autodl-tmp/train}"
TEST="${TEST:-/root/autodl-tmp/test}"
OUT="${OUT:-outputs_cand}"
PROBE_OUT="${PROBE_OUT:-probe_cand}"
SUB="${SUB:-sub_cand}"
BATCH_SIZE="${BATCH_SIZE:-72}"          # this host: 416 NaN'd at >=80 (HANDOFF §52.2)
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-20}"
CLEAN_EPOCHS="${CLEAN_EPOCHS:-2}"
GATE_INIT="${GATE_INIT:-0.02}"
TAUS="${TAUS:-0 0.75 1.0 1.25 1.5}"
N_CLASSES="${N_CLASSES:-500}"
N_TRAIN="${N_TRAIN:-90197}"
N_TEST="${N_TEST:-49857}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NVIDIA_TF32_OVERRIDE=0

step() { echo; echo "=== [$(date '+%H:%M:%S')] $* ==="; }

# ---------------------------------------------------------------- stage 0 ---
step "stage 0: guards"
for f in train.py infer.py losses.py noise.py semantic_regularization.py selftest.py \
         sf_gate.py validate_candidate.py validate_tta.py select_tau.py pick_checkpoint.py; do
  test -f "$f" || { echo "FATAL: missing $f" >&2; exit 1; }
done
test -d "$DATA" || { echo "FATAL: DATA=$DATA not found" >&2; exit 1; }
test -d "$TEST" || { echo "FATAL: TEST=$TEST not found" >&2; exit 1; }
for p in "$OUT" "$SUB" "$SUB.zip"; do
  test ! -e "$p" || { echo "FATAL: refusing to overwrite $p" >&2; exit 1; }
done
# the probe dir is scratch: always regenerated, never an artefact
rm -rf "$PROBE_OUT" "${PROBE_OUT}.log"
if pgrep -af '[t]rain\.py' >/dev/null; then
  echo "FATAL: another train.py is already running" >&2; pgrep -af '[t]rain\.py'; exit 1
fi
"$PY" --version; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current 2>/dev/null || true
df -h /root/autodl-tmp | tail -1

# ---------------------------------------------------------------- stage 1 ---
step "stage 1: dataset gate"
"$PY" sf_gate.py data --train "$DATA" --test "$TEST" \
  --classes "$N_CLASSES" --n-train "$N_TRAIN" --n-test "$N_TEST" || exit 1

# ---------------------------------------------------------------- stage 2 ---
step "stage 2: smoke (compile + selftest + real 416 tower fwd/bwd + budget)"
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  echo "  SKIPPED (SKIP_SMOKE=1; it already printed SMOKE PASSED on this host)"
else
  bash smoke_416_se_localattn.sh || exit 1
fi

# ---------------------------------------------------------------- stage 3 ---
step "stage 3: preflight at batch ${BATCH_SIZE} (4 train batches + full hold-out val)"
"$PY" -u train.py --data "$DATA" --out "$PROBE_OUT" \
  --epochs 1 --warmup-epochs 1 --limit-batches 4 \
  --batch-size "$BATCH_SIZE" --workers 2 --img-size 416 \
  --train-pos-embed --local-head --local-attn --local-se \
  --local-gate-init "$GATE_INIT" \
  --lora-rank 64 --lora-qkv --augment-extra clip_color_erasing \
  --save-every 1 --nan-guard 2>&1 | tee "${PROBE_OUT}.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "FATAL: preflight exited non-zero" >&2; exit 1; }
"$PY" - "${PROBE_OUT}.log" "$GATE_INIT" <<'PY' || exit 1
import re, sys
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
gate = float(sys.argv[2])
if "Traceback" in t:
    raise SystemExit("PREFLIGHT FAILED: traceback -- lower BATCH_SIZE and rerun")
if re.search(r"(?:loss|val_loss|val_acc)=nan", t, re.I):
    raise SystemExit("PREFLIGHT FAILED: NaN at this batch size (HANDOFF §52.2)")
# `f'{n/1e6:.3f}M'` prints 10.391M for 10,390,581 -- accept the whole 10.39 prefix
NEED = ("trainable params=10.39", "layers=48", "img_size=416", "local_attn=True",
        "local_se=True", f"local_gate_init={gate:g}", "lora_qkv=True",
        "augment_extra=clip_color_erasing", "positional grid is now trainable",
        "positional grid resampled to 13x13", "classes=500", "best val_acc =")
missing = [tok for tok in NEED if tok not in t]
if missing:
    raise SystemExit(f"PREFLIGHT FAILED: missing {missing}")
print("PREFLIGHT PASSED: 10.39M params, 13x13 grid, finite loss and validation")
PY

# ---------------------------------------------------------------- stage 4 ---
step "stage 4: full training -- ${EPOCHS} epochs + ${CLEAN_EPOCHS} clean-stage epochs"
"$PY" -u train.py --data "$DATA" --out "$OUT" \
  --epochs "$EPOCHS" --warmup-epochs 3 \
  --batch-size "$BATCH_SIZE" --workers "$WORKERS" --img-size 416 \
  --train-pos-embed --local-head --local-attn --local-se \
  --local-gate-init "$GATE_INIT" \
  --lora-rank 64 --lora-qkv --augment-extra clip_color_erasing \
  --clean-finetune-epochs "$CLEAN_EPOCHS" \
  --clean-lr-scale 0.05 --clean-weight-decay 0.005 \
  --clean-quantile 0.50 --clean-min-confidence 0.35 \
  --clean-min-per-class 4 --clean-uncertain-weight 0.15 \
  --clean-min-global-frac 0.10 \
  --save-every 2 --nan-guard 2>&1 | tee "${OUT}.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "FATAL: training exited non-zero" >&2; exit 1; }
test -s "$OUT/best.pt" || { echo "FATAL: no best.pt" >&2; exit 1; }
"$PY" validate_candidate.py --checkpoint "$OUT/best.pt" --log "${OUT}.log" \
  --classes "$N_CLASSES" --gate-init "$GATE_INIT" || exit 1

# ---------------------------------------------------------------- stage 5 ---
step "stage 5: clean-stage report"
if [ -f "$OUT/clean_stage_report.json" ]; then
  "$PY" - "$OUT/clean_stage_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
ks = ("n", "clean", "clean_fraction", "agree", "relabel", "low_trust",
      "classes_with_clean", "class_clean_min", "class_clean_median", "class_clean_max",
      "mean_confidence", "stage2_epochs", "stage2_lr", "sampler", "policy")
for k in ks:
    if k in r:
        print(f"  {k:22s} {r[k]}")
PY
else
  echo "  (no clean_stage_report.json -- stage 2 was skipped)"
fi
ls -la "$OUT"

# ---------------------------------------------------------------- stage 6 ---
step "stage 6: 16-view scales TTA on the hold-out, per candidate checkpoint"
mkdir -p "$OUT/holdout"
CANDS=()
for ck in best.pt clean_ep1.pt clean_ep2.pt; do
  [ -f "$OUT/$ck" ] || continue
  js="$OUT/holdout/${ck%.pt}.json"
  echo "--- $ck ---"
  OMP_NUM_THREADS=1 "$PY" -u validate_tta.py --data "$DATA" \
    --checkpoint "$OUT/$ck" --views scales --tau 0 1.0 \
    --batch-size 16 --workers 4 --amp bf16 \
    --per-class-csv "$OUT/holdout/${ck%.pt}_classes.csv" \
    --output-json "$js" 2>&1 | tail -8
  test -f "$js" && CANDS+=("$OUT/$ck=$js")
done
test ${#CANDS[@]} -gt 0 || { echo "FATAL: no hold-out report produced" >&2; exit 1; }

# ---------------------------------------------------------------- stage 7 ---
step "stage 7: pick the checkpoint by 16-view micro (NOT by val_acc)"
PICK="$("$PY" pick_checkpoint.py --out "$OUT" --classes "$N_CLASSES" \
        --holdout "${CANDS[@]}" 2>"$OUT/holdout/pick.log")"
cat "$OUT/holdout/pick.log"
CHOSEN_CK="$(echo "$PICK" | awk '/^CHOSEN/{print $2}')"
CHOSEN_SCORE="$(echo "$PICK" | awk '/^CHOSEN/{print $3}')"
test -n "$CHOSEN_CK" || { echo "FATAL: checkpoint selection failed" >&2; exit 1; }
echo "chosen checkpoint: $CHOSEN_CK  (16-view micro $CHOSEN_SCORE)"
echo "reference: 416+SE@tau1.0 = 0.7948 on the same split (HANDOFF §53.4)"

# ---------------------------------------------------------------- stage 8 ---
step "stage 8: ONE inference pass -- 16 views, taus [$TAUS], logits saved"
mkdir -p "$SUB"
OMP_NUM_THREADS=1 "$PY" -u infer.py --test "$TEST" \
  --checkpoint "$CHOSEN_CK" \
  --output "$SUB/pred_results.csv" \
  --workers 4 --batch-size 64 --tta scales \
  --logit-adjust $TAUS \
  --save-logits "$PWD/${OUT}_test_logits.npz" 2>&1 | tee "$SUB/infer.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "FATAL: inference exited non-zero" >&2; exit 1; }
ls -la "$SUB"

# ---------------------------------------------------------------- stage 9 ---
step "stage 9: choose tau by marginal violation (label-free), never by micro"
"$PY" select_tau.py --test "$TEST" --csv "$SUB"/pred_results*.csv 2>&1 | tee "$SUB/select_tau.log"
CHOSEN_CSV="$(awk '/lowest marginal violation/{print $4}' "$SUB/select_tau.log")"
if [ -z "${CHOSEN_CSV:-}" ] || [ ! -f "$SUB/$CHOSEN_CSV" ]; then
  echo "  WARNING: could not parse a winner; falling back to tau=1.0"
  CHOSEN_CSV="pred_results_tau100.csv"
fi
test -f "$SUB/$CHOSEN_CSV" || { echo "FATAL: chosen CSV $CHOSEN_CSV missing" >&2; exit 1; }
echo "chosen submission: $CHOSEN_CSV"

# --------------------------------------------------------------- stage 10 ---
step "stage 10: CSV gate, pack, decision record"
"$PY" sf_gate.py csv --csv "$SUB/$CHOSEN_CSV" --test "$TEST" --classes "$N_CLASSES" || exit 1
"$PY" - "$SUB" "$CHOSEN_CSV" "$CHOSEN_CK" "$CHOSEN_SCORE" "$OUT" "$GATE_INIT" "$TAUS" <<'PY'
import hashlib, json, pathlib, sys, zipfile
sub, csv_name, ck, score, out, gate, taus = sys.argv[1:8]
src = pathlib.Path(sub) / csv_name
dst = pathlib.Path(sub).with_suffix('.zip')
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(src, arcname='pred_results.csv')
rep = pathlib.Path(out) / 'clean_stage_report.json'
rec = dict(
    checkpoint=str(ck),
    checkpoint_sha256=hashlib.sha256(pathlib.Path(ck).read_bytes()).hexdigest()[:16],
    holdout_16view_micro=float(score),
    reference_416se_tau1_micro=0.7948,
    chosen_csv=csv_name,
    csv_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    csv_bytes=src.stat().st_size,
    rows=sum(1 for _ in src.open(encoding='utf-8')),
    zip=dst.name, zip_bytes=dst.stat().st_size,
    zip_md5=hashlib.md5(dst.read_bytes()).hexdigest(),
    local_gate_init=float(gate),
    taus_tried=taus.split(),
    selection='checkpoint by 16-view micro; tau by marginal violation',
    removed_rules=['tau by hold-out micro (HANDOFF 51.3/52.8)',
                   'checkpoint by val_acc (HANDOFF 53.4)'],
    clean_stage=json.loads(rep.read_text(encoding='utf-8')) if rep.is_file() else None,
)
pathlib.Path(sub, 'decision.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
print(json.dumps({k: rec[k] for k in ('checkpoint', 'holdout_16view_micro', 'chosen_csv',
                                      'rows', 'csv_bytes', 'zip', 'zip_md5')},
                 ensure_ascii=False, indent=2))
PY

step "DONE"
echo "submission package : $SUB.zip"
echo "   contains        : $CHOSEN_CSV renamed to pred_results.csv"
echo "decision record    : $SUB/decision.json"
echo "logits for any tau : ${OUT}_test_logits.npz  (re-argmax via retau_from_logits.py)"
date
