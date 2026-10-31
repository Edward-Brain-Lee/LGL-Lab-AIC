#!/usr/bin/env bash
# =============================================================================
# Post-training chain ONLY: resume from finished checkpoints, never retrain.
#
# This exists because the host went away mid-run (AutoDL instance stopped at
# ~16:15, handoff §39.6: "训练跑完后被欠费关机打断在推理阶段 —— 不用重训").
# Checkpoints live on the persistent data disk, so everything after stage 4 can
# be replayed without spending 2.5 h of GPU again.
#
# Two judgment rules are the same as run_cand.sh, and they are the reason this
# script exists at all:
#   * checkpoint -> 16-view `scales` micro on the fixed hold-out, NOT val_acc
#     (HANDOFF §53.4: 416+SE had lower val_acc and +0.1832 on the board);
#   * tau        -> marginal violation (`select_tau.py`), NOT hold-out micro
#     (HANDOFF §51.3/§52.8: micro always prefers tau=0 and throws away +0.17).
#
# The two-stage clean fine-tune is deliberately NOT resumed: `--resume` rebuilds
# the EMA teacher from the student (§29.7-5), so "continuing" would be a
# different model, and the component was rated ~0 anyway.
#
#   bash run_post.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
unset OMP_NUM_THREADS || true

PY="${PY:-/root/miniconda3/bin/python}"
export PATH=/root/miniconda3/bin:$PATH
DATA="${DATA:-/root/autodl-tmp/train}"
TEST="${TEST:-/root/autodl-tmp/test}"
OUT="${OUT:-outputs_cand}"
SUB="${SUB:-sub_cand}"
TAUS="${TAUS:-0 0.75 1.0 1.25 1.5}"
N_CLASSES="${N_CLASSES:-500}"
CKPTS="${CKPTS:-}"          # override, e.g. CKPTS="best.pt ep18.pt"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NVIDIA_TF32_OVERRIDE=0

step() { echo; echo "=== [$(date '+%H:%M:%S')] $* ==="; }

step "stage 0: guards (no training is performed by this script)"
for f in train.py infer.py losses.py noise.py semantic_regularization.py sf_gate.py \
         validate_tta.py select_tau.py pick_checkpoint.py; do
  test -f "$f" || { echo "FATAL: missing $f" >&2; exit 1; }
done
test -d "$DATA" || { echo "FATAL: DATA=$DATA not found" >&2; exit 1; }
test -d "$TEST" || { echo "FATAL: TEST=$TEST not found" >&2; exit 1; }
test -d "$OUT" || { echo "FATAL: $OUT not found -- nothing to resume from" >&2; exit 1; }
test -s "$OUT/best.pt" || { echo "FATAL: no $OUT/best.pt" >&2; exit 1; }
if pgrep -af '[t]rain\.py' >/dev/null; then
  echo "FATAL: a train.py is already running" >&2; pgrep -af '[t]rain\.py'; exit 1
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
df -h /root/autodl-tmp | tail -1

step "stage 1: what the interrupted run actually left behind"
ls -la "$OUT" | sed -n '1,40p'
echo "--- last completed epochs (from the log, if it survived) ---"
if [ -f "${OUT}.log" ]; then
  grep -E '^epoch ' "${OUT}.log" | tail -3
  grep -cE '^epoch ' "${OUT}.log" | sed 's/^/epochs logged: /'
  grep -E '^noise stats' "${OUT}.log" | tail -1
else
  echo "  (no ${OUT}.log -- checkpoints are still valid, just no curve)"
fi
echo "--- stage-2 report? ---"
if [ -f "$OUT/clean_stage_report.json" ]; then
  "$PY" - "$OUT/clean_stage_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
for k in ("n", "clean_fraction", "classes_with_clean", "stage2_epochs", "stage2_lr"):
    if k in r:
        print(f"  {k:20s} {r[k]}")
PY
else
  echo "  absent -> the clean stage never started; this run is stage-1 only"
fi

step "stage 2: validate_candidate on best.pt (structure gate)"
"$PY" validate_candidate.py --checkpoint "$OUT/best.pt" --classes "$N_CLASSES" \
  --gate-init 0.02 || echo "  WARNING: structural gate failed; inspect before submitting"

step "stage 3: 16-view scales TTA on the hold-out, per candidate checkpoint"
mkdir -p "$OUT/holdout"
if [ -n "$CKPTS" ]; then
  LIST="$CKPTS"
else
  LIST="best.pt"
  for ck in ep20.pt ep19.pt ep18.pt ep16.pt ep14.pt clean_ep1.pt clean_ep2.pt; do
    [ -f "$OUT/$ck" ] && LIST="$LIST $ck"
  done
fi
echo "candidates: $LIST"
CANDS=()
for ck in $LIST; do
  [ -f "$OUT/$ck" ] || { echo "  skip $ck (missing)"; continue; }
  js="$OUT/holdout/${ck%.pt}.json"
  echo "--- $ck ---"
  OMP_NUM_THREADS=1 "$PY" -u validate_tta.py --data "$DATA" \
    --checkpoint "$OUT/$ck" --views scales --tau 0 1.0 \
    --batch-size 16 --workers 4 --amp bf16 \
    --per-class-csv "$OUT/holdout/${ck%.pt}_classes.csv" \
    --output-json "$js" 2>&1 | tail -6
  [ -f "$js" ] && CANDS+=("$OUT/$ck=$js")
done
test ${#CANDS[@]} -gt 0 || { echo "FATAL: no hold-out report produced" >&2; exit 1; }

step "stage 4: pick the checkpoint by 16-view micro (NOT by val_acc)"
PICK="$("$PY" pick_checkpoint.py --out "$OUT" --classes "$N_CLASSES" \
        --holdout "${CANDS[@]}" 2>"$OUT/holdout/pick.log")"
cat "$OUT/holdout/pick.log"
CHOSEN_CK="$(echo "$PICK" | awk '/^CHOSEN/{print $2}')"
CHOSEN_SCORE="$(echo "$PICK" | awk '/^CHOSEN/{print $3}')"
test -n "$CHOSEN_CK" || { echo "FATAL: checkpoint selection failed" >&2; exit 1; }
echo "chosen checkpoint: $CHOSEN_CK  (16-view micro $CHOSEN_SCORE)"
echo "reference: 416+SE@tau1.0 = 0.7948 on the same split (HANDOFF §53.4)"

step "stage 5: ONE inference pass -- 16 views, taus [$TAUS], logits saved"
test ! -e "$SUB" || { echo "FATAL: $SUB exists; move it away first" >&2; exit 1; }
test ! -e "$SUB.zip" || { echo "FATAL: $SUB.zip exists; move it away first" >&2; exit 1; }
mkdir -p "$SUB"
OMP_NUM_THREADS=1 "$PY" -u infer.py --test "$TEST" \
  --checkpoint "$CHOSEN_CK" --output "$SUB/pred_results.csv" \
  --workers 4 --batch-size 64 --tta scales \
  --logit-adjust $TAUS \
  --save-logits "$PWD/${OUT}_test_logits.npz" 2>&1 | tee "$SUB/infer.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "FATAL: inference exited non-zero" >&2; exit 1; }
ls -la "$SUB"

step "stage 6: choose tau by marginal violation (label-free), never by micro"
"$PY" select_tau.py --test "$TEST" --csv "$SUB"/pred_results*.csv 2>&1 | tee "$SUB/select_tau.log"
CHOSEN_CSV="$(awk '/lowest marginal violation/{print $4}' "$SUB/select_tau.log")"
if [ -z "${CHOSEN_CSV:-}" ] || [ ! -f "$SUB/$CHOSEN_CSV" ]; then
  echo "  WARNING: could not parse a winner; falling back to tau=1.0"
  CHOSEN_CSV="pred_results_tau100.csv"
fi
test -f "$SUB/$CHOSEN_CSV" || { echo "FATAL: chosen CSV $CHOSEN_CSV missing" >&2; exit 1; }
echo "chosen submission: $CHOSEN_CSV"

step "stage 7: CSV gate, pack, decision record"
"$PY" sf_gate.py csv --csv "$SUB/$CHOSEN_CSV" --test "$TEST" --classes "$N_CLASSES" || exit 1
"$PY" - "$SUB" "$CHOSEN_CSV" "$CHOSEN_CK" "$CHOSEN_SCORE" "$OUT" "$TAUS" <<'PY'
import hashlib, json, pathlib, sys, zipfile
sub, csv_name, ck, score, out, taus = sys.argv[1:7]
src = pathlib.Path(sub) / csv_name
dst = pathlib.Path(sub).with_suffix('.zip')
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(src, arcname='pred_results.csv')
rep = pathlib.Path(out) / 'clean_stage_report.json'
log = pathlib.Path(f'{out}.log')
epochs = []
if log.is_file():
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('epoch '):
            epochs.append(line.split()[1])
rec = dict(
    resumed_from='interrupted run (host stopped); no retraining performed',
    epochs_available=epochs[-3:],
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
    taus_tried=taus.split(),
    clean_stage=json.loads(rep.read_text(encoding='utf-8')) if rep.is_file() else None,
    selection='checkpoint by 16-view micro; tau by marginal violation',
)
pathlib.Path(sub, 'decision.json').write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
print(json.dumps({k: rec[k] for k in ('epochs_available', 'checkpoint', 'holdout_16view_micro',
                                      'chosen_csv', 'rows', 'csv_bytes', 'zip', 'zip_md5')},
                 ensure_ascii=False, indent=2))
PY

step "DONE"
echo "submission package : $SUB.zip  (contains $CHOSEN_CSV renamed to pred_results.csv)"
echo "decision record    : $SUB/decision.json"
date
