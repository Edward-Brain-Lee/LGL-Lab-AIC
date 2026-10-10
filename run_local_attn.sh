#!/usr/bin/env bash
set -euo pipefail
ARM=local_attn_junk
PY="${PY:-python}"
DATA="${DATA:-/root/autodl-tmp/train}"
TEST="${TEST:-/root/autodl-tmp/test}"
OUT="${OUT:-outputs_384pe_local_attn_junk}"
SUB="${SUB:-sub_local_attn_junk_scales}"
RANK=64
FLOOR=0.2
MAX_FRAC=0.05
N_ROWS="${N_ROWS:-37444}"
N_BYTES="${N_BYTES:-1610092}"
for f in train.py infer.py losses.py noise.py semantic_regularization.py; do
  test -f "$f" || { echo "FATAL: missing dependency $f" >&2; exit 1; }
done
test ! -e "$OUT" || { echo "FATAL: refusing to overwrite existing output $OUT" >&2; exit 1; }
test ! -e "$SUB" || { echo "FATAL: refusing to overwrite existing output $SUB" >&2; exit 1; }
test ! -e "$SUB.zip" || { echo "FATAL: refusing to overwrite existing output $SUB.zip" >&2; exit 1; }
test -f "smoke_${ARM}/OK" || { echo "FATAL: run smoke_local_attn.sh first" >&2; exit 1; }
want="$(sha256sum train.py | cut -d' ' -f1)"
grep -q "^$want " "smoke_${ARM}/OK" || { echo 'FATAL: smoke hash is stale' >&2; exit 1; }
# NOT `bash -n a.sh b.sh`: bash reads the first non-option word as the command
# file and everything after it as that script's positional parameters, so a
# second file on the same line is never parsed.
for s in smoke_local_attn.sh run_local_attn.sh; do
  bash -n "$s" && echo "  bash -n $s: ok"
done
"$PY" -m py_compile train.py infer.py losses.py noise.py semantic_regularization.py
# nothing else may be on this GPU (HANDOFF §33.3): two 384px jobs do not fit and
# the failure mode is a confusing cuDNN error rather than an OOM
if pgrep -af 'train\.py' | grep -v grep; then
  echo 'FATAL: a train.py is already running on this box' >&2; exit 1
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
df -h "$PWD" | tail -1
echo "=== [$ARM] $(date) ==="
sha256sum train.py infer.py losses.py noise.py semantic_regularization.py
"$PY" -u train.py --data "$DATA" --out "$OUT" --epochs 20 --warmup-epochs 3 --batch-size 128 --workers 8 --img-size 384 --train-pos-embed --local-head --local-attn --lora-rank "$RANK" --lora-qkv --robust-loss apl --save-every 4 --junk-filter margin --junk-floor "$FLOOR" --junk-max-frac "$MAX_FRAC" 2>&1 | tee "$OUT.log"
test -f "$OUT/best.pt" || { echo 'FATAL: training produced no best.pt' >&2; exit 1; }

# ---- stage 1 gate: the flag reached train.py, and the branch is alive --------
"$PY" - <<PYEOF
import re, sys, torch
t = open(r"$OUT.log", encoding="utf-8", errors="replace").read()
g = lambda p: re.search(p, t)
checks = [
    ("local_attn=True",        g(r"local_attn=True\b")),
    ("local_head=True",        g(r"local_head=True\b")),
    ("train_pos_embed=True",   g(r"train_pos_embed=True\b")),
    ("lora_qkv=True",          g(r"lora_qkv=True\b")),
    ("lora_rank=64",           g(r"lora_rank=64\b")),
    ("attn_temp=False",        g(r"attn_temp=False\b")),
    ("img_size=384",           g(r"img_size=384")),
    ("junk_filter=margin",     g(r"junk_filter=margin\b")),
    ("loss_schedule=fixed",    g(r"loss_schedule=fixed\b")),
    ("robust_loss=apl",        g(r"robust_loss=apl\b")),
    ("layers=48",              g(r"layers=48\b")),
    ("pos grid trainable",     g(r"positional grid is now trainable \(111\.4k")),
    ("trainable params 10.4x", g(r"trainable params=10\.4\d\d?M")),
    ("junk report",            g(r"\[junk\] flagged \d+/\d+ = ")),
    ("reached the last epoch", g(r"best val_acc = ")),
]
bad = [c for c, m in checks if not m]
if bad:
    print("GATE FAIL, missing from the log:", bad); sys.exit(1)
if "Traceback" in t:
    print("GATE FAIL: traceback in the log"); sys.exit(1)
# --local-attn has no per-epoch line of its own, so the residual gate is the only
# visible sign the branch is learning at all: with gate == 0 the entire local
# branch is multiplied by zero, every local-attn parameter receives a zero
# gradient, and the arm is the incumbent plus dead weights.  A checkpoint cannot
# show that -- read the trajectory, exactly as the attn-temp arm does for tau.
gl = re.findall(r"local_gate=([-\d.eE+]+)", t)
if not gl:
    print("GATE FAIL: no local_gate line anywhere in the log"); sys.exit(1)
peak = max(abs(float(v)) for v in gl)
print(f"  local_gate: {len(gl)} epochs, last {gl[-1]}, max|g| {peak:.3e}")
if peak < 1e-6:
    print("GATE FAIL: local_gate never left zero -- the local attention branch never "
          "received a gradient, so this arm is the incumbent plus dead parameters and "
          "its result cannot be read as evidence about --local-attn"); sys.exit(1)
if peak < 1e-3:
    print(f"  WARNING: local_gate only reached {peak:.2e} -- the branch barely opened")
ck = torch.load(r"$OUT/best.pt", map_location="cpu", weights_only=False)
assert ck.get("local_attn") is True, f"best.pt says local_attn={ck.get('local_attn')}"
keys = list(ck["model"])
miss = [n for n in ("local_head.sim_scale", "local_head.query_proj.weight",
                    "local_head.key_proj.weight", "local_head.score.weight")
        if not any(k.endswith(n) for k in keys)]
assert not miss, f"trained tensors absent from best.pt: {miss}"
print(f"  ckpt ok: epoch {ck['epoch']} val_acc {ck['val_acc']:.4f} "
      f"val_acc_hi {ck['val_acc_hi']:.4f} "
      f"sim_scale {[round(v, 4) for v in ck['model']['local_head.sim_scale'].tolist()]}")
print("GATE OK [arm=$ARM]:", ", ".join(c for c, _ in checks))
PYEOF

mkdir "$SUB"
"$PY" -u infer.py --test "$TEST" --checkpoint "$OUT/best.pt" --output "$SUB/pred_results.csv" --tta scales --workers 4 --batch-size 64
csv="$SUB/pred_results.csv"
bytes=$(stat -c %s "$csv")
rows=$(sed 's/\r$//' "$csv" | awk 'END{print NR}')
uniq=$(sed 's/\r$//' "$csv" | cut -d, -f1 | sort -u | wc -l)
bad=$(sed 's/\r$//' "$csv" | grep -cvE '^[^,]+\.jpg,[0-9]{4}$' || true)
echo "scales: bytes=$bytes rows=$rows unique=$uniq malformed=$bad"
[ "$bytes" = "$N_BYTES" ] || { echo "FATAL: expected $N_BYTES bytes" >&2; exit 1; }
[ "$rows" = "$N_ROWS" ] || { echo "FATAL: expected $N_ROWS rows" >&2; exit 1; }
[ "$uniq" = "$N_ROWS" ] || { echo "FATAL: duplicate filenames" >&2; exit 1; }
[ "$bad" = 0 ] || { echo 'FATAL: malformed prediction rows' >&2; exit 1; }
(cd "$SUB" && zip -q "../$SUB.zip" pred_results.csv)
echo "produced $SUB.zip from one checkpoint: $(stat -c %s "$SUB.zip") bytes"
