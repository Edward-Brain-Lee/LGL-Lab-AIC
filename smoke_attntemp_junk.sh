#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 发车前闸门 for the two NEW single variables:
#
#   attn-temp : --attn-temp           per-head learnable softmax temperature
#   junk      : --junk-filter both    frozen-tower 杂图 / 重复图 screening
#
#   cd /root/autodl-tmp
#   bash smoke_attntemp_junk.sh      # read the output BEFORE starting run_attntemp_junk.sh
#
# Neither flag has ever touched the real ViT-B/32 tower.  Both are new code on a
# path the stub selftest cannot reach, and both fail in the shape this project
# keeps paying for -- silently: an unused parameter (rho frozen by Net's blanket
# freeze), a moving anchor, a permutation between the frozen features and the
# training indices.  So this is three gates, cheapest first, fail-loud.
#
#   0. bash -n on both scripts + selftest.py   ~1 min, CPU, no data
#   1. the real ViT-B/32 tower, both flags     ~2 min, no data
#   2. 20 real batches, --attn-temp            ~2-4 min, GPU to itself
#   3. 20 real batches, --junk-filter both     ~8-12 min, GPU to itself
#
# Gate 3 is the expensive one and it is on purpose: --limit-batches only truncates
# the *training* loop, so the frozen feature pass still sweeps the entire training
# split.  That is the whole risk of the junk arm -- coverage, the per-class report,
# and a 134k x 134k chunked matmul's memory -- and it is exactly what this gate buys
# before a 3.4-hour run commits to it.
# ---------------------------------------------------------------------------
set -euo pipefail

ARM="attntemp_junk"
if [ $# -gt 0 ]; then
  echo "usage: bash smoke_attntemp_junk.sh   (no arm argument)" >&2; exit 2
fi

RANK=64
# The recipe both arms sit on: 384 + --train-pos-embed + --local-head + rank 64 + qkv.
RECIPE="--img-size 384 --train-pos-embed --local-head --lora-rank $RANK --lora-qkv"

# 48 = 12 blocks x (qkv + out_proj + c_fc + c_proj).  --attn-temp adds no module --
# it lives on the block the qkv flag already rebuilds -- so this number must be
# identical for the qkv, attn-temp and junk arms.  If it moves, a stale tree is
# being gated.
WANT_LAYERS=48
# 12 x (3072 + 1536 + 3840 + 3840) x 64.  Exact, so it pins the rank down where
# the 3-decimal printed total cannot.
WANT_LORA=9437184
# rank64+qkv printed 10.326M on 2026-10-06 (HANDOFF §42/§43).  --attn-temp adds 144
# scalars, which cannot reach the next 3-decimal step.
WANT_PARAMS_RE='trainable params=(?:10\.325|10\.326)M'
# Both are train.py's own defaults, passed explicitly so the config line in the
# log shows what ran.  §21.3 records that the strength of centroid filtering was
# never calibrated; a run must not stack a second unmeasured choice on top of an
# unmeasured one, so this arm moves the *mechanism* and nothing else.
JUNK_FLOOR=0.2
JUNK_MAX_FRAC=0.05

PY="${PY:-python}"
DATA="${DATA:-/root/autodl-tmp/train}"
SMOKE_AT="smoke_attntemp"
SMOKE_JK="smoke_junk"

echo "=== smoke [$ARM] $(date) ==="
echo "tree    : $PWD"
echo "expected: layers=$WANT_LAYERS adapter_params=$WANT_LORA trainable=10.325-10.326M"
sha256sum train.py infer.py | cut -c1-16

# HANDOFF §29.7-3: a smoke sharing the box with a 384px training run died with a
# spurious cuDNN error.  The smoke needs the GPU to itself.
if pgrep -af 'train\.py' | grep -v grep; then
  echo "FATAL: a train.py is already running -- the smoke needs the GPU alone" >&2; exit 1
fi

# ---- gate 0: shell syntax, then selftest -------------------------------------
echo "--- gate 0a: bash -n ---"
bash -n smoke_attntemp_junk.sh && echo "  smoke_attntemp_junk.sh: syntax ok"
if [ -f run_attntemp_junk.sh ]; then
  bash -n run_attntemp_junk.sh && echo "  run_attntemp_junk.sh: syntax ok"
else
  echo "  WARNING: run_attntemp_junk.sh not found beside this script" >&2
fi

# The stub selftest is where the *contracts* are pinned: rho == 0 is bit-for-bit
# the flagless run, tau scales a head's query rows against an MHA whose
# in_proj_weight was edited by hand, the anchor stays the frozen tower with
# tau != 1, /rho is not frozen by Net.__init__, the robust centroids drop the
# mislabelled samples, the cap bounds the filter, and the frozen pass keeps row i
# attached to training index i.  What it cannot say anything about is the real
# open_clip call pattern -- that is gate 1.
echo "--- gate 0b: selftest ---"
"$PY" selftest.py | tail -25

# ---- gate 1: the real tower, which the stub never touches ---------------------
echo "--- gate 1: real tower, both flags ---"
"$PY" - <<PYEOF
import copy
import torch
import torch.nn.functional as F
import open_clip
from train import Net, resize_positional_embedding

torch.manual_seed(0)
base = open_clip.create_model('ViT-B-32-quickgelu', pretrained='openai').eval()
# open_clip's tower does NOT interpolate its positional grid on its own -- at 384
# it raises "size of tensor a (82) must match tensor b (50)" until this runs.  Do
# it once, on `base` *before* the deepcopy, exactly as main() does on clip_model;
# calling it twice would bicubically resample an already-resampled grid.
grid = resize_positional_embedding(base.visual, 384)
print(f'  positional grid resampled to {grid}x{grid}')
x = torch.randn(2, 3, 384, 384)

# --- attn-temp: the model the run trains, built the way the run builds it
m = Net(copy.deepcopy(base), 7, $RANK, 'all', local_head=True, lora_qkv=True,
        attn_temp=True)
assert m.n_lora == $WANT_LAYERS, f'n_lora={m.n_lora}, want $WANT_LAYERS'
n_lora_p = sum(p.numel() for n, p in m.named_parameters()
               if n.endswith('.A') or n.endswith('.B'))
assert n_lora_p == $WANT_LORA, f'LoRA params={n_lora_p}, want $WANT_LORA'
# --attn-temp must not have quietly widened the adapter sweep: rho is
# num_heads per block, so 12 tensors of 12 -- 144 scalars, not 12.
rho_t = [p for n, p in m.named_parameters() if n.endswith('attn_rho')]
n_temp = sum(p.numel() for p in rho_t)
assert len(rho_t) == 12 and n_temp == 144, \
    f'attn_rho: {len(rho_t)} tensors / {n_temp} scalars, want 12 x 12 = 144'
print(f'  adapter params exact: {n_lora_p} over {m.n_lora} modules; '
      f'rho: {len(rho_t)} x {rho_t[0].numel()} = {n_temp}')

# step 0 must be the official tower, and the temperatures must start at exactly 1
with torch.no_grad():
    z_ref = F.normalize(base.visual(x).float(), dim=-1)
    z_new = F.normalize(m.clip.visual(x).float(), dim=-1)
    tau0 = m.attn_temperature()
d = (z_ref - z_new).abs().max().item()
assert d < 1e-5, f'wrapped tower != official tower at init: max|dz|={d:.2e}'
assert tau0.numel() == 144 and float((tau0 - 1.0).abs().max()) == 0.0, \
    f'tau does not start at exactly 1.0 over 144 heads: {tau0.tolist()}'
print(f'  init equivalence on the real tower: max|dz|={d:.2e}; tau == 1.0 exactly')

# rho must be trainable -- Net.__init__ freezes the backbone by name, and this is
# the silent failure that would make the whole flag a 3.4-hour no-op
rhos = [p for n, p in m.named_parameters() if n.endswith('attn_rho')]
assert len(rhos) == 12 and all(p.requires_grad for p in rhos), \
    'attn_rho is frozen: the flag would do nothing'
# and it must be in the checkpoint, otherwise infer.py rebuilds a model without it
sd = m.trainable_state_dict()
assert sum(1 for k in sd if k.endswith('attn_rho')) == 12, 'rho is not checkpointed'

# the real question: does tau reach the logits?  Move it, and the live forward must
# change while the anchor must not -- anchor_feat means "the frozen official tower",
# and a temperature leaking into it would drift every statistic built on it.
with torch.no_grad():
    for p in rhos:
        p.fill_(0.5)                      # tau = e^0.5 ~ 1.65 per head
    tau1 = m.attn_temperature()
    z_moved = F.normalize(m.clip.visual(x).float(), dim=-1)
    z_anch = m.anchor_feat(x)
d_move = (z_ref - z_moved).abs().max().item()
d_anch = (z_ref - z_anch).abs().max().item()
assert d_move > 1e-3, f'tau changed nothing on the real tower: max|dz|={d_move:.2e}'
assert d_anch < 1e-5, f'anchor is no longer the frozen tower: max|dz|={d_anch:.2e}'
print(f'  tau=1.65 moves the live tower by {d_move:.3e}, the anchor by {d_anch:.2e}')

# gradients must reach rho through F.multi_head_attention_forward
with torch.no_grad():
    for p in rhos:
        p.zero_()
m.train()
m(x).sum().backward()
g = [p.grad for p in rhos if p.grad is not None]
assert len(g) == 12, f'gradient reached {len(g)}/12 rho tensors'
assert all(torch.isfinite(t).all() for t in g), 'non-finite gradient at rho'
assert max(float(t.abs().max()) for t in g) > 0, \
    'every rho gradient is exactly zero -- the temperature is detached'
print(f'  rho gradients: 12/12, max|g|={max(float(t.abs().max()) for t in g):.3e}')

# --- junk: anchor_feat on the real tower, which is what the frozen pass consumes
with torch.no_grad():
    m2 = Net(copy.deepcopy(base), 7, $RANK, 'all', local_head=True,
             lora_qkv=True).eval()
    f = m2.anchor_feat(x)
assert f.shape == (2, 512) and torch.isfinite(f).all(), f.shape
assert float(f.norm(dim=-1).sub(1).abs().max()) < 1e-5, 'anchor_feat is not unit-norm'
print(f'  anchor_feat on the real tower: {tuple(f.shape)}, unit-norm ok')
PYEOF

# ---- gate 2: 20 real batches with --attn-temp --------------------------------
# warmup-epochs is left at its default 3 so the single smoke epoch is a warm-up
# epoch; --limit-batches truncates the inner loop only, so the epoch still ends,
# validation still runs and best.pt is written.
echo "--- gate 2: 20 real batches, --attn-temp -> $SMOKE_AT ---"
if [ -d "$SMOKE_AT" ]; then echo "clearing stale $SMOKE_AT"; rm -rf "$SMOKE_AT"; fi
"$PY" -u train.py --data "$DATA" --out "$SMOKE_AT" \
  --epochs 1 --limit-batches 20 --batch-size 128 --workers 4 --save-every 1 \
  $RECIPE --attn-temp 2>&1 | tee "$SMOKE_AT.log"

"$PY" - <<PYEOF
import re, sys, torch
t = open(r"$SMOKE_AT.log", encoding="utf-8", errors="replace").read()
g = lambda p: re.search(p, t)
checks = [
    ("trainable params=10.325M or 10.326M", g(r"$WANT_PARAMS_RE")),
    ("layers=$WANT_LAYERS",        g(rf"layers=$WANT_LAYERS\b")),
    ("lora_qkv=True",              g(r"lora_qkv=True\b")),
    ("attn_temp=True",             g(r"attn_temp=True\b")),
    ("lora_rank=$RANK",            g(rf"lora_rank=$RANK\b")),
    ("[attn-temp] tau report",     g(r"\[attn-temp\] tau min=")),
    ("epoch completed (val_acc)",  g(r"val_acc")),
]
bad = [c for c, m in checks if not m]
if bad:
    print("SMOKE FAIL, missing from the log:", bad); sys.exit(1)
if "Traceback" in t:
    print("SMOKE FAIL: traceback in the log"); sys.exit(1)
m = g(r"\[attn-temp\] tau min=([\d.]+) mean=([\d.]+) max=([\d.]+)")
print(f"  tau after 20 batches: min={m.group(1)} mean={m.group(2)} max={m.group(3)}")
ck = torch.load(r"$SMOKE_AT/best.pt", map_location="cpu", weights_only=False)
assert ck["attn_temp"] is True, ck["attn_temp"]
n = sum(1 for k in ck["model"] if k.endswith("attn_rho"))
nt = sum(v.numel() for k, v in ck["model"].items() if k.endswith("attn_rho"))
assert n == 12 and nt == 144, f"attn_rho in smoke ckpt: {n} tensors / {nt} scalars, want 12/144"
# 20 batches at lr 1e-4 cannot move rho far, but it must have moved *somewhere*:
# an exactly-zero delta means nothing is connected to the parameter.
dr = [float(v.abs().max()) for k, v in ck["model"].items() if k.endswith("attn_rho")]
print(f"  rho in the checkpoint: {n} tensors / {nt} scalars, max|rho|={max(dr):.3e} (init 0)")
print("SMOKE GATE OK [attn-temp]:", ", ".join(c for c, _ in checks))
PYEOF

# ---- gate 3: 20 real batches with --junk-filter both -------------------------
# The frozen pass ignores --limit-batches: it sweeps all of $DATA.  That is the
# point -- it is the first time the filter sees 134k real images, and the things
# that can go wrong there (a hole in the index coverage, a class silently emptied,
# the cap pinning, the chunked dedup OOM) are all invisible on the stub.
echo "--- gate 3: 20 real batches, --junk-filter both -> $SMOKE_JK ---"
echo "    (the frozen pass covers the WHOLE training split; this gate is ~10 min)"
if [ -d "$SMOKE_JK" ]; then echo "clearing stale $SMOKE_JK"; rm -rf "$SMOKE_JK"; fi
"$PY" -u train.py --data "$DATA" --out "$SMOKE_JK" \
  --epochs 1 --limit-batches 20 --batch-size 128 --workers 4 --save-every 1 \
  $RECIPE --junk-filter both --junk-floor $JUNK_FLOOR \
  --junk-max-frac $JUNK_MAX_FRAC 2>&1 | tee "$SMOKE_JK.log"

"$PY" - <<PYEOF
import re, sys, torch
t = open(r"$SMOKE_JK.log", encoding="utf-8", errors="replace").read()
g = lambda p: re.search(p, t)
checks = [
    ("frozen pass progress",      g(r"\[junk\] frozen pass \d+/\d+ batches")),
    ("centroid rounds",           g(r"\[junk\] centroid round 1/")),
    ("margin report",             g(r"\[junk\] mode=both: judged \d+/\d+")),
    ("margin quantiles",          g(r"\[junk\] margin quantiles")),
    ("duplicate report",          g(r"\[junk\] near-duplicates >")),
    ("flag rate + mean weight",   g(r"\[junk\] flagged \d+/\d+ = ")),
    ("per-class flag rate",       g(r"\[junk\] per-class flag rate")),
    ("no coverage hole",          not g(r"冻结特征 pass 漏了")),
    ("epoch completed (val_acc)", g(r"val_acc")),
]
bad = [c for c, m in checks if not m]
if bad:
    print("SMOKE FAIL, missing from the log:", bad); sys.exit(1)
if "Traceback" in t:
    print("SMOKE FAIL: traceback in the log"); sys.exit(1)
for line in t.splitlines():
    if line.strip().startswith("[junk]") or "!! 警告" in line:
        print("  " + line.strip())
# The risk of a filter is never that it filters too much in aggregate -- the cap
# bounds that -- it is that it empties a small class, and §14.1 says the smallest
# classes here hold 159+ images with 7 below 50.  5% of the split is 6708 samples
# and the rare classes are 7 folders, so "all of the damage in a few folders" is a
# real failure mode, not a hypothetical.
mpc = g(r"per-class flag rate: max ([\d.]+)%, (\d+) classes over 20%")
pc_max, pc_over = float(mpc.group(1)), int(mpc.group(2))
assert pc_max < 1.0, (f"a class had {pc_max:.1%} of its samples flagged -- that class is "
                      "effectively emptied.  Fix the filter, do not start the run.")
assert pc_over <= 10, f"{pc_over} classes have >20% of their samples flagged"
mad = g(r"\[junk\] flagged (\d+)/(\d+) = ")
flagged, total = int(mad.group(1)), int(mad.group(2))
assert flagged > 0, ("the filter flagged NOTHING on the real training split.  Either the "
                     "margin rule does not transfer or the frozen pass is broken -- read "
                     "the quantile line above before starting the 3.4-hour run.")
ck = torch.load(r"$SMOKE_JK/best.pt", map_location="cpu", weights_only=False)
jw = ck["junk_w"]
assert jw is not None, "junk_w is not in the checkpoint"
n_tr = sum(ck["class_counts"])
assert jw.shape == (n_tr,), f"junk_w {tuple(jw.shape)} vs training split {n_tr}"
if total != n_tr:
    print(f"  WARNING: the log says {total} training images, the ckpt says {n_tr}")
hit = int((jw < 1).sum())
assert 0 < hit <= int($JUNK_MAX_FRAC * n_tr) + 1, f"{hit} flagged, cap {int($JUNK_MAX_FRAC*n_tr)}"
# The floor is not exactly representable in float32 (0.2 -> 0.20000000298...), and
# the weights are stored at float32, so this cannot be an == against the python
# literal -- compare to a tolerance instead.
assert abs(float(jw.min()) - $JUNK_FLOOR) < 1e-6, \
    f"min junk weight {float(jw.min())!r} != $JUNK_FLOOR"
assert float(jw.max()) == 1.0, "some unflagged sample is not at weight 1.0"
assert ck["junk_report"] is not None and ck["junk_report"]["mode"] == "both"
print(f"  junk_w: {hit}/{n_tr} = {hit/n_tr:.2%} at {float(jw.min())}, "
      f"mean {float(jw.mean()):.4f} (cap $JUNK_MAX_FRAC)")
print("SMOKE GATE OK [junk]:", ", ".join(c for c, _ in checks))
PYEOF

# ---- handshake for run_attntemp_junk.sh ---------------------------------------
# Written only for the junk arm because that is the one whose gate costs 10 minutes.
# One marker per arm keeps a stale certification from covering a different tree.
mkdir -p "$SMOKE_AT" "$SMOKE_JK"
echo "$(sha256sum train.py | cut -c1-16) $(sha256sum infer.py | cut -c1-16) $ARM" > "$SMOKE_JK/OK"
echo "$(sha256sum train.py | cut -c1-16) $(sha256sum infer.py | cut -c1-16) $ARM" > "$SMOKE_AT/OK"
echo "=== smoke [$ARM] PASSED $(date) -- now safe to start run_attntemp_junk.sh ==="
