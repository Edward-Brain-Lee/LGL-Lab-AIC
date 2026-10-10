#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 发车前闸门 for the COMBINED arm: --lora-rank 64 --lora-qkv.
#
#   cd /root/autodl-tmp
#   bash smoke_lr64qkv.sh          # read the output BEFORE starting run_lr64qkv.sh
#
# One arm only -- the qkv/ctrl matched pair keeps its own smoke_qkv.sh.  This is
# the P0 of HANDOFF §39.5.  Both axes now have measurements and they disagree:
#
#   placement (rank 32 -> +--lora-qkv)  val_acc +0.42 -> 73.9077  (2.19x, board)
#   capacity  (rank 32 -> rank 64)      val_acc +0.68 -> 73.47    (0.48x, proxy)
#
# Twice the hold-out gain, half the conversion: teammate §42's rank-64 arm reads
# below the incumbent on score.py.  Nothing here should be read off val_acc.
# 0.675 is the rank 16 -> 32 rate on the out_proj path; it is NOT a rank 32 -> 64
# rate.  This run is where the two axes are either compounded or shown to collide.
#
# Multi-model fusion is forbidden by the rules, so a single recipe carrying both
# changes is the ONLY legal way to "combine" the three arms.
#
# Three gates, cheapest first, fail-loud at the first mismatch:
#   0. bash -n on both scripts + selftest.py    ~30 s, CPU, no data
#   1. the real ViT-B/32 tower                  ~1 min, no data
#   2. 20 real batches through train.py         ~2-4 min, needs the GPU to itself
#
# Gate 2 runs the *actual* pipeline: ImageFolderNoisy's split and 750-class
# medoid library, 384 augmentation, resize_positional_embedding, the noise
# tracker, the EMA teacher's named_parameter zip (which must line up across the
# .qkv.A/.B names and is asserted with `tn == sn`), and checkpoint writing.
# ---------------------------------------------------------------------------
set -euo pipefail

# single arm; the argument is accepted so a typo cannot silently pick a default
ARM="${1:-lr64qkv}"
if [ "$ARM" != "lr64qkv" ]; then
  echo "usage: bash smoke_lr64qkv.sh   (no arm argument)" >&2; exit 2
fi

RANK=64
QKV_FLAG="--lora-qkv"
WANT_QKV="True"
WANT_LAYERS=48
# exact, not rounded: 48 adapter modules x 3,072 params per rank unit x rank 64.
# 12 x (qkv 3072 + out_proj 1536 + c_fc 3840 + c_proj 3840) = 12 x 12,288.
WANT_LORA=9437184
# The TOTAL carries the rounding, not the adapter part.  Backing the fixed base
# (classifier + --local-head + the 111.4k pos grid) out of five 3-decimal prints
# -- rank 16 2.658M / rank 32 4.428M / rank 64 7.966M (§42.1), our rank 32+qkv
# 5.607M and ctrl 5.644M -- brackets it at [888,556, 888,612), so the total lands
# at 10.326M (observed 2026-10-06).  Note f'{n/1e6:.3f}' flips 10.325 -> 10.326 at
# 10,325,500, not 10,325,999; 10.325M stays accepted only as a rounding guard.
# WANT_LORA above is what actually pins the adapter down.
WANT_PARAMS_RE='trainable params=(?:10\.325|10\.326)M'

PY="${PY:-python}"
DATA="${DATA:-/root/autodl-tmp/train}"
SMOKE="smoke_$ARM"

echo "=== smoke [$ARM] $(date) ==="
echo "tree    : $PWD"
echo "expected: layers=$WANT_LAYERS adapter_params=$WANT_LORA trainable=10.325-10.326M"
sha256sum train.py infer.py | cut -c1-16

# The smoke needs the GPU to itself: HANDOFF §29.7-3 records a smoke sharing the
# box with a 384px training run dying with a spurious cuDNN error.
if pgrep -af 'train\.py' | grep -v grep; then
  echo "FATAL: a train.py is already running -- the smoke needs the GPU alone" >&2; exit 1
fi

# ---- gate 0: shell syntax, then selftest -------------------------------------
# bash has no local compile step, and these two files cannot be checked on the
# Windows box the editing happens on.  `bash -n` on both is the substitute, and
# it is run here so a syntax error in the 3.4-hour driver is caught by the smoke
# rather than by `nohup` at 3am.
echo "--- gate 0a: bash -n ---"
bash -n smoke_lr64qkv.sh && echo "  smoke_lr64qkv.sh: syntax ok"
if [ -f run_lr64qkv.sh ]; then
  bash -n run_lr64qkv.sh && echo "  run_lr64qkv.sh: syntax ok"
else
  echo "  WARNING: run_lr64qkv.sh not found beside this script" >&2
fi

# selftest proves the qkv path on the stub tower: numerics vs
# nn.MultiheadAttention, zero-init no-op, the module tree, checkpoint contents
# (qkv.A/B in, frozen qkv weight out), and a train -> best.pt -> infer round
# trip.  It runs on a stub, so it says nothing about the real open_clip call
# pattern -- that is gate 1.
echo "--- gate 0b: selftest ---"
"$PY" selftest.py

# ---- gate 1: the real tower, which the stub never touches ---------------------
# Everything here is load-bearing and, on the real ViT-B/32, otherwise unproven:
#   * ResidualAttentionBlock.attention() calls self.attn(q, k, v, need_weights=,
#     attn_mask=) -- three positional args (_oc_src/transformer.py:283);
#   * --local-head makes every training step go through
#     forward_intermediates -> Transformer.forward_intermediates -> blk(x,
#     attn_mask=) -> that same three-arg call;
#   * anchor_feat() must still return the frozen tower's embedding with the
#     adapters disabled;
#   * and the point of the whole arm: gradients must reach attn.qkv.A/B through
#     F.multi_head_attention_forward.
echo "--- gate 1: real tower next ---"
"$PY" - <<PYEOF
import copy
import torch
import torch.nn.functional as F
import open_clip
from train import Net

torch.manual_seed(0)
base = open_clip.create_model('ViT-B-32-quickgelu', pretrained='openai').eval()
x = torch.randn(2, 3, 224, 224)

# qkv built exactly as the run builds it, so this is the model that will train
mq = Net(copy.deepcopy(base), 7, $RANK, 'all', local_head=True,
         lora_qkv=$WANT_QKV).eval()
assert mq.n_lora == $WANT_LAYERS, f'n_lora={mq.n_lora}, want $WANT_LAYERS'

# "layers=" is the number of LoRALinear instances add_lora built.  With
# --lora-qkv each attention block contributes 4 (qkv + out_proj + c_fc + c_proj)
# instead of 3, so 12 blocks give 48, not 36+1: the qkv projection is one extra
# module *per block*, not one for the whole tower.
#
# n_lora alone cannot tell a correct tree from one where out_proj got wrapped a
# second time (the guard at train.py:407 is what stops that), but the exact
# adapter parameter count can -- a double wrap would add rank*(768+768)*12.
# At rank 64 that is 9,437,184: this is the assertion that actually pins the
# rank down, since the printed total is only good to 1,000.
n_lora_p = sum(p.numel() for n, p in mq.named_parameters()
               if n.endswith('.A') or n.endswith('.B'))
assert n_lora_p == $WANT_LORA, f'LoRA params={n_lora_p}, want $WANT_LORA'
print(f'  adapter params exact: {n_lora_p} over {mq.n_lora} modules')

# the wrapped tower must be the official tower at init (B is zero-initialised)
with torch.no_grad():
    z_ref = F.normalize(base.visual(x).float(), dim=-1)
    z_new = F.normalize(mq.clip.visual(x).float(), dim=-1)
d = (z_ref - z_new).abs().max().item()
assert d < 1e-5, f'wrapped tower != official tower at init: max|dz|={d:.2e}'
print(f'  init equivalence on the real tower: max|dz|={d:.2e}')

# the --local-head path every training step takes, and the frozen-anchor path
with torch.no_grad():
    out = mq(x)
assert out.shape == (2, 7), out.shape
with torch.no_grad():
    anc = mq.anchor_feat(x)
assert anc.shape == (2, 512), anc.shape
print('  forward_intermediates + anchor_feat ok')

# training must actually be able to learn the adapter -- the whole point of the
# arm.  Watch the order: LoRA's B is zero-initialised, so grad_A = B^T @ G is
# exactly zero on the first backward; asserting a non-zero A gradient here would
# fail on a correct implementation.  B takes the signal at step 0; then B is made
# non-zero and A is checked, which is what step 2 onwards looks like.
mq.train()
mq(x).sum().backward()

def grads(suffix):
    return [p.grad for n, p in mq.named_parameters()
            if n.endswith(suffix) and p.grad is not None]

probe = 'attn.qkv.B'
gb = grads(probe)
assert gb, f'no gradient reached ...{probe}'
assert all(torch.isfinite(p).all() for p in gb), f'non-finite gradient at {probe}'
tb = sum(p.abs().sum().item() for p in gb)
assert tb > 0, f'gradient at {probe} is exactly zero -- the adapter is detached'
print(f'  {probe}: {len(gb)} tensors, sum|g|={tb:.3e}')

probe_a = probe[:-1] + 'A'
with torch.no_grad():
    for n, p in mq.named_parameters():
        if n.endswith(probe):
            p.copy_(torch.randn_like(p) * 0.02)
mq.zero_grad(set_to_none=True)
mq(x).sum().backward()
ga = grads(probe_a)
assert ga, f'no gradient reached ...{probe_a}'
ta = sum(p.abs().sum().item() for p in ga)
assert ta > 0, f'gradient at {probe_a} is zero even with B != 0'
print(f'  {probe_a}: {len(ga)} tensors, sum|g|={ta:.3e}')
assert len(gb) == 12 and len(ga) == 12, f'{len(gb)}/{len(ga)} of 12 qkv blocks'
PYEOF

# ---- gate 2: 20 real batches through the actual pipeline ----------------------
# --limit-batches only truncates the inner batch loop (train.py:889), so the epoch
# still ends, validation still runs against the real hold-out, and best.pt is
# written -- which pulls the checkpoint assertion in run_lr64qkv.sh hours earlier.
# warmup-epochs is left at its default 3 so the single smoke epoch is a warmup
# epoch, exactly as §29's gate 1 ran it: the anchor/proto path is exercised too.
echo "--- gate 2: 20 real batches -> $SMOKE ---"
if [ -d "$SMOKE" ]; then echo "clearing stale $SMOKE"; rm -rf "$SMOKE"; fi
"$PY" -u train.py --data "$DATA" \
  --out "$SMOKE" \
  --epochs 1 --limit-batches 20 --batch-size 128 --workers 4 --save-every 1 \
  --img-size 384 --train-pos-embed --local-head \
  --lora-rank "$RANK" $QKV_FLAG 2>&1 | tee "$SMOKE.log"

"$PY" - <<PYEOF
import re, sys, torch
t = open(r"$SMOKE.log", encoding="utf-8", errors="replace").read()
g = lambda p: re.search(p, t)
checks = [
    ("trainable params=10.325M or 10.326M", g(r"$WANT_PARAMS_RE")),
    ("layers=$WANT_LAYERS",           g(r"layers=$WANT_LAYERS ")),
    ("lora_qkv=$WANT_QKV",            g(r"lora_qkv=$WANT_QKV ")),
    ("lora_rank=$RANK",               g(r"lora_rank=$RANK ")),
    ("img_size=384",                  g(r"img_size=384")),
    ("local_head=True",               g(r"local_head=True")),
    ("train_pos_embed=True",          g(r"train_pos_embed=True")),
    ("epoch completed (val_acc)",     g(r"val_acc")),
]
bad = [c for c, m in checks if not m]
if bad:
    print("SMOKE FAIL, missing from the log:", bad); sys.exit(1)
if "Traceback" in t:
    print("SMOKE FAIL: traceback in the log"); sys.exit(1)
m = g(r"trainable params=([\d.]+)M")
print(f"  trainable params as printed: {m.group(1)}M")
ck = torch.load(r"$SMOKE/best.pt", map_location="cpu", weights_only=False)
assert ck["lora_rank"] == $RANK, ck["lora_rank"]
assert ck["lora_qkv"] is True, ck["lora_qkv"]
n = sum(1 for k in ck["model"] if k.endswith(("attn.qkv.A", "attn.qkv.B")))
assert n == 24, f"qkv tensors in smoke ckpt: {n}"
print("SMOKE GATE OK:", ", ".join(c for c, _ in checks), f"| ckpt qkv tensors={n}")
PYEOF

# ---- handshake for run_lr64qkv.sh ---------------------------------------------
# Records which train.py this gate passed against, so a later edit invalidates it
# instead of silently certifying a different tree.
mkdir -p "$SMOKE"
echo "$(sha256sum train.py | cut -c1-16) $(sha256sum infer.py | cut -c1-16) $ARM" > "$SMOKE/OK"
echo "=== smoke [$ARM] PASSED $(date) -- now safe to start run_lr64qkv.sh ==="
