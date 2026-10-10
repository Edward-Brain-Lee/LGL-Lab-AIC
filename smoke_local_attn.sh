#!/usr/bin/env bash
set -euo pipefail
ARM=local_attn_junk
PY="${PY:-python}"
DATA="${DATA:-/root/autodl-tmp/train}"
TEST="${TEST:-/root/autodl-tmp/test}"
OUT="smoke_${ARM}"
RANK=64
MAX_FRAC=0.05
FLOOR=0.2
for f in train.py infer.py losses.py noise.py semantic_regularization.py; do
  test -f "$f" || { echo "FATAL: missing dependency $f" >&2; exit 1; }
done
test ! -e "$OUT" || { echo "FATAL: refusing to overwrite $OUT" >&2; exit 1; }
test ! -e "$OUT.log" || { echo "FATAL: refusing to overwrite $OUT.log" >&2; exit 1; }
echo "=== smoke [$ARM] $(date) ==="
sha256sum train.py infer.py losses.py noise.py semantic_regularization.py
# NOT `bash -n a.sh b.sh`: bash reads the first non-option word as the command
# file and everything after it as that script's positional parameters, so a
# second file on the same line is never parsed.  Prove it with
# `bash -n smoke_local_attn.sh /nonexistent.sh` -- it exits 0.
for s in smoke_local_attn.sh run_local_attn.sh; do
  bash -n "$s" && echo "  bash -n $s: ok"
done
"$PY" -m py_compile train.py infer.py losses.py noise.py semantic_regularization.py
# the contract test, including check_local_attention_and_schedule, which nothing
# else runs -- without this line the new self-test only executes if someone
# remembers to run selftest.py by hand.
echo '--- contract self-test ---'
"$PY" selftest.py | tail -n 3
echo '--- local attention real-tower forward/backward ---'
"$PY" - <<'PYEOF'
import copy, torch, torch.nn.functional as F, open_clip
from train import Net, resize_positional_embedding
torch.manual_seed(17)
base = open_clip.create_model('ViT-B-32-quickgelu', pretrained='openai').eval()
resize_positional_embedding(base.visual, 384)
m = Net(copy.deepcopy(base), 7, 64, 'all', local_head=True, lora_qkv=True, local_attn=True).train()
x = torch.randn(2, 3, 384, 384)
with torch.no_grad():
    h = m.local_head
    assert h is not None and h.query_proj is not None and h.key_proj is not None
    assert torch.equal(h.sim_scale, torch.ones_like(h.sim_scale))
    tok = torch.randn(2, 9, h.proj.in_features)
    assert not torch.allclose(h(tok), F.normalize(h.proj(tok.mean(1)).float(), dim=-1), atol=1e-5)
    # The production residual gate, rather than the inner attention, preserves
    # the exact incumbent output at construction.
    z0 = F.normalize(m.clip.encode_image(x).float(), dim=-1)
    z1 = m(x, return_feat=True)[1]
    assert torch.allclose(z0, z1, atol=1e-5)
    # Open the residual only for this gradient check; production starts at zero.
    h.gate.fill_(0.1)
m(x).sum().backward()
g = m.local_head.sim_scale.grad
assert g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0
assert m.local_head.query_proj.weight.grad is not None
sd = m.trainable_state_dict()
assert any('local_head.query_proj.weight' in k for k in sd)
assert any('local_head.key_proj.weight' in k for k in sd)
assert any('local_head.sim_scale' in k for k in sd)
fresh = Net(copy.deepcopy(base), 7, 64, 'all', local_head=True, lora_qkv=True, local_attn=True)
missing, unexpected = fresh.load_state_dict(sd, strict=False)
assert not unexpected and not ({n for n,p in fresh.named_parameters() if p.requires_grad} & set(missing))
print('forward/backward/checkpoint ok; sim_scale_grad=', g.tolist())
PYEOF
echo '--- junk cap/floor/per-class quota ---'
"$PY" - <<PYEOF
import torch
from train import robust_centroids, junk_weights
torch.manual_seed(3)
nclass, per, d = 7, 20, 16
truth = torch.nn.functional.normalize(torch.randn(nclass,d), dim=-1)
z = truth.repeat_interleave(per,0) + .01*torch.randn(nclass*per,d)
y = torch.arange(nclass).repeat_interleave(per)
y[:4] = 1
# Class 6's folder is entirely foreign, and deliberately split across TWO other
# classes.  A folder that is uniformly one foreign class has a centroid that IS
# that class, so its members agree with it and are never flagged: the never-empty
# guard only fires when a class is left with nothing.  Split across two, the
# centroid lands between them and all twenty members disagree with it at once --
# which is what the per-class quota has to hold down.
z6 = torch.cat([truth[0] + .01*torch.randn(per//2, d),
                truth[1] + .01*torch.randn(per//2, d)])
z = torch.cat([z, z6])
y = torch.cat([y, torch.full((per,), 6)])
means, present, _ = robust_centroids(z, y.tolist(), nclass, rounds=2, verbose=False)
w, rep = junk_weights(z, y, means, present, mode='margin', max_frac=$MAX_FRAC, floor=$FLOOR, dedup_tau=.999, verbose=False)
counts = torch.bincount(y, minlength=nclass)
hit = int((w < 1).sum())
assert w.shape[0] == z.shape[0] and float(w.min()) >= $FLOOR - 1e-6
assert hit <= int($MAX_FRAC*z.shape[0]), f'{hit} flagged, budget {int($MAX_FRAC*z.shape[0])}'
# the line every downstream gate reads must agree with the weights actually written
assert rep['flagged'] == hit, (rep['flagged'], hit)
# class 6 is 20/20 suspicious and its quota is ceil(5% of 20) = 1.  A global
# top-k with no quota spends its first slots on class 1 (margin ~1.0) and then
# on class 6 (margin ~0.3), so it flags 3 of the 20 -- that is what this catches.
assert int((w[y==6] < 1).sum()) <= 1, int((w[y==6] < 1).sum())
assert rep['classes_over_20pct'] == 0, rep
mass = torch.zeros(nclass).index_add_(0, y, w)
assert torch.all(mass >= counts.float()*$FLOOR - 1e-5), (mass, counts)
assert int((w[y==1] < 1).sum()) < int(counts[1])
print('junk quota ok:', {k: rep[k] for k in ('judged','margin_pos','flagged','per_class_max','classes_over_20pct')})
PYEOF
echo '--- short real train + infer reconstruction ---'
test -d "$DATA" || { echo "FATAL: DATA=$DATA not found" >&2; exit 1; }
test -d "$TEST" || { echo "FATAL: TEST=$TEST not found" >&2; exit 1; }
"$PY" -u train.py --data "$DATA" --out "$OUT" --epochs 1 --limit-batches 2 --batch-size 32 --workers 2 --save-every 1 --img-size 384 --train-pos-embed --local-head --local-attn --lora-rank "$RANK" --lora-qkv --robust-loss apl --junk-filter margin --junk-floor "$FLOOR" --junk-max-frac "$MAX_FRAC" 2>&1 | tee "$OUT.log"
test -f "$OUT/best.pt" || { echo 'FATAL: no best.pt' >&2; exit 1; }

# The flag has to have reached train.py, not just the command line.  If any of
# these is missing the arm silently becomes "incumbent + junk margin" and still
# trains for 3.4 hours.
"$PY" - <<PYEOF
import re, sys
t = open(r"$OUT.log", encoding="utf-8", errors="replace").read()
g = lambda p: re.search(p, t)
checks = [
    ("config local_attn=True",    g(r"local_attn=True\b")),
    ("config local_head=True",    g(r"local_head=True\b")),
    ("config junk_filter=margin", g(r"junk_filter=margin\b")),
    ("trainable params grew",     g(r"trainable params=10\.4\d\d?M")),
    ("local_gate printed",        g(r"local_gate=")),
    ("junk report line",          g(r"\[junk\] flagged \d+/\d+ = ")),
]
bad = [c for c, m in checks if not m]
if bad:
    print("SMOKE FAIL, missing from the log:", bad); sys.exit(1)
if "Traceback" in t:
    print("SMOKE FAIL: traceback in the log"); sys.exit(1)
print("  log gate ok:", ", ".join(c for c, _ in checks))
PYEOF

"$PY" -u infer.py --test "$TEST" --checkpoint "$OUT/best.pt" --output "$OUT/pred_results.csv" --batch-size 8 --workers 0 >/dev/null
test -s "$OUT/pred_results.csv"
echo "$(sha256sum train.py | cut -d' ' -f1) $(sha256sum infer.py | cut -d' ' -f1) $ARM" > "$OUT/OK"
echo "=== smoke [$ARM] PASSED ==="
