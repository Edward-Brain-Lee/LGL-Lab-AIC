"""Gate on the *real* open_clip ViT-B/32 tower, before a 2-hour run is started.

`selftest.py` proves the maths on a 32-dim stub.  It cannot prove any of this,
because every one of these facts lives in open_clip's actual module tree:

  * `ResidualAttentionBlock.attention()` calls `self.attn(q, k, v, need_weights=,
    attn_mask=)`, i.e. three positional args, so a LoRA wrapper that only
    implements `forward` breaks on the real tower and nowhere else;
  * `--local-head` routes every training step through
    `visual.forward_intermediates` -> `Transformer.forward_intermediates`;
  * `--lora-qkv` has to replace the *fused* `in_proj` of
    `nn.MultiheadAttention` and still feed `F.multi_head_attention_forward` a
    merged weight matrix;
  * `--train-pos-embed` at 384 has to resample the 7x7 grid to 12x12 and make it
    a `Parameter` the optimiser will see;
  * `anchor_feat()` must return the frozen tower's embedding with adapters off.

The expected numbers come from `sf_gate.expected_trainable`, which self-checks
against the 750-class run that actually printed 10.326M (HANDOFF 43.2), so this
gate and the log gate cannot disagree about what "the right model" is.

    python sf_tower_gate.py --rank 64 --classes 500 --img-size 384
"""
import argparse
import copy
import sys

import torch
import torch.nn.functional as F

import open_clip

import sf_gate
from train import Net, check_backbone, enable_pos_embed_training, \
    resize_positional_embedding


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rank', type=int, default=64)
    p.add_argument('--classes', type=int, default=sf_gate.SF_CLASSES)
    p.add_argument('--img-size', type=int, default=384)
    p.add_argument('--model', default='ViT-B-32-quickgelu')
    a = p.parse_args(argv)

    check_backbone(a.model)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[tower gate] device={dev} model={a.model} rank={a.rank} '
          f'classes={a.classes} img_size={a.img_size}')

    torch.manual_seed(0)
    base = open_clip.create_model(a.model, pretrained='openai')
    if a.img_size != 224:
        grid = resize_positional_embedding(base.visual, a.img_size)
        want_grid = a.img_size // 32
        assert grid == want_grid, f'grid {grid}, expected {want_grid}'
        print(f'  positional grid: 7x7 -> {grid}x{grid} (bicubic)')
    base = base.eval()

    net = Net(copy.deepcopy(base), a.classes, a.rank, 'all',
              local_head=True, lora_qkv=True)
    if a.img_size != 224:
        n_pe = enable_pos_embed_training(net)
        tokens = (a.img_size // 32) ** 2 + 1
        assert n_pe == tokens * 768, f'pos grid {n_pe}, expected {tokens * 768}'
        print(f'  --train-pos-embed: {n_pe:,} params, {tokens} tokens')

    # --- module tree -------------------------------------------------------
    assert net.n_lora == 48, f'n_lora={net.n_lora}, expected 48 (12 blocks x 4)'
    n_adapter = sum(pp.numel() for n, pp in net.named_parameters()
                    if n.endswith(('.A', '.B')))
    exp = sf_gate.expected_trainable(a.classes, rank=a.rank, qkv=True,
                                     local_head=True,
                                     pos_tokens=(a.img_size // 32) ** 2 + 1)
    assert n_adapter == exp['adapters'], \
        f'adapter params {n_adapter:,}, expected {exp["adapters"]:,}'
    n_train = sum(pp.numel() for pp in net.parameters() if pp.requires_grad)
    assert n_train == exp['total'], \
        f'trainable {n_train:,}, expected {exp["total"]:,} ' \
        f'(printed {exp["total"] / 1e6:.3f}M)'
    print(f'  adapters {n_adapter:,} over {net.n_lora} modules; '
          f'trainable total {n_train:,} = {exp["total"] / 1e6:.3f}M')

    # --- the wrapped tower must still be the official tower at step 0 ------
    # Both sides must be fed the resolution the grid now expects: after
    # `resize_positional_embedding(…, 384)` the official tower's PE is 145 tokens,
    # so a 224x224 probe is a shape error, not a mismatch.  (That bug used to sit
    # right here and only surfaced on the first GPU host.)
    net = net.to(dev).eval()
    base = base.to(dev)
    x_eq = torch.randn(2, 3, a.img_size, a.img_size, device=dev)
    with torch.no_grad():
        z_ref = F.normalize(base.visual(x_eq).float(), dim=-1)
        z_new = F.normalize(net.clip.visual(x_eq).float(), dim=-1)
    d = (z_ref - z_new).abs().max().item()
    assert d < 1e-5, f'wrapped tower != official tower at init: max|dz|={d:.2e}'
    print(f'  init equivalence on the real tower @{a.img_size}px: max|dz|={d:.2e}')

    # --- the paths every training step takes, at the trained resolution ----
    x = x_eq
    with torch.no_grad():
        out = net(x)
        anc = net.anchor_feat(x)
    assert out.shape == (2, a.classes), out.shape
    assert anc.shape == (2, 512), anc.shape
    assert torch.isfinite(out).all() and torch.isfinite(anc).all()
    print(f'  forward_intermediates @{a.img_size}: logits {tuple(out.shape)}, '
          f'anchor {tuple(anc.shape)}')

    # --- gradients must reach the qkv adapter ------------------------------
    # LoRA's B is zero-initialised, so grad_A = B^T G is exactly zero on the very
    # first backward; B takes the signal at step 0, A from step 1 onwards.
    net = net.train()
    net.zero_grad(set_to_none=True)
    net(x).sum().backward()

    def grads(suffix):
        return [pp.grad for n, pp in net.named_parameters()
                if n.endswith(suffix) and pp.grad is not None]

    gb = grads('attn.qkv.B')
    assert len(gb) == 12, f'{len(gb)} of 12 qkv.B tensors have a gradient'
    assert all(torch.isfinite(g).all() for g in gb), 'non-finite gradient at qkv.B'
    sb = sum(g.abs().sum().item() for g in gb)
    assert sb > 0, 'gradient at attn.qkv.B is exactly zero — adapter is detached'
    print(f'  attn.qkv.B: {len(gb)} tensors, sum|g|={sb:.3e}')

    with torch.no_grad():
        for n, pp in net.named_parameters():
            if n.endswith('attn.qkv.B'):
                pp.copy_(torch.randn_like(pp) * 0.02)
    net.zero_grad(set_to_none=True)
    net(x).sum().backward()
    ga = grads('attn.qkv.A')
    assert len(ga) == 12, f'{len(ga)} of 12 qkv.A tensors have a gradient'
    sa = sum(g.abs().sum().item() for g in ga)
    assert sa > 0, 'gradient at attn.qkv.A is zero even with B != 0'
    print(f'  attn.qkv.A: {len(ga)} tensors, sum|g|={sa:.3e}')

    print('TOWER OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
