"""Probe: what does it take to run ViT-B/32 at a resolution other than 224?

Run this BEFORE touching train.py / infer.py.  ViT-B/32 has patch size 32, so a
legal input is a multiple of 32: 224 = 7x7 patches, 288 = 9x9, 320 = 10x10.
336 is NOT one of them (336/32 = 10.5) -- the "336" people quote belongs to
CLIP ViT-L/14@336px, whose patch is 14.

What was measured on this build (open_clip 3.3.0, torch 2.8.0):

* the tower is ``open_clip.transformer.VisionTransformer``, not a timm model, so
  the positional grid is ``positional_embedding`` with shape
  ``(1 + grid*grid, width)`` -- CLS first, *no* leading batch dimension.  timm
  models instead expose ``pos_embed``.  Both spellings are handled below.
* there is **no automatic interpolation**: 224 succeeds, every other size raises
  ``The size of tensor a (82) must match the size of tensor b (50)``.  So a
  resolution change means interpolating the grid by hand.

This script therefore does two things: it re-measures the above (so the finding
is reproduced rather than assumed), and it *tries* the manual fix -- bicubic
interpolation of the 7x7 patch grid to the target grid, CLS token untouched --
then checks that the resulting features still line up with the 224 ones.  An
interpolation that runs without raising but scrambles the positions would be
worse than no interpolation at all, because training would quietly degrade.

Read-only: mutates the in-memory model only.  CPU is fine, no data, no GPU.

    python probe_resolution.py
"""
import warnings

import torch

import open_clip

MODEL = 'ViT-B-32-quickgelu'
CANDIDATES = (224, 256, 288, 320, 352)
TARGET = 288                     # the size worth actually chasing (see datastats.py)

# Cap the intra-op threads *before* torch spins anything up.  A no-GPU AutoDL
# container is capped at ~2 GB RAM by its cgroup but os.cpu_count() still reports
# every core of the host, so the default setting spawns one worker per core and
# each glibc malloc arena reserves its own heap -- enough to get this script
# OOM-killed (observed: bare `Killed`, with no traceback) before the first
# forward returns.  It is also pointless work at 0.5 CPU.
torch.set_num_threads(2)

try:                                    # reuse the training pipeline's own helpers
    from train import CLIP_MEAN, CLIP_STD, resize_positional_embedding
except Exception:                       # noqa: BLE001 -- probe must still run alone
    CLIP_MEAN, CLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    resize_positional_embedding = None


def rss_mb():
    """Resident set size, so an OOM kill leaves a number behind instead of a
    bare `Killed` the reader has to guess about."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024
    except Exception:                       # noqa: BLE001
        pass
    return float('nan')


def attrs(obj, names):
    """Print every attribute that exists; name what is missing rather than guess."""
    for n in names:
        if not hasattr(obj, n):
            print(f'    {n}: <absent>')
            continue
        v = getattr(obj, n)
        if torch.is_tensor(v):
            print(f'    {n}: tensor shape={tuple(v.shape)} dtype={v.dtype}')
        else:
            print(f'    {n}: {v!r}')


def forward_with_warnings(visual, size):
    """Return (ok, description). Warnings are captured, not just printed -- a
    resolution change that only whispers a warning is exactly the failure mode
    this probe exists to catch."""
    x = torch.randn(1, 3, size, size)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            with torch.inference_mode():
                z = visual(x)
            desc = f'OK  -> output {tuple(z.shape)}'
            ok = True
        except Exception as e:                          # noqa: BLE001
            desc = f'FAILED  {type(e).__name__}: {str(e)[:160]}'
            ok = False
    for w in caught:
        desc += f'\n        WARN: {str(w.message)[:160]}'
    return ok, desc


def find_positional(visual):
    """open_clip names it `positional_embedding`; timm names it `pos_embed`.
    Returns (attribute_name, tensor) or (None, None)."""
    for cand in ('positional_embedding', 'pos_embed'):
        if hasattr(visual, cand):
            return cand, getattr(visual, cand)
    return None, None


def restore(visual, name, pe):
    """Put the trained grid back.  The helper *replaces* the attribute (it has to:
    the grid grows, and `copy_` refuses a shape change), so restoring means
    re-assigning the original object, not writing into it."""
    setattr(visual, name, pe)


def main():
    print(f'open_clip {getattr(open_clip, "__version__", "?")}   torch {torch.__version__}')
    try:
        import timm
        print(f'timm {timm.__version__}')
    except Exception as e:                                  # noqa: BLE001
        print(f'timm not importable: {e}')

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained='openai')
    model.eval()
    visual = model.visual

    print('\n=== visual tower ===')
    print(f'  {type(visual).__module__}.{type(visual).__name__}')
    attrs(visual, ('image_size', 'patch_size', 'grid_size', 'output_dim',
                   'positional_embedding', 'pos_embed'))

    print('\n=== what open_clip hands out as preprocessing ===')
    print(f'  {preprocess}')
    print('  NOTE: this project trains and infers with Resize(256) + CenterCrop(224),')
    print('        which crops to 87.5% of the short side.  open_clip\'s own recipe')
    print('        is Resize(224) + CenterCrop(224), i.e. the whole frame -- that is')
    print('        exactly the `wide` view in infer.py --tta, so it is already a')
    print('        one-flag A/B for whether CLIP\'s native framing helps here.')

    print('\n=== forward at each candidate size (batch 1, random input) ===')
    print(f'  after loading the model: RSS {rss_mb():.0f} MB')
    results = {}
    for size in CANDIDATES:
        ok, desc = forward_with_warnings(visual, size)
        results[size] = ok
        print(f'  {size:>3}x{size:<3} (grid {size // 32}x{size // 32}): {desc}'
              f'   [RSS {rss_mb():.0f} MB]')

    name, pe = find_positional(visual)
    if pe is None:
        print('\nno positional embedding found on the tower -- inspect it by hand.')
        return
    batched = pe.dim() == 3
    flat = pe[0] if batched else pe
    base_grid = int(round((flat.shape[0] - 1) ** 0.5))
    print(f'\npositional embedding: `{name}` shape={tuple(pe.shape)} '
          f'-> {base_grid}x{base_grid} grid + 1 cls')

    def content_feature(size):
        """Feature of one smooth synthetic image at ``size``.  Same content, so a
        correct interpolation keeps the features correlated; a scrambled one sends
        the similarity towards chance."""
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 256), torch.linspace(0, 1, 256),
                                indexing='ij')
        img = torch.stack([yy, xx, (yy + xx) / 2]).unsqueeze(0)
        x = torch.nn.functional.interpolate(img, size=(size, size), mode='bilinear',
                                            align_corners=False)
        x = (x - torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)) / torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        with torch.inference_mode():
            z = visual(x)
        return torch.nn.functional.normalize(z.float(), dim=-1)

    ref = content_feature(224)                       # measured at the trained grid

    broken = [s for s in CANDIDATES if not results[s]]
    if not broken:
        print('\nevery candidate already works -- no interpolation needed.')
        return

    print(f'\n=== retrying {broken} after manual positional interpolation ===')
    if resize_positional_embedding is None:
        print('  train.py is not importable from here, so the helper cannot be reused.')
        print('  Run this from the repository directory.')
        return
    pristine = pe                                  # the trained grid, untouched
    for size in broken:
        # each attempt must start from the *trained* grid; resampling a grid a
        # previous attempt already replaced would compound the error
        restore(visual, name, pristine)
        try:
            resize_positional_embedding(visual, size)
        except SystemExit as e:                       # rejected size, or no grid found
            print(f'  {size}: {e}')
            continue
        ok, desc = forward_with_warnings(visual, size)
        line = f'  {size}: {desc}'
        if ok:
            sim = float((ref * content_feature(size)).sum(dim=-1))
            line += f'\n        cosine(feat@224, feat@{size}) = {sim:+.4f}'
            if sim > 0.5:
                line += '   <- positions line up; fine-tuning here is viable'
            elif sim > 0.0:
                line += '   <- weakly aligned; risky, prefer 256 or stay at 224'
            else:
                line += '   <- NOT aligned; do not train this way'
        print(line)
    restore(visual, name, pristine)                 # leave the model as we found it

    print(f'\n=== verdict for {TARGET} ===')
    ok, desc = forward_with_warnings(visual, TARGET)
    print(f'  with the trained grid restored, {TARGET} -> {desc}')
    print('  (expected to fail again -- the interpolation above was reverted on purpose)')
    print('\n  The plumbing is already in place (train.py / infer.py / analyze.py /')
    print('  valmetrics.py all read --img-size, and the grid is resampled through')
    print('  train.resize_positional_embedding).  What is still open is only the')
    print('  verdict above: a high cosine means a 288 run is worth its 2 GPU hours,')
    print('  a low one means stay at 224 and spend the time elsewhere.')


if __name__ == '__main__':
    main()
