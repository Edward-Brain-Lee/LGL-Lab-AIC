"""Is the local token branch doing anything, and do the confident mistakes look like junk?

Zero training, one pass over the held-out split:

    python attn_audit.py --checkpoint outputs_sf_base/best.pt --data /root/autodl-tmp/train

Why this exists.  HANDOFF 45.8 puts attention work on the table, but 45.5 says the
errors are diffuse (the top 20 confusion pairs are only 5.7% of them) and 8.9% of
the hold-out is confidently wrong -- the signature of label noise, not of missing
parts.  So before paying ~2 h of GPU for an attention arm, this prints the two
numbers that decide whether the axis has a target at all:

  * ``tanh(local_head.gate)`` -- if it is ~0 the entire local branch is multiplied
    by zero, every ``--local-attn`` / ``--local-se`` parameter gets zero gradient,
    and no variant of either can matter.  This is checkable in one second.
  * token dispersion vs correctness -- if the confidently-wrong images are no more
    degenerate than the correct ones, "attention that suppresses junk patches" has
    nothing to suppress.

Read-only: it never writes to the checkpoint, the output dirs, or the submission
files.  The only file it creates is its own ``--out`` CSV.

Token dispersion is computed in the tower's final token space as
``mean_p (1 - cos(token_p, mean_token))``: one number per image for "how unlike the
image's own average is a typical patch".  A uniform patch set gives ~0; a photo
with a busy subject and flat background gives something clearly positive.  It is
deliberately a *frozen-tower* statistic, so it does not depend on the trained
projection drifting.
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import open_clip

import train as T


def model_args(ck, data, img_size, workers, batch_size):
    """A train.py args namespace with training's own defaults, then overridden.

    The hold-out split is rebuilt by ``train.build_datasets``, so ``--val-ratio``
    and ``--split-seed`` have to come from the same place training read them.
    Hand-rolling the split here would silently produce a different hold-out set and
    every per-class number downstream would be quietly wrong.
    """
    ck_args = ck.get('args', {}) or {}
    argv = ['--data', str(data)]
    # `seed` matters as much as `split_seed`: build_datasets falls back to
    # `a.seed` whenever `--split-seed` is None, so replaying only the latter would
    # rebuild a *different* hold-out for any run that used a non-default --seed,
    # and every per-class recall below would be measured on the wrong images.
    for key in ('seed', 'val_ratio', 'split_seed', 'crop_min', 'randaug_n', 'randaug_m'):
        val = ck_args.get(key)
        if val is not None:
            argv += ['--' + key.replace('_', '-'), str(val)]
    a = T.parse_args(argv)
    a.data, a.img_size, a.workers, a.batch_size = str(data), img_size, workers, batch_size
    return a


def load_model(path, device):
    """Rebuild the exact architecture the checkpoint was trained with.

    Mirrors infer.py's loader on purpose: any flag that changes the module tree
    (qkv LoRA, attn-temp, local-head, local-attn, local-se) has to be replayed or
    the trained tensors land in `unexpected` and the model runs as a different net.
    """
    ck = torch.load(str(path), map_location='cpu', weights_only=False)
    ca = ck.get('args', {}) or {}
    classes = ck['classes'] if 'classes' in ck else ck['class_to_idx']
    rank = ck.get('lora_rank', ca.get('lora_rank', 8))
    target = ck.get('lora_target', ca.get('lora_target', 'all'))
    flags = {k: bool(ck.get(k, ca.get(k, False)))
             for k in ('lora_qkv', 'attn_temp', 'local_head', 'local_attn', 'local_se')}
    alpha = float(ck.get('lora_alpha', ca.get('lora_alpha', 0.0)) or 0.0)
    model_name = ck.get('model_name', ca.get('model', 'ViT-B-32-quickgelu'))
    img_size = int(ck.get('img_size', ca.get('img_size', 224)))

    clip_model = open_clip.create_model(model_name, pretrained=ck.get('pretrained', 'openai'))
    if img_size != 224:
        T.resize_positional_embedding(clip_model.visual, img_size)
    model = T.Net(clip_model, len(classes), rank, target,
                  local_head=flags['local_head'], lora_qkv=flags['lora_qkv'],
                  lora_alpha=alpha, attn_temp=flags['attn_temp'],
                  local_attn=flags['local_attn'], local_se=flags['local_se'])
    missing, unexpected = model.load_state_dict(ck.get('model', ck), strict=False)
    if unexpected:
        raise SystemExit('checkpoint carries trained weights this model cannot hold: '
                         f'{sorted(unexpected)[:5]}')
    lost = {n for n, p in model.named_parameters() if p.requires_grad} & set(missing)
    if lost:
        raise SystemExit(f'checkpoint has no trained weights for: {sorted(lost)[:5]}')
    del clip_model
    return ck, model.to(device).eval(), flags, img_size, classes


@torch.no_grad()
def audit(a):
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available()
                          else ('cpu' if a.device == 'auto' else a.device))
    ck, model, flags, img_size, classes = load_model(a.checkpoint, device)
    if model.local_head is None:
        raise SystemExit('this checkpoint has no --local-head branch, so there is nothing '
                         'to audit; attention work needs a local-head checkpoint')

    args_a = model_args(ck, a.data, img_size, a.workers, a.batch_size)
    tr, va = T.build_datasets(args_a)
    if a.limit:
        va.items = va.items[:a.limit]
        va.targets = [y for _, y in va.items]
    loader = DataLoader(va, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=True)

    gate = float(torch.tanh(model.local_head.gate))
    m = model.local_head
    print(f'checkpoint : {a.checkpoint} (epoch {ck.get("epoch", "?")})')
    print(f'recipe     : local_head={flags["local_head"]} local_attn={flags["local_attn"]} '
          f'local_se={flags["local_se"]} attn_temp={flags["attn_temp"]} qkv={flags["lora_qkv"]}')
    print(f'hold-out   : {len(va)} images, {len(classes)} classes, img_size={img_size}')
    print(f'LOCAL GATE : raw={float(model.local_head.gate):+.6f} tanh={gate:+.6f} '
          f'|weight|={abs(gate):.6f}')
    if abs(gate) < 1e-3:
        print('  >>> the local branch is multiplied by ~0: every local-attn/local-se '
              'parameter is receiving ~zero gradient.  Attention inside this branch '
              'cannot move the score until the gate opens.')
    print(f'pooling    : {"SimPool Q/K attention" if m.score is not None else "plain mean"}'
          f'{", + SE recalibration" if m.se_expand is not None else ""}')

    captured = {}

    def grab(_module, inputs, _out):
        captured['toks'] = inputs[0].detach()

    handle = model.local_head.register_forward_hook(grab)

    rows = []
    for x, y, idx in loader:
        captured.clear()
        logits = model(x.to(device, non_blocking=True))
        toks = captured['toks'].float()                      # [B, N, width]
        prob = logits.float().softmax(-1)
        conf, pred = prob.max(-1)

        normed = F.normalize(toks, dim=-1)
        mean_tok = F.normalize(normed.mean(1), dim=-1)
        cos_p = torch.einsum('bnd,bd->bn', normed, mean_tok)  # [B, N]
        dev = 1.0 - cos_p                                     # [B, N], 0 = typical patch
        disp_mean = dev.mean(1)
        disp_p90 = dev.quantile(0.90, dim=1)

        ent = None
        if m.score is not None:
            s = m.score(toks).transpose(1, 2)                 # [B, H, N]
            q = m.query_proj(toks.mean(1))
            k = m.key_proj(toks)
            sim = torch.einsum('bd,bnd->bn', q, k) / math.sqrt(m.attn_dim)
            s = s + torch.tanh(m.sim_scale)[0] * sim[:, None, :]
            s = s * m.log_temp.exp()[None, :, None]
            w = s.softmax(-1)
            e = -(w * w.clamp_min(1e-12).log()).sum(-1) / math.log(w.shape[-1])
            ent = e.mean(1)

        for i in range(toks.shape[0]):
            rows.append(dict(
                idx=int(idx[i]), label=int(y[i]), pred=int(pred[i].item()),
                conf=float(conf[i].item()), correct=int(pred[i].item() == int(y[i])),
                disp_mean=float(disp_mean[i].item()), disp_p90=float(disp_p90[i].item()),
                attn_entropy=('' if ent is None else float(ent[i].item())),
            ))
    handle.remove()

    if not rows:
        raise SystemExit('no images were audited')

    # ---- class-level context -------------------------------------------------
    train_counts = Counter(tr.targets)
    s = summarize(rows, train_counts)
    print_summary(s, limited=bool(a.limit))

    out = Path(a.out or f'attn_audit_{Path(a.checkpoint).stem}.csv')
    with out.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    side = out.with_suffix('.classes.json')
    write_class_context(side, train_counts, s['weak'])
    print()
    print(f'per-image rows -> {out}')
    print(f'class context  -> {side}')


def write_class_context(path, train_counts, weak):
    """Sidecar JSON so a second arm can be judged against the *same* weak-class set.

    HANDOFF 45.5 defines "weak" as >=100 training images AND recall < 0.6, so the
    training counts have to travel with the audit; recomputing them from the arm
    being compared would move the target between the two runs.
    """
    import json
    with Path(path).open('w', encoding='utf-8') as fh:
        json.dump({'train_counts': {str(k): v for k, v in sorted(train_counts.items())},
                   'weak': sorted(weak)}, fh, ensure_ascii=False, indent=1)


def summarize(rows, train_counts, weak_min_train=100, weak_recall=0.6,
              hcw_conf=0.7):
    """Every number the report prints, as plain data.

    Kept free of torch and of printing so it can be exercised on synthetic rows --
    an aggregation bug here does not crash, it just reports the wrong conclusion,
    which is the expensive kind.
    """
    if not rows:
        raise SystemExit('no images were audited')

    per_cls_total, per_cls_hit = defaultdict(int), defaultdict(int)
    for r in rows:
        per_cls_total[r['label']] += 1
        per_cls_hit[r['label']] += r['correct']
    recall = {c: per_cls_hit[c] / per_cls_total[c] for c in per_cls_total}
    # HANDOFF 45.5's definition: enough data, yet recall below 0.6
    weak = {c for c in per_cls_total
            if train_counts.get(c, 0) >= weak_min_train and recall[c] < weak_recall}

    n = len(rows)
    wrong = [r for r in rows if not r['correct']]
    right = [r for r in rows if r['correct']]
    hcw = [r for r in rows if r['conf'] >= hcw_conf and not r['correct']]
    weak_rows = [r for r in rows if r['label'] in weak]
    weak_errors = sum(1 for r in weak_rows if not r['correct'])

    def median(key, subset):
        vals = sorted(r[key] for r in subset)
        return vals[len(vals) // 2] if vals else float('nan')

    ordered = sorted(rows, key=lambda r: r['disp_mean'])
    k = max(1, len(ordered) // 4)
    quartiles = []
    for q in range(4):
        part = ordered[q * k:(q + 1) * k] if q < 3 else ordered[3 * k:]
        if not part:
            continue
        quartiles.append(dict(
            q=q + 1, n=len(part),
            lo=part[0]['disp_mean'], hi=part[-1]['disp_mean'],
            err=1.0 - sum(r['correct'] for r in part) / len(part),
            conf_wrong=sum(1 for r in part
                           if r['conf'] >= hcw_conf and not r['correct']) / len(part),
        ))

    entropy = [r['attn_entropy'] for r in rows if r['attn_entropy'] != '']
    return dict(
        n=n, acc=sum(r['correct'] for r in rows) / n,
        n_wrong=len(wrong), n_hcw=len(hcw),
        weak=sorted(weak), n_weak_rows=len(weak_rows), n_weak_errors=weak_errors,
        weak_recall=(sum(r['correct'] for r in weak_rows) / len(weak_rows)
                     if weak_rows else float('nan')),
        # share of ALL errors, not of all images -- the denominators differ and
        # getting this wrong flatters the weak classes by a factor of ~3
        weak_error_share=(weak_errors / len(wrong)) if wrong else float('nan'),
        disp=dict(
            correct=median('disp_mean', right), wrong=median('disp_mean', wrong),
            hcw=median('disp_mean', hcw), weak=median('disp_mean', weak_rows)),
        quartiles=quartiles,
        entropy=dict(mean=sum(entropy) / len(entropy) if entropy else float('nan'),
                     median=median('attn_entropy', rows) if entropy else float('nan')),
    )


def print_summary(s, limited=False):
    print()
    print(f'accuracy   : {s["acc"]:.4f}  ({s["n"] - s["n_wrong"]}/{s["n"]})')
    print(f'conf-wrong : {s["n_hcw"]} ({100.0 * s["n_hcw"] / s["n"]:.1f}%) with conf>=0.7')
    print(f'weak class : {len(s["weak"])} classes (train>=100, recall<0.6), '
          f'{s["n_weak_rows"]} images, '
          f'{100.0 * s["weak_error_share"]:.1f}% of all errors')
    print(f'            recall on those: {s["weak_recall"]:.4f}')

    print()
    print("token dispersion (mean 1-cos to the image's own mean token)")
    for name in ('correct', 'wrong', 'hcw', 'weak'):
        label = {'hcw': 'conf-wrong', 'weak': 'weak class'}.get(name, name)
        print(f'  {label:<11} median={s["disp"][name]:.4f}')

    print()
    print('error rate by dispersion quartile (q1 = most uniform patches)')
    for q in s['quartiles']:
        print(f'  q{q["q"]}  n={q["n"]:5d}  dispersion {q["lo"]:.4f}..{q["hi"]:.4f}  '
              f'err={q["err"]:.4f}  conf-wrong={q["conf_wrong"]:.4f}')

    if s['entropy']['mean'] == s['entropy']['mean']:       # not NaN -> available
        print()
        print(f'attention entropy (normalised by log N): mean={s["entropy"]["mean"]:.4f} '
              f'median={s["entropy"]["median"]:.4f}   [1.0 = uniform = no information]')

    if limited:
        print()
        print('NOTE: --limit was used, so per-class recall and the weak-class set are '
              'computed from a truncated hold-out.  Re-run without --limit before '
              'quoting any of those numbers.')



def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True, help='the training folder (class per sub-dir)')
    p.add_argument('--out', default=None, help='per-image CSV path')
    p.add_argument('--device', default='auto')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--limit', type=int, default=0, help='audit only the first N hold-out images')
    return p.parse_args(argv)


if __name__ == '__main__':
    audit(parse_args())
