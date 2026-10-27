"""Semifinal-stage gates: dataset, parameter budget, checkpoint, submission CSV, log.

Why this file exists instead of more bash heredocs
--------------------------------------------------
The 复赛 chain (`run_lr64qkv.sh` / `smoke_lr64qkv.sh`) put its assertions inline in
bash+heredoc.  That worked once, but every constant in it is 复赛-specific
(N_ROWS=37444, N_BYTES=1610092, `trainable params=10.325|10.326M`) and none of it
can be executed on the Windows box the code is edited on, so a typo is only found
three hours into a cloud run.

This module is stdlib-only for `data` / `csv` / `log` / `params` (no torch, no
PIL), importable by the runner, and runnable on this laptop against the same
fixtures the cloud will use.  `ckpt` needs torch and is imported lazily.

The 500-class numbers are *derived*, not copied
-----------------------------------------------
`expected_trainable` reconstructs the parameter budget from module shapes and
self-checks against the one observation the project actually has (the 750-class
run printed 10.326M, HANDOFF 43.2).  Change the class count and the expected
total moves with it, so the gate cannot silently rot when the stage changes.
"""
import argparse
import csv
import io
import os
import re
import sys
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Semifinal stage, measured from the delivered zips (2026-10-25):
#   500 class folders `0000`..`0499`, 90,197 images, 49,857 test images,
#   every image 500x500 RGB, no EXIF, no exact duplicates.
SF_CLASSES = 500
SF_TRAIN = 90197
SF_TEST = 49857

# Test file names are 32 lowercase hex chars + '.jpg' -> 36 chars, which is what
# makes one CSV row exactly 43 bytes (36 + ',' + 4-digit label + CRLF).
SF_NAME = re.compile(r'^[0-9a-f]{32}\.jpg$')

# The recipe being reproduced: 复赛 74.249 (HANDOFF 43.2).
# `save_every` is the one deliberate change: 2 instead of 4 writes two extra
# snapshots per 10 epochs and does not touch the optimisation at all (same steps,
# same order, same LR), but it makes the *free* epoch-saturation curve measurable
# later with `checkpoint_search.py` — no GPU, no submission slot.
RECIPE = dict(rank=64, qkv=True, img_size=384, local_head=True,
              train_pos_embed=True, epochs=20, warmup_epochs=3, batch_size=128,
              save_every=2)


def fail(msg):
    print('FATAL: ' + msg, file=sys.stderr)
    raise SystemExit(1)


def ok(msg):
    print('  ok: ' + msg)


def _images(root, recursive):
    root = Path(root)
    it = root.rglob('*') if recursive else root.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


def class_dirs(train):
    """Numeric class folders, in the order `train.py` will sort them into ids."""
    train = Path(train)
    if not train.is_dir():
        fail(f'training root is not a directory: {train}')
    dirs = sorted(d.name for d in train.iterdir() if d.is_dir())
    return dirs


def gate_data(train, test, classes=SF_CLASSES, n_train=SF_TRAIN, n_test=SF_TEST,
              verbose=True):
    """Layout + counts + the invariant that makes class ids identity-mapped."""
    dirs = class_dirs(train)
    bad = [d for d in dirs if not (len(d) == 4 and d.isdigit())]
    if bad:
        fail(f'training folders must be 4-digit numeric, got {bad[:5]}')
    if len(dirs) != classes:
        fail(f'{len(dirs)} class folders, expected {classes} '
             f'(wrong stage, or a class folder failed to extract)')
    want = [f'{i:04d}' for i in range(classes)]
    if dirs != want:
        missing = sorted(set(want) - set(dirs))[:5]
        fail(f'class folders are not a contiguous 0000..{classes - 1:04d} set; '
             f'missing {missing} -- sorted() would shift every class id')

    counts = {}
    for d in dirs:
        counts[d] = len(_images(Path(train) / d, recursive=False))
    empty = [d for d, c in counts.items() if c == 0]
    if empty:
        fail(f'empty class folders: {empty[:5]}')
    total = sum(counts.values())
    if total != n_train:
        fail(f'{total} training images, expected {n_train}')
    if verbose:
        c = sorted(counts.values())
        ok(f'train: {len(dirs)} classes, {total} images, '
           f'per-class min {c[0]} / median {c[len(c) // 2]} / max {c[-1]}')

    files = _images(test, recursive=True)
    if len(files) != n_test:
        fail(f'{len(files)} test images, expected {n_test}.  Before submitting, '
             f'settle 49,857 vs the 24,912 printed in README.md — the row count is '
             f'derived from this directory, so a wrong expectation here invalidates '
             f'the submission rather than the run.')
    names = [p.name for p in files]
    dup = len(names) - len(set(names))
    if dup:
        fail(f'{dup} duplicate test file names; the grader matches by name')
    odd = [n for n in names if not SF_NAME.match(n)]
    if odd:
        fail(f'{len(odd)} test names are not 32-hex + .jpg (e.g. {odd[:3]}); '
             f'the 43-byte row assumption and the CLI output would both change')
    rows, nbytes = expected_csv(names)
    if verbose:
        ok(f'test : {len(files)} images, unique, all 32-hex .jpg')
        ok(f'csv  : {rows} rows / {nbytes} bytes (43 B/row, CRLF, no header)')
    return dict(classes=len(dirs), train=total, test=len(files),
                rows=rows, bytes=nbytes, counts=counts, dirs=dirs)


def expected_csv(names):
    """Rows and bytes that `infer.py` must produce for exactly these names."""
    return len(names), sum(len(n) for n in names) + len(names) * (1 + 4 + 2)


def expected_trainable(nclass, width=768, dim=512, rank=64, qkv=True,
                       local_head=True, pos_tokens=145):
    """Trainable-parameter budget of the 74.249 recipe, from module shapes.

    Adapters (LoRA on every `nn.Linear` of all 12 blocks):
        qkv      : in 768  -> out 2304 : rank*(768+2304)
        out_proj : 768 -> 768          : rank*(768+768)
        c_fc     : 768 -> 3072         : rank*(768+3072)
        c_proj   : 3072 -> 768         : rank*(3072+768)
    x12 blocks.  Head is a cosine classifier (nclass x 512 + 1 temperature);
    `--local-head` adds a 768x512 projection and a scalar gate;
    `--train-pos-embed` adds the position grid (145 tokens at 384px) as a Parameter.
    """
    per_block = (rank * (width + 3 * width) if qkv else 0) + rank * (width + width) \
        + rank * (width + 4 * width) + rank * (4 * width + width)
    adapters = 12 * per_block
    head = nclass * dim + 1
    local = (width * dim + 1) if local_head else 0
    pos = (pos_tokens * width) if pos_tokens else 0
    return dict(adapters=adapters, head=head, local=local, pos=pos,
                total=adapters + head + local + pos)


def gate_params(nclass, rank=64, qkv=True, img_size=384, local_head=True,
                train_pos_embed=True, verbose=True):
    tokens = (img_size // 32) ** 2 + 1
    e = expected_trainable(nclass, rank=rank, qkv=qkv, local_head=local_head,
                           pos_tokens=tokens if train_pos_embed else 0)
    printed = f'{e["total"] / 1e6:.3f}M'
    if verbose:
        ok(f'expected trainable params = {e["total"]:,} ({printed})')
        ok(f'  adapters {e["adapters"]:,} + head {e["head"]:,} + '
           f'local {e["local"]:,} + pos {e["pos"]:,} ({tokens} tokens)')
    return e, printed


def _selfcheck_params():
    """The formula must reproduce the one number the project has measured."""
    e750 = expected_trainable(750)
    # 2026-10-06 run printed 10.326M; the bracket from five independent prints was
    # [10,325,740, 10,325,796).
    assert 10_325_740 <= e750['total'] < 10_325_796, e750['total']
    e500 = expected_trainable(500)
    assert e500['total'] == 10_197_762, e500['total']
    assert f"{e500['total'] / 1e6:.3f}" == '10.198'
    return True


def gate_log(log_path, classes=SF_CLASSES, recipe=RECIPE, verbose=True):
    """Everything the run itself must print, checked against the same recipe."""
    text = Path(log_path).read_text(encoding='utf-8', errors='replace')
    if 'Traceback' in text:
        fail(f'traceback in {log_path}')
    nclass = recipe['rank'], recipe['qkv']
    e = expected_trainable(classes, rank=recipe['rank'], qkv=recipe['qkv'],
                           local_head=recipe['local_head'],
                           pos_tokens=(recipe['img_size'] // 32) ** 2 + 1)
    printed = f"{e['total'] / 1e6:.3f}M"
    # accept the neighbouring rounding too: f'{n/1e6:.3f}' flips at n/1e6 = .xxx5,
    # and the print is the only place the total appears.
    lo = f"{e['total'] / 1e6 - 0.001:.3f}M"
    def kv(key, value):
        """`key=value` as a whole token, so `lora_rank=64` never matches 640."""
        return re.search(rf'\b{key}={value}(?=\s|$)', text)

    checks = [
        (f'trainable params={printed} (or {lo})',
         re.search(r'trainable params=(?:' + printed.replace('.', r'\.') + '|'
                   + lo.replace('.', r'\.') + ')', text)),
        ('layers=48', kv('layers', 48)),
        ('lora_qkv=True', kv('lora_qkv', 'True')),
        (f'lora_rank={recipe["rank"]}', kv('lora_rank', recipe['rank'])),
        (f'img_size={recipe["img_size"]}', kv('img_size', recipe['img_size'])),
        ('local_head=True', kv('local_head', 'True')),
        ('train_pos_embed=True', kv('train_pos_embed', 'True')),
        (f'classes={classes}', kv('classes', classes)),
        ('positional grid trainable', re.search(r'positional grid is now trainable', text)),
    ]
    bad = [name for name, m in checks if not m]
    if bad:
        fail(f'{log_path} is missing: {bad}')
    if verbose:
        ok(f'{log_path}: ' + ', '.join(n for n, _ in checks))
    return True


def gate_ckpt(path, classes=SF_CLASSES, recipe=RECIPE, verbose=True):
    """Checkpoint must carry exactly the trained recipe, on the right class set."""
    import torch                                    # lazy: data/csv gates stay CPU-only
    ck = torch.load(str(path), map_location='cpu', weights_only=False)
    cls = ck.get('classes') or ck.get('class_to_idx')
    if cls is None:
        fail(f'{path} carries no class mapping')
    if len(cls) != classes:
        fail(f'{path} carries {len(cls)} classes, expected {classes}')
    names = sorted(cls)
    if names != [f'{i:04d}' for i in range(classes)] or \
            [cls[n] for n in names] != list(range(classes)):
        fail(f'{path} class mapping is not identity 0000..{classes - 1:04d}')
    rank = ck.get('lora_rank', ck.get('args', {}).get('lora_rank'))
    if rank != recipe['rank']:
        fail(f'{path} lora_rank={rank}, expected {recipe["rank"]}')
    if bool(ck.get('lora_qkv')) != bool(recipe['qkv']):
        fail(f'{path} lora_qkv={ck.get("lora_qkv")}, expected {recipe["qkv"]}')
    args = ck.get('args', {})
    for key in ('img_size', 'local_head', 'train_pos_embed'):
        got = ck.get(key, args.get(key))
        if got != recipe[key]:
            fail(f'{path} {key}={got}, expected {recipe[key]}')
    n_qkv = sum(1 for k in ck['model'] if k.endswith(('attn.qkv.A', 'attn.qkv.B')))
    if n_qkv != 24:
        fail(f'{path}: {n_qkv} qkv adapter tensors, expected 24')
    pe = [v for k, v in ck['model'].items() if 'positional_embedding' in k]
    tokens = (recipe['img_size'] // 32) ** 2 + 1
    if not pe:
        fail(f'{path}: no trainable positional grid — --train-pos-embed did not take')
    if tuple(pe[0].shape) != (tokens, 768):
        fail(f'{path}: positional grid {tuple(pe[0].shape)}, expected {(tokens, 768)}')

    # Assert per part, not on the total.  `ProtoHead` registers the EMA prototypes
    # as *buffers*, so they ride along in `trainable_state_dict()` and would make a
    # whole-checkpoint sum disagree with the printed parameter count by
    # nclass x 512 for no real reason.
    model = ck['model']
    e = expected_trainable(classes, rank=recipe['rank'], qkv=recipe['qkv'],
                           local_head=recipe['local_head'], pos_tokens=tokens)
    adapters = sum(v.numel() for k, v in model.items() if k.endswith(('.A', '.B')))
    if adapters != e['adapters']:
        fail(f'{path}: {adapters:,} adapter weights, expected {e["adapters"]:,}')
    parts = {'head.weight': (classes, 512), 'head.logit_scale': (),
             # nn.Linear(width, dim) -> weight is (out_features, in_features)
             'local_head.proj.weight': (512, 768), 'local_head.gate': ()}
    for name, shape in parts.items():
        if name not in model:
            fail(f'{path}: missing {name} — the checkpoint was not trained with '
                 f'the 74.249 recipe')
        if tuple(model[name].shape) != shape:
            fail(f'{path}: {name} {tuple(model[name].shape)}, expected {shape}')
    n_trained = adapters + sum(model[n].numel() for n in parts) + pe[0].numel()
    if n_trained != e['total']:
        fail(f'{path}: {n_trained:,} trained weights, expected {e["total"]:,}')
    if verbose:
        proto = next((v.numel() for k, v in model.items() if k.endswith('proto.proto')), 0)
        ok(f'{Path(path).name}: epoch {ck.get("epoch")} val_acc {ck.get("val_acc", 0):.4f} '
           f'val_acc_hi {ck.get("val_acc_hi", 0):.4f}')
        ok(f'  trained {n_trained:,} = {e["total"] / 1e6:.3f}M '
           f'(adapters {adapters:,}, {n_qkv} qkv tensors, grid {tuple(pe[0].shape)})'
           + (f'; +{proto:,} prototype buffer' if proto else ''))
    return ck


def gate_csv(csv_path, test, classes=SF_CLASSES, verbose=True):
    """The submission file: every test image exactly once, 4-digit labels, 43 B/row."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        fail(f'missing {csv_path}')
    raw = csv_path.read_bytes()
    text = raw.decode('utf-8')
    if text.startswith('\ufeff'):
        fail(f'{csv_path} starts with a BOM; the grader reads the first name literally')
    lines = text.split('\r\n')
    if lines and lines[-1] == '':
        lines.pop()
    elif '\n' in text.replace('\r\n', ''):
        fail(f'{csv_path} mixes line endings; infer.py writes CRLF only')
    files = _images(test, recursive=True)
    names_on_disk = {p.name for p in files}
    seen, labels = set(), set()
    bad = []
    for i, line in enumerate(lines, 1):
        if ',' not in line:
            bad.append((i, line[:40]))
            continue
        name, label = line.rsplit(',', 1)
        if name not in names_on_disk:
            bad.append((i, f'unknown name {name[:40]}'))
        if name in seen:
            bad.append((i, f'duplicate name {name[:40]}'))
        seen.add(name)
        if not (len(label) == 4 and label.isdigit()):
            bad.append((i, f'label {label!r} is not 4 digits'))
        elif not 0 <= int(label) < classes:
            bad.append((i, f'label {label} outside 0..{classes - 1}'))
        labels.add(label)
    if bad:
        for i, why in bad[:8]:
            print(f'    line {i}: {why}', file=sys.stderr)
        fail(f'{len(bad)} malformed rows in {csv_path}')
    missing = names_on_disk - seen
    if missing:
        fail(f'{len(missing)} test images missing from {csv_path} '
             f'(e.g. {sorted(missing)[:3]}) — a missing row invalidates the submission')
    rows, nbytes = expected_csv([p.name for p in files])
    if len(lines) != rows:
        fail(f'{csv_path}: {len(lines)} rows, expected {rows}')
    if len(raw) != nbytes:
        fail(f'{csv_path}: {len(raw)} bytes, expected {nbytes} (43 B/row)')
    if verbose:
        ok(f'{Path(csv_path).name}: {rows} rows / {nbytes} bytes / '
           f'{len(labels)} distinct labels / every test name present once')
    return dict(rows=rows, bytes=nbytes, labels=len(labels))


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)

    d = sub.add_parser('data', help='dataset layout and counts')
    d.add_argument('--train', required=True)
    d.add_argument('--test', required=True)
    d.add_argument('--classes', type=int, default=SF_CLASSES)
    d.add_argument('--n-train', type=int, default=SF_TRAIN)
    d.add_argument('--n-test', type=int, default=SF_TEST)

    s = sub.add_parser('params', help='expected trainable-parameter budget')
    s.add_argument('--classes', type=int, default=SF_CLASSES)
    for k, v in RECIPE.items():
        s.add_argument('--' + k.replace('_', '-'), type=type(v), default=v)

    c = sub.add_parser('ckpt', help='checkpoint carries the right recipe')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--classes', type=int, default=SF_CLASSES)

    v = sub.add_parser('csv', help='submission CSV is complete and well formed')
    v.add_argument('--csv', required=True)
    v.add_argument('--test', required=True)
    v.add_argument('--classes', type=int, default=SF_CLASSES)

    g = sub.add_parser('log', help='training log printed the expected config')
    g.add_argument('--log', required=True)
    g.add_argument('--classes', type=int, default=SF_CLASSES)

    a = p.parse_args(argv)
    if a.cmd == 'data':
        gate_data(a.train, a.test, a.classes, a.n_train, a.n_test)
    elif a.cmd == 'params':
        gate_params(a.classes, a.rank, a.qkv, a.img_size, a.local_head,
                    a.train_pos_embed)
    elif a.cmd == 'ckpt':
        gate_ckpt(a.checkpoint, a.classes)
    elif a.cmd == 'csv':
        gate_csv(a.csv, a.test, a.classes)
    elif a.cmd == 'log':
        gate_log(a.log, a.classes)
    return 0


if __name__ == '__main__':
    _selfcheck_params()
    raise SystemExit(cli())
