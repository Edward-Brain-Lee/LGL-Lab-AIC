"""Semifinal baseline runner: the 复赛 74.249 recipe, re-trained from the official
OpenAI ViT-B/32 weights on the 500-class semifinal set.

    python sf_run.py plan                 # print every path/count/command, run nothing
    python sf_run.py prep                 # locate (and unzip) + verify the dataset
    python sf_run.py smoke                # gates; the last one needs a free GPU
    python sf_run.py train                # ~2 h; refuses to start without a fresh smoke
    python sf_run.py infer                # 8-view + 16-view CSVs, each packed and verified
    python sf_run.py all                  # prep -> smoke -> train -> infer

Why a Python runner instead of another bash chain
-------------------------------------------------
`run_lr64qkv.sh` and `smoke_lr64qkv.sh` are bash + heredoc; none of it can run on
the Windows box the code is edited on, so a typo costs a cloud round-trip.  Here
the assertions live in `sf_gate.py` (stdlib, fixture-tested on this laptop) and
the runner is one process that `--dry-run` prints in full.

Compliance (README 十一), unchanged by the stage switch
-------------------------------------------------------
* backbone: official OpenAI CLIP ViT-B/32 only;
* data: the semifinal zips only -- no 初赛/复赛 images, no 复赛 checkpoint, no
  extra dataset, no test labels anywhere in training;
* one model / one inference path: a single `best.pt` plus deterministic
  multi-view TTA over that same checkpoint (organizer-approved); view sets are
  never averaged into a third answer;
* automatic noise handling (`noise.py`), so 五.6 holds;
* seed, args and RNG state are recorded in the checkpoint (五.5).
"""
import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import sf_gate

REPO = Path(__file__).resolve().parent
RECIPE = sf_gate.RECIPE
TAG = 'sf_base'
OUT = f'outputs_{TAG}'


def log(msg):
    print(f'[sf] {msg}', flush=True)


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def has_classes(p):
    try:
        dirs = [d for d in Path(p).iterdir() if d.is_dir()]
    except OSError:
        return False
    return len(dirs) == sf_gate.SF_CLASSES and all(
        len(d.name) == 4 and d.name.isdigit() for d in dirs)


def has_flat_images(p):
    try:
        return len(sf_gate._images(p, recursive=True)) >= sf_gate.SF_TEST
    except OSError:
        return False


def _candidates(root, key):
    """Directories or zips that could hold `train` or `test`, root first."""
    root = Path(root)
    out = [root / key, root / f'{key}.zip']
    out += sorted(root.glob(f'*/{key}')) + sorted(root.glob(f'*/{key}.zip'))
    out += sorted(root.glob(f'*{key}*.zip')) + sorted(root.glob(f'*/*{key}*.zip'))
    return out


def resolve(root, key, explicit=None):
    if explicit:
        return Path(explicit)
    for cand in _candidates(root, key):
        if not cand.exists():
            continue
        if cand.is_dir():
            if key == 'train' and not has_classes(cand):
                continue
            if key == 'test' and not has_flat_images(cand):
                continue
        return cand
    return None


def extract(zip_path, dest, dry=False):
    zip_path, dest = Path(zip_path), Path(dest)
    if dry:
        log(f'DRY would extract {zip_path} -> {dest}')
        return
    dest.mkdir(parents=True, exist_ok=True)
    df = subprocess.run(['df', '-h', str(dest)], capture_output=True, text=True)
    if df.returncode == 0:
        log('disk: ' + df.stdout.strip().splitlines()[-1])
    unzip = shutil.which('unzip')
    log(f'extracting {zip_path.name} -> {dest}/ using '
        f'{"unzip" if unzip else "python zipfile (slower)"}')
    if unzip:
        rc = subprocess.run([unzip, '-q', '-o', str(zip_path), '-d', str(dest)]).returncode
    else:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        rc = 0
    if rc:
        sys.exit(f'extract failed: {zip_path}')


def stage_prep(a):
    root = Path(a.root)
    src = {k: resolve(root, k, getattr(a, k)) for k in ('train', 'test')}
    for k, p in src.items():
        if p is None:
            sys.exit(f'could not find {k} (dir or .zip) under {root}; '
                     f'pass --{k} explicitly')
        log(f'{k:5s} source : {p}')

    dirs = {}
    for k in ('train', 'test'):
        p = src[k]
        if p.suffix.lower() == '.zip':
            target = root / k
            if not target.is_dir():
                extract(p, target, a.dry_run)      # into root/train, root/test
            dirs[k] = target
        else:
            dirs[k] = p
    if a.dry_run:
        return dirs

    # an archive may unzip into a single nested folder; search one level down
    for k, want in (('train', has_classes), ('test', has_flat_images)):
        if not want(dirs[k]):
            nested = [c for c in dirs[k].glob('*') if c.is_dir() and want(c)]
            if len(nested) == 1:
                log(f'{k}: using nested folder {nested[0]}')
                dirs[k] = nested[0]
            else:
                sys.exit(f'{dirs[k]} does not hold {k} data; pass --{k}')
    g = sf_gate.gate_data(dirs['train'], dirs['test'])
    sf_gate.gate_params(g['classes'], **plan_param_kwargs())
    log('PREP OK — layout, counts and the expected budget all match')
    return dirs


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
# Extra train.py switches for the arm being run (`--local-se`, `--local-attn`,
# `--attn-temp`, `--junk-*`, ...).  Set once from --extra in main(); train_cmd
# appends it to *every* invocation, so the 20-batch smoke inside the gate trains
# the same arm the full run does instead of silently testing the baseline.
TRAIN_EXTRA = []


def arm_recipe():
    """RECIPE plus whichever attention switches this run's --extra requests.

    Without this the log gate would still validate the 10.198M baseline budget and
    reject a correct --local-se run *after* it has spent two hours training.
    """
    r = dict(RECIPE)
    for key in sf_gate.ARM_KEYS:      # every arm key must exist, defaulting off
        r.setdefault(key, False)
    for tok in TRAIN_EXTRA:
        key = tok.lstrip('-').replace('-', '_')
        if key in sf_gate.ARM_KEYS:
            r[key] = True
    return r


def plan_param_kwargs():
    """The RECIPE keys gate_params accepts, with this run's arm switches applied.

    Without it `plan --extra '--local-se'` would preview the 10.198M baseline
    budget next to a command that trains 10.272M — an off-by-one-recipe in the
    one place an operator looks to confirm the arm is wired.
    """
    r = arm_recipe()
    keys = ('rank', 'qkv', 'img_size', 'local_head', 'train_pos_embed',
            'local_attn', 'local_se', 'attn_temp')
    return {k: r[k] for k in keys}


def train_cmd(data, out, epochs=None, limit_batches=0):
    cmd = [sys.executable, '-u', 'train.py', '--data', str(data), '--out', str(out),
           '--epochs', str(1 if limit_batches else (epochs or RECIPE['epochs'])),
           '--warmup-epochs', str(RECIPE['warmup_epochs']),
           '--batch-size', str(RECIPE['batch_size']),
           '--workers', '4' if limit_batches else '8',
           '--img-size', str(RECIPE['img_size']),
           '--train-pos-embed', '--local-head',
           '--save-every', '1' if limit_batches else str(RECIPE['save_every']),
           '--lora-rank', str(RECIPE['rank']), '--lora-qkv']
    cmd += list(TRAIN_EXTRA)
    if limit_batches:
        cmd += ['--limit-batches', str(limit_batches)]
    return cmd


def infer_cmd(test, ckpt, csv_out, views):
    cmd = [sys.executable, '-u', 'infer.py', '--test', str(test),
           '--checkpoint', str(ckpt), '--output', str(csv_out),
           '--workers', '4', '--batch-size', '64', '--tta']
    return cmd + list(views)


def run(cmd, log_path=None, dry=False):
    cmd = [str(c) for c in cmd]
    log(' '.join(cmd))
    if dry:
        return 0
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8', errors='replace') as fh:
            p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding='utf-8', errors='replace')
            for line in p.stdout:
                sys.stdout.write(line)
                fh.write(line)
            p.wait()
            return p.returncode
    return subprocess.run(cmd, cwd=REPO).returncode


def pack(csv_path, zip_path, dry=False):
    if dry:
        log(f'DRY would pack {csv_path} -> {zip_path}')
        return
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname='pred_results.csv')
    log(f'packed {Path(zip_path).name} ({Path(zip_path).stat().st_size} B) '
        f'holding pred_results.csv ({Path(csv_path).stat().st_size} B)')


def _assert_gpu_free():
    try:
        p = subprocess.run(['pgrep', '-af', r'[t]rain\.py'], capture_output=True, text=True)
    except FileNotFoundError:
        return
    if p.stdout.strip():
        sys.exit(f'a train.py is already running — the GPU is not free:\n{p.stdout}')


def _need_dir(p, what, dry=False):
    """A stage that trains or infers needs an extracted directory, not a zip."""
    if dry:
        return
    p = Path(p)
    if p.is_file():
        sys.exit(f'{what} is still an archive ({p}); run `python sf_run.py prep` '
                 f'first so it can be unpacked and verified')
    if not p.is_dir():
        sys.exit(f'{what} directory not found: {p}; pass --{what} explicitly')


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def env_report(warn_no_gpu=True):
    """Say which interpreter/library versions the run will actually use.

    Worth one line: the AutoDL conda interpreter is not on the non-interactive
    PATH, and `open_clip` is not in the image (it has to be pip-installed), so
    the first thing that can go wrong is invisible from the command line.
    """
    import torch
    import open_clip
    log(f'python {sys.version.split()[0]} | torch {torch.__version__} | '
        f'open_clip {open_clip.__version__} | cuda {torch.cuda.is_available()}')
    if warn_no_gpu and not torch.cuda.is_available():
        log('WARNING: no CUDA device — training/inference will be unusably slow')


# The transitive import closure of train.py/infer.py/selftest.py.  A missing one
# of these is a one-second check here and a 10-minute failed run at 0.5 CPU (this
# list exists because `build_semantic_reference.py` was missing on the box and the
# selftest only said so after importing torch).
NEEDED_MODULES = ['train', 'infer', 'noise', 'losses', 'semantic_regularization',
                  'analyze', 'class_search', 'checkpoint_search',
                  'build_semantic_reference', 'selftest']


def check_modules():
    missing = [m for m in NEEDED_MODULES if not (REPO / f'{m}.py').is_file()]
    if missing:
        sys.exit(f'missing local module(s) {missing} in {REPO} — upload them before '
                 f'running anything: the import chain of train.py/selftest.py needs '
                 f'all of {NEEDED_MODULES}')
    log(f'module closure present ({len(NEEDED_MODULES)} files)')


def stage_smoke(a, data, test):
    _need_dir(data, 'train', a.dry_run)
    _need_dir(test, 'test', a.dry_run)
    check_modules()
    if not a.dry_run:
        env_report(warn_no_gpu=False)
    log('--- gate 0: selftest (stub tower, CPU, no data) ---')
    if run([sys.executable, 'selftest.py'], REPO / f'smoke_{TAG}/selftest.log', a.dry_run):
        sys.exit('selftest failed')

    log('--- gate 1: dataset + expected parameter budget ---')
    if a.dry_run:
        sf_gate.gate_params(sf_gate.SF_CLASSES, **plan_param_kwargs())
    else:
        sf_gate.gate_data(data, test)
        sf_gate.gate_params(sf_gate.SF_CLASSES, **plan_param_kwargs())

    log('--- gate 2: the real ViT-B/32 tower ---')
    rc = run([sys.executable, '-u', 'sf_tower_gate.py',
              '--rank', RECIPE['rank'], '--classes', sf_gate.SF_CLASSES,
              '--img-size', RECIPE['img_size']],
             REPO / f'smoke_{TAG}/tower.log', a.dry_run)
    if rc:
        sys.exit('tower gate failed')

    log('--- gate 3: 20 real batches through the actual pipeline ---')
    out = f'smoke_{TAG}/run'
    if not a.dry_run and Path(out).exists():
        shutil.rmtree(out)
    if run(train_cmd(data, out, limit_batches=20),
           REPO / f'smoke_{TAG}/train.log', a.dry_run):
        sys.exit('20-batch smoke failed')

    if not a.dry_run:
        sf_gate.gate_log(REPO / f'smoke_{TAG}/train.log', recipe=arm_recipe())
        sf_gate.gate_ckpt(Path(out) / 'best.pt')
        (REPO / f'smoke_{TAG}').mkdir(parents=True, exist_ok=True)
        Path(REPO / f'smoke_{TAG}' / 'OK').write_text(
            f'{sha16(REPO / "train.py")} {sha16(REPO / "infer.py")} '
            f'{TAG} rank={RECIPE["rank"]} {" ".join(TRAIN_EXTRA)}\n')
    log('SMOKE PASSED')


def stage_train(a, data):
    _need_dir(data, 'train', a.dry_run)
    marker = REPO / f'smoke_{TAG}' / 'OK'
    if not a.dry_run:
        if not marker.is_file():
            sys.exit(f'no smoke gate at {marker}; run `python sf_run.py smoke` first')
        fields = marker.read_text().split()
        got, got_extra = fields[:2], fields[4:]
        got_rank = fields[3] if len(fields) > 3 else 'rank=?'
        if got_rank != f'rank={RECIPE["rank"]}':
            sys.exit(f'smoke gate was passed at {got_rank} but this run is '
                     f'rank={RECIPE["rank"]} — the gates validated a different budget. '
                     f'Re-run `python sf_run.py smoke --tag {TAG} --rank {RECIPE["rank"]}`.')
        if got != [sha16(REPO / 'train.py'), sha16(REPO / 'infer.py')]:
            sys.exit('smoke gate is STALE — train.py/infer.py changed since it passed. '
                     'Re-run `python sf_run.py smoke`.')
        if got_extra != TRAIN_EXTRA:
            sys.exit(f'smoke gate was passed for arm {got_extra} but this run is '
                     f'{TRAIN_EXTRA} — the gates validated a different recipe. '
                     'Re-run `python sf_run.py smoke --extra "..."`.')
        _assert_gpu_free()
    if run(train_cmd(data, OUT), REPO / f'{OUT}.log', a.dry_run):
        sys.exit('training failed')
    if a.dry_run:
        return
    sf_gate.gate_log(REPO / f'{OUT}.log', recipe=arm_recipe())
    sf_gate.gate_ckpt(Path(OUT) / 'best.pt')


def stage_infer(a, test):
    _need_dir(test, 'test', a.dry_run)
    ckpt = Path(OUT) / 'best.pt'
    if not a.dry_run and not ckpt.is_file():
        sys.exit(f'missing {ckpt}')
    # `scales` first, deliberately.  The 复赛 chain sealed the 8-view file before
    # starting the 16-view pass, on the grounds that "the safe file is already
    # packed if the box dies".  That ordering costs the thing we actually want:
    # 16 views is the 74.249 submission, and 8 views is only a fallback/A-B on the
    # same checkpoint — so the file worth having should exist as early as possible,
    # and the fallback can finish later (or be skipped with --no-emit-8v).
    jobs = [('scales', ['scales'])]
    if a.emit_8v:
        jobs.append(('8v', []))
    for tag, views in jobs:
        d = Path(f'sub_{TAG}_{tag}')
        d.mkdir(parents=True, exist_ok=True)
        csv_path = d / 'pred_results.csv'
        if run(infer_cmd(test, ckpt, csv_path, views),
               REPO / f'infer_{TAG}_{tag}.log', a.dry_run):
            sys.exit(f'inference [{tag}] failed')
        if not a.dry_run:
            sf_gate.gate_csv(csv_path, test)
        pack(csv_path, Path(f'sub_{TAG}_{tag}.zip'), a.dry_run)
    log(f'next: submit sub_{TAG}_scales.zip (16 views = the 74.249 view set); '
        f'the 8v file is the same checkpoint under fewer views, never average them')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('stage', choices=['plan', 'prep', 'smoke', 'train', 'infer', 'all'])
    p.add_argument('--root', default=os.environ.get('ROOT', '/root/autodl-tmp'))
    p.add_argument('--train', default=os.environ.get('DATA', ''))
    p.add_argument('--test', default=os.environ.get('TEST', ''))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-emit-8v', dest='emit_8v', action='store_false')
    p.add_argument('--rank', type=int, default=None,
                   help='override the LoRA rank from RECIPE (e.g. 32 for the '
                        'reduced-capacity arm).  This moves RECIPE itself so the '
                        'parameter gates follow it; --extra cannot do that.')
    p.add_argument('--tag', default='sf_base',
                   help='arm name; drives outputs_<tag>, sub_<tag>_* and '
                        'smoke_<tag> so an arm can never overwrite the baseline')
    p.add_argument('--extra', default='',
                   help='extra train.py switches for this arm, e.g. '
                        '--extra="--local-se --junk-filter margin".  Any unrecognised '
                        'flag is treated the same way, so `sf_run.py train --local-se` '
                        'works too (argparse cannot take a --prefixed --extra value).')
    a, extras = p.parse_known_args(argv)
    global TRAIN_EXTRA, TAG, OUT
    if a.rank:
        RECIPE['rank'] = int(a.rank)          # same dict object the gates read
        log(f'arm: lora rank overridden to {RECIPE["rank"]}')
    # Unknown flags are the honest way to pass a --prefixed switch: `--extra --local-se`
    # raises "expected one argument" because argparse reads the value as an option.
    TRAIN_EXTRA = shlex.split(a.extra) + list(extras)
    TAG = a.tag
    OUT = f'outputs_{TAG}'
    if TRAIN_EXTRA:
        log(f'arm: tag={TAG} train.py ' + ' '.join(TRAIN_EXTRA))
    a.emit_8v = getattr(a, 'emit_8v', True)

    root = Path(a.root)

    if a.stage == 'plan':
        # Deliberately no `resolve()`: it stats five digits of files, which is ~1 s
        # on a normal box and minutes on a 0.5-core container — a dry run that takes
        # minutes is a dry run nobody uses.
        check_modules()
        sf_gate._selfcheck_params()
        log(f'repo {REPO}')
        log(f'root {root}')
        for k, explicit in (('train', a.train), ('test', a.test)):
            p = Path(explicit) if explicit else root / k
            log(f'{k:5s} source: {p} ({"exists" if p.exists() else "MISSING"})')
        sf_gate.gate_params(sf_gate.SF_CLASSES, **plan_param_kwargs())
        log(f'csv: {sf_gate.SF_TEST} rows / {sf_gate.SF_TEST * 43} bytes '
            f'(43 B/row = 36-char name + comma + 4 digits + CRLF)')
        log('--- train ---')
        run(train_cmd(root / 'train', OUT), dry=True)
        log('--- infer ---')
        for tag, views in (('8v', []), ('scales', ['scales'])):
            run(infer_cmd(root / 'test', Path(OUT) / 'best.pt',
                          Path(f'sub_{TAG}_{tag}') / 'pred_results.csv', views), dry=True)
        return 0

    src_train = resolve(root, 'train', a.train)
    src_test = resolve(root, 'test', a.test)
    data = src_train or root / 'train'
    test = src_test or root / 'test'

    if a.stage in ('prep', 'all'):
        dirs = stage_prep(a)
        if dirs:
            data, test = dirs['train'], dirs['test']
    if a.stage in ('smoke', 'all'):
        stage_smoke(a, data, test)
    if a.stage in ('train', 'all'):
        stage_train(a, data)
    if a.stage in ('infer', 'all'):
        stage_infer(a, test)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
