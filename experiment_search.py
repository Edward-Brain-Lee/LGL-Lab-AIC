"""Plan or run sequential, reproducible ablations; no reconstructed labels.

With no --execute this prints commands and writes a plan. One arm changes one
training knob. Runs always start from official OpenAI weights, never another
arm's checkpoint. No test inference or submissions are made by this runner.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


ARMS = {
    'control': [],
    'fixed_anchor': ['--fixed-anchor'],
    'consistency_cosine': ['--consistency-mode', 'cosine'],
    'shuffle': ['--sampler', 'shuffle'],
    'uniform': ['--sampler', 'uniform'],
    'conf_gamma1': ['--conf-gamma', '1'],
    'no_decay_small': ['--optimizer-groups', 'no_decay_small'],
    'anchor02': ['--anchor-weight', '0.2'],
    'smooth010': ['--label-smooth', '0.10'],
    'decay010': ['--weight-decay', '0.10'],
    'lr010': ['--lr', '0.0001'],
    'lr030': ['--lr', '0.0003'],
    'alpha64': ['--lora-alpha', '64'],
    'augment12': ['--randaug-m', '12'],
    'crop040': ['--crop-min', '0.40'],
    'tail025': ['--class-weight-beta', '0.25', '--class-weight-min', '0.75', '--class-weight-max', '1.5'],
    'margin': ['--class-margin', '0.5', '--class-margin-weight', '0.05',
               '--class-margin-power', '0.25', '--class-margin-conf', '0.5'],
    'semantic005': ['--semantic-weight', '0.05', '--semantic-conf', '0.5'],
    'semantic001': ['--semantic-weight', '0.01', '--semantic-conf', '0.5'],
    'semantic003': ['--semantic-weight', '0.03', '--semantic-conf', '0.5'],
}


def plans(a):
    repo = Path(__file__).resolve().parent
    root = Path(a.out).resolve()
    result = []
    for arm in a.arms:
        target = root / arm
        train = [sys.executable, '-u', str(repo / 'train.py'), '--data', a.data,
                 '--out', str(target), '--pretrained', a.pretrained,
                 '--epochs', '20', '--warmup-epochs', '3', '--batch-size', str(a.batch_size),
                 '--workers', str(a.workers), '--img-size', '384', '--train-pos-embed',
                 '--local-head', '--lora-rank', '64', '--lora-qkv', '--seed', '3407',
                 '--split-seed', '3407',
                 '--save-every', '4', *ARMS[arm]]
        if arm.startswith('semantic'):
            if not a.semantic_reference:
                raise ValueError(f'{arm} requires --semantic-reference from official training split')
            train.extend(['--semantic-reference', a.semantic_reference])
        rank = [sys.executable, '-u', str(repo / 'checkpoint_search.py'), '--data', a.data,
                '--checkpoints', *[str(target / f'ep{ep}.pt') for ep in (4, 8, 12, 16, 20)],
                '--recipes', 'scales', '--taus', '0', '0.1', '0.25', '0.5',
                '--workers', str(a.workers), '--batch-size', '64',
                '--out', str(root / f'analysis_{arm}')]
        result.append(dict(arm=arm, train=train, rank=rank))
    return result


def run_logged(command, log, repo):
    with log.open('w', encoding='utf-8') as f:
        process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f'command failed ({code}), see {log}')


def main(a):
    repo, root = Path(__file__).resolve().parent, Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = plans(a)
    plan_path = root / 'plan.json'
    if a.execute and plan_path.exists():
        raise ValueError('output already contains a plan; choose a new output directory for execution')
    document = dict(training_source=a.data, pretrained=a.pretrained, jobs=jobs,
                    source_sha256={p: hashlib.sha256((repo / p).read_bytes()).hexdigest()
                                   for p in ('train.py', 'infer.py', 'losses.py', 'checkpoint_search.py',
                                             'experiment_search.py', 'semantic_regularization.py',
                                             'build_semantic_reference.py')})
    plan_path.write_text(json.dumps(document, indent=2), encoding='utf-8')
    for job in jobs:
        print(job['arm'])
        for stage in ('train', 'rank'):
            print('  ' + shlex.join(job[stage]))
    if not a.execute:
        print(f'Plan saved to {plan_path}. Add --execute on a fresh output directory to run.')
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for the full 384px training sweep')
    if not Path(a.data).is_dir():
        raise ValueError('official training directory is unavailable')
    run_logged([sys.executable, str(repo / 'selftest.py')], root / 'selftest.log', repo)
    for job in jobs:
        if (root / job['arm']).exists():
            raise ValueError('refusing to overwrite an existing training output')
        for stage in ('train', 'rank'):
            run_logged(job[stage], root / f"{job['arm']}_{stage}.log", repo)
    print('Finished ablations. Compare training-holdout rankings and evaluate finalists independently.')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--pretrained', default='openai')
    p.add_argument('--semantic-reference', default='')
    p.add_argument('--out', default='outputs_search76')
    p.add_argument('--arms', nargs='+', choices=list(ARMS),
                   default=['control', 'conf_gamma1', 'shuffle'])
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--execute', action='store_true')
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
