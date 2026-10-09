"""Search single-checkpoint recipes on the official training hold-out only.

No test path or reconstructed labels are accepted. Each candidate contains one
checkpoint, one fixed view set and a global train-frequency adjustment tau.
The table ranks macro recall by default; V* remains diagnostic if supplied.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import analyze
import infer
from class_search import measure, write_csv
from train import ImageFolderNoisy


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metadata_digest(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def runtime_identity():
    versions = {'python': platform.python_version(), 'torch': str(torch.__version__),
                'numpy': np.__version__, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version()}
    versions['cudnn_benchmark'] = torch.backends.cudnn.benchmark
    versions['cudnn_deterministic'] = torch.backends.cudnn.deterministic
    if torch.cuda.is_available():
        versions['gpu'] = torch.cuda.get_device_name()
        versions['matmul_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
        versions['cudnn_allow_tf32'] = torch.backends.cudnn.allow_tf32
    for package in ('torchvision', 'open_clip_torch', 'Pillow'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'uninstalled'
    return versions


def cached_logits(path, metadata, names, y, counts, compute):
    """Exact recipe logits from official hold-out only; reject corrupt caches."""
    expected_y = np.asarray(y, dtype=np.int64)
    expected_counts = np.asarray(counts, dtype=np.int64)
    expected_names = np.asarray(names, dtype=str)
    key = metadata_digest(metadata)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['metadata'].item()) != json.dumps(metadata, sort_keys=True):
                raise ValueError(f'logits cache metadata differs: {path}')
            if str(saved['metadata_sha256'].item()) != key:
                raise ValueError(f'logits cache metadata digest differs: {path}')
            for field, expected in (('names', expected_names), ('y', expected_y),
                                    ('counts', expected_counts)):
                if not np.array_equal(saved[field], expected):
                    raise ValueError(f'logits cache {field} differ: {path}')
            values = saved['logits']
            if values.dtype != np.float32 or values.shape != (len(names), len(counts)):
                raise ValueError(f'logits cache dtype/shape differ: {path}')
            if not np.isfinite(values).all():
                raise ValueError(f'logits cache contains nonfinite values: {path}')
            if str(saved['logits_sha256'].item()) != hashlib.sha256(values.tobytes()).hexdigest():
                raise ValueError(f'logits cache payload digest differs: {path}')
        print(f'  logits cache hit: {path}', flush=True)
        return torch.from_numpy(values.copy()), list(names)
    logits, actual_names = compute()
    if actual_names != list(names):
        raise ValueError('hold-out filename order drift')
    values = logits.float().cpu().numpy()
    if values.shape != (len(names), len(counts)) or not np.isfinite(values).all():
        raise ValueError('hold-out logits shape differs or contains nonfinite values')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, logits=values, names=expected_names, y=expected_y, counts=expected_counts,
                 metadata=json.dumps(metadata, sort_keys=True), metadata_sha256=key,
                 logits_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    temporary.replace(path)
    return logits, actual_names


@torch.no_grad()
def collect(model, files, transforms, size, device, batch_size, workers):
    ds = infer.TTADataset(files, transforms, size)
    kw = {'prefetch_factor': 1, 'worker_init_fn': infer._one_thread_per_worker} if workers else {}
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers, shuffle=False,
                        pin_memory=device.type == 'cuda', **kw)
    result, names = [], []
    for views, batch_names, _ in loader:
        accum = None
        for v in range(views.shape[1]):
            # Match infer.py numerics; no independently trained model is combined.
            logits = model(views[:, v].to(device)).float().cpu()
            accum = logits if accum is None else accum + logits
        result.append(accum / views.shape[1])
        names.extend(batch_names)
    return torch.cat(result), names


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repo = Path(__file__).resolve().parent
    source_hashes = {name: file_sha256(repo / name)
                     for name in ('checkpoint_search.py', 'train.py', 'infer.py', 'analyze.py')}
    cache_dir = Path(getattr(a, 'cache_dir', '') or out / 'logits_cache')
    data_fingerprints = {}
    entries = []
    for checkpoint in a.checkpoints:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        args = ck.get('args', {})
        data = a.data or args.get('data', '')
        if not data:
            raise ValueError('official training directory required')
        model = None
        classes = ck['classes']
        size = ck.get('img_size', args.get('img_size', 224))
        split_seed = args.get('split_seed')
        seed = int(args.get('seed', 3407) if split_seed is None else split_seed)
        ratio = float(args.get('val_ratio', .1))
        plain_tf = infer.build_view_transform(*infer.TTA_VIEWS['plain'], size)
        ds = ImageFolderNoisy(data, plain_tf, True, ratio, seed, 'val', size)
        tr = ImageFolderNoisy(data, plain_tf, False, ratio, seed, 'train', size)
        if ds.class_to_idx != classes:
            raise ValueError('training-folder class ordering differs from checkpoint')
        files = [Path(p) for p, _ in ds.items]
        if len({p.name for p in files}) != len(files):
            raise ValueError('duplicate basenames in training hold-out')
        labels = {i: f'{int(c):04d}' for c, i in classes.items()}
        ref = {Path(p).name: labels[y] for p, y in ds.items}
        freq = Counter(tr.targets)
        counts = {labels[i]: freq[i] for i in range(len(classes))}
        if any(v <= 0 for v in counts.values()):
            raise ValueError('class has no training examples')
        saved_counts = ck.get('class_counts')
        if saved_counts is not None and list(saved_counts) != [freq[i] for i in range(len(classes))]:
            raise ValueError('official training counts differ from checkpoint; wrong data or split')
        split_id = hashlib.sha256(json.dumps(list(ref.items())).encode()).hexdigest()[:12]
        if entries and entries[0]['split'] != split_id:
            raise ValueError('checkpoint splits differ; compare on a common untouched split')
        reference = out / f'holdout_{split_id}.csv'
        with reference.open('w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerows(ref.items())
        (out / f'counts_{split_id}.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
        mask = None
        if a.vstar:
            with np.load(a.vstar, allow_pickle=False) as v:
                if not np.array_equal(v['paths'], np.asarray([str(p) for p in files])):
                    raise ValueError('V* paths/order do not match this checkpoint split')
                if not np.array_equal(v['y'], np.asarray(ds.targets)):
                    raise ValueError('V* labels do not match official training hold-out')
                mask = v['vstar'].astype(bool)
                if mask.shape != (len(files),):
                    raise ValueError('V* mask shape differs from hold-out size')
        fingerprint_key = tuple(str(p.resolve()) for p in files)
        if fingerprint_key not in data_fingerprints:
            print('Hashing official hold-out image content for logits-cache provenance...', flush=True)
            data_fingerprints[fingerprint_key] = metadata_digest(
                [(str(p.resolve()), file_sha256(p)) for p in files])
        pretrained = a.pretrained or ck.get('pretrained', 'openai')
        pretrained_id = {'name': pretrained,
                         'model': ck.get('model_name', args.get('model', 'ViT-B-32-quickgelu'))}
        if Path(pretrained).is_file():
            pretrained_id['sha256'] = file_sha256(pretrained)
        ck_hash = file_sha256(checkpoint)
        ck_tag = f'{Path(checkpoint).parent.name}_{Path(checkpoint).stem}_{ck_hash[:8]}'
        n = torch.tensor([counts[labels[i]] for i in range(len(classes))], dtype=torch.float32)
        prior = (n / n.sum()).log()
        for recipe in a.recipes:
            views = infer.resolve_views([recipe])
            if set(views) - set(infer.TTA_VIEWS):
                raise ValueError(f'unknown view recipe: {recipe}')
            tfs = [infer.build_view_transform(*infer.TTA_VIEWS[v], size) for v in views]
            print(f'{checkpoint}: {recipe} ({len(views)} views), {len(files)} hold-out images', flush=True)
            metadata = dict(schema=1, checkpoint_sha256=ck_hash, split=split_id,
                            holdout_content_sha256=data_fingerprints[fingerprint_key],
                            classes=classes, labels=ds.targets, counts=list(freq[i] for i in range(len(classes))),
                            source_sha256=source_hashes, views=[(v, infer.TTA_VIEWS[v]) for v in views],
                            img_size=size, pretrained=pretrained_id, runtime=runtime_identity(),
                            device=str(device), batch_size=a.batch_size)
            # JSON round-trip normalizes tuples, so stored metadata compares exactly.
            metadata = json.loads(json.dumps(metadata))
            def compute():
                nonlocal model
                if model is None:
                    model_args = argparse.Namespace(checkpoint=checkpoint, pretrained=a.pretrained, lora_rank=0)
                    model, loaded_classes, loaded_size = analyze.load_model(model_args, device)
                    if loaded_classes != classes or loaded_size != size:
                        raise ValueError('checkpoint model metadata differs')
                return collect(model, files, tfs, size, device, a.batch_size, a.workers)
            logits, names = cached_logits(cache_dir / f'{metadata_digest(metadata)}.npz', metadata,
                                          list(ref), ds.targets,
                                          [freq[i] for i in range(len(classes))], compute)
            if names != list(ref):
                raise ValueError('hold-out filename order drift')
            for tau_index, tau in enumerate(a.taus):
                pred_index = (logits - tau * prior).argmax(1).tolist()
                pred = {name: labels[i] for name, i in zip(names, pred_index)}
                stats, rows, _ = measure(pred, ref, pred, counts=counts)
                tag = f'{ck_tag}_{recipe}_tau{tau_index}'
                path = out / f'{tag}.csv'
                with path.open('w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(pred.items())
                write_csv(out / f'{tag}_classes.csv', rows, list(rows[0]))
                metric = stats[a.metric]
                if mask is not None:
                    hit = np.asarray([pred[name] == ref[name] for name in names])
                    vstar_score = float(hit[mask].mean()) if mask.any() else None
                else:
                    vstar_score = None
                entries.append(dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=ck_hash,
                                    epoch=int(ck['epoch']) + 1, recipe=recipe, tau=tau, split=split_id,
                                    selection_metric=a.metric, selection_score=metric,
                                    accuracy=stats['accuracy'], macro=stats['macro'],
                                    tail20_macro=stats['tail20_macro'], vstar=vstar_score,
                                    prediction=str(path.resolve()), reference=str(reference.resolve())))
                print(f"  tau={tau:g} micro={stats['accuracy']:.4f} macro={stats['macro']:.4f} "
                      f"tail20={stats['tail20_macro']:.4f}", flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if len({r['split'] for r in entries}) != 1:
        raise ValueError('checkpoint splits differ; compare on a common untouched split')
    entries.sort(key=lambda r: r['selection_score'], reverse=True)
    write_csv(out / 'ranking.csv', entries, list(entries[0]))
    winner = entries[0]
    selection = dict(winner=winner, candidates=entries, provenance='official-training-holdout',
                     caution='Hold-out labels are noisy; this ranking alone does not establish leaderboard gains.')
    (out / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print(f"Selected {winner['checkpoint']} / {winner['recipe']} / tau={winner['tau']} "
          f"on {a.metric}; selection.json contains the reproducible recipe.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--data', default='')
    p.add_argument('--pretrained', default='')
    p.add_argument('--recipes', nargs='+', default=['plain', 'center', 'scales'])
    p.add_argument('--taus', nargs='+', type=float, default=[0, .1, .25, .5])
    p.add_argument('--metric', choices=['macro', 'accuracy', 'tail20_macro'], default='macro')
    p.add_argument('--vstar', default='', help='optional frozen-CLIP V* file, same official training split')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default='analysis_checkpoint_search')
    p.add_argument('--cache-dir', default='', help='persistent official hold-out logits cache; default OUT/logits_cache')
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
