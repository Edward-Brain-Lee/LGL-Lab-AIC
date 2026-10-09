"""Build visual semantic prototypes on disjoint official training images only.

Uses a newly loaded OpenAI CLIP ViT-B/32 at its native 224px resolution. No test
path, external images, guessed species names or leaderboard proxy are inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import open_clip

from train import ImageFolderNoisy, CLIP_MEAN, CLIP_STD


def split_fingerprint(ds):
    rows = [(Path(p).relative_to(ds.root).as_posix(), int(y)) for p, y in ds.items]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


@torch.no_grad()
def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    ds = ImageFolderNoisy(a.data, tf, False, a.val_ratio, a.seed, 'train')
    if not len(ds):
        raise ValueError('empty official training split')
    model = open_clip.create_model('ViT-B-32-quickgelu', pretrained=a.pretrained).to(device).eval()
    features, labels = [], []
    for x, y, _ in DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                              num_workers=a.workers, pin_memory=device.type == 'cuda'):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == 'cuda'):
            z = model.encode_image(x.to(device))
        features.append(F.normalize(z.float(), dim=-1).cpu())
        labels.append(y)
    z, y = torch.cat(features), torch.cat(labels)
    count = torch.bincount(y, minlength=len(ds.class_to_idx))
    if bool((count == 0).any()):
        raise ValueError('a class has no official training samples')
    means = torch.stack([z[y == c].mean(0) for c in range(len(count))])
    prototypes = F.normalize(means, dim=-1)
    artifact = dict(provenance='official-training-only', backbone='ViT-B-32-quickgelu',
                    pretrained=a.pretrained, native_size=224, classes=ds.class_to_idx,
                    train_fingerprint=split_fingerprint(ds), seed=a.seed, val_ratio=a.val_ratio,
                    counts=count, prototypes=prototypes,
                    concentration=means.norm(dim=-1),
                    comment='Raw frozen visual class means, no relabel/filter; not species names.')
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('refusing to overwrite existing semantic reference')
    torch.save(artifact, target)
    print(f'Wrote {len(count)} official-training prototypes to {target.resolve()}')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--pretrained', default='openai', help='official OpenAI tag or official local weight file')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--val-ratio', type=float, default=.1)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', default='analysis_semantic/reference.pt')
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
