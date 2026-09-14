"""Generate ``pred_results.csv`` for the official test set.

Single model, single forward pass (no ensemble, no TTA) -- CLIP ViT-B/32 with
the LoRA adapters and the cosine head produced by ``train.py``.

Usage::

    python infer.py --test /root/autodl-tmp/test --checkpoint outputs/best.pt \
                    --output pred_results.csv

Every image found under ``--test`` produces exactly one CSV row.  An image that
Pillow cannot decode is replaced by a grey image instead of being skipped, so a
partially truncated file (the competition warns about those) can never make the
submission row count mismatch the test set.
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torchvision import transforms

import open_clip

from train import CLIP_MEAN, CLIP_STD, IMG_EXTS, Net

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    classes = ck['classes'] if 'classes' in ck else ck['class_to_idx']
    ck_args = ck.get('args', {})
    rank = a.lora_rank or ck.get('lora_rank', ck_args.get('lora_rank', 8))
    target = ck.get('lora_target', ck_args.get('lora_target', 'all'))
    pretrained = a.pretrained or ck.get('pretrained', 'openai')
    model_name = ck.get('model_name', ck_args.get('model', 'ViT-B-32-quickgelu'))

    clip_model = open_clip.create_model(model_name, pretrained=pretrained)
    model = Net(clip_model, len(classes), rank, target)
    missing, _ = model.load_state_dict(ck.get('model', ck), strict=False)
    trained = {n for n, p in model.named_parameters() if p.requires_grad}
    lost = trained & set(missing)
    assert not lost, f'checkpoint has no trained weights for: {sorted(lost)[:5]}'
    del clip_model
    model.to(device).eval()
    print(f'loaded {a.checkpoint} (epoch {ck.get("epoch", "?")}, '
          f'{len(classes)} classes, {model_name}, lora rank {rank}/{target})')

    # folder name -> four-digit submission label
    idx_to_label = {}
    for name, i in classes.items():
        try:
            idx_to_label[i] = f'{int(name):04d}'
        except ValueError:
            idx_to_label[i] = name
            print(f'WARNING: class folder "{name}" is not numeric, passing it through unchanged')

    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
    files = sorted(p for p in Path(a.test).rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS)
    assert files, f'no images found under {a.test}'
    print(f'{len(files)} test images found')

    rows, unreadable = [], 0
    with torch.no_grad():
        for s in range(0, len(files), a.batch_size):
            ims, names = [], []
            for p in files[s:s + a.batch_size]:
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:                       # truncated / unreadable
                    img = Image.new('RGB', (224, 224), (127, 127, 127))
                    unreadable += 1
                ims.append(tf(img))
                names.append(p.name)
            pred = model(torch.stack(ims).to(device)).argmax(1).cpu().tolist()
            rows += [(n, idx_to_label[i]) for n, i in zip(names, pred)]

    with open(a.output, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)

    names = [r[0] for r in rows]
    print(f'wrote {len(rows)} rows to {a.output} '
          f'(unreadable images replaced by grey: {unreadable})')
    if len(rows) != len(files):
        print('WARNING: row count does not match the number of test images')
    if len(set(names)) != len(names):
        print('WARNING: duplicate file names found -- the grader may match by name only')


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--test', required=True, help='test image folder (searched recursively)')
    p.add_argument('--checkpoint', required=True, help='best.pt / last.pt from train.py')
    p.add_argument('--output', default='pred_results.csv')
    p.add_argument('--pretrained', default='', help='override the CLIP weights (default: as in the checkpoint)')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lora-rank', type=int, default=0, help='override (default: as in the checkpoint)')
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
