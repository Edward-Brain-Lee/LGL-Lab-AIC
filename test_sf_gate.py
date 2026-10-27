"""Fixture tests for `sf_gate.py` — stdlib only, runs on the laptop in a second.

    python test_sf_gate.py

Covers the gates that decide whether a 2-hour run may start and whether a
submission file is valid, and checks that each one actually *fails* on the
mistake it exists to catch (a gate that cannot say no is not a gate).
"""
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import sf_gate


@contextlib.contextmanager
def must_fail(why):
    """Run a gate that is expected to refuse; silence its (correct) complaint."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            yield
    except SystemExit:
        return
    raise AssertionError(f'gate accepted: {why}')


def _mk(path, data=b'x', mode='wb'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as fh:
        fh.write(data)


def _fake_dataset(root, classes=3, per_class=2, n_test=2):
    """Minimal layout that obeys every structural invariant the gates check."""
    for c in range(classes):
        for i in range(per_class):
            _mk(root / 'train' / f'{c:04d}' / f'img{i}.jpg')
    names = [f'{i:032x}.jpg' for i in range(n_test)]
    for n in names:
        _mk(root / 'test' / n)
    return root / 'train', root / 'test', names


def _csv_text(names, labels=None):
    labels = labels or ['0001'] * len(names)
    return ''.join(f'{n},{l}\r\n' for n, l in zip(names, labels))


def test_param_budget():
    assert sf_gate._selfcheck_params()
    e = sf_gate.expected_trainable(500)
    # adapters: 48 modules x 110,592 params per rank unit... pinned by the 复赛 run
    assert e['adapters'] == 9_437_184, e['adapters']
    assert e['total'] == 10_197_762, e['total']
    assert f"{e['total'] / 1e6:.3f}" == '10.198'
    # the 750-class observation the formula is anchored on
    assert 10_325_740 <= sf_gate.expected_trainable(750)['total'] < 10_325_796
    # rank scales the adapter part linearly and nothing else
    a32 = sf_gate.expected_trainable(500, rank=32)
    assert a32['adapters'] * 2 == e['adapters']
    assert a32['head'] == e['head'] == 500 * 512 + 1
    # no local head -> no 393,216 + no gate
    assert sf_gate.expected_trainable(500, local_head=False)['total'] == \
        e['total'] - (768 * 512 + 1)


def test_data_gate():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train, test, names = _fake_dataset(root)
        g = sf_gate.gate_data(train, test, classes=3, n_train=6, n_test=2,
                              verbose=False)
        assert (g['classes'], g['train'], g['test']) == (3, 6, 2)
        rows, nbytes = sf_gate.expected_csv(names)
        assert (g['rows'], g['bytes']) == (rows, nbytes) == (2, 2 * 43)

        # wrong class count
        with must_fail('wrong class count'):
            sf_gate.gate_data(train, test, classes=4, n_train=6, n_test=2, verbose=False)

        # wrong image count
        with must_fail('wrong image count'):
            sf_gate.gate_data(train, test, classes=3, n_train=7, n_test=2, verbose=False)

        # a gap in the class folders silently shifts every id -> must be refused
        (train / '0001').rename(train / '0009')
        with must_fail('non-contiguous class set'):
            sf_gate.gate_data(train, train, classes=3, n_train=6, n_test=6, verbose=False)
        (train / '0009').rename(train / '0001')

        # test names that are not 32-hex would change the 43-byte row assumption
        _mk(test / 'short.jpg')
        with must_fail('non-standard test name'):
            sf_gate.gate_data(train, test, classes=3, n_train=6, n_test=3, verbose=False)
        (test / 'short.jpg').unlink()


def test_csv_gate():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train, test, names = _fake_dataset(root, classes=3, per_class=2, n_test=3)
        good = root / 'pred_results.csv'
        _mk(good, _csv_text(names, ['0000', '0001', '0002']).encode(), 'wb')
        g = sf_gate.gate_csv(good, test, classes=3, verbose=False)
        assert (g['rows'], g['bytes'], g['labels']) == (3, 3 * 43, 3)

        bad_cases = {
            'missing row': _csv_text(names[:2]),
            'duplicate row': _csv_text([names[0]] * 3),
            'unknown name': _csv_text(names[:2] + ['a' * 32 + '.jpg']),
            'label too short': _csv_text(names, ['001', '0001', '0002']),
            'label out of range': _csv_text(names, ['0000', '0001', '0003']),
            'header line': 'filename,label\r\n' + _csv_text(names),
            'lf only': _csv_text(names).replace('\r\n', '\n'),
        }
        for why, text in bad_cases.items():
            p = root / 'bad.csv'
            _mk(p, text.encode(), 'wb')
            with must_fail(why):
                sf_gate.gate_csv(p, test, classes=3, verbose=False)


def test_log_gate():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        classes, rank, img = 3, 64, 384
        e = sf_gate.expected_trainable(classes, rank=rank, qkv=True, local_head=True,
                                       pos_tokens=(img // 32) ** 2 + 1)
        good = (f'classes={classes} train=6 val=2 device=cuda\n'
                f'model=ViT-B-32-quickgelu/openai layers=48 lora_alpha=128 '
                f'lora_qkv=True attn_temp=False trainable params='
                f'{e["total"] / 1e6:.3f}M\n'
                f'img_size={img}: positional grid resampled to 12x12\n'
                f'--train-pos-embed: positional grid is now trainable (111.4k params)\n'
                f'config: img_size={img} local_head=True train_pos_embed=True '
                f'lora_rank={rank}\n')
        log = root / 'run.log'
        _mk(log, good.encode(), 'wb')
        assert sf_gate.gate_log(log, classes=classes, verbose=False)

        for why, text in {
            'wrong param total': good.replace(f'{e["total"] / 1e6:.3f}M', '9.999M'),
            'missing qkv flag': good.replace('lora_qkv=True', 'lora_qkv=False'),
            'wrong rank': good.replace(f'lora_rank={rank}', 'lora_rank=32'),
            'missing pos-embed line': good.replace(
                'positional grid is now trainable', 'nothing here'),
            'wrong class count': good.replace(f'classes={classes}', 'classes=750'),
            'traceback': good + 'Traceback (most recent call last):\n',
        }.items():
            _mk(log, text.encode(), 'wb')
            with must_fail(why):
                sf_gate.gate_log(log, classes=classes, verbose=False)


def test_ckpt_gate():
    """Needs torch; skipped on the plain system Python, run with the CPU venv."""
    try:
        import torch
    except ImportError:
        print('  skip  test_ckpt_gate (no torch in this interpreter)')
        return
    classes, rank, img, tokens = 20, 64, 384, (384 // 32) ** 2 + 1
    # 12 blocks x 4 wrapped Linear modules, shapes as in the real tower
    blocks = [('attn.qkv', 768, 2304), ('attn.out_proj', 768, 768),
              ('mlp.c_fc', 768, 3072), ('mlp.c_proj', 3072, 768)]

    def make():
        model = {}
        for b in range(12):
            for name, inp, out in blocks:
                model[f'transformer.resblocks.{b}.{name}.A'] = torch.zeros(rank, inp)
                model[f'transformer.resblocks.{b}.{name}.B'] = torch.zeros(out, rank)
        model['head.weight'] = torch.zeros(classes, 512)
        model['head.logit_scale'] = torch.zeros(())
        model['local_head.proj.weight'] = torch.zeros(512, 768)
        model['local_head.gate'] = torch.zeros(())
        model['clip.visual.positional_embedding'] = torch.zeros(tokens, 768)
        # buffers that ride along in trainable_state_dict(); must not be counted
        model['proto.proto'] = torch.zeros(classes, 512)
        model['proto.filled'] = torch.zeros(classes, dtype=torch.bool)
        return {'model': model,
                'classes': {f'{i:04d}': i for i in range(classes)},
                'lora_rank': rank, 'lora_qkv': True,
                'args': {'img_size': img, 'local_head': True, 'train_pos_embed': True},
                'epoch': 20, 'val_acc': 0.77, 'val_acc_hi': 0.9}

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'best.pt'
        torch.save(make(), p)
        ck = sf_gate.gate_ckpt(p, classes=classes, verbose=False)
        assert ck['epoch'] == 20

        def must_refuse(mutate, why):
            ck2 = make()
            mutate(ck2)
            q = Path(td) / 'bad.pt'
            torch.save(ck2, q)
            with must_fail(why):
                sf_gate.gate_ckpt(q, classes=classes, verbose=False)

        must_refuse(lambda c: c.update(lora_rank=32), 'wrong lora rank')
        must_refuse(lambda c: c.update(lora_qkv=False), 'missing --lora-qkv')
        must_refuse(lambda c: c['args'].update(img_size=416), 'wrong img_size')
        must_refuse(lambda c: c['model'].pop('clip.visual.positional_embedding'),
                    'no trainable positional grid')
        must_refuse(lambda c: c['model'].pop('local_head.proj.weight'), 'no local head')
        must_refuse(lambda c: c['classes'].pop('0003'), 'missing class')
        must_refuse(lambda c: c['model'].pop('transformer.resblocks.0.attn.qkv.A'),
                    'missing qkv adapter')
        must_refuse(lambda c: c['model'].update(
            {'head.weight': torch.zeros(classes + 1, 512)}), 'wrong head shape')


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'  ok  {t.__name__}')
    print(f'ALL {len(tests)} SF-GATE TESTS PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
