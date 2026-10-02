"""Post-run diagnostics for robust CLIP training.

This module deliberately does not import :mod:`train`: it can inspect a thin
checkpoint on a CPU-only machine and therefore cannot accidentally rebuild or
alter the official backbone.  It is useful both as a command-line report and
as a small collection of pure helpers that can be called from the training
loop in a later experiment.

Examples
--------
    python diagnostics.py --checkpoint outputs/last.pt
    python diagnostics.py --checkpoint outputs/last.pt --out outputs/diag.json

The report contains only information already present in a checkpoint.  It does
not read test data and does not perform model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    """Convert argparse/torch/numpy scalars into deterministic JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    # torch and numpy scalar values expose ``item`` without importing either.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def config_fingerprint(config: Mapping[str, Any], *, exclude: tuple[str, ...] =
                       ("out", "resume", "limit_batches")) -> str:
    """Return a short SHA-256 fingerprint for a training configuration.

    Paths and smoke-test controls are excluded by default because they do not
    change the learned function.  Callers may pass ``exclude=()`` when they
    need a literal fingerprint of every argument.
    """
    clean = {str(k): _jsonable(v) for k, v in config.items() if k not in exclude}
    blob = json.dumps(clean, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _to_list(x):
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _quantiles(values, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    vals = sorted(float(v) for v in _to_list(values))
    if not vals:
        return {str(q): 0.0 for q in qs}
    out = {}
    for q in qs:
        pos = q * (len(vals) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
        frac = pos - lo
        out[str(q)] = vals[lo] * (1 - frac) + vals[hi] * frac
    return out


def summarize_tracker(state: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize a serialized ``LabelTrustTracker`` state.

    State labels follow the implementation in ``noise.py``.  A relabelled row
    is identified by ``mix > 0``; a noisy row by its stored low weight.  Rows
    rescued by the global noise cap have weight 1 and are consequently counted
    as clean, matching ``LabelTrustTracker.refresh``'s documented accounting.
    """
    y = _to_list(state.get("y"))
    # Older checkpoints do not store y in tracker.state_dict; callers can pass
    # it as ``state['_y']`` or rely on counts from the checkpoint itself.
    n = len(y)
    seen = [bool(v) for v in _to_list(state.get("seen"))]
    weight = [float(v) for v in _to_list(state.get("weight"))]
    mix = [float(v) for v in _to_list(state.get("mix"))]
    pred = [int(v) for v in _to_list(state.get("pred"))]
    label = [int(v) for v in _to_list(state.get("label"))]
    suspect_raw = state.get("suspect")
    suspect = [bool(v) for v in _to_list(suspect_raw)] if suspect_raw is not None else []
    if not n:
        n = max(len(seen), len(weight), len(mix), len(pred), len(label))
        y = [None] * n
    if not seen:
        seen = [False] * n
    if not weight:
        weight = [1.0] * n
    if not mix:
        mix = [0.0] * n
    relabel = [s and m > 1e-8 for s, m in zip(seen, mix)]
    # ``w_noise`` is not stored separately.  Any seen row with sub-unit weight
    # and no relabel mixture is a demoted/noisy row, including judge vetoes.
    noisy = [s and not r and w < 1.0 - 1e-6 for s, r, w in zip(seen, relabel, weight)]
    clean = [s and not r and not z for s, r, z in zip(seen, relabel, noisy)]
    unseen = [not s for s in seen]

    def count(mask):
        return int(sum(bool(v) for v in mask))

    trust = []
    prob = state.get("prob")
    if prob is not None and hasattr(prob, "gather") and y and all(v is not None for v in y):
        try:
            import torch
            yy = torch.as_tensor(y, dtype=torch.long, device=prob.device)
            trust = prob.gather(1, yy[:, None]).squeeze(1)[torch.as_tensor(seen, device=prob.device)]
        except Exception:
            trust = []

    transitions = Counter()
    if y and pred and len(y) == len(pred) and all(v is not None for v in y):
        transitions.update((int(a), int(b)) for a, b in zip(y, pred) if a != b)
    top_transitions = [
        {"given": int(a), "pred": int(b), "count": int(c)}
        for (a, b), c in transitions.most_common(30)
    ]

    result = {
        "n": n,
        "counts": {"clean": count(clean), "relabel": count(relabel),
                   "noisy": count(noisy), "unseen": count(unseen)},
        "fractions": {k: count(v) / max(n, 1) for k, v in
                      (("clean", clean), ("relabel", relabel),
                       ("noisy", noisy), ("unseen", unseen))},
        "suspect": int(sum(suspect)) if suspect else 0,
        "trust_quantiles": _quantiles(trust),
        "weight_quantiles": _quantiles(weight),
        "top_given_to_teacher_transitions": top_transitions,
    }
    if y and n == len(y) and all(v is not None for v in y):
        result["per_class"] = per_class_states(y, clean, relabel, noisy, unseen,
                                                pred=pred, suspect=suspect)
    return result


def per_class_states(y, clean, relabel, noisy, unseen, *, pred=None, suspect=None):
    """Return compact per-class state counts and teacher disagreement rates."""
    nclass = max((int(v) for v in y if v is not None), default=-1) + 1
    rows = [{"class": c, "n": 0, "clean": 0, "relabel": 0,
             "noisy": 0, "unseen": 0} for c in range(nclass)]
    has_pred = pred is not None and len(pred) == len(y)
    has_suspect = suspect is not None and len(suspect) == len(y)
    for i, c in enumerate(y):
        row = rows[int(c)]
        row["n"] += 1
        row["clean"] += bool(clean[i])
        row["relabel"] += bool(relabel[i])
        row["noisy"] += bool(noisy[i])
        row["unseen"] += bool(unseen[i])
        if has_pred:
            row["teacher_disagree"] = row.get("teacher_disagree", 0) + (pred[i] != c)
        if has_suspect:
            row["suspect"] = row.get("suspect", 0) + bool(suspect[i])
    if has_pred:
        for row in rows:
            row["teacher_disagree_rate"] = row.get("teacher_disagree", 0) / max(row["n"], 1)
    return rows


def summarize_jsd(values) -> dict[str, Any]:
    """Aggregate a tensor/list of per-sample JSD values for an epoch."""
    vals = [float(v) for v in _to_list(values)]
    positive = [v for v in vals if v > 1e-7]
    return {"n": len(vals), "mean": sum(vals) / max(len(vals), 1),
            "p50": _quantiles(vals, (0.5,))["0.5"],
            "p90": _quantiles(vals, (0.9,))["0.9"],
            "max": max(vals, default=0.0), "downweighted": len(positive),
            "downweighted_fraction": len(positive) / max(len(vals), 1)}


def epoch_record(*, epoch: int, loss: float | None = None,
                 val_loss: float | None = None, val_acc: float | None = None,
                 val_acc_hi: float | None = None, lr: float | None = None,
                 noise: Mapping[str, Any] | None = None,
                 jsd: Mapping[str, Any] | None = None,
                 frozen_judge: Mapping[str, Any] | None = None,
                 **extra) -> dict[str, Any]:
    """Create one JSON-safe epoch row for a sidecar ``diagnostics.jsonl``.

    Keeping this as a pure helper lets the training loop append a row without
    importing any reporting or plotting code.  ``extra`` is intentionally
    accepted for experiment-specific measurements (distillation KL, target
    flip rate, prototype coverage, and so on).
    """
    row = {"epoch": int(epoch)}
    for key, value in (("loss", loss), ("val_loss", val_loss),
                       ("val_acc", val_acc), ("val_acc_hi", val_acc_hi),
                       ("lr", lr), ("noise", noise), ("jsd", jsd),
                       ("frozen_judge", frozen_judge)):
        if value is not None:
            row[key] = _jsonable(value)
    # Omit disabled/unknown optional measurements instead of writing ``null``.
    # A present key is used when auditing whether a method actually ran, so a
    # null placeholder would be indistinguishable from an enabled measurement
    # in a grep-based postmortem.
    row.update({str(k): _jsonable(v) for k, v in extra.items() if v is not None})
    return row


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one diagnostics row atomically enough for a single process."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(row), ensure_ascii=False,
                           sort_keys=True) + "\n")


def checkpoint_report(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable report from a loaded checkpoint mapping."""
    args = checkpoint.get("args") or {}
    if not isinstance(args, Mapping):
        args = vars(args)
    tracker = dict(checkpoint.get("tracker") or {})
    # ``y`` is intentionally not part of train.py's tracker state to avoid a
    # second copy in full checkpoints.  Recover it when present in a diagnostic
    # extension, otherwise report global counts and omit per-class rows.
    if "y" not in tracker and "targets" in checkpoint:
        tracker["y"] = checkpoint["targets"]
    report = {
        "checkpoint": {"epoch": checkpoint.get("epoch"),
                       "val_acc": checkpoint.get("val_acc"),
                       "val_acc_hi": checkpoint.get("val_acc_hi"),
                       "model_name": checkpoint.get("model_name"),
                       "image_size": args.get("image_size"),
                       "pos_embed_trained": checkpoint.get("pos_embed_trained"),
                       "config_fingerprint": config_fingerprint(args)},
        "tracker": summarize_tracker(tracker),
    }
    # Report an explicit warning instead of silently pretending that a thin
    # inference snapshot contains tracker diagnostics.
    if not checkpoint.get("tracker"):
        report["tracker"] = {"available": False,
                             "note": "thin checkpoint: use last.pt for tracker diagnostics"}
    else:
        report["tracker"]["available"] = True
    return report


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="", help="write JSON here (default: print only)")
    args = p.parse_args(argv)
    try:
        import torch
    except ImportError as e:
        raise SystemExit("diagnostics.py requires PyTorch to load a checkpoint") from e
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = checkpoint_report(ck)
    text = json.dumps(_jsonable(report), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out} (config={report['checkpoint']['config_fingerprint']})")
    else:
        print(text)


if __name__ == "__main__":
    _main()
