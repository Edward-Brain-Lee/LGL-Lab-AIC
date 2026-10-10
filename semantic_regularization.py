"""Training-only geometry from the official training split and official CLIP.

This is visual semantic distance, not an inferred species-name annotation or a
second classifier. It never combines prototype logits with submission logits.
"""
import torch
import torch.nn.functional as F


def semantic_cost(prototypes):
    z = F.normalize(prototypes.float(), dim=-1)
    cost = ((1 - z @ z.T) / 2).clamp(0, 1)
    cost.fill_diagonal_(0)
    return cost


def expected_semantic_cost(logits, targets, cost):
    """Per-example expected distance from target to all predicted classes.

    CE remains the identification objective. This penalty only supplies relative
    geometry: predicting a distant concept costs more than a near neighbour.
    """
    target_cost = cost[targets.long()] if targets.ndim == 1 else targets.float() @ cost
    return (logits.float().softmax(1) * target_cost).sum(1)
