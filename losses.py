"""Robust losses for learning with noisy labels.

Every loss returns a **per-sample** vector of shape ``(B,)`` so the caller can
apply noise filtering / re-weighting sample by sample; ``reduction='mean'``
gives the plain scalar objective.

* ``ce``  -- cross entropy, used during warm-up only;
* ``gce`` -- generalized cross entropy (Zhang & Sabuncu, NeurIPS 2018);
* ``nce`` -- normalized cross entropy (Ma et al., ICML 2020, Eq. 6);
* ``rce`` -- reverse cross entropy (Ma et al., ICML 2020, Table 1);
* ``apl`` -- active passive loss = NCE + RCE (Ma et al., ICML 2020).

All losses cast the logits to fp32 first: NCE raises ``p`` to a power, which is
not numerically safe in fp16.
"""
import math

import torch
import torch.nn.functional as F

EPS = 1e-6


def _p_of_label(logits, y):
    """Probability assigned to the given label, differentiable."""
    return F.softmax(logits.float(), 1).gather(1, y[:, None]).squeeze(1)


def ce(logits, y, reduction='none'):
    return F.cross_entropy(logits.float(), y, reduction=reduction)


def gce(logits, y, q=0.7, reduction='none'):
    """Generalized cross entropy ``(1 - p_y**q) / q``.

    Bounded in ``[0, 1/q]`` and its gradient vanishes as ``p_y -> 1``, so a
    memorised wrong sample stops dominating the update.
    """
    loss = (1.0 - _p_of_label(logits, y).clamp_min(EPS).pow(q)) / q
    return loss.mean() if reduction == 'mean' else loss


def rce(logits, y, eps=1e-4, reduction='none'):
    """Reverse cross entropy: cross entropy with the roles of prediction and
    label swapped (Ma et al., ICML 2020).

    The one-hot target is smoothed to ``eps``, which collapses the off-target
    terms::

        RCE = -Σ_k p_k log q_k = -log(eps) * (1 - p_y)

    i.e. a mean-absolute-error-like term scaled by ``-log(eps) ≈ 9.21``.  Like
    NCE its gradient does not vanish as the model grows confident, which is what
    keeps it acting on samples CE has already stopped caring about.

    (An earlier version returned the prediction *entropy*.  Minimising entropy
    *sharpens* the posterior -- the opposite of damping an over-confident model
    -- so despite the docstring claiming otherwise it was never APL's RCE.)
    """
    p = _p_of_label(logits, y)
    loss = -math.log(eps) * (1.0 - p)
    return loss.mean() if reduction == 'mean' else loss


def nce(logits, y, k=0.2, B=1.0, reduction='none'):
    """Normalized cross entropy (Ma et al., ICML 2020).

    ``-log(p)`` for ``p <= k``; for ``p > k`` it is replaced by the straight line
    tangent to ``-log(p)`` at ``p = k``: ``A * p**B + C`` with
    ``A = -1 / (B * k**B)`` and ``C = 1 / B - log(k)`` (matching value and slope
    at ``p = k``).  The gradient therefore never vanishes on confident samples,
    which is what makes the loss *active* against memorised label noise.

    Note: the loss is unbounded above but bounded below (for ``k = 0.2`` the
    minimum is ``-1/k + 1 - log k ~= -2.39``), so after the warm-up phase the
    printed training loss may legitimately be negative -- that is by design and
    is not a sign of divergence.
    """
    p = _p_of_label(logits, y).clamp(EPS, 1 - EPS)
    A, C = -1.0 / (B * k ** B), 1.0 / B - math.log(k)
    loss = torch.where(p <= k, -torch.log(p), A * p.pow(B) + C)
    return loss.mean() if reduction == 'mean' else loss


def apl(logits, y, k=0.2, B=1.0, rce_scale=1.0, reduction='none'):
    """Active passive loss: ``NCE + rce_scale * RCE``.

    Both halves push ``p_y`` up, so this is still a classification loss; the
    robustness comes from the *shape* of the two gradients (neither vanishes on
    a confident sample), not from a different direction.
    """
    loss = nce(logits, y, k, B) + rce_scale * rce(logits, y)
    return loss.mean() if reduction == 'mean' else loss


def make_robust_loss(name='apl', q=0.7, k=0.2, B=1.0, rce_scale=1.0):
    """Return ``f(logits, y) -> (B,)`` for the requested robust loss."""
    if name == 'ce':
        return lambda lg, y: ce(lg, y)
    if name == 'gce':
        return lambda lg, y: gce(lg, y, q)
    if name == 'nce':
        return lambda lg, y: nce(lg, y, k, B)
    if name == 'apl':
        return lambda lg, y: apl(lg, y, k, B, rce_scale)
    raise ValueError(f'unknown robust loss: {name}')
