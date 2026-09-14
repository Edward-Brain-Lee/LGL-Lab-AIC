"""Automatic label-noise handling for the training set.

``LabelTrustTracker`` keeps an exponential moving average of the teacher's class
posterior for every training sample (updated online, for free, from the teacher
forward that the confidence re-weighting already needs).  At the start of every
epoch after warm-up the posterior is turned into

* a **possibly corrected** label (pseudo-label) for the sample, and
* a per-sample **loss weight**,

which together play the role of the explicit sample-selection / label-correction
step of DivideMix (Li et al., ICLR 2020) and co-teaching (Han et al., NeurIPS
2018).  Nothing is removed from the dataset, so the sampling schedule stays
valid for the whole run; untrusted samples are simply trusted *less*.
"""
import torch
import torch.nn.functional as F


class LabelTrustTracker:
    """Per-sample teacher-posterior EMA -> soft labels + loss weights.

    The rule is driven by the teacher's **top-1 pick**, not by its absolute
    confidence -- see :meth:`refresh` for why an absolute gate does not work.
    Decision rule applied in :meth:`refresh` (after warm-up):

    ==================================================  ==============  ==============
    condition                                           label           weight
    ==================================================  ==============  ==============
    ``argmax p == y`` (teacher agrees)                  given ``y``     ``1.0``
    disagrees and ``max(p) >= tau_conf``                ``argmax p``    ``w_relabel``
    disagrees and ``max(p) <  tau_conf``                given ``y``     ``w_noise``
    ==================================================  ==============  ==============

    Samples the sampler has not drawn yet keep the given label at weight 1.
    ``max_noise_frac`` caps how much of the set may be declared untrusted, so a
    cold teacher early in training cannot throw away most of the data.
    """

    def __init__(self, targets, nclass, momentum=0.9, tau_conf=0.8,
                 w_noise=0.1, w_relabel=0.5, max_noise_frac=0.4, device='cpu'):
        self.device = device
        self.y = torch.as_tensor(targets, dtype=torch.long, device=device)
        self.n, self.nclass = self.y.numel(), nclass
        self.momentum = momentum
        self.tau_conf = tau_conf
        self.w_noise, self.w_relabel = w_noise, w_relabel
        self.max_noise_frac = max_noise_frac

        self.prob = torch.zeros(self.n, nclass, device=device)
        self.seen = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.label = self.y.clone()
        self.weight = torch.ones(self.n, device=device)
        self.stats = {}

    @torch.no_grad()
    def update(self, idx, prob):
        """Fold the teacher posterior of one batch into the running average."""
        idx = idx.to(self.device).long()
        p = prob.to(self.device, dtype=self.prob.dtype)
        m = self.momentum
        # first observation is stored as-is, later ones are averaged in
        self.prob[idx] = torch.where(self.seen[idx, None], m * self.prob[idx] + (1 - m) * p, p)
        self.seen[idx] = True

    @torch.no_grad()
    def refresh(self):
        """Recompute soft labels and weights.  Call once per epoch, post warm-up."""
        trust = self.prob.gather(1, self.y[:, None]).squeeze(1)
        conf, pred = self.prob.max(1)
        judged = self.seen

        # What carries the signal is whether the teacher's top-1 pick *is* the
        # given label, not how confident it is.  Gating the clean set on an
        # absolute ``p[y] >= 0.8`` asked the teacher to be certain before it was
        # believed; on a 500-class problem the logit scale only grows slowly, so
        # ~83% of the set -- including samples the teacher actively *agreed*
        # with at p[y] = 0.5 -- fell through into the reject pile and the 40%
        # cap pinned on every single epoch.  Agreement is evidence for the
        # label; only disagreement is evidence against it.
        agree = pred == self.y
        relabel = judged & ~agree & (conf >= self.tau_conf)
        noisy = judged & ~agree & (conf < self.tau_conf)

        # cap the rejected fraction: keep the highest-trust ones, reject the rest
        n_noisy = int(noisy.sum())
        cap = int(self.max_noise_frac * self.n)
        capped = 0
        if n_noisy > cap:
            order = trust[noisy].argsort()                    # ascending trust
            keep = noisy.nonzero().squeeze(1)[order[:cap]]    # lowest trust -> untrusted
            capped = n_noisy - cap
            noisy = torch.zeros_like(noisy)
            noisy[keep] = True

        # ``clean`` = "trained at full weight": the leftovers of the two branches
        # above, so that clean + relabel + noisy + unseen partitions the set and
        # the printed statistics add up to N.
        clean = judged & ~relabel & ~noisy

        self.label = torch.where(relabel, pred, self.y)
        self.weight = torch.where(relabel, torch.full_like(trust, self.w_relabel),
                                  torch.where(noisy, torch.full_like(trust, self.w_noise),
                                              torch.ones_like(trust)))
        self.stats = {
            'clean': int(clean.sum()), 'relabel': int(relabel.sum()),
            'noisy': int(noisy.sum()), 'unseen': int((~judged).sum()),
            'capped': capped,           # rescued from `noisy` by max_noise_frac
            'mean_trust': float(trust[judged].mean()) if bool(judged.any()) else 0.0,
        }
        return self.stats

    def state_dict(self):
        return {'prob': self.prob, 'seen': self.seen, 'label': self.label, 'weight': self.weight}

    def load_state_dict(self, sd):
        self.prob.copy_(sd['prob'].to(self.device))
        self.seen.copy_(sd['seen'].to(self.device))
        self.label.copy_(sd['label'].to(self.device))
        self.weight.copy_(sd['weight'].to(self.device))


def prototype_bootstrap(sum_feats, counts):
    """Class means of the frozen-CLIP features collected during warm-up.

    Returns ``(means, present)``; prototypes are only used for the classes that
    were actually seen.
    """
    present = counts > 0
    means = sum_feats / counts.clamp_min(1)[:, None]
    return F.normalize(means.float(), dim=-1), present
