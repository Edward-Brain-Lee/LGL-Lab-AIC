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

    ==================================================  ==================  ==============
    condition                                           target              weight
    ==================================================  ==================  ==============
    ``argmax p == y`` (teacher agrees)                  ``y``               ``1.0``
    disagrees and ``max(p) >= tau_conf``                ``mix(y, argmax)``  ``w_relabel``
    disagrees and ``max(p) <  tau_conf``                ``y``               ``w_noise``
    ==================================================  ==================  ==============

    The first row is the one place the rule can feed on itself: the evidence is
    the EMA of the *student's* teacher, so a sample the student has memorised --
    including one whose given label is wrong -- ends up agreeing with itself and
    is rewarded with full weight and a hard target, forever.  A :class:`FrozenJudge`
    (see below) can veto that row; the veto only demotes the weight, it never
    changes the label or the target.

    Samples the sampler has not drawn yet keep the given label at weight 1.
    ``max_noise_frac`` caps how much of the set may be declared untrusted, so a
    cold teacher early in training cannot throw away most of the data.

    The middle row is a **mixture**, not an overwrite: ``relabel_mix`` of the
    target mass moves onto the teacher's pick and ``1 - relabel_mix`` stays on
    the given label.  The brief describes the noise as *weakly correlated*
    annotation -- the given label is wrong but related -- and in that regime a
    confident disagreement is frequently not a wrong label at all but a
    genuinely confusable neighbour.  A hard overwrite would promote that
    confusion to ground truth and teach the model exactly the error we are
    trying to avoid; hedging keeps the given label's evidence alive while still
    letting the teacher pull.
    """

    def __init__(self, targets, nclass, momentum=0.9, tau_conf=0.8,
                 w_noise=0.1, w_relabel=0.5, max_noise_frac=0.4,
                 relabel_mix=0.5, class_tau_delta=0.0, device='cpu'):
        self.device = device
        self.y = torch.as_tensor(targets, dtype=torch.long, device=device)
        self.n, self.nclass = self.y.numel(), nclass
        self.momentum = momentum
        self.tau_conf = tau_conf
        self.w_noise, self.w_relabel = w_noise, w_relabel
        self.max_noise_frac = max_noise_frac
        self.relabel_mix = relabel_mix
        self.class_tau_delta = float(class_tau_delta)

        self.prob = torch.zeros(self.n, nclass, device=device)
        self.seen = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.label = self.y.clone()
        self.weight = torch.ones(self.n, device=device)
        self.pred = self.y.clone()                       # teacher top-1, per sample
        self.mix = torch.zeros(self.n, device=device)    # target mass on `pred`
        self.suspect = None                              # FrozenJudge veto, or None
        self.stats = {}

    @torch.no_grad()
    def reset(self):
        """Drop the posterior EMA and its ``seen`` mask.

        Called when the warm-up ends.  The EMA is updated from the very first
        batch, but until warm-up finishes the teacher is an EMA of a randomly
        initialised head, so what it accumulated is noise -- and because the
        first observation of a sample is stored *verbatim*, that noise is not
        washed out by later updates, it is decayed from.  Restarting here means
        the next batch's posterior becomes the first observation, from a teacher
        that has actually been trained.
        """
        self.prob.zero_()
        self.seen.zero_()

    @torch.no_grad()
    def set_judge(self, suspect):
        """Install the frozen-CLIP veto: ``suspect[i]`` -> never rewarded as clean.

        See :class:`FrozenJudge`.  Only the agree branch is affected; the vetoed
        samples fall through to the disagreement branches, which keep their label
        and drop their weight.
        """
        if suspect is None:
            self.suspect = None
            return
        suspect = suspect.to(self.device).bool()
        if suspect.ndim != 1 or suspect.numel() != self.n:
            raise ValueError(f'FrozenJudge suspect mask must have shape ({self.n},), '
                             f'got {tuple(suspect.shape)}')
        self.suspect = suspect

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
        vetoed = 0
        veto_mask = torch.zeros_like(agree)
        if self.suspect is not None:
            veto_mask = judged & self.suspect
            vetoed = int(veto_mask.sum())
            # An independent judge (frozen CLIP, which never saw the labels) says
            # this sample is better explained by another class.  It may not be
            # confirmed as clean, however confidently the student agrees with
            # itself -- that is the loop this exists to break.
            agree = agree & ~veto_mask

        # Optional class-conditional calibration.  The adjustment is estimated
        # only from the current training-set posterior and is disabled by the
        # default (delta=0), so the historical rule remains unchanged unless
        # explicitly requested.
        tau = torch.full_like(conf, float(self.tau_conf))
        if self.class_tau_delta > 0 and bool(judged.any()):
            med = torch.full((self.nclass,), float(self.tau_conf), device=self.device)
            global_med = conf[judged].median()
            for c in range(self.nclass):
                sel = judged & (self.y == c)
                if bool(sel.any()):
                    # Harder classes with lower observed confidence receive a
                    # slightly lower threshold, bounded to avoid label takeover.
                    med[c] = (self.tau_conf + self.class_tau_delta *
                              (conf[sel].median() - global_med)).clamp(0.5, 0.95)
            tau = med[self.y]

        # A frozen-CLIP veto is a demotion only: it must never enter the
        # relabel branch, even when the EMA teacher disagrees with the folder.
        relabel = judged & ~veto_mask & ~agree & (conf >= tau)
        noisy = judged & (veto_mask | (~agree & (conf < tau)))

        # cap the rejected fraction: keep the highest-trust ones, reject the rest
        n_noisy = int(noisy.sum())
        cap = int(self.max_noise_frac * self.n)
        capped = 0
        if n_noisy > cap:
            # Vetoed rows are an independent negative signal and must remain
            # demoted; only ordinary low-confidence disagreements are eligible
            # for the global cap's rescue path.
            ordinary = noisy & ~veto_mask
            forced = veto_mask
            room = max(0, cap - int(forced.sum()))
            order = trust[ordinary].argsort()
            keep = ordinary.nonzero().squeeze(1)[order[:room]]
            capped = int(ordinary.sum()) - int(keep.numel())
            noisy = torch.zeros_like(noisy)
            noisy[keep] = True
            noisy[forced] = True

        # ``clean`` = "trained at full weight": the leftovers of the two branches
        # above, so that clean + relabel + noisy + unseen partitions the set and
        # the printed statistics add up to N.
        clean = judged & ~relabel & ~noisy

        self.label = torch.where(relabel, pred, self.y)
        self.weight = torch.where(relabel, torch.full_like(trust, self.w_relabel),
                                  torch.where(noisy, torch.full_like(trust, self.w_noise),
                                              torch.ones_like(trust)))
        self.pred = pred
        self.mix = torch.where(relabel, torch.full_like(trust, self.relabel_mix),
                               torch.zeros_like(trust))
        self.stats = {
            'clean': int(clean.sum()), 'relabel': int(relabel.sum()),
            'noisy': int(noisy.sum()), 'unseen': int((~judged).sum()),
            'capped': capped,           # rescued from `noisy` by max_noise_frac
            'vetoed': vetoed,           # demoted by the frozen-CLIP judge
            'mean_trust': float(trust[judged].mean()) if bool(judged.any()) else 0.0,
            # How much of the training signal survives the weighting.  A collapse
            # here means the filtering is throwing the set away, which is worth
            # seeing in the log rather than inferring from the accuracy.
            'mean_weight': float(self.weight.mean()),
        }
        return self.stats

    @torch.no_grad()
    def target(self, idx, y):
        """Soft target ``(B, C)`` for one batch, **without** label smoothing.

        Non-relabelled samples get a one-hot on the given label; a relabelled one
        gets ``(1 - mix) * given + mix * teacher_pick`` -- see the class
        docstring for why the mixture rather than an overwrite.
        """
        c = self.nclass
        t = F.one_hot(y, c).to(torch.float32)
        m = self.mix[idx][:, None]
        if bool((m > 0).any()):
            t = (1 - m) * t + m * F.one_hot(self.pred[idx], c).to(torch.float32)
        return t

    def state_dict(self):
        return {'prob': self.prob, 'seen': self.seen, 'label': self.label, 'weight': self.weight,
                'pred': self.pred, 'mix': self.mix, 'suspect': self.suspect}

    def load_state_dict(self, sd):
        self.prob.copy_(sd['prob'].to(self.device))
        self.seen.copy_(sd['seen'].to(self.device))
        self.label.copy_(sd['label'].to(self.device))
        self.weight.copy_(sd['weight'].to(self.device))
        # tolerating their absence keeps checkpoints written before the mixed
        # target existed loadable
        if 'pred' in sd:
            self.pred.copy_(sd['pred'].to(self.device))
        if 'mix' in sd:
            self.mix.copy_(sd['mix'].to(self.device))
        # ... and a checkpoint written before the judge existed must not be
        # *silently* resumed with the old (self-confirming) rule either: `None`
        # here just means no judge, which is what that checkpoint was trained with
        self.suspect = None if sd.get('suspect') is None else sd['suspect'].to(self.device)


def prototype_bootstrap(sum_feats, counts):
    """Class means of the frozen-CLIP features collected during warm-up.

    Returns ``(means, present)``; prototypes are only used for the classes that
    were actually seen.
    """
    present = counts > 0
    means = sum_feats / counts.clamp_min(1)[:, None]
    return F.normalize(means.float(), dim=-1), present


class FrozenJudge:
    """An independent second opinion on every training sample, from frozen CLIP.

    ``LabelTrustTracker`` judges a sample by the EMA of the *student's* teacher,
    so once the student has memorised a label -- including a wrong one -- the
    teacher agrees with it and :meth:`LabelTrustTracker.refresh` rewards it with
    full weight and a hard target.  On this dataset that is not hypothetical: the
    class folders are named by English search keywords, so folder ``0000/`` holds
    bluebirds *and* a red "bluebird pure Sialia" water heater, and the only thing
    that can contradict a memorised keyword is a model that never saw the labels.

    Frozen CLIP is that model.  It is stored as one frozen feature per sample
    (collected during the warm-up, from the same forward pass the prototype
    bootstrap already needs) and scored against the frozen class centroids: a
    sample is *suspect* when a different class's centroid is closer to it than
    its own, by more than ``margin``.

    Two caveats, both real:

    * frozen CLIP is not right about every sample -- on a 750-class fine-grained
      problem its NCC accuracy is far from 1, so a veto has a false-positive rate
      that must be *measured* (``probe.py`` reports exactly this distribution)
      before the flag is turned on in earnest;
    * the veto therefore only ever **demotes** a sample.  It never relabels, and
      the margin keeps near-ties out of it.
    """

    def __init__(self, n, dim, margin=0.0, device='cpu'):
        self.device = device
        self.margin = margin
        self.feat = torch.zeros(n, dim, device=device)
        self.seen = torch.zeros(n, dtype=torch.bool, device=device)

    @torch.no_grad()
    def add(self, idx, feats):
        """Record the frozen features of one batch.  Later draws overwrite."""
        idx = idx.to(self.device).long()
        self.feat[idx] = feats.detach().float()
        self.seen[idx] = True

    @torch.no_grad()
    def add_all(self, feats, seen=None):
        """Bulk variant for a full-coverage pass (``train.frozen_feature_pass``).

        ``add`` can only fill in the samples the sampler happened to draw during
        warm-up, so the judge silently abstains on the rest -- and the samples it
        abstains on are the rare ones, which are exactly the ones a class-erasing
        filter hurts most.  This takes the whole split at once.
        """
        feats = feats.to(self.device, dtype=self.feat.dtype)
        assert feats.shape == self.feat.shape, (
            f'full-coverage features are {tuple(feats.shape)}, judge holds '
            f'{tuple(self.feat.shape)} -- the pass and the tracker disagree on the split')
        self.feat.copy_(feats)
        self.seen.copy_(torch.ones_like(self.seen) if seen is None
                        else seen.to(self.device).bool())

    @torch.no_grad()
    def judge(self, means, given, present, chunk=8192):
        """``(suspect, top1, margin)`` for every sample.

        ``means``/``present`` come from :func:`prototype_bootstrap`.  A class with
        no warm-up samples has a zero centroid, and a zero centroid has cosine 0
        against everything -- which would make it a *neutral* competitor that
        beats a genuinely anti-correlated real class.  So its similarity is
        forced to -2 rather than left implicit, and a sample whose own class is
        absent is never judged either.
        """
        present = present.to(self.device).bool()
        z = F.normalize(self.feat.float(), dim=-1)
        m = F.normalize(means.float(), dim=-1)
        n = z.shape[0]
        suspect = torch.zeros(n, dtype=torch.bool, device=self.device)
        top1 = torch.zeros(n, dtype=torch.long, device=self.device)
        margins = torch.zeros(n, dtype=torch.float32, device=self.device)
        rows = torch.arange(n, device=self.device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sim = z[s:e] @ m.t()
            sim[:, ~present] = -2.0
            g = given[s:e]
            own = sim.gather(1, g[:, None]).squeeze(1)
            other_sim = sim.clone()
            other_sim[torch.arange(e - s, device=self.device), g] = -2.0
            other = other_sim.max(1).values
            margin = other - own
            margins[s:e] = margin
            top1[s:e] = sim.argmax(1)
            suspect[s:e] = (margin > self.margin) & self.seen[s:e] & present[g]
        return suspect, top1, margins
