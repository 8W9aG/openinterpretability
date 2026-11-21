# Decoupled Weight Decay Regularization

**Citation.** Ilya Loshchilov and Frank Hutter, University of Freiburg. "Decoupled Weight Decay Regularization". [arXiv:1711.05101](https://arxiv.org/abs/1711.05101). Versions 1 (14 November 2017) and 2 (14 February 2018) are titled "Fixing Weight Decay Regularization in Adam". Version 3, of 4 January 2019, has the current title. The arXiv comment says "Published as a conference paper at ICLR 2019". OpenReview showed a browser check, so we didn't check the venue there. Code is at [github.com/loshchil/AdamW-and-SGDW](https://github.com/loshchil/AdamW-and-SGDW).

**Mismatch.** None found.

**Status.** Used. `papers/why-neurons-rest` trains its conditions with AdamW, except the two SGD ones (`scripts/train_resting.py`). It cites this paper for the optimiser.

**Read.** On 2026-11-21 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the abstract pages of all three versions and the HTML of version 3. It had the abstract and Sections 1 to 4.3 in the source's own words, with Algorithms 1 and 2 and Propositions 1 to 3. The rest of Section 4, Section 5 and the appendices came back only as a summary, and this page uses nothing from them. The PDF's text didn't extract.

## Summary

**Claim.** L2 regularisation and weight decay are the same for SGD, once rescaled by the learning rate, but not for Adam. Decoupling the decay from the gradient step (AdamW) makes Adam generalise better, and makes the best weight decay largely independent of the learning rate.

**Method.**

- **The update.** Algorithm 2 gives Adam with L2 regularisation and AdamW side by side. AdamW's step is θ_t = θ_{t−1} − η_t(α m̂_t / (√v̂_t + ε) + λθ_{t−1}). The decay λθ_{t−1} is outside the adaptive ratio, scaled by the schedule multiplier η_t but not by α.
- **Theory.** Proposition 1 shows that, for SGD, weight decay λ equals an L2 term with coefficient λ/α. The equivalence fails for adaptive methods.
- **Experiments.** ResNets with 11.6M and 25.6M parameters on CIFAR-10, and ImageNet32x32. Batch size 128, budgets from 100 to 1,800 epochs, fixed, step and cosine schedules.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Default learning rate 0.001, CIFAR-10 and ImageNet32x32 (Section 4.4) | AdamW has a "15% relative improvement in test error compared to Adam" |
| Grid of learning rate and weight decay, CIFAR-10, 100 epochs (Section 4.2, Figure 2) | "AdamW largely decouples weight decay and learning rate" |
| Learning-rate schedules (Figure 1) | "Cosine annealing yields clearly superior results" |

## Where the claims go beyond the evidence

- **Image classification only.** Every experiment in the sections we read is a ResNet on CIFAR-10 or ImageNet32x32.
- **No seeds or intervals.** The sections we read give no number of runs and no error bars. The decoupling claim rests on heat maps.
- **Relative numbers.** The 15% is a relative gain. The absolute error rates are only in the figures.

## Use in this repo

- **Our citations check out.** `papers/why-neurons-rest` says it trains with AdamW, and cites this paper for the optimiser. The two authors, 2019 and ICLR 2019 match the arXiv listing.
- **The step size is Adam's.** `papers/why-neurons-rest` says that under AdamW a parameter moves by about the learning rate times a ratio of gradient moments. Algorithm 2 shows that step, α m̂/√v̂. Our reading: the property comes from Adam itself, and this paper adds only the decay term. The citation is still right, because the paper gives the update that the runs use.
- **With no weight decay, AdamW is Adam.** The baseline runs of `papers/why-neurons-rest` use no weight decay. With λ = 0, Algorithm 2's two variants are the same. So the decoupling matters only to the conditions with weight decay, such as `weight_decay_0.1` and Pythia's setup.
- **Weight decay doesn't set the rest.** That paper finds that weight decay 0.1 leaves the rest close to the baseline (−0.94 against −1.06 at L2, Table 2). So whether the decay is coupled or decoupled can't change its main result much.
- **No question for the maintainer.**
- **Related pages.** [`pythia.md`](pythia.md) reviews the suite whose AdamW settings the `pythia_` conditions copy.
