# The Local Learning Coefficient: A Singularity-Aware Complexity Measure

**Citation.** Edmund Lau (University of Melbourne), Zach Furman and George Wang (Timaeus), Daniel Murfet (University of Melbourne) and Susan Wei (Monash University). "The Local Learning Coefficient: A Singularity-Aware Complexity Measure". [arXiv:2308.12108](https://arxiv.org/abs/2308.12108), version 1 of 23 August 2023, revised as version 2 on 30 September 2024. Version 2 merges a related paper (arXiv:2402.03698) and adds new results. The arXiv page names no venue, so this page treats it as a preprint. The text we read links no code.

**Mismatch.** [Hoogland et al.](stagewise-development.md) cite this work as "Lau et al. (2025)", which suggests a later publication. We found no venue that shows it.

**Status.** Planned. `papers/neuron-development` names the local learning coefficient as a measure that may show when neurons form, and #4156 plans that test.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read Sections 1 to 7 and Appendices A, B and parts of H in the source's own words. The other appendices came back only as a summary, and this page uses nothing from them.

## Summary

**Claim.** The local learning coefficient (LLC) is a measure of a model's complexity near a set of weights. Unlike the parameter count, it takes account of degeneracy: the many ways to change weights without changing the function. It can be estimated for large networks.

**Method.**

- **Definition.** The LLC λ(w\*) is the exponent with which the volume of near-optimal weights around w\* shrinks as the allowed loss falls. It extends Watanabe's learning coefficient, which equals the real log canonical threshold of singular learning theory. For a regular model it is d/2, half the parameter count.
- **Estimator.** Sample weights from a posterior tied to w\* by a localising term, with stochastic-gradient Langevin dynamics (SGLD), at inverse temperature β\* = 1/log n. The estimate is nβ\* times the mean sampled loss minus the loss at w\*.
- **Tests.** Deep linear networks up to 100M parameters, whose true values are known from theory, and ResNet18 on CIFAR-10 under different optimiser settings.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Deep linear networks (Section 5.1) | The estimator is accurate "in DLNs up to 100M parameters", including at minima found by SGD |
| ResNet18 on CIFAR-10 (Section 5.2) | "stronger implicit regularization = higher test accuracy = lower LLC" |
| Calibration (Section 5.2) | The ResNet values "cannot be calibrated", so only comparisons between them are used |

## Where the claims go beyond the evidence

- **Calibrated only on linear networks.** The estimator is checked against true values only where theory gives them. Elsewhere, only relative values are meaningful.
- **Sensitive to the sampler.** The authors say the step size is "the most important hyperparameter". A step that is too large is unstable, and one that is too small underestimates.
- **Assumptions about real data.** The authors say it is "unclear how reasonable it is to assume" one of the theory's conditions for real data.
- **Minima only, in theory.** The LLC is defined at local minima. Using it on checkpoints during training, as later work does, goes beyond the definition.

## Use in this repo

- **Our citations check out.** `papers/neuron-development` calls the LLC "a measure of the degeneracy of the loss landscape". That matches the definition. It also names the LLC as a measure that may show late changes the loss doesn't show. The paper marks that as an idea for future work. The five authors, 2023 and "Preprint" match the arXiv listing.
- **The estimator is the hard part of #4156.** An LLC study of pythia-70m's checkpoints would need the SGLD step size tuned and checked per checkpoint. Our reading: the source's own advice, to tune for an acceptance rate of 0.9 to 0.95, is the place to start.
- **Away from minima.** Training checkpoints aren't minima, so #4156 would use the LLC as [Hoogland et al.](stagewise-development.md) do, beyond its definition. Both papers say so.
- **No question for the maintainer.**
- **Related pages.** [`stagewise-development.md`](stagewise-development.md) applies the LLC across training.
