# Loss Landscape Degeneracy and Stagewise Development in Transformers

**Citation.** Jesse Hoogland and George Wang (Timaeus), Matthew Farrugia-Roberts (University of Oxford), Liam Carroll (independent), Susan Wei (Monash University) and Daniel Murfet (University of Melbourne). "Loss Landscape Degeneracy and Stagewise Development in Transformers". [arXiv:2402.02364](https://arxiv.org/abs/2402.02364), version 1 of 4 February 2024, last revised as version 3 on 1 August 2025. The arXiv comment says "To appear, TMLR", and our reference gives TMLR 2025 ([OpenReview](https://openreview.net/forum?id=45qJyBG8Oj)). OpenReview showed a browser check, so we didn't check the date of publication. Code is at [github.com/timaeus-research/icl](https://github.com/timaeus-research/icl).

**Mismatch.** None found. Version 1's material on "essential dynamics" moved to a separate paper (arXiv:2501.17745), as the arXiv comment says.

**Status.** Planned. `papers/neuron-development` names the local learning coefficient as a measure that may show when neurons form, and #4156 plans that test.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). In five fetches, it had Sections 1 to 8 and Appendices A to C in the source's own words. The model sizes in Appendix F, and Appendices E and F, came back only as a summary, and this page uses nothing from them.

## Summary

**Claim.** Training a transformer passes through stages. Plateaus in the local learning coefficient (LLC), a measure of how degenerate the loss landscape is near the current weights, mark the boundaries. Each stage matches a change in what the model computes, and some changes don't show in the loss.

**Method.**

- **Models.** A two-layer attention-only language model "with around 3M parameters trained on a subset of the Pile", and an in-context linear-regression transformer of about 50k parameters.
- **The LLC.** A positive number that "measures the amount of degeneracy" of the loss near a minimum. It is estimated by sampling weights near the checkpoint with SGLD and comparing their loss with the checkpoint's.
- **Stages.** Smooth the LLC curve over training, and take its plateaus, the "approximate zeros" of its derivative, as stage boundaries. The authors call the result "approximate developmental stages".
- **Checks.** For each stage, behavioural and structural measures, such as n-gram scores, previous-token heads and induction heads.

**Findings, as the source reports them.**

| Stage of the language model (Figure 1) | Training steps | What forms |
| --- | --- | --- |
| LM1 | 0 to 900 | Bigram statistics |
| LM2 | 900 to 6,500 | Use of position, common n-grams, the start of the induction circuit |
| LM3 | 6,500 to 8,500 | Previous-token heads. The LLC falls |
| LM4 | 8,500 to 17,000 | Induction heads |
| LM5 | 17,000 to 50,000 | "no significant changes" found |

Across five seeds of the language model, every seed shows LM1 to LM4. A Hessian-based measure "fails to detect three of the four stage boundaries" that the LLC finds.

## Where the claims go beyond the evidence

- **The estimator is used outside its definition.** The LLC is defined for local minima of a likelihood. The authors say they apply it "throughout training", where checkpoints aren't minima, and call that "an interesting open problem".
- **Boundaries depend on choices.** The authors say stage boundaries are sensitive to the checkpoint spacing, the smoothing, the time axis and the tolerance.
- **Two small settings.** A 3M-parameter attention-only model and a 50k-parameter regression model. The authors don't claim the stages are universal.
- **Only large changes show.** The authors say their analysis "is not exhaustive", and only "macroscopic" changes may appear as plateaus.

## Use in this repo

- **Our citations check out.** `papers/neuron-development` cites the paper three times.
  - "a small attention-only language model, of about 3 million parameters" matches Sections 1 and 3.
  - Stage boundaries found "with the local learning coefficient … not with the loss" matches the method, which uses plateaus of the LLC curve. The source says such measures "can reveal developments invisible in the loss".
  - "approximate developmental stages" is the source's own phrase.
  - The six authors match. The year and TMLR follow our reference, and the arXiv listing says "to appear".
- **The neuron-development finding fits the motivation.** That paper finds poles changing late in training with no matching feature in the loss. The source shows the same kind of thing for whole circuits. Our reading: whether the LLC shows single neurons' changes is still untested, and #4156 holds that test.
- **Pythia, not an attention-only model.** The source's language model has no MLP, so it has no neurons. The registry's neurons are in Pythia's MLPs. So an LLC study for #4156 would have to repeat the method on a model with MLPs.
- **No question for the maintainer.**
- **Related pages.** [`pythia.md`](pythia.md) reviews the checkpoints that #4156 would use. [`induction-heads-in-context-learning.md`](induction-heads-in-context-learning.md) reviews the induction heads that form in LM4.
