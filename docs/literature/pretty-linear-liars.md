# Pretty Linear Liars: Unregularized Probes Favor Intervention-Unfaithful Features

**Citation.** Karl Vilhelmsson Emneby, Julian Yocum and Cameron Allen, of the Center for Human-Compatible AI at UC Berkeley. The text doesn't map each author to an affiliation. "Pretty Linear Liars: Unregularized Probes Favor Intervention-Unfaithful Features". [arXiv:2611.20371](https://arxiv.org/abs/2611.20371), version 1 of 16 November 2026, the only version. The arXiv comment says it was accepted at the NeurIPS 2026 Workshop on Interpretability as a Science. The paper links no code. The text gives no funding statement and no disclosure.

**Mismatch.** Section 4 says the probes are fitted "with no train/test split", but Section 4.1 reports 80/20 held-out results. The abstract calls ridge and DAS directions strikingly similar "in both direction and variance". The angles between them are 26° to 29°, and on Llama-2 their variances differ by about two times.

**Status.** Planned. It bears on the repo's own linear probes, which the [probe comparison](../analyses/probe-comparison.md) fits with a weak L2 penalty. No verdict rests on those probes, so no protocol rule depends on this paper.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in two fetches: Sections 1 to 6, the references and Appendices A to D. We didn't see any figure as an image. Some numbers in the HTML conversion were garbled, and Table 1 was taken from the HTML, not checked against the PDF.

## Summary

**Claim.** A linear probe with no regularization decodes a concept best, but its direction does nothing when the model is intervened on along it. A ridge probe with a well-chosen penalty decodes a little worse. Its direction moves the model much as a direction found by causal search does.

**Method.**

- **Testbeds.** A 4-layer toy transformer that tracks the belief state of a 3-state hidden Markov model. And Llama-2-7B at layer 8, on 8,000 prompts "{name} was born in 19". Its target is the true birth decade.
- **Probes.** Unregularized least squares (OLS) and ridge regression (L2) only. The penalty is chosen from a sweep, as the value with the lowest intervention loss.
- **The intervention test.** Move the activations along the probe's direction until the probe reads a target value. Then measure how far the model's output is from that target. On Llama this is an interchange between two prompts.
- **The comparison.** Distributed alignment search (DAS), which learns a direction by minimising the same intervention loss.
- **Theory.** For a direction of small variance, the OLS weight grows without bound as the variance shrinks. So OLS leans on low-variance directions that the model doesn't use. Ridge caps that weight.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Toy model: probe loss / intervention loss (Table 1) | OLS 9.46e-5 / 0.214. Ridge (λ = 0.1) 3.99e-3 / 1.95e-3. DAS 3.21e-3 / 1.88e-3 |
| Llama-2-7B, same (Table 1) | OLS 0.0671 / 13.7. Ridge (λ = 10⁴) 0.251 / 7.07. DAS 0.689 / 6.42 |
| Llama, OLS intervention (Section 4) | 13.7, the same loss as no intervention at all |
| Angle to the DAS direction (Table 1) | OLS 89.8° and 84.7°. Ridge 28.9° and 26.3° |
| Held out, one 80/20 split (Section 4.1) | Llama OLS loss rises from 0.05 to 0.38. Ridge stays at 0.27 to 0.28 |
| Llama principal components (Appendix C) | 3,976 of 4,096 lie within 3 standard errors of zero covariance with the target |

## Where the claims go beyond the evidence

- **The penalty is chosen on the test.** λ, and the Llama layer, are picked by intervention loss on the same interventions used to report faithfulness. No held-out causal check exists. So "ridge is faithful" is partly selected, not found.
- **Two testbeds, regression only.** One toy model and one 7B model at one layer. Only OLS and ridge, for continuous targets. Logistic, sparse and mean-difference probes aren't tested. The authors call logistic probes a "natural next step."
- **No controls or statistics.** No random-direction baseline, no seeds and no intervals. One train/test split.
- **Ridge leaves much unexplained.** The authors say themselves that even the best ridge probes leave intervention loss "well above zero." On Llama, ridge halves the loss of not intervening (7.07 against 13.7).
- **The theory needs sampling noise.** It explains OLS's failure by noise in sampled data, but the toy model's data is the full set of sequences.

The authors state limits themselves. Failure of OLS doesn't prove ridge faithful. Interventions may move activations off their usual distribution.

## Use in this repo

- **It names a risk in the probe comparison.** The [probe comparison](../analyses/probe-comparison.md) fits logistic probes on standardised features with an L2 penalty of only 1e-3. It finds that their directions barely match the poles: cosine at most 0.07, against at most 0.19 for mass-mean probes. Our reading, not the paper's: a weak penalty lets a probe lean on low-variance directions, which fits that small cosine. The mass-mean probe has no such freedom.
- **The registry's verdicts don't rest on linear probes.** A registry "probe" is a prompt written to test an explanation, and the verdict rests on the unit's own activation and on causal checks. So this paper's failure mode doesn't reach the decision rule. It reaches the analyses that compare a pole with a probe's direction.
- **Sparse probes over neurons are untested here.** The `sparse-probing-search` skill uses k-sparse logistic probes over neurons. A k-sparse probe can only put weight on the k chosen neurons, so it can't spread over thousands of low-variance directions. But the paper doesn't test it, so this is our reasoning, not evidence.
- **It agrees with the gate study's choice.** The [copy-gate analysis](../analyses/copy-gate.md) fits L2-regularised logistic probes and sets thresholds out of fold, with a random-direction null. This paper adds a reason to also report each gate's effect when the model is intervened on along it, not only its decoding.
- **Question for the maintainer, recorded on this page only.** Should the probe comparison sweep its L2 penalty and report the intervention effect of each probe direction, as this paper does?
- **Related pages.** [`linear-probe-evidence.md`](linear-probe-evidence.md) and [`activation-probes-few-positives.md`](activation-probes-few-positives.md) review other probing work. [`neurons-in-a-haystack.md`](neurons-in-a-haystack.md) is the sparse probing that the skill uses.
