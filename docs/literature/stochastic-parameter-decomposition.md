# Stochastic Parameter Decomposition

**Citation.** Lucius Bushnaq, Dan Braun and Lee Sharkey, of Goodfire. Bushnaq did most of the work at Apollo Research. "Stochastic Parameter Decomposition". [arXiv:2506.20790](https://arxiv.org/abs/2506.20790), version 1 of 25 June 2025, revised as version 2 on 4 September 2025. Preprint, no venue given. Code is at [github.com/goodfire-ai/spd](https://github.com/goodfire-ai/spd/tree/spd-paper).

**Mismatch.** The importance loss is written as a sum over subcomponents in Equation 7, but the pseudocode averages it over the batch. Both describe the same minimisation.

**Status.** Used. `papers/spd-input-projections` runs SPD on GPT-2 small, and [`docs/analyses/spd-baseline.md`](../analyses/spd-baseline.md) reports the earlier runs. `papers/native-vs-learned` cites it.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words, in three fetches. That covers Sections 1 to 5, the references, Appendices A.1 to A.6 and the pseudocode. We didn't see the figures.

## Summary

**Claim.** A network's weights can be split into rank-one subcomponents, each local to one layer, so that few of them are needed on any input. SPD does this more stably than attribution-based parameter decomposition (APD), and it recovers the known mechanisms of toy models.

**Method.**

- **Subcomponents.** Each matrix is approximated by C rank-one pieces, W ≈ Σ_c U_c V_cᵀ. C can exceed the matrix's rank.
- **Causal importance.** A tiny MLP for each subcomponent reads its own inner activation and outputs g ∈ [0, 1]. A leaky hard sigmoid lets g be exactly 0 or 1.
- **Stochastic masks.** The mask is m = g + (1 − g)·r, with r uniform on [0, 1]. A subcomponent with low importance can then be removed at random.
- **Losses.** Faithfulness asks that the pieces sum to the weights. Stochastic reconstruction asks that the masked model's output stays close, in KL for a language model, and it has a layerwise version. Importance minimality is Σ|g|ᵖ with p > 0. The source sets p for each model, as 1 or 2, and doesn't change it during training.
- **Against APD.** Rank-one pieces replace full-size components, and a learned importance replaces gradient attributions. Random masks replace a top-k choice, and there is no simplicity loss.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Models tested (Section 3) | Toy models only: superposition models (5 features in 2 dimensions, 40 in 10), a residual MLP and 2- and 3-layer models |
| Toy superposition, recovery of the true directions (Table 1) | Mean max cosine 1.000 in all four models. APD's size ratio was about 0.9, and SPD's is 0.99 to 1.03 |
| Hidden identity (Section 3) | "APD fails to learn ground truth mechanisms in this model", and SPD succeeds |
| Three-layer model (Section 3) | APD struggled with more than two layers, and SPD succeeds |

## Where the claims go beyond the evidence

- **Toy models only.** The paper doesn't decompose a language model. It calls scaling to larger models future work that "will likely require further improvements in training stability".
- **Seeds matter.** The authors say "there is sensitivity to random seeds in these results".
- **Rank-one pieces need clustering.** A mechanism spread over several pieces must be grouped afterwards, and in general the right grouping isn't known.
- **Single-layer bias.** The authors note that both APD and SPD favour mechanisms within one layer.

## Use in this repo

- **Our citations check out, with one fix.** `papers/spd-input-projections` (preprint) describes SPD in its Related work and Method.
  - Rank-one subcomponents that sum to the matrix, a causal importance per token, the mask m = g + (1 − g)r, and the three losses all match. So does SPD following APD, and testing only toy models.
  - It said the importance exponent p "falls during training", as part of SPD. The source fixes p. Our runs lower it from 2 to 0.4, which is our choice. This PR rewrites that sentence. No number changes.
  - `papers/native-vs-learned` cites SPD as a parameter decomposition, which matches.
  - The three authors and 2025 match.
- **The GPT-2 runs are the scaling the source leaves open.** `papers/spd-input-projections` decomposes both MLP projections of GPT-2 small. It finds that SPD's losses favour the neurons, but training doesn't find them from a random start. That fits the source's warnings about training stability and seed sensitivity.
- **C equal to the width can't split a neuron.** The source allows C above the rank. The GPT-2 runs use C = 3,072, the MLP width, and `papers/spd-input-projections` already lists that limit.
- **No question for the maintainer.**
- **Related pages.** [`toy-models-of-superposition.md`](toy-models-of-superposition.md) reviews the toy models that SPD is tested on.
