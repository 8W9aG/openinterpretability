# Interpretability in Parameter Space: Attribution-based Parameter Decomposition

**Citation.** Dan Braun, Lucius Bushnaq, Stefan Heimersheim, Jake Mendel and Lee Sharkey, of Apollo Research. "Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition". [arXiv:2501.14926](https://arxiv.org/abs/2501.14926), version 1 of 24 January 2025, last revised as version 4 on 7 February 2025. Preprint, no venue given. Code is at [github.com/ApolloResearch/apd](https://github.com/ApolloResearch/apd). The text gives no funding statement.

**Mismatch.** Appendix D.1.2 describes the model with 40 features, but its text says "each of the 5 input features". This looks like a slip from D.1.1.

**Status.** Not applicable. `papers/spd-input-projections` runs SPD, the later method from the same group, not APD. APD's top-k choice, gradient attributions and full-size components appear nowhere in the repo's code.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 4 for this page's author, a model (Claude Opus 5.5). In six fetches it had the whole text in the source's own words, with the appendices and the references. It saw the figure captions but not the figures.

## Summary

**Claim.** A network's parameters can be split into components, each a mechanism, such that few of them are needed on any input. The decomposition with the shortest description recovers the known mechanisms of toy models.

**Method.**

- **Components.** The parameters are split into C components, each the same shape as the whole parameter set. The components may be full rank.
- **Three losses.** *Faithfulness*: the components sum to the parameters. *Minimality*: the top-k components by gradient attribution give nearly the model's output. The top k are chosen across a batch. *Simplicity*: active components have a low Schatten quasi-norm, so a low rank.
- **Models.** Toy models of superposition with 5 and 40 features. A one-layer residual MLP that computes 100 functions with 50 neurons, and the same task spread over two layers. A hand-coded model is in Appendix B.
- **Metrics.** The mean maximum cosine similarity (MMCS) to the true directions, and the mean ratio of their norms (ML2R).

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Toy superposition, 5 features in 2 dimensions (Table 1) | MMCS 0.998 ± 0.000, ML2R 0.893 ± 0.004 |
| Toy superposition, 40 features in 10 dimensions (Table 1) | MMCS 0.996 ± 0.003, ML2R 0.935 ± 0.001 |
| All three toy settings (abstract) | "APD successfully identifies the ground truth mechanisms up to a small error" |

The two residual-MLP results are qualitative, shown in figures only.

## Where the claims go beyond the evidence

- **Toy models only.** The authors say APD "is currently only appropriate for studying toy models because of its computational cost and hyperparameter sensitivity".
- **Heavy tuning.** The authors say "Extensive hyperparameter tuning was often required" to find the right solution. They call the 5-feature model "brittle".
- **No seeds and no runs counted.** Table 1 gives standard deviations, but the paper doesn't say over what. The hand-coded model's figure is one run picked "out of many runs".
- **Baselines argued, not run.** SVD and NMF are ruled out by argument. Sparse autoencoders and transcoders are discussed, not tested. No random or untrained network is decomposed.
- **First-order attributions.** The authors say gradient attributions can fail where gradients are small, such as a saturated softmax.
- **Layers stay privileged.** The authors say that "Layers are therefore still slightly privileged by APD".

## Use in this repo

- **Our citations check out.** `papers/spd-input-projections` says SPD "follows" APD. The SPD paper compares itself with APD throughout ([`stochastic-parameter-decomposition.md`](stochastic-parameter-decomposition.md)). The five authors, 2025 and the arXiv link match. The paper's description of rank-one subcomponents, causal importance and random masks is cited to SPD, not to APD, which is correct.
- **APD's limits are why SPD exists.** SPD replaces the top-k choice and the gradient attributions with a learned importance and random masks, and its components are rank one. Our reading: a finding from `papers/spd-input-projections` is about SPD's losses, and says nothing about APD's.
- **The neuron basis.** APD's introduction calls a neuron-by-neuron description "unnecessarily long", with polysemantic components. `papers/spd-input-projections` tests that view on GPT-2's input projections. There SPD's losses favour the neurons, though a random start doesn't find them. Our reading: APD's argument may not hold at every layer of a real model.
- **No question for the maintainer.**
- **Related pages.** [`stochastic-parameter-decomposition.md`](stochastic-parameter-decomposition.md) reviews the successor that the repo runs. [`toy-models-of-superposition.md`](toy-models-of-superposition.md) reviews the toy model that both papers test.
