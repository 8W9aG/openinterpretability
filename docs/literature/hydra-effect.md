# The Hydra Effect: Emergent Self-repair in Language Model Computations

**Citation.** Thomas McGrath, Matthew Rahtz, Janos Kramár, Vladimir Mikulik and Shane Legg, of Google DeepMind. "The Hydra Effect: Emergent Self-repair in Language Model Computations". [arXiv:2307.15771](https://arxiv.org/abs/2307.15771), version 1 of 28 July 2023, the only version. Preprint, no venue given. The paper links no code and states no funding.

**Mismatch.** None in the citation. Section 2.5.2 calls its own evidence on erasure "largely anecdotal", while the abstract states the erasure function without that caveat.

**Status.** Used. The causal harness freezes the outputs of later MLPs when it patches a unit, to block self-repair (`openinterpretability/harness/patcher.py`, the `hydra_mitigated` field of `core/causal.py`, and `docs/PROTOCOL.md`). `papers/default-on-poles` cites the paper for compensation by later layers.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words, in two fetches: Sections 1 to 6, Appendix A, the references and every figure caption. The figures themselves are images, so no value on this page is read off a plot.

## Summary

**Claim.** When one attention layer of a language model is ablated, a later attention layer takes over part of its function. The authors call this the Hydra effect. Late MLP layers also push down the most likely token, and they push less when an attention layer upstream is ablated. Both happen in a model trained without dropout.

**Method.**

- **Model and task.** Chinchilla 7B, with 32 layers, on the 1,209 factual-recall prompts of Counterfact.
- **Ablation.** Resample ablation, with patches from 15 other prompts. Appendix A compares zero, mean, noise and resample ablation, and prefers resample.
- **Two effects.** The *total effect* lets every later layer respond to the ablation. The *direct effect* holds every other unit at its clean value. The direct effect is read through the unembedding, with the final norm's scale held at its clean value.
- **Compensation.** The change in each later layer's direct effect after an ablation, compared with the ablated layer's own direct effect.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Which layers compensate (abstract, Section 1) | Attention: "another later layer appears to take over its function" |
| Late MLPs (abstract, Section 2.5.2) | They "downregulate the maximum-likelihood token", less so when the promoting attention layer is ablated |
| Layer 23 of 32 (Section 4.2) | The compensatory response explains 92% of the variance in downstream changes. There, most of the response is in the erasure MLPs |
| Slope of the response (Section 4.2) | Below 1 at every layer past 13, so compensation never restores the output in full |
| Middle layers (Section 6) | Compensation and reduced erasure together restore about 70% of the drop in the token's logit |
| Early layers (Section 4.2) | Large total effects, almost no direct effect |
| Dropout (abstract, Section 6) | The model was trained with no dropout, which the authors say rules out dropout as the cause |

## Where the claims go beyond the evidence

- **One model, one task.** All results are on Chinchilla 7B and Counterfact. The authors don't test a model trained with dropout, or another family.
- **No statistics.** The paper gives no seeds, intervals or tests. Each figure shows a single run over the 1,209 prompts.
- **Whole layers only.** The analysis ablates layers, not heads or neurons. The authors say a layer is "likely to be too coarse", and a footnote notes that sometimes one head does the compensating.
- **The erasure story is anecdotal.** Section 2.5.2 says so, and asks for multiple-ablation studies to tell the hypotheses apart.
- **"Natural dropout" is a conjecture.** The authors label it so themselves.

## Use in this repo

- **Our citations check out.** `papers/default-on-poles` cites the paper twice.
  - The quote "another later layer appears to take over its function" is in Section 1, and it concerns attention layers, as the paper says.
  - The quote "downregulate the maximum-likelihood token" is in the abstract, about late MLP layers.
  - The paper marks its step from attention ablations to MLP clamps as its own. The reference's authors, year and title match. No change is needed.
- **The harness blocks the erasure, not the Hydra effect.** `patcher.py` freezes every later MLP's output, and its docstring calls that blocking "the hydra effect". In the paper, the Hydra effect is compensation by later *attention* layers. Frozen MLPs can't respond, so the patch measures more of the unit's effect than an unfrozen one would. But later attention layers can still compensate. Our reading: the measured effect is neither the total effect nor the paper's direct effect, which would hold every other unit at its clean value. The causal scores aren't gated, so no verdict depends on this.
- **The default-on-poles result fits the erasure half.** That paper finds that freezing later MLPs raises an early clamp's KL by 1.38–2.65×. That is what the paper's erasure MLPs predict. The name "Hydra effect" belongs to the attention half, which that test doesn't measure.
- **The same names would help the protocol.** `docs/PROTOCOL.md` says downstream MLPs are frozen "so that they cannot compensate (the hydra effect)". Follow-up #5654 rewords the docstring, the protocol line and the `hydra_mitigated` description, and asks whether to also freeze later attention.
- **Question for the maintainer, recorded on this page only.** Should the causal harness offer a direct-effect mode, holding every later layer at its clean value as in the paper's Definition 3.3?
- **Related pages.** [`copy-suppression-head.md`](copy-suppression-head.md) reviews a head that does a similar suppression. [`ffn-key-value-memories.md`](ffn-key-value-memories.md) reviews what MLP layers write.
