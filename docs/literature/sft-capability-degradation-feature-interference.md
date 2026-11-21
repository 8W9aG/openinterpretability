# Pervasive Capability Degradation in SFT from Feature Interference

**Citation.** Hang Chen (Nanyang Technological University), Jiaying Zhu (The Chinese University of Hong Kong) and Wenya Wang (Nanyang Technological University, corresponding author). "Unifying and Beyond the Theory of Pervasive Capability Degradation in LLM Post-training SFT from Feature Interference". arXiv:2611.05003 [cs.LG], v1 3 November 2026, v2 5 November 2026. [Paper](https://arxiv.org/abs/2611.05003), [code](https://github.com/Zodiark-ch/SFT-Conflict). Preprint, no venue given. We found no funding or tool disclosure in the text we read.

**Status.** Planned. #4782 would use it in its strand on catastrophic interference, and its prediction is one that the interference report (#5230) and #5125 could test. No code, skill or protocol in this repo depends on it. The maintainer designs the campaign to fix models, so this page reports evidence and records questions only.

**Read.** On 2026-11-21 a `web-fetch` subagent (Claude) read the abstract page and the HTML of v2 in five parts. We have the main text, from the abstract to the conclusion, in the source's own words, with Tables 1–5 and the figure captions. The appendices (A–I) came back only as summaries, so the hyperparameters, the 44 benchmarks, the data sizes and Theorem 1 are not checked here. We did not see the figures or read the code. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** Supervised fine-tuning on a narrow task damages general skills through feature interference. Pre-training leaves features in near-orthogonal superposition. Narrow SFT data push correlated features onto a shared direction. A general input then gives "ghost" activations on dormant features, which push the output across decision boundaries and flip predictions. The paper says this one mechanism explains six families of existing mitigations, and it proposes a new one, AEGIS.

**Method.**

- **Theory.** Features in superposition, after Elhage et al. (2022), with interference Σ a_j (f_iᵀf_j). The output layer read as a power (Laguerre–Voronoi) diagram. An "idealized" hypothesis that each block's linear map factors through an overcomplete virtual space. The collapse of correlated features is argued by analogy with toy models.
- **The six paradigms.** Regularisation to the base model, data mixing, parameter-space orthogonality, sparse parameter updates, gradient orthogonality, and freezing localised parameters. One baseline for each: PSFT, SMART, KnOTS, DARE, DiFT and CLUE. CLUE is the authors' own earlier method.
- **AEGIS.** For each token pair, the frozen base model gives a margin and an allowed width. A Huber penalty then limits how far the margins drift on general data during SFT. The loss is CE on the target data, CE on general data, and λ times that penalty.
- **Experiments.** Full fine-tuning of Mistral-7B, Qwen2.5-7B and Gemma-2-9B, and of Qwen2.5 3B–72B for cost. There are 14 target tasks: eight GLUE tasks, GSM8K, three code sets, Evol-Instruct and LogiQA2. General data come from FLAN 2022. General skill (P-Acc) is the mean accuracy on 44 held-out benchmarks.
- **The baseline.** "Vanilla" trains on the target data and the FLAN general data together. So every row already mixes data, and the tables have no row for target-only SFT.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Qwen2.5-7B, GSM8K | Vanilla: accuracy 0.714, P-Acc 0.668. AEGIS: 0.739 and 0.683. Best baselines on P-Acc: SMART 0.676, CLUE 0.669 |
| Gemma-2-9B, GSM8K | Vanilla 0.755 and 0.553. AEGIS 0.801 and 0.584 |
| Mistral-7B, GSM8K | Vanilla 0.375 and 0.391. AEGIS 0.418 and 0.429 |
| Qwen2.5-7B, RTE | Vanilla 0.859 and 0.667. AEGIS 0.893 and 0.678 |
| Qwen2.5-7B, decision-boundary crossings on the held-out benchmarks (10⁶) | Base 18.569. New crossings after SFT, Vanilla against AEGIS: RTE 5.700 against 4.309, GSM8K 2.321 against 1.546, CodeAlpaca 1.941 against 1.370 |
| Cost, Qwen2.5-72B | 474.7 GPU-hours against 361.4. FLOPs 1.27 times Vanilla's |
| Seeds | Five, with "variance across seeds … consistently below 0.03" |

## Where the claims go beyond the evidence

- **No feature is measured.** The feature dictionary is assumed, not estimated. The experiments read accuracies, loss curves and margin crossings, with no SAE latent, neuron or feature direction. So the collapse onto a shared direction and the ghost activations are the theory's account, not an observation. The abstract's "mechanistic interpretability" overstates this.
- **The phase transition is an analogy.** The step from narrow data to collapsed features rests on toy-model results, not on an LLM measurement.
- **Gaps within the stated noise.** The seed variance is "below 0.03", but several gains over the best baseline are smaller. On Qwen2.5-7B GSM8K the gap to SMART is 0.007 on P-Acc. The paper gives no intervals or tests, and does not say whether the cells are means.
- **The baseline already mixes data.** Vanilla trains on general data too, so the tables cannot show how much damage plain SFT does, or how much data mixing alone removes.
- **One of the six baselines is the authors' own.** The paper does not discuss this.
- **The cost claim varies with size.** The abstract says "about 1.3 times". At 3B the time ratio is about 2.1, from Table 4 (our arithmetic).
- **Small inconsistencies.** Table 5's caption counts the top 50% of competing tokens, and the text counts all of them. Equation 36 sums over the target channels where the text says general ones.

The authors state some limits themselves. AEGIS does not reach a lower target loss than Vanilla. Most margin crossings are already in the base model, as "an intrinsic trade-off" of superposition. They call Hypothesis 1 idealized. They leave RL, preference tuning and merging for later.

## Use in this repo

- **A testable prediction about units.** The theory predicts ghost activations: after narrow SFT, a unit fires on inputs that it ignored before. For a VERIFIED pole, that shows as new false positives, with the trigger unchanged. The interference report (#5230, [`docs/analyses/interference.md`](../analyses/interference.md)) re-reads each VERIFIED unit's probes on a changed model, against two noise floors. Its hard negatives are the place where ghost activations would show. This is our reading of the theory, not a claim of the paper.
- **The base-instruct result sets the scale.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) finds that 38 of 43 Qwen2.5-1.5B hypotheses keep on the instruct model, with every pole's activation correlating at 0.953 or more. So broad instruction tuning moved these units little. The paper's mechanism needs narrow SFT, which is what #5125 plans.
- **A neuron is not a feature here.** The theory works in a feature basis in superposition. The registry's units are native neurons. A pole that keeps its explanation could still carry a collapsed feature direction, so a keep is not evidence against the theory.
- **AEGIS is a margin constraint, for #4784.** It limits how far the output margins move on general data. That is a fix at the output layer, with no unit in it. The fix campaign is the maintainer's, so this page only records it.
- **No new issue.** The test belongs in #5125 and #5230, which already re-read units after a change.
- **A question for the maintainer.** Should #5125's narrow fine-tunes record each VERIFIED pole's hard-negative activations before and after, as a direct test of the ghost-activation prediction? The interference report already measures them, so this is a choice of which result to report.
