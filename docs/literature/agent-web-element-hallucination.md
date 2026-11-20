# Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination

**Citation.** Mohamed Aghzal (George Mason University), Mor Geva (Tel Aviv University), Gregory J. Stein (George Mason University) and Ziyu Yao (George Mason University). "Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination". [arXiv:2611.18514](https://arxiv.org/abs/2611.18514), version 1 of 13 November 2026, the only version. Preprint, no venue given. The paper links no code, and doesn't say that its trajectories or labels are released. Funding: NSF grants 2418580, 2540626, 2232733 and 2018631, the Foresight Institute, and Army Research Lab W911NF2520011. The text gives no disclosure of tool use.

**Mismatch.** The abstract puts the 85% steering result after a sentence about the probe on the last prompt token. But 85% comes from a probe at another position (the start of the selector). Steering with the last-token probe reached only 41.5%. Section 4.2 gives selector agreement as "around 25%" at step 1, and Table 4 gives 28.1%.

**Status.** Planned. It is evidence for #5220. It predicts a failure with a probe and fixes it by steering, two of the routes that the fix campaign weighs. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in four fetches: Sections 1 to 6 and Appendices A to L. We didn't see any figure as an image. So the probe's accuracy by layer, in Figure 4, appears here only as the text describes it.

## Summary

**Claim.** A web agent's hallucinated actions can be predicted before it writes them, and steering along the probe's direction fixes most of them. But the standard tools don't explain how the model decides to hallucinate.

**Method.**

- **Agent and model.** Qwen2.5-7B-Instruct writes one thought and one action at each step, choosing from a listed set of actions. Tasks come from Online-Mind2Web, with at most 10 steps each.
- **Hallucination.** An action that isn't in the list. The data has 300 episodes and 2,071 actions, 32.6% of them hallucinated.
- **Probes.** Linear probes at each of the 28 layers. Each layer has three positions: the last prompt token, the start of the selector, and the start of its value. The split is 70/15/15 by episode, so 356 test actions.
- **Steering.** The probe's weights, scaled by the activations' spread, added at one layer and position with strength α from 10 to 80. "Fix" is the share of hallucinations that become valid, and "break" the share of valid actions that become hallucinations.
- **Explanation tools.** Logit lens, amplifying single components, activation patching, counterfactual prompts, and a pretrained sparse autoencoder at layer 27.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Probe at the last prompt token, layer 18 (Section 4) | test AUC 0.80 |
| Steering, selector start, layer 17 (Table 5) | fix / break: α = 10 43.9 / 0.0, α = 20 75.6 / 3.1, α = 40 85.4 / 7.0, α = 60 48.8 / 30.2, α = 80 0.0 / 96.9 |
| Steering from the last prompt token (Table 5) | best 41.5 fixed, 10.8 broken |
| Same vector, another prompt format (Table 6) | at α = 40, 58.1 fixed and 7.0 broken |
| Best joint change to heads and MLPs (Table 3) | 8.90 fixed, 2.37 broken |
| Activation patching for "Search" (Table 2) | at most 0.360 recovered, from the residual stream at layer 27 |
| Removing the chosen valid action, step 0 (Appendix H) | gives a hallucination in only 38% of cases |
| Sparse autoencoder features (Appendix K) | the top features by class "mostly promote tokens unrelated" to the hallucination |

## Where the claims go beyond the evidence

- **85% is the best cell of a sweep.** It is the best of six positions and five strengths, and the paper doesn't say whether that cell was chosen on development or test data. All the fix rates are multiples of 1/41, which suggests only 41 hallucinations were steered.
- **No random-direction control.** No steering along a random direction of the same norm, and no random baseline for the probe's AUC.
- **No intervals or seeds.** Every number is a single run. Our estimate, not the paper's: if the 85% is 35 of 41, its 95% interval is about 72% to 93%.
- **Breaks are measured, side effects aren't.** The 7% of valid actions that break is reported. Task success and other effects of steering are not.
- **One model, one dataset.** The authors call it a case study.
- **Steering depends on the prompt format.** With the same vector and another list format, the fix rate falls from 85.4% to 58.1%. The authors say this themselves.

The authors state that the results "only provide a partial explanation." Detection and steering work, but how the decision is made stays unexplained.

## Use in this repo

- **It is the same shape as the copy-gate study.** The [copy-gate analysis](../analyses/copy-gate.md) for #5233 asks whether a probe can tell a wrong answer from a right one before the model gives it. It has a random-direction null and a threshold set out of fold. This paper's probe reaches AUC 0.80 with neither. The gate study's controls are the ones this paper lacks.
- **Its break rate is the interference the fix plan measures.** The fix plan (#5220) and the interference report (#5230) score a fix by what it changes where it should do nothing. Here that is the 7% of valid actions broken at α = 40, rising to 30% at α = 60. The steep rise shows that one strength is a narrow window, not a safe range.
- **Paired patching failed here, which bears on the registry's patching.** Removing the chosen action produced a hallucination in only 38% of cases. And the corrupted prompts changed more than the variable of interest. So the authors found no clean pair for activation patching. The registry's causal checks use paired prompts too, so the [activation-patching skill](../../.claude/skills/activation-patching-practice/SKILL.md) applies: a pair must change one thing.
- **A probe direction isn't a unit.** The steering vector is a direction across all of a layer's residual stream. The SAE features at layer 27 weren't readable either. Nothing here names a neuron or head that a registry record could hold. The best joint change to named heads and MLPs fixed only 8.9%.
- **Questions for the maintainer, recorded on this page only.**
  - Should #5220's steering arm report fix and break rates over a range of strengths, with a random-direction control, rather than its best cell?
  - Should a steering fix be scored on prompts in another format, as this paper's Table 6 does?
- **Related pages.** The activation-steering section of the [index](README.md) holds the other steering reviews. [`steering-vector-reliability.md`](steering-vector-reliability.md) and [`does-steering-break-your-model.md`](does-steering-break-your-model.md) report how steering fails.
