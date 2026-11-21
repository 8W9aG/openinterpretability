# Compressed Sensing for Capability Localization in Large Language Models

**Citation.** Anna Bair, Yixuan Even Xu, Mingjie Sun and J. Zico Kolter, Carnegie Mellon University. "Compressed Sensing for Capability Localization in Large Language Models". [arXiv:2603.03335](https://arxiv.org/abs/2603.03335), version 1 of 11 February 2026, revised as version 2 on 29 June 2026. Code is at [github.com/locuslab/llm-components](https://github.com/locuslab/llm-components). The text names no funding.

**Mismatch.** Our issue named ICML 2026 as the venue. Neither the arXiv page nor the text names one, so this page treats it as a preprint. The text's numbers also disagree across tables. For the same model, task and method, Table 2 gives a GSM8K drop of −40.6 ± 6.8, averaged over seeds. Table 4 gives −48.4 from one run.

**Status.** Planned. Research question #5602 would use random subsets and sparse recovery to find an answer's units in `explain`, in place of the greedy search.

**Read.** On 2026-11-21 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the arXiv abstract page, the HTML of version 2 and the code's README. In three fetches it had the whole of version 2 in the source's own words, Sections 1 to 7 and Appendices A to E. Version 1 came back only in part, and this page uses it only for the change log.

## Summary

**Claim.** A small set of attention heads carries a capability, such as math or rhyming. It can be found from a few hundred model runs with random heads knocked out, instead of thousands of one-at-a-time tests.

**Method.**

- **Units.** Attention heads only, zero-ablated. Mean ablation is a robustness check (Appendix B.1).
- **Tests.** M random masks of heads to knock out, 100 to 400 per task. The main design is stratified, so each head is knocked out in about the same number of masks.
- **Signal.** Task accuracy over 100 prompts for each mask, not the output for one input.
- **Recovery.** A Lasso on y = Φx + ε, with Φ the masks. The top 5 heads by coefficient are the result. Interactions between heads are treated as noise: "Modeling head interactions is unnecessary for our use case".
- **Models.** Llama-3.2-1B and 3B, Llama-3.1-8B, and Qwen-2.5-3B, 7B and 14B.

**Findings, as the source reports them.**

| Setting (Llama-3.1-8B, top 5 heads) | Model runs | Drop in task score |
| --- | --- | --- |
| GSM8K, greedy one at a time (Table 4) | 5,120 | −55.4 |
| GSM8K, compressed sensing, stratified (Table 4) | 100 | −48.4 |
| GSM8K, compressed sensing, mean of 3 seeds (Table 2) | 100 | −40.6 ± 6.8, general tasks −1.0 ± 0.8 |
| Five random heads (Table 10) | – | GSM8K −5.99 ± 5.82 |
| Zero against mean ablation, GSM8K heads (Table 9) | – | −40.6 against −36.3 |

The top 5 heads by compressed sensing match those of a one-shot greedy search poorly: recall at 5 is 0.33 to 0.53 (Table 7). The authors note that "all values remain relatively low".

## Where the claims go beyond the evidence

- **Necessary, not sufficient.** The authors say the heads are "necessary, but not sufficient for task performance", and "they do not make up a self-contained circuit".
- **Interactions dropped by design.** A first-order Lasso can't find units that matter only together.
- **Selected on the test prompts.** Hyperparameters are chosen on the same 100 prompts used to find the heads, and the final scores include them.
- **Spread across tables.** The same setting gives −40.6 and −48.4 in two tables, and more model runs give larger drops (−52.4 at 300 runs, Table 6).
- **Few seeds.** Three seeds, with standard deviations and no tests.

## Use in this repo

- **Its search is the cheap half of #5602.** `minimal_set` in `openinterpretability/explain_coverage.py` is greedy, at about two forward passes per unit. Random masks with a Lasso would cost a few hundred passes whatever the number of units. That is the trade #5602 asks about.
- **Knock-out against keep-only.** The paper zeroes the chosen heads and keeps the rest. `explain` mean-ablates every unit except the kept set (`keep_hooks`), so it asks what suffices, not what is needed. The paper's own caveat, "necessary, but not sufficient", marks the gap. Our reading: #5602 should draw random keep-sets, as its plan says, not random knock-out sets.
- **Interactions matter here.** For the twelve-days prompt on pythia-70m, heads alone keep 0.42 of the coverage and poles alone 0.03, but together 0.94. A first-order Lasso would miss much of that. #5602 already plans pairwise terms or SPEX (#5599).
- **One prompt, not a task.** The paper's signal is accuracy over 100 prompts. `explain` works on one prompt and one answer token. Our reading: one prompt's log-probability has no noise from sampling prompts. But it can depend on many interacting units, so M may need to be larger.
- **Heads only.** The paper never knocks out MLP neurons. The registry's units are mostly neuron poles, thousands per model, so N is far larger than the paper's 1,024 heads.
- **No question for the maintainer.**
- **Related pages.** [`attribution-patching.md`](attribution-patching.md) reviews the gradient shortcut that `explain` uses to rank units before testing them.
