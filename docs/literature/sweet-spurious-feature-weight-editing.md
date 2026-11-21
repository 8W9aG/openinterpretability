# SWEET: Mitigating Spurious Feature Reliance in LLM Reasoning via Weight Editing

**Citation.** Zhizhang Fu, Lucas Zhao, Michalis Korakakis, Mengru Ding, Lukasz Szpruch and Yue Zhang (corresponding author). Their affiliations are Westlake University, Yale University, the University of Cambridge, The Alan Turing Institute, Xianyu of Alibaba and the University of Edinburgh. "SWEET: Mitigating Spurious Feature Reliance in LLM Reasoning via Weight Editing". arXiv:2611.10974 [cs.CL], v1, 7 November 2026. [Paper](https://arxiv.org/abs/2611.10974), with an anonymous code link in the introduction. Licence CC BY 4.0. Preprint, no venue given. We found no funding statement. GPT-5.4-mini wrote the distractor clauses, and three LLMs judged the reasoning traces.

**Status.** Planned. #4784 reviews fixes guided by an explanation, and SWEET is a fix by weight edit on selected neurons and heads. No code, skill or protocol here depends on it. The maintainer designs the campaign to fix models (#5220), so this page reports evidence and records questions only.

**Read.** On 2026-11-21 a `web-fetch` subagent (Claude) read the abstract page and the whole HTML of v1 by offset. We have every section, appendix and table in the source's own words. We saw the four figures only as captions, and did not read the code. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** Reasoning models lose accuracy when a math problem gains an irrelevant clause (NoOp, after Mirzadeh et al., 2025). Two directions in the residual stream carry this. One is at the tokens where the model takes up the clause (TL). The other is over the whole trace, and is linked to overthinking (SL). Removing those directions from the outputs of selected MLP neurons and attention heads, once and with no training, closes most of the gap.

**Method.**

- **Directions.** For each layer, a variance-normalised mean difference between NoOp and original traces. It uses 20 Math500 pairs a model, where the model is right on the original and wrong with the clause.
- **Units.** Each MLP neuron's down-projection column and each head's output block. A unit's score is the mean projection of its output onto the direction, so a direct attribution with no patching or gradient. The top 25% of positive-scoring neurons get the TL edit, and every positive-scoring head gets the SL edit.
- **The edit.** W′ = (I − ddᵀ)W for each selected unit: the direction is projected out of its output. The budgets are chosen on the first 100 Math500 items.
- **Models.** DeepSeek-R1-Distill-Qwen 1.5B, 7B and 32B, DeepSeek-R1-Distill-Llama-8B and Qwen3-4B. Greedy decoding, up to 16,384 tokens.
- **Benchmarks.** Math500, AMC, Gaokao-Math, AIME 2024–26 and GSM-Symbolic, each in original and NoOp form.
- **Baselines.** Activation steering, ThinkEdit and Steer2Edit, adapted to the same directions. Four methods against overthinking. No fine-tuning, no ROME or MEMIT, and no edit of random units.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Five models, five benchmarks (Table 4) | NoOp 53.7 → 65.5. Original 65.2 → 67.2. Relative gap 17.6% → 2.5% |
| DeepSeek-R1-Distill-Qwen-32B, average | NoOp 66.4 → 75.7, original 75.4 → 78.1. Steer2Edit: 71.0 and 76.9 |
| DeepSeek-R1-Distill-Llama-8B, average | NoOp 46.1 → 61.5, original 60.0 → 63.3. Steer2Edit: 56.4 and 65.4 |
| Reuse of the distractor's 3-grams in NoOp traces | TL −77.5%, SL −0.6%, both −78.8% |
| Judged over-integration, NoOp, 500 traces, wins against losses | GPT-6-Luna: TL 51.6% against 4.4%. Haiku 5.5: 34.2% against 10.2%. DeepSeek V4.1 Flash: 41.6% against 21.0% |
| Attention from TL positions to the distractor clause | 1.6–3.5 times depth-matched controls, on all five models |
| Against overthinking methods (Table 4) | SWEET NoOp 65.5 and gap 2.5, at −23.3% tokens. Budget forcing 62.4 and 4.4, at −46.5% |
| DeepSeek-R1-Distill-Qwen-32B, AIME, TL alone | NoOp 42.2 → 42.2, original 54.4 → 47.8 |

## Where the claims go beyond the evidence

- **No random control.** No edit of random units, random heads or a random direction is run. Projecting any direction out of 25% of the neurons and every positive head could change accuracy. So the gain is not yet tied to these directions.
- **The units are found, not explained.** A unit is chosen because its output projects onto the direction. No unit is checked on its own, by its activations on text or by ablating it. The "mechanistic evidence" in the abstract is the vocabulary projection of the directions, at 75% depth, and the attention ratio. Section 5 calls the link a hypothesis, and the readouts at 25% and 50% depth are much less clear.
- **One greedy run, no intervals.** Each cell is one run. AIME has 90 questions a model, so one question is 1.1 points, and a standard error near 5 points (our arithmetic).
- **Side effects are measured only on math.** No general-capability benchmark appears. Some cells drop, such as GSM-Symbolic for the Llama model and AIME for the 32B model.
- **The judge result depends on the judge.** The main text gives the GPT-6-Luna numbers, the most favourable of three judges.
- **Budgets follow a shared rule, not each model's best.** Qwen3-4B's own best head budget is 5%, and the shared 100% is used. One table shows the MLP variant above the attention variant for SL, but the paper picks attention on architectural grounds.

The authors state some limits themselves. The work covers only irrelevant clauses in math. Edits "can reduce performance on some model–dataset combinations". The calibration's transfer to other data is "unverified".

## Use in this repo

- **The same unit, a different test.** SWEET edits single MLP neurons, the registry's unit, by their down-projection column. But it chooses them by direct attribution to a direction, with no explanation and no check of what each one does. A VERIFIED pole has a tested explanation. A fix that edits only VERIFIED poles whose explanation names the distractor's role would be the explanation-guided version. That choice is the maintainer's (#5220).
- **The edit fits the patch format.** Projecting a direction out of selected columns of W_down is a small weight patch, the kind that [`docs/patches.md`](../patches.md) records (#5229). Such a patch's interference checks (#5230) would also cover what SWEET leaves out. They read the units it did not mean to change, against a matched random edit as the noise floor.
- **The missing control is the one #5230 already runs.** The interference report compares a change with "a matched random change of the same size". Applied to SWEET's edit, that is the random-unit control the paper lacks.
- **A question for the maintainer.** Should a fix campaign that edits neurons choose them from VERIFIED explanations, or by attribution as SWEET does, with explanations checked afterwards? The first ties the fix to the registry. The second can reach neurons that no one has studied.
- **No new issue.**
