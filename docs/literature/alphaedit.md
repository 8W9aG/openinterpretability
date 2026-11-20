# AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models

**Citation.** Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Shi Jie, Xiang Wang, Xiangnan He and Tat-Seng Chua. The affiliations are the University of Science and Technology of China and the National University of Singapore. The text we read doesn't map them to authors. Xiang Wang and Xiangnan He are the corresponding authors. "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models". [arXiv:2410.02355](https://arxiv.org/abs/2410.02355), version 4 of 22 April 2025. Versions 1, 2 and 3 are of 3 October 2024, 21 October 2024 and 7 March 2025. Venue: ICLR 2025, as an oral, as the arXiv journal-reference field states. The code is at [github.com/jianghoucheng/AlphaEdit](https://github.com/jianghoucheng/AlphaEdit). Funding: the National Science and Technology Major Project (2023ZD0121102) and the National Natural Science Foundation of China (92270114, U24B20180).

**Mismatches.**

- **Author order.** The arXiv listing names the fifth author "Shi Jie". The HTML names "Jie Shi".
- **Headline gains.** The abstract's "36.7%" has no stated basis, and it isn't the 28.24% of the plug-in result. Observation 1's "32.85%" on LLaMA3 and Observation 2's "18.33%" on fluency don't follow from Table 1 by a plain ratio. For example, fluency on GPT2-XL is 597.88 against 530.47, about 12.7%.
- **Appendix C.7.** The text says the best multi-hop results are "9.14 and 9.75". Table 5 has no 9.75, and 9.14 is a CoT value.
- **"GLUE".** The general-ability tests are called six GLUE tasks, but one of them is MMLU.
- **Standard deviations.** One cell has ±0.10 in Table 1 and ±1.21 in Table 6.

**Status.** Planned. The fix plan (#5220) uses MEMIT as the baseline of its editing arm, and AlphaEdit changes MEMIT by one projection. #5167 runs a ROME edit on pythia-70m. No code, skill or protocol here depends on it. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 4 for this page's author, a model (Claude Opus 5.5). It read the source's own text in four chunks by offset. That covers Sections 1 to 7, the ethics, reproducibility and acknowledgement sections, Appendices A to C, and the figure captions. We didn't see any figure as an image. So the general-ability results (Figures 4 and 8) and the shift metrics (Figure 9) appear here only as the text describes them. We didn't read versions 1 to 3, the PDF or the code.

## Summary

**Claim.** Locate-then-edit methods break a model over many edits. The edit's two goals pull against each other: write the new fact, and keep the old ones. AlphaEdit projects the update onto the null space of the preserved knowledge's keys before it is applied. The update then can't move the outputs for that knowledge, and the edit only has to write the new fact.

**Method.**

- **The key-value view.** As in MEMIT, a layer's MLP output matrix $W_{\mathrm{out}}$ is a store. A key $k$ is the MLP's hidden activation for a subject and relation. It maps to a value $v = W_{\mathrm{out}} k$. $K_1$ and $V_1$ hold the new facts' keys and values. $K_0$ and $V_0$ hold the knowledge to keep.
- **MEMIT's update.** $\Delta = (V_1 - W K_1) K_1^\top (K_0 K_0^\top + K_1 K_1^\top)^{-1}$, which balances the new facts against the preserved ones.
- **The projection.** Take the SVD of $K_0 K_0^\top$, a $d_0 \times d_0$ matrix, with $d_0$ the MLP's hidden width. Keep the eigenvectors whose eigenvalues are at most $10^{-2}$ in $\hat U$, and set $P = \hat U \hat U^\top$. Then $\Delta P K_0 \approx 0$: the update can't move the preserved keys' outputs. $K_0$ comes from "100,000 (s,r,o) triplets from Wikipedia", as in MEMIT. $P$ is computed once and reused.
- **AlphaEdit's update.** For sequential editing, $K_p$ holds the keys of earlier edits, and $R = V_1 - W K_1$. The update is $\Delta = R K_1^\top P (K_p K_p^\top P + K_1 K_1^\top P + I)^{-1}$. The authors call this "adding just a single line of code".
- **Setting.** 2,000 facts in batches of 100, so 20 sequential edits. The general-ability tests run to 3,000. LLaMA3-8B (layers 4 to 8), GPT-J (layers 3 to 8) and GPT2-XL (layers 13 to 17), the layers chosen by causal tracing as in MEMIT. Counterfact and ZsRE.
- **Baselines.** ROME, MEMIT, PRUNE, RECT, fine-tuning, MEND and InstructEdit. PRUNE has no public code, so the authors implemented it from its paper.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| LLaMA3-8B, Counterfact, efficacy, generalisation, specificity (Table 1) | AlphaEdit 98.90, 94.22, 67.88. MEMIT 65.65, 64.65, 51.56. RECT 66.05, 63.62, 61.41 |
| LLaMA3-8B, ZsRE, same (Table 1) | AlphaEdit 94.47, 91.13, 32.55. MEMIT 34.62, 31.28, 18.49 |
| GPT-J, Counterfact, same (Table 1) | AlphaEdit 99.75, 96.38, 75.48. MEMIT 98.55, 95.50, 63.64 |
| GPT2-XL, Counterfact, same (Table 1) | AlphaEdit 99.50, 93.95, 66.39. MEMIT 94.70, 85.82, 60.50 |
| Time per 100 edits, LLaMA3-8B (Table 7) | MEMIT 222.51 s, AlphaEdit 223.24 s |
| Memory-based methods, LLaMA3-8B (Table 2) | GRACE has higher Counterfact specificity (72.23) than AlphaEdit, but ZsRE generalisation of 1.03. The text concedes AlphaEdit "does not always surpass the memory-based methods" |
| A tenth of the preserved knowledge (Table 6) | Counterfact efficacy 95.21 and generalisation 90.12 stay close. Specificity falls from 67.88 to 56.12 |
| Hidden states after 2,000 edits, 1,000 prompts (Section 4.4) | Under t-SNE, AlphaEdit's states "remain consistent with the original distribution". The baselines' show "a significant shift" |
| General ability after 3,000 edits (Section 4.3) | AlphaEdit "maintains the original model performance across all metrics". The baselines' metrics are "rapidly approaching zero" by 2,000. Values only in figures |

## Where the claims go beyond the evidence

- **The headline percentages can't be checked from the tables.** "36.7%", "32.85%" and "18.33%" have no stated aggregation, and plain ratios of Table 1 don't give them. Table 1 itself is clear: the gaps over MEMIT on LLaMA3 are large, and on GPT-J they're small.
- **The guarantee covers the sampled keys, not all knowledge.** $P$ protects the span of 100,000 Wikipedia triplets' keys, with directions below an eigenvalue of $10^{-2}$ treated as empty. Knowledge whose keys fall in that near-null space isn't protected. With a tenth of the keys, specificity falls by 11.8 points, so the protection depends on how well $K_0$ covers what the model knows.
- **Spread without seeds.** Every cell has a standard deviation, but the paper doesn't say how many runs give it. It reports no intervals or tests.
- **The general-ability claims are figures only.** The text gives no number for SST, MRPC, CoLA, RTE, NLI or MMLU.
- **t-SNE is a picture of the shift, not a measure.** Appendix C.3 adds overlap and Hausdorff measures, but their values are only in a figure.
- **Model choice.** On Gemma and phi-1.5 (Table 3), AlphaEdit's Counterfact efficacy is 75.21 and 70.79, far below its numbers on the three main models. The text doesn't discuss the gap.
- **No units.** The paper treats the MLP's hidden activation as a key vector. It doesn't ask which neurons the projection leaves free.

The authors state limits themselves: multimodal models and "large reasoning models" are untested.

## Use in this repo

- **The keys are neuron activations.** $K_0$'s rows are the edited layer's MLP neurons, and each column is one prompt's activations. So the null space of $K_0 K_0^\top$ is the set of directions in neuron space that the preserved prompts never use. Our reading, not the paper's claim: AlphaEdit writes each new fact only through combinations of neurons that are near silent on the preserved corpus. Dormant and never-firing poles (#5134) are where such directions live. A VERIFIED pole that fires on common text is in $K_0$'s span, so an AlphaEdit update can't read it.
- **What it guarantees for the registry, and what it doesn't.** The edit changes only $W_{\mathrm{out}}$ of the chosen layers. So every pole at or below those layers keeps its activations exactly, as on the [MEMIT page](memit.md). Above them, the guarantee holds for prompts whose keys sit in $K_0$'s span. The registry's probes and top examples aren't Wikipedia triplets. So the interference report ([`docs/analyses/interference.md`](../analyses/interference.md)) tests whether the guarantee reaches the explanations. Run it on an AlphaEdit edit and on a MEMIT edit of the same facts. The paper's own evidence for "preserved" is Counterfact specificity and t-SNE, both at the level of whole outputs.
- **The fix plan's editing arm.** #5220 names MEMIT as the baseline of arm F. AlphaEdit is MEMIT plus one projection, with about the same cost (223 against 223 seconds per 100 edits). So arm F could carry it as a second baseline at almost no cost. That's the maintainer's choice: this page records it as a question and opens no issue.
- **The patch format can hold it.** An AlphaEdit edit is a dense change to $W_{\mathrm{out}}$, a `dense_add` op ([`docs/patches.md`](../patches.md)). $P$ would belong in the evidence, beside the record of which keys it protects. Without $K_0$, nobody can check what the edit promises to keep.
- **Sequential edits and forgetting.** [`editing-at-scale-forgetting.md`](editing-at-scale-forgetting.md) shows ROME and MEMIT collapse after hundreds of sequential edits. AlphaEdit's $K_p$ term protects earlier edits, and its results stop at 3,000. #5167's rank-one edit on pythia-70m is the registry's first edit. A sequence of edits there, with and without the projection, would show whether the collapse reaches VERIFIED poles above the edited layers.
- **Related pages.** [`locating-editing-factual-associations.md`](locating-editing-factual-associations.md) (ROME) and [`memit.md`](memit.md) are the methods it builds on. [`aging-with-grace.md`](aging-with-grace.md) is GRACE, the memory-based method that beats it on specificity. [`ffn-key-value-memories.md`](ffn-key-value-memories.md) is the key-value view. #4876 reviews "Knowledge in Superposition", which explains why edits interfere in the first place.
- **Question for the maintainer.** Should arm F of #5220 run AlphaEdit beside MEMIT, given that it costs the same and claims to protect preserved keys by construction? The page records the question and opens no issue.
