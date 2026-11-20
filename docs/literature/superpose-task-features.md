# Superpose Task-specific Features for Model Merging

**Citation.** Haiquan Qiu and You Wu (Tsinghua University), Dong Li and Jianmin Guo (Huawei), and Quanming Yao. Yao is at Tsinghua University and the State Key Laboratory of Space Network and Communications. "Superpose Task-specific Features for Model Merging". [arXiv:2502.10698](https://arxiv.org/abs/2502.10698), version 2 of 18 September 2025. Version 1 is of 15 February 2025. Venue: EMNLP 2025, as the arXiv comment states. The code is at [github.com/LARS-research/STF](https://github.com/LARS-research/STF). Funding: the National Key R&D Program of China (2023YFB2903904), NSFC (92270106), the Beijing Natural Science Foundation (4242039) and the CCF-Huawei Populus Grove Fund.

**Mismatches.**

- **The system's shape.** Remark 2 says that keeping every feature exactly would be "an overdetermined system". Equation 3 has as many equations as unknowns.
- **An undefined symbol.** Appendix C writes the system with a $\Sigma$ that the text we read doesn't define.
- **The percentages.** "1.4%", "2.2%" and "4.8%" mix absolute points and relative gains. The T5-base "1.4%" is over PCB. The best baseline, Localize-and-Stitch, is 0.8 points behind.

**Status.** Planned. #5298 holds the merge rules of the patch format, beside [task arithmetic](task-arithmetic.md) and [TIES](ties-merging.md). No code, skill or protocol here depends on it. The fix campaign (#5220) is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the HTML in two chunks by offset. We have the source's own text for Sections 1 to 5, the algorithms and theorem, the limitations, the acknowledgements and Appendices A to D. We didn't see any figure as an image, so the feature-preservation plot and the ablation curves appear here only as the text describes them. We didn't read version 1, the reference list or the code.

## Summary

**Claim.** A linear layer's change under fine-tuning, the *task matrix*, carries its task's features as its singular vectors. Merging should keep each task's features: for each one, the merged layer must give the same output along that feature as the task did. That condition is one linear system, solved per layer, with no data.

**Method.**

- **Task matrix and features.** For task $i$ and one linear layer, $M_i = P_i - P_{\mathrm{pre}}$. Its SVD is $M_i = \sum_k \sigma_i^{(k)} u_i^{(k)} v_i^{(k)\top}$. The right singular vectors $v_i^{(k)}$ are the input features. The left ones, scaled by $\sigma_i^{(k)}$, are the output features. Remark 1 calls the right singular vectors "feature detectors".
- **The condition.** The merged matrix $M = \sum_i \sum_k \alpha_i^{(k)} u_i^{(k)} v_i^{(k)\top}$ must satisfy $u_i^{(k)\top} M v_i^{(k)} = \sigma_i^{(k)}$ for every task and feature (Appendix C).
- **The solution.** That gives $(U \circ V)\alpha = \sigma$ (Theorem 1). $U$ and $V$ are the Gram matrices of all tasks' left and right singular vectors, and $\circ$ is the elementwise product. It costs $O(r^3)$ for $r$ features in total. It needs no data, no activations and no covariance.
- **The rest.** Each task matrix is first trimmed to its largest entries ($\eta$ of them). The merged layer is $P_{\mathrm{pre}} + \gamma M$. Biases and embeddings use task arithmetic, and norm layers the mean.
- **Settings.** T5-base on 7 tasks, GPT-2 Medium LoRA adapters on 3 generation sets, Llama-2-7B on 3 domains, and CLIP ViT-B/32 and ViT-L/14 on 8 image tasks. $\eta$ and $\gamma$ are set per setting. For ViT they come from a grid, and the text doesn't say on which data.
- **Baselines.** Averaging, Task Arithmetic, Fisher, RegMean, TIES, PCB, MetaGPT, Knots, Consensus TA and Localize-and-Stitch.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| T5-base, 7 tasks, mean (Table 1) | STF 73.7. Localize-and-Stitch 72.9, PCB 72.3, TIES 72.2, Task Arithmetic 71.6. Fine-tuned models 81.0 |
| ViT-B/32 and ViT-L/14, 8 tasks, mean (Table 6) | STF 77.4 and 87.8. PCB 76.3 and 87.5. TIES 73.6 and 86.0. Task Arithmetic 70.1 and 84.5 |
| Llama-2-7B, CMMLU, GSM8K, HumanEval, mean (Table 3) | STF 36.5, 63.0, 14.0, 37.8. PCB 36.4, 52.3, 16.5, 35.1 |
| T5-base, 6 held-out T0 tasks (Table 5) | STF 38.2. PCB 37.3. Pre-trained 31.1 |
| Time to merge (Table 4) | Llama-2-7B: STF 623 s, TIES 1,280 s, PCB 1,593 s (on CPU). T5-base: STF 127 s, TIES 34 s |
| Singular vectors removed (Figure 5, by the text) | Removing up to 80% of the smallest has "minimal impact". Removing 5% of the largest causes "significant degradation" |

## Where the claims go beyond the evidence

- **"Outperforms on all metrics" is false in its own tables.** STF is below a baseline on four of seven T5 tasks and on HumanEval (14.0 against 16.5). It is also below on some ViT tasks, and on three of the GPT-2 LoRA scores.
- **Small margins, single runs.** The averages lead by 0.3 to 1.1 points on ViT and 0.8 on T5-base, over the best baseline. No table has a seed, an interval or a test.
- **The settings may be tuned on test data.** $\gamma$ for ViT comes from a grid whose data the text doesn't name. The T5 sweep in Section 4.3 is on the reported test set.
- **"Features" means singular vectors.** The paper frames its method by the linear representation hypothesis, and cites superposition and sparse-autoencoder work. But it never shows that a task's singular vectors are features in that sense. The authors say so themselves: STF "does not explicitly identify task-specific features corresponding to semantic meanings".
- **An ablation claims more than it tested.** It says task matrices merge better than fine-tuned matrices "across all architectures", but only T5-base was tested.
- **Unclear scope.** Which matrices were merged in the LoRA, Llama and ViT runs isn't listed. Whether the baseline numbers were rerun or copied isn't said.

The authors state limits themselves. Their features have no semantic reading, and the SVD and the linear solve cost time. And STF uses "a simple scaling approach to manage interference".

## Use in this repo

- **A third merge rule for #5298, and the one that fits low-rank patches.** STF works on each linear layer's change through its SVD. A patch's `lowrank_add` op already holds a change in factored form, $AB^\top$. Its SVD is cheap, so STF suits the patch format's low-rank edits better than [TIES](ties-merging.md). TIES trims and elects signs on single entries of the full delta. Like the other rules, a merge must bring its own evidence ([`docs/patches.md`](../patches.md)). STF's $\gamma$ would be chosen there.
- **Its features aren't the registry's units.** STF keeps each task's top singular directions of a matrix. For an MLP's output matrix, the right singular vectors are directions in neuron space: mixes of neurons, not single neurons. Our reading, not the paper's: STF can keep a task's top directions while moving a VERIFIED pole that lies across several of them. So the interference report ([`docs/analyses/interference.md`](../analyses/interference.md)) is still the test of whether a merge keeps the registry's explanations. STF's own measure, the output along each kept direction, is a weight-level check.
- **Merging a patch onto itself is exact.** With one task, STF's system has the solution $\alpha = 1$, so it returns the task's own change. Two patches with orthogonal singular vectors also merge exactly. This gives a cheap test for an implementation of the rule.
- **The data-free claim fits the patch format.** STF needs only the weights. A merge of two patches could then run without the evidence prompts, but the patch format still asks for evidence on the merged result.
- **Related pages.** [`task-arithmetic.md`](task-arithmetic.md) is the baseline. [`ties-merging.md`](ties-merging.md) and [`tangent-space-task-arithmetic.md`](tangent-space-task-arithmetic.md) give other answers to the same interference. [`lorahub.md`](lorahub.md) composes adapters with learned weights. #4878 reviews Task Singular Vectors, a closely related SVD-based method.
- **Question for the maintainer.** For #5298, should a merge of two `lowrank_add` patches use STF by default, given that it works on their factors directly? Or should a full-delta rule like TIES be the default for every op? This page records the question and opens no issue.
