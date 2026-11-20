# TIES-Merging: Resolving Interference When Merging Models

**Citation.** Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel and Mohit Bansal. Yadav, Tam, Raffel and Bansal are at the University of North Carolina at Chapel Hill. Choshen is at IBM Research and MIT. "TIES-Merging: Resolving Interference When Merging Models". [arXiv:2306.01708](https://arxiv.org/abs/2306.01708), version 2 of 27 October 2023. Version 1 is of 2 June 2023. Venue: NeurIPS 2023, as the arXiv comment states. The code is at [github.com/prateeky2806/ties-merging](https://github.com/prateeky2806/ties-merging). Funding: the acknowledgements name NSF-AI Engage Institute DRL211263, NSF-CAREER Award 1846185, DARPA MCS Grant N66001-19-2-4031 and NSF Grant 2145822.

**Mismatches.**

- **Task Arithmetic on T5-Base.** Table 1 gives it 73.2 with no validation, but Table 4 gives 73.9 for the same row. The bracket of −3.2 for TIES on T5-Base with no validation matches neither number.
- **The task count.** Section 6 says T5-Large merges seven tasks, but Appendix C.5 samples subsets "of the 8 tasks".
- **RegMean in Table 3.** Its row repeats the Averaging row in every column.

**Status.** Planned. #5298 names TIES as a merge rule for the patch format, beside the task-vector rules of [`task-arithmetic.md`](task-arithmetic.md). No code, skill or protocol here depends on it. The fix campaign (#5220) is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the source's own text in three chunks by offset: Sections 1 to 8, the acknowledgements, Appendices A to C, and the references. We didn't see any figure as an image. So Figures 4, 6, 9 and 10 appear here only as the text describes them. We didn't read version 1, the PDF or the code.

## Summary

**Claim.** Merging task vectors loses accuracy because of two kinds of interference between parameters: small, redundant changes, and changes whose signs disagree across models. Trimming the small changes, electing one sign per parameter, and averaging only the changes that agree with it reduces both. The result beats earlier merges.

**Method.**

- **Task vector.** $\tau_t = \theta^t_{\mathrm{ft}} - \theta_{\mathrm{init}}$, as in Ilharco et al. (2023).
- **Trim.** Keep the top $k$% of each task vector's entries by magnitude, and set the rest to 0.
- **Elect.** For each parameter $p$, the elected sign is $\gamma^p = \mathrm{sgn}(\sum_t \hat\tau^p_t)$: the sign with the larger total magnitude.
- **Disjoint merge.** For each parameter, average only the trimmed entries whose sign matches the elected one: $\tau^p_m = \frac{1}{|A^p|}\sum_{t \in A^p} \hat\tau^p_t$. "The disjoint mean always ignores the zero values."
- **Apply.** $\theta_m = \theta_{\mathrm{init}} + \lambda \tau_m$. With a validation set, $k$ and $\lambda$ are tuned on it. Without one, the recipe is $k = 20$ and $\lambda = 1$. That recipe was itself chosen on (IA)³ models, not with no data.
- **Models.** (IA)³ adapters on T0-3B (11 tasks). Full fine-tunes of T5-Base and T5-Large (7 tasks). CLIP ViT-B/32 and ViT-L/14 (8 tasks). Also 10 BERT checkpoints of each of three GLUE tasks.
- **Baselines.** Averaging, Task Arithmetic, Fisher merging and RegMean.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| With validation, mean accuracy over tasks (Table 1) | TIES against Task Arithmetic: (IA)³ 66.4 and 63.9, T5-Base 73.9 and 73.2, T5-Large 76.9 and 73.3, ViT-B/32 73.6 and 70.1, ViT-L/14 86.0 and 84.5 |
| Same, the single fine-tuned models and multitask training (Table 1) | Fine-tuned 71.4, 82.8, 88.8, 90.5, 94.2. Multitask 73.1, 83.6, 88.1, 88.9, 93.5 |
| No validation, ViT-B/32 and ViT-L/14 (Table 1) | TIES 72.4 and 86.0. Task Arithmetic at $\lambda = 0.4$: 60.4 and 83.3 |
| Out of domain, 6 held-out T0 tasks, T5-Base and T5-Large (Figure 5) | TIES 35.3 and 40.4. Best other method 34.3 and 36.0 (RegMean) |
| Ablations, T5-Base and (IA)³ (Figure 12) | Full 74.5 and 70.7. Without trim 73.0 and 70.6, without elect 73.1 and 69.6, without the disjoint mean 72.6 and 67.5, without the scale 72.0 and 65.5 |
| Oracle sign, from a multitask model, (IA)³ (Figure 11) | 72.0, against 66.4 with the elected sign and 73.1 for multitask training |
| Sign from 32 examples per task (Table 2) | 67.7 when the sign vector starts from the mean, +1.2 over the elected sign |
| Trimming (Figure 3) | "Keeping only the top-20% of the parameter does not degrade the performance" |
| Sign conflicts (Figure 4, B.3) | "the likelihood of a sign conflict increases with the number of models being merged". With more parameters kept, it "reaches almost 80%" for 10 BERT checkpoints |

## Where the claims go beyond the evidence

- **"Interference" is a word for two parameter statistics.** The paper shows that trimming and sign election raise merged accuracy. It doesn't show that redundant or conflicting parameters are what interferes inside the model. The ablation says the scale and the disjoint mean matter most. Trimming alone adds 0.1 on (IA)³.
- **No intervals.** The main tables have no seeds, standard errors or intervals. Only the sign-flip study (Figure 10) averages three runs. Gains of 0.7 to 3.6 points sit on single runs.
- **The validation-free recipe isn't validation-free in origin.** $k = 20$ and $\lambda = 1$ were chosen on (IA)³ models, then applied to the full fine-tunes. And with no validation on T5-Base, TIES is below Task Arithmetic.
- **The oracle needs what merging tries to avoid.** The +5.6 gain of the oracle sign needs a multitask model trained on all 11 tasks. With 32 examples per task, the gain is 1.2.
- **Inconsistencies.** Three are listed under the citation. The RegMean row of Table 3 in particular can't be trusted without the code.
- **No units.** The paper studies parameters, magnitudes and signs. It measures no activation, feature or neuron.

The authors state limits themselves. Merging "relies on common initialization and model architecture". It "still lags behind the simultaneous multitask training". And "it is not clear how to select the checkpoints for merging".

## Use in this repo

- **TIES is a resolution rule for parameter conflicts.** The patch format refuses a merge while two sides edit the same parameters, unless `--resolution additive` is given ([`docs/patches.md`](../patches.md)). `additive` is $\theta_0 + \Delta_1 + \Delta_2$. TIES would be a second resolution: trim each side, elect a sign per parameter, and average the agreeing entries. `RESOLUTIONS` in `openinterpretability/patches/history.py` lists the built ones, and #5298 holds this one. Low-rank ops must first become full deltas, because trimming and sign election act on single entries of the full delta. [`lorahub.md`](lorahub.md) gives the same reason for its merges.
- **A patch is usually small, so trimming does less.** TIES trims whole fine-tunes, where most of the change is small. A fix patch in #5220 edits a few parameters on purpose (`sparse_add`, `lowrank_add`, `add_neurons`). Its "redundant" entries may be the edit itself. So the default $k = 20$ shouldn't carry over to patches without its own test. The ablation agrees that trimming matters least.
- **Sign conflicts can be read by unit.** The paper counts conflicts per parameter. The registry can group them by neuron: for each VERIFIED pole, the share of its input and output weights whose signs disagree across two patches. Our reading, not the paper's claim: a pole with many conflicts is where the merge most likely breaks its explanation. The interference report ([`docs/analyses/interference.md`](../analyses/interference.md)) would then test that guess on the merged model. The traffic map (#5231) asks the same question from the other side: which units a change at one site disturbs.
- **The disjoint mean is a soup of agreeing entries.** With one task and all signs agreeing, TIES's merge is the mean of [`model-soups.md`](model-soups.md), which #5298 also lists. So one implementation can serve both: a mean with an optional trim and sign election.
- **Disentanglement offers a check, and TIES a fix.** [`tangent-space-task-arithmetic.md`](tangent-space-task-arithmetic.md) measures whether two changes stay apart on each other's inputs. TIES changes the merge so that they interfere less. A merge's evidence could report both: the disentanglement error before and after TIES, and the interference report's keep test.
- **Related pages.** [`task-arithmetic.md`](task-arithmetic.md) is the method that TIES builds on. #4877 and #4878 review later merge rules. [`wise-lifelong-editing.md`](wise-lifelong-editing.md) merges edit shards with TIES.
- **Question for the maintainer.** Should a TIES merge of patches choose $k$ and $\lambda$ in its evidence run, or use the paper's validation-free recipe ($k = 20$, $\lambda = 1$)? The recipe was tuned on whole fine-tunes, not small fixes. This page records the question and opens no issue.
