# Task Singular Vectors: Reducing Task Interference in Model Merging

**Citation.** Antonio Andrea Gargiulo and Donato Crisostomi (equal contribution), Maria Sofia Bucarelli, Simone Scardapane, Fabrizio Silvestri and Emanuele Rodolà, of Sapienza University of Rome. The HTML header names only that university. "Task Singular Vectors: Reducing Task Interference in Model Merging". [arXiv:2412.00081](https://arxiv.org/abs/2412.00081), version 3 of 4 April 2025. Versions 1 and 2 are of 26 November 2024 and 2 January 2025. Venue: CVPR 2025, as the arXiv comment states. The code is at [github.com/AntoAndGar/task_singular_vectors](https://github.com/AntoAndGar/task_singular_vectors). Funding: the PRIN 2022 projects AdVVent (20227YET9B) and NEREO (2022AEFHAZ), and FAIR (PE0000013) and SERICS (PE00000014) under Italy's National Recovery and Resilience Plan.

**Mismatch.** The abstract says the compressed task vectors keep "10% of their original size". Section 5.2 says TSV-C needs "approximately twice the size of the original model, regardless of the number of tasks". The paper doesn't reconcile the two. Table 4's bracketed gains also use different baselines in different rows.

**Status.** Planned. #5298 holds the merge rules of the patch format. This paper gives a merge rule and an interference measure computed from weights alone. No code, skill or protocol here depends on it. The fix campaign (#5220) is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in three fetches: Sections 1 to 7, the acknowledgements, the references and Appendices A to E. We didn't see any figure as an image, so the low-rank curves and the per-layer interference appear here only as the text describes them. We didn't read versions 1 and 2, or the code.

## Summary

**Claim.** A task's change to each layer is low-rank, and two tasks interfere where their top singular vectors overlap. So keep a few singular vectors per task, and make the tasks' vectors orthogonal to each other before adding them. That merge beats task arithmetic by a wide margin, with no tuning.

**Method.**

- **Task matrices.** For task $i$ and layer $l$, $\Delta_i = \theta^{(l)}_{\mathrm{ft},i} - \theta^{(l)}_{\mathrm{pre}}$, with SVD $U_i \Sigma_i V_i^\top$. The singular vectors are the task singular vectors. Only 2D matrices are merged this way. Norm layers and other parameters use task arithmetic.
- **Low rank.** Keeping 3% of each task's singular components loses 1.5 points of mean accuracy on ViT-B/32 over 20 tasks (Section 3.2).
- **Interference.** Singular Task Interference, $\mathrm{STI} = \lVert (U^\top U - I)\, \Sigma\, (V^\top V - I) \rVert_1$, with $U$, $\Sigma$ and $V$ the concatenated factors of all tasks. Overlap between different tasks' singular vectors raises it.
- **TSV-Merge.** Keep each task's top $1/T$ of the components, for $T$ tasks. Make the concatenated $U$ and $V$ orthogonal by whitening, solved as an orthogonal Procrustes problem (Proposition 4.1), then add the result with $\alpha = 1$. No validation data and no labels are used.
- **TSV-Compress.** Keep each task's low-rank factors apart, and pick them at test time by the task's index, known or given by a router.
- **Setting.** CLIP ViT-B/32, ViT-B/16 and ViT-L/14, merging 8, 14 and 20 image tasks, with the checkpoints of Wang et al. The baselines' scale is tuned on validation data. TSV-M uses $\alpha = 1$ untuned.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| ViT-B/32, 8, 14 and 20 tasks, mean accuracy (Table 2) | TSV-M 85.86, 80.06, 77.07. Consensus TA 75.03, 70.39, 65.43. Task Arithmetic 70.79, 65.32, 60.52 |
| ViT-L/14, same (Table 2) | TSV-M 92.98, 89.17, 87.72. Consensus TA 86.34, 82.22, 79.00. Task Arithmetic 84.93, 79.41, 74.01 |
| ViT-L/14, 8 tasks, normalised accuracy (Table 2) | TSV-M 96.98, against the single fine-tuned models |
| Ablation, ViT-B/32, normalised, 8, 14 and 20 tasks (Table 4) | Task Arithmetic 76.5, 72.1, 66.8. Interference reduction alone 82.6, 75.7, 69.9. Both steps 92.3, 87.9, 84.3 |
| TSV-C, ViT-B/32, 8 tasks (Table 3) | 92.62 against 92.83 for the fine-tuned models, 99.74% normalised |
| Interference by depth (Section 6.2) | "highest in the initial transformer layers and decreases significantly in the deeper layers" |

## Where the claims go beyond the evidence

- **The link from interference to accuracy is a description, not a measure.** Section 6.4 says lower interference "is closely associated with" higher accuracy. No correlation, and no test over models or layers, is given. The authors say themselves that "a formal definition of interference in model merging is not yet established".
- **Single runs, one model family, one domain.** Every table gives one number per cell, with no seed or interval. All results are CLIP ViTs on image classification. No language model is tested.
- **$\alpha = 1$ and rank $1/T$ are found, not derived.** The authors call the $\alpha$ finding "purely empirical", and the uniform rank is a stated limit. Their baselines have a tuned scale, which is fair to the baselines.
- **The compression claim conflicts with itself.** "10% of their original size" against "twice the size of the original model" is the mismatch above.
- **No units.** The paper studies singular vectors of weight changes. It mentions "shared features in the weight space" and "destructive overlaps in the activations", but measures no activation or neuron.

The authors state limits themselves. The rank is uniform across tasks, and interference has no formal definition yet. And Gram–Schmidt doesn't work for the orthogonalisation.

## Use in this repo

- **STI is a weight-level interference test for patch merges.** It needs only the weights of the parent patches. For two `lowrank_add` patches, the factors are already stored ([`docs/patches.md`](../patches.md)), so STI costs almost nothing. It could flag a merge before any model run. The interference report ([`docs/analyses/interference.md`](../analyses/interference.md)) then tests what STI can't: whether VERIFIED poles and answer routes keep. This page proposes STI as a cheap first check, not a replacement.
- **TSV-M is a third rule for #5298.** Beside `additive`, [TIES](ties-merging.md) and [STF](superpose-task-features.md), TSV-M is the rule built for low-rank changes. It orthogonalises them instead of solving for their coefficients, as STF does, or trimming entries, as TIES does. Its rank $1/T$ comes from whole fine-tunes. A fix patch is already low-rank by design, so the cut may keep all of it.
- **Its depth finding can be checked against units.** The paper finds interference highest in early layers. Our reading, not the paper's: the interference report could test it. Does a merge flip more VERIFIED poles in early layers than in late ones? The registry records each pole's layer, so the report can split its flips by depth.
- **Singular vectors aren't units.** As on the STF page: a task's top singular vectors in an MLP's output matrix are mixes of neurons. Orthogonal singular vectors don't mean that two patches leave each other's VERIFIED poles alone.
- **Related pages.** [`task-arithmetic.md`](task-arithmetic.md) is the baseline. [`superpose-task-features.md`](superpose-task-features.md) is the other SVD-based merge. [`tangent-space-task-arithmetic.md`](tangent-space-task-arithmetic.md) measures disentanglement on outputs, and STI measures overlap on weights. [`lorahub.md`](lorahub.md) composes adapters.
- **Question for the maintainer.** Should a merge in the patch format report STI for its parents before the evidence run, as a weight-only warning beside the interference report? The page records the question and opens no issue.
