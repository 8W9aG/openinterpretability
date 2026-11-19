# Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning

**Citation.** "Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning". arXiv:2601.18699 [cs.LG], version 2, 13 June 2026. [arXiv](https://arxiv.org/abs/2601.18699).

- **Author.** Gustav Olaf Yunus Laitinen-Fredriksson Lundstrom-Imanov, the only author. Version 2 gives his affiliation as the Division of Statistics and Machine Learning (STIMA), Linköping University, Sweden.
- **Versions.** Version 1 is from 26 January 2026 (1,378 KB). Version 2 is from 13 June 2026 (92 KB). They are two different papers under one arXiv ID. The first point of *Where the claims go beyond the evidence* gives the differences.
- **Venue.** Preprint, no venue given. Each page of version 2 has the footer "Preprint submitted to Elsevier".
- **Code and data.** Version 2 says that its code and analysis scripts "are publicly accessible", but it gives no link. Its appendix says that the supplementary material is at "the project's repository", with no link. Version 1 gives `github.com/olaflaitinen/mechanistic-forgetting-analysis`. On 2026-11-19 that address gave HTTP 404.
- **Funding and tools.** Version 2 says that the work had no specific grant. It gives no compute statement. Version 1 thanks DTU Compute for its computational infrastructure.
- **Mismatches.** Version 1 gives the author as "Olaf Yunus Laitinen Imanov", at DTU Compute, Technical University of Denmark. Version 2 gives a different form of the name and a different university. Version 2 has its CRediT statement twice.

**Status.** Not applicable. We can't check the paper's numbers. No code or data is available, and the two versions report different experiments. Open issues #5127 (layer CKA) and #5133 (weight change at a pole) already cover the measures that it would bear on.

**Read.** On 2026-11-19 we downloaded the PDFs of version 2 and version 1 from arXiv with `curl`, and extracted their text with `pdftotext`. A model (claude-opus-5-5) read them, not a person.

- **Version 2, the latest.** We read all 13 pages of the PDF in the source's own text. That is the highlights, the abstract, sections 1 to 9, the statements, Appendix A, the references and the author's biography. Tables 1 to 5 and Algorithm 1 came through as text.
- **Version 2 figures.** We did not look at Figures 1 to 8 as images. The text extraction gives only their axis titles, legends and captions. So this page takes no numbers from the figures.
- **Version 1.** We read its abstract, its introduction, its methods and selected passages of its results in the source's own text. We did not read it in full, and we did not check its figures or all of its numbers.
- **Checks of our own.** We checked the arithmetic of Tables 1 and 2, and computed correlations between their columns. We searched Crossref for five of the cited works, and two searches gave no matching record. The other three searches failed with server errors. We also read the version 1 code link.

## Summary

This summary is of version 2. It describes what the paper says, not what we could check.

**Claim.** When a large language model is fine-tuned on a sequence of tasks, attention heads in early layers become more diffuse. The MLP and expert blocks in the middle layers change their representations most. A method that the paper introduces, Low-Rank Circuit Projection (LRCP), reduces the forgetting at a small training cost.

**Method.**

- **Closed models.** Ten models, read through their APIs. Anthropic: Claude Fable 5, Claude Mythos 5 and Claude Opus 4.8. OpenAI: GPT-5.5 High and GPT-4.5. Google: Gemini 3.5 Flash and Gemini 3.1 Pro. xAI: Grok 4.3 Beta and Grok 3 (Think). Meta: Meta Muse Spark.
- **Open-weight models.** Ten models whose weights the author can read. They are DeepSeek-V4-Pro, Kimi K2.6, GLM-5.1, Qwen 3.5-397B, Llama 4 Maverick, DeepSeek-V4-Flash, DeepSeek-V3.2, Llama 4 Scout, Qwen 3.6-27B and Qwen 3.6-35B-A3B. All except Qwen 3.6-27B are mixture-of-experts (MoE) models.
- **Sequence.** Task A is code refactoring and software engineering, with SWE-bench cited. Task B is mathematics, with MATH and GSM8K cited. Task C is biomedical question answering.
- **Training.** AdamW, with a learning rate given as "a constant learning rate of 2 × 10−5" and also "a cosine decay schedule". The batch size is 128 for all runs.
- **Measures for open-weight models.** Linear CKA between baseline and fine-tuned activations for each layer. *Routing Gate Drift* (RGD), the mean Jensen-Shannon divergence between the routing distributions before and after. The Frobenius drift of each projection matrix. The entropy of attention patterns. Activation patching, which puts the baseline's activations into the fine-tuned model.
- **Measure for closed models.** *Semantic Output Drift* (SOD), one minus the cosine similarity between sentence embeddings of the outputs before and after fine-tuning.
- **Localisation.** Algorithm 1 (RADL) marks each layer whose CKA is below a threshold, or whose RGD is above one. The paper doesn't give the thresholds.
- **LRCP.** It takes the SVD of activations from a fixed calibration set of the old task. Then it projects the gradient onto the orthogonal complement of the top-k singular directions, in the layers that RADL marks.
- **Baselines for LRCP.** Standard fine-tuning, LoRA at rank 16, EWC and replay of 10% of old data, on Llama 4 Scout only, for the sequence A → B.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Closed models, Task A accuracy before and after A → B → C (Table 1) | Losses from 27.7 points (Meta Muse Spark) to 45.9 points (GPT-4.5). GPT-5.5 High: 94.1 → 65.4. Claude Fable 5: 92.4 → 61.2 |
| Closed models, mean SOD over A → C (Table 1) | 0.241 (Meta Muse Spark) to 0.498 (GPT-4.5) |
| Open-weight models, Task A accuracy before and after A → B → C (Table 2) | Losses from 34.3 points (DeepSeek-V4-Pro) to 47.2 points (Qwen 3.6-27B) |
| Open-weight models, mid-layer CKA (Table 2) | 0.412 (Qwen 3.6-27B) to 0.612 (DeepSeek-V4-Pro) |
| MoE models, RGD (Table 2) | 0.314 (DeepSeek-V4-Pro) to 0.512 (Qwen 3.6-35B-A3B) |
| Frobenius drift, units of 10⁻³, all ten open-weight models (Table 3) | Wdown is highest in every model, from 6.89 to 12.89. Wvalue is lowest, from 1.21 to 2.45 |
| Llama 4 Scout after A → B: Task A / Task B accuracy (Table 4) | Standard fine-tuning 35.1 / 76.8. LoRA 58.4 / 70.1. EWC 56.1 / 71.2. Replay 72.1 / 74.3. LRCP 74.8 / 75.1 |
| Llama 4 Scout, training cost and trained share of parameters (Table 4) | LoRA 1.06×, 0.14%. EWC 1.62×, 100%. Replay 1.28×, 100%. LRCP 1.09×, 1.62% |
| LRCP rank ratio k/d from 0.05 to 0.40, Llama 4 Scout (Table 5) | Task A 48.4 → 78.1. Task B 76.4 → 61.4. Epochs to converge 3.1 → 5.8 |
| Activation patching, 500 SWE-bench samples (Figure 3) | Recovery by layer group is in the figure only. The text gives no number |
| Attention entropy, Llama 4 Scout (Figure 5) | Early-layer heads show "the largest increase in entropy". The text gives no number |
| Abstract | LRCP "mitigates up to 94.2% of ancestral capabilities" and "matches the adaptation velocity of standard PEFT baselines" |

The paper says nothing about single neurons, explanations of units, probes or random baselines.

## Where the claims go beyond the evidence

- **The two versions are different studies.** Version 1 studies six models: Llama 4 Scout, Llama 4 Maverick, DeepSeek-V3.1, GPT-5.1, Claude Opus 4.5 and Gemini 2.5 Pro. It uses twelve task sequences of four to six tasks each. It reports three mechanisms, including loss-landscape flattening and a gradient-alignment correlation of r = 0.87. Version 2 has twenty different models, one sequence of three tasks, and LRCP. It doesn't report the loss-landscape or gradient results, and it doesn't say why they went. The issue's description of the paper, "gradient interference in attention weights", matches version 1, not version 2.
- **The paper doesn't say how it fine-tuned the closed models.** Section 4.3 says that all twenty models go through the same AdamW pipeline. It names no fine-tuning service for the closed models. It also doesn't say how it set AdamW and the batch size through an API. The paper describes Claude Mythos 5 as a model for "trusted red-teaming tasks", and gives no route of access.
- **The learning-rate schedule contradicts itself.** Section 4.3 gives a constant learning rate and a cosine decay in one sentence.
- **The SOD definition doesn't match its use.** Equation 3 defines SOD on output token distributions. The text under it embeds "top-k sampled tokens". Section 8.3 uses greedy decoding at temperature 0, with five runs.
- **The abstract's 94.2% is in no table.** LRCP keeps 74.8 of Llama 4 Scout's initial 79.5 on Task A, which is 94.1%. It recovers 39.7 of the 44.4 points that standard fine-tuning loses, which is 89.4%. These are our computations, and neither gives 94.2. The phrase "mitigates up to 94.2% of ancestral capabilities" also has no clear meaning.
- **No result supports the claim about adaptation speed.** Table 5 gives convergence epochs for LRCP only. No table gives them for LoRA or for the other baselines.
- **One number has two settings.** Standard fine-tuning leaves Llama 4 Scout at 35.1 on Task A in Table 4, after A → B. Table 2 gives the same 35.1 after A → B → C.
- **The tables are unusually tight.** Across the ten open-weight models, mid-layer CKA correlates with Task A loss at Pearson r = −0.99. Across the nine MoE models, CKA and RGD correlate at r = −0.998, and their ranks match exactly. Across the ten closed models, SOD and Task A loss correlate at r = 0.996. These are our computations from Tables 1 and 2. They are unusual for ten models of different size, design and training. The paper gives no per-run data to check them.
- **No seeds, intervals or tests.** Version 2 reports one number for each cell. It gives no seeds, no spread and no significance tests. Version 1 says that its error bars come from five seeds with a Bonferroni correction. Version 2 doesn't say this.
- **No null for CKA or drift.** Section 5.2 says that mid-layer CKA of 0.412 to 0.612 is a large drop. It gives no reference, such as CKA between two runs of the same fine-tuning or against a model at initialisation.
- **Claims without an experiment.** Section 6.2 says that freezing the router harms adaptation, and that an L2 penalty on gate logits is "the optimal routing strategy". The paper reports no experiment for either. Section 8.1 explains the middle-layer drift as a "semantic routing hub", with no test.
- **Wide claims from one model.** The LRCP comparison, the rank sweep and the entropy result each use Llama 4 Scout alone. Figure 3 doesn't name the model that it patches.
- **The O(1) claim is not a cost analysis.** Section 7 says that LRCP reduces the SVD cost "to O(1)", because it takes the SVD once. The projection at every step still costs a matrix product.
- **Several references don't support their claims, or may not exist.** The paper cites Sutton and Barto (2018) for MoE routing, and Lopez-Paz and Ranzato (2017) for online SVD in projection methods. Kirkpatrick et al. (2017) has two authors' names misprinted. The Schwarz et al. (2018) entry gives an author "Modifiers" and a venue "Micro-Conference on Machine Learning". Crossref has no record that matches Chame and Demir (2024) or Rostova and Rostova (2024). We did not search for the others.
- **No code.** Version 2 says that its code is public but gives no link. Version 1's link gave HTTP 404 on 2026-11-19.
- **Credit.** Section 8.3 states the limits of API models: version drift and non-deterministic sampling. It also reports the LRCP rank sweep, which shows the cost to the new task as well as the gain on the old one.

## Use in this repo

- **The paper is not evidence for the repo.** Version 2 and version 1 report different experiments under one ID. The code is not available, and the tables are tighter than ten different models usually give. So no analysis or paper in this repo should cite its numbers as a result.
- **It measures layers and matrices, not units.** CKA, RGD and Frobenius drift describe a whole layer or matrix. The registry tests what one pole computes. [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) does this after instruction tuning. On Qwen2.5-0.5B, 252 of 311 VERIFIED hypotheses keep (81%), at median activation correlation r 0.974. A layer can keep a high CKA while a pole in it changes.
- **The middle-layer claim agrees with our data only by direction.** In `base-instruct.md`, 104 of 112 Qwen2.5-0.5B poles keep in the first third of the layers. In the middle third, 63 of 92 keep, and in the last third 85 of 107. The paper also puts the most change in the middle layers. Its evidence for this is not checkable, so we count it as no support. [`anatomy-of-catastrophic-forgetting.md`](anatomy-of-catastrophic-forgetting.md) puts the most change at the top, in small vision networks.
- **The open issues already hold the measures.** #5127 computes layer CKA for Qwen2.5-0.5B and its instruct model, with a random-initialisation null. #5133 reads the change in each pole's Wgate, Wup and Wdown, against a null of shuffled poles. Each has the null that this paper lacks. The paper's finding that Wdown changes most is a claim that #5133 can test on our models. We add no comment to either issue, because this paper adds no checkable evidence.
- **Our drift measure has a reference that the paper's lacks.** [`papers/neuron-development/README.md`](../../papers/neuron-development/README.md) follows Pythia-70m's 86 VERIFIED poles through training. Their median correlation with the final form is 0.62 at step 16,000. Adjacent checkpoints correlate at a median of 0.78 to 0.82. So poles move during ordinary training too. A drift measure must have that reference before a value reads as "collapse". The paper's Figure 2 shows weight paths with t-SNE and gives no numbers.
- **The sister search shows the kind of null that the paper lacks.** [`docs/analyses/sister-neurons.md`](../analyses/sister-neurons.md) compares each match with a randomly rotated basis, at median r 0.08. The paper's CKA, RGD and SOD values have no such comparison.
- **LRCP is an older idea, with weak evidence here.** It projects updates away from the old task's activation subspace, as orthogonal projection methods do. [`orthogonal-gradient-descent.md`](orthogonal-gradient-descent.md) and [`gradient-episodic-memory.md`](gradient-episodic-memory.md) review that family. The paper's evidence for LRCP is one model, one task pair and one run. The campaign to fix models is the maintainer's, so this page records only that this evidence is weak.
- **Activation patching across checkpoints is a sound design, but not shown here.** Putting the old model's activations into the new one, layer by layer, would locate forgetting causally. The `activation-patching-practice` skill covers how to choose the metric and the corruption. The paper gives no number from its patching, so it adds nothing to that skill.
- **Question for the maintainer.** Should the literature index mark a source whose versions report different studies, or whose code is missing? This page records it as a question only.

**Proposed follow-up (not opened).** None. #5127 and #5133 already cover the measures that this paper uses, each with a null.
