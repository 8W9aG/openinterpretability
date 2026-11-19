# LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition

**Citation.** Chengsong Huang (Sea AI Lab, Singapore, and Washington University in St. Louis), Qian Liu (Sea AI Lab), Bill Yuchen Lin (Allen Institute for AI), Tianyu Pang, Chao Du and Min Lin (all Sea AI Lab). The first three authors made an equal contribution. "LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition". [arXiv:2307.13269](https://arxiv.org/abs/2307.13269), version 3 of 19 August 2024. Version 1 is of 25 July 2023. Venue: COLM 2024, as the PDF header and the arXiv comment state. The code is at [github.com/sail-sg/lorahub](https://github.com/sail-sg/lorahub), and the trained LoRA modules are at `huggingface.co/lorahub`. The text names no funding.

**Mismatch.** The abstract says that the code is available. The Reproducibility Statement still says that the code "will be" released after acceptance. That statement is older text that version 3 did not change. We did not open the repository.

**Status.** Planned. #4782 would use it in its strand on LoRA and low-rank fine-tuning. #5125 would use its composition rule as one arm, if that issue's fine-tunes use LoRA. No code, skill or protocol here depends on it. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the version 3 PDF (22 pages) with `curl` and converted it with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own text the abstract, Sections 1 to 7, the Reproducibility Statement and Appendices A to I. We read Tables 1 to 9 as extracted text. We did not see Figures 1 to 3 as images. Figure 3 (the effect of the number of modules) came through only as axis labels, so this page gives no number from it. We read the references only to check names. We did not read versions 1 and 2, the COLM proceedings page, the code or the released modules.

## Summary

**Claim.** One can train LoRA modules on many tasks, then compose them for a new task from about five examples. The composition needs no gradients and no new parameters. On Big-Bench Hard (BBH) with FLAN-T5-large, it nearly matches in-context learning, and it uses the zero-shot prompt length at inference.

**Method.**

- **Base models.** FLAN-T5-large for the main runs. Appendix C uses T5-large, which has no instruction tuning. Appendix D uses FLAN-T5-xl.
- **The modules.** LoRA modules of rank 16, built with the Hugging Face PEFT library. Section 3.2 says that LoRA decomposes "the attention weight matrix update". Each module trains for 10 epochs at batch size 64 and learning rate 1e-4.
- **Upstream tasks.** The tasks that trained FLAN-T5, "nearly 200 distinct tasks". The text says the authors "trained several LoRA modules" on them, and gives no exact count.
- **Compose.** Each run picks 20 modules at random. With module i written as Aᵢ Bᵢ, the composed module is m̂ = (Σ wᵢ Aᵢ)(Σ wᵢ Bᵢ) (Equation 1). Each wᵢ is a scalar and can be negative.
- **Adapt.** A gradient-free search sets w. The objective is the cross-entropy on the few-shot examples Q plus α·Σ|wᵢ|, with α = 0.05. The optimiser is Shiwa from Nevergrad, mostly CMA-ES. It runs K = 40 steps from w = 0, with |wᵢ| at most 1.5.
- **Data and metric.** BBH, 27 tasks, scored by exact match. Each run uses 5 examples per task. Table 1 averages three runs, each with different examples.
- **Baselines.** Zero-shot, in-context learning (ICL) on the same 5 examples, and three methods trained by gradient on those 5 examples for 40 epochs. These are full fine-tuning (FFT), LoRA and IA3. The authors report each at the end of training, "without any model selection".

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| BBH mean of 27 tasks, FLAN-T5-large, mean of three runs (Table 1) | Zero 27.0, ICL 37.3, IA3 31.6, LoRA 37.7, FFT 42.1, LoraHub 34.7 |
| Tokens per example (Table 1) | 597.8 for ICL. 111.6 for every other method |
| Best of three runs, same setting (Table 4) | ICL 38.4, IA3 32.1, LoRA 40.9, FFT 46.2, LoraHub 41.2 |
| Single module ranked by few-shot loss (Table 2) | 31.7, against LoraHub mean 34.7 and best 41.2 |
| Dyck Languages (Table 1) | LoraHub 1.0 and ICL 0.9, against LoRA 22.2 and FFT 19.5 |
| T5-large with no instruction tuning, mean of 27 tasks (Table 5) | Rank 4: 16.14. Rank 16: 20.78. Rank 64: 14.76. Zero-shot is 0.0 on all tasks but Dyck Languages (0.67) |
| FLAN-T5-xl, five seeds (Table 6) | Zero 25.8, LoraHub mean 36.5, best 41.3 |
| The 20 lowest-loss modules in place of 20 random ones (Table 7) | 35.4, against 34.7. Logical Deduction with seven objects falls from 36.8 to 22.0 |
| No bound on wᵢ (Table 9) | 33.5, against 34.7. Date Understanding falls from 32.9 to 17.7 |
| The five modules with the largest mean weight, each alone (Tables 3 and 8) | 28.10, 30.78, 28.10, 25.14 and 27.04. Change from FLAN-T5-large: +1.10, +3.78, +1.10, −1.86, +0.04 |
| Memory, after the LoRA-FA setup (Section 5) | About 40GB for FFT, 34GB for LoRA, 5GB for LoraHub |

Section 5 also reports a test on WikiTableQuestions (WTQ). The authors add a WTQ-trained module to the pool, and the search gives it the largest weight. The text says the composed module was "marginally" better than that module alone, and gives no number. Appendix H says that more modules give more variance and a higher best score. It puts the variance down to the choice of modules, not to the optimiser's seed.

## Where the claims go beyond the evidence

- **"Upper bound" is a best-of-three on the test set.** Table 4 takes the largest test score over three runs. Section 4.2 then says the upper bound "can surpass ICL on 18 tasks". A maximum over runs, read on test data, is selection after seeing the data. Table 1's mean, 34.7, is below ICL's 37.3.
- **No intervals or tests.** Table 1 averages three runs and gives no spread. Many per-task gaps are a point or two. The text gives "a notable improvement of 3.1%" over IA3. That is 3.1 points (34.7 against 31.6), with no test.
- **One family of models and one benchmark.** All runs use T5 models and BBH. The modules come from FLAN-T5's own training tasks. So the transfer is within one instruction-tuning mixture.
- **The rank claim for FLAN-T5 has no table.** Section 5 says that rank has "minimal impact" for FLAN-T5. Appendix C gives a table for T5-large only.
- **The skill reading of the weights has no null.** Appendix A reads the top five modules as reading comprehension and reasoning. No shuffled or random-module baseline tests that reading. Table 3's caption says "average weight values", but the text says average absolute weight.
- **The WTQ test gives no numbers.** "Marginal superiority" over the WTQ module has no score, seed count or interval.
- **No measure of other tasks.** The authors test each composed module only on its target task. The paper does not measure what the composition costs on the upstream tasks or on general text.

The authors state some limits themselves. LoraHub falls behind LoRA and FFT, mostly on tasks far from the upstream tasks. ICL is the better choice for a one-time task. More modules widen the variance.

## Use in this repo

- **Nothing here runs it.** The repo has no LoRA, PEFT or adapter code. `git grep` finds `lora` only in `scripts/moe_experts_check.py`, as attention rank settings of a test model. FLAN-T5 is not a registry model. [`fixing-models.md`](fixing-models.md) and [`lora-low-rank-adaptation.md`](lora-low-rank-adaptation.md) note the same gap.
- **A composed module is not the sum of its parts.** This point is our inference from Equation 1. Expanded, m̂ is Σ wᵢ² AᵢBᵢ plus the cross terms wᵢwⱼ AᵢBⱼ for every i ≠ j. No upstream task trained a cross term. A one-module LoraHub applies w²·AB, so a negative w does not flip that module's sign. So "does a pole keep under adapter i, and under adapter j?" does not answer "does it keep under their composition?". Task arithmetic, a plain sum of updates, has no cross terms. #4872 reviews it.
- **The merge question is the same as for one adapter.** As [`lora-low-rank-adaptation.md`](lora-low-rank-adaptation.md) argues, a merged update computes the same function as the adapter beside the weights. Section 3.1 merges the final module into the model. So the composed update is the thing to test, not the act of merging.
- **The paper's modules may leave the registry's units in place.** Section 3.2 places LoRA on "the attention weight matrix". The text does not list the target modules, and we did not open the code. If only attention moves, a pole's own Wgate, Wup and Wdown do not change. A weight-only reading at the pole, as #5133 plans, would then see zero change. [`text-to-lora.md`](text-to-lora.md) makes the same point for T2L's adapters.
- **The base-instruct numbers are for a wider change.** In [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md), 38 of 43 Qwen2.5-1.5B hypotheses keep on the instruct model (88%, 95% CI 76–95%). For Qwen2.5-0.5B, 252 of 311 keep (81%, 76–85%). Inside the chat template, the shares fall to 70% and 72%. That tuning changed the MLP weights too. The repo has no evidence on an attention-only adapter, and none on a composed one.
- **A composition arm for #5125.** If #5125's fine-tunes use LoRA, one arm can test composition. Train two adapters on two narrow tasks, and merge each alone. Then merge their plain sum, and their Equation 1 product. Compare the keep shares as `scripts/base_instruct.py` measures them. If the product breaks poles that neither adapter breaks alone, the cross terms are the cause. This belongs on #5125 as a comment, not a new issue.
- **A middle null for #5154.** #5154 adds a random Δ′ of the instruct change's Frobenius norm, and compares keep shares with 81% and 88%. A composition with random wᵢ gives a second null. It is random in its coefficients, but it lies in the span of task-trained updates. A pole that keeps under isotropic noise and fails under random task updates would point to the task subspace. This also belongs on #5154 as a comment.
- **Weight magnitude is not function, here too.** Appendix A reads the modules with the largest weight as reasoning skills. Each of them alone moves BBH by −1.86 to +3.78 points (Table 8). The repo's evidence agrees that weight statistics alone say little. In [`docs/analyses/weight-kurtosis.md`](../analyses/weight-kurtosis.md), output and input kurtosis score AUCs of 0.51 (0.45–0.57) and 0.54 (0.48–0.60). Both intervals include 0.5. A registry explanation needs probes, and the paper's reading has none.
- **Pre-training shows the same gap between score and units.** In [`papers/neuron-development/`](../../papers/neuron-development/README.md), Pythia-70m's 86 VERIFIED poles correlate with their final form at a median of 0.62 at step 16,000. That is after the loss settles. Adjacent checkpoints correlate at a median of 0.78 to 0.82. LoraHub also moves task scores with about 20 scalars, and says nothing on what those scalars do to units.
- **Evidence for the fix campaign, not a design.** About 20 coefficients, fit on 5 examples, move BBH from 27.0 to 34.7 on FLAN-T5-large. The paper measures no other skill after the change. [`complementary-learning-systems.md`](complementary-learning-systems.md) and [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md) explain why a narrow update needs a measure of what it damages. A fix for the #4781 " days" error would need that measure too.
- **Leave model merging to its own pages.** #4872 (task arithmetic), #4873 (task arithmetic in the tangent space), #4877 and #4878 cover merging and its interference. This page does not restate them.
- **No follow-up issue.** The two tests above belong on #5125 and #5154, which already plan the forward-pass and weight-change reads.
- **Question for the maintainer.** Should #4782 count a composed adapter as its own case, apart from one merged adapter? The cross terms of Equation 1 are a change that no single task trained. The page records this and opens no issue.
