# Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time

**Citation.** Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith and Ludwig Schmidt. Carmon, Kornblith and Schmidt made an equal contribution. The affiliations are the University of Washington, Columbia University, Google Research (Brain Team), Meta AI Research and Tel Aviv University. "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time". [arXiv:2203.05482](https://arxiv.org/abs/2203.05482), version 3 of 1 July 2022. Version 1 is of 10 March 2022, and version 2 of 21 June 2022. Venue: ICML 2022 (PMLR 162), as the PDF's first page states. The code is at [github.com/mlfoundations/model-soups](https://github.com/mlfoundations/model-soups). Funding: the acknowledgements name the Israeli Science Foundation (grant 2486/21), the Blavatnik Family foundation and the Yandex Initiative for Machine Learning. They also name NSF IFML, Open Philanthropy, NSF and DARPA grants, and gifts from the Allen Institute for AI.

**Mismatch.** Version 3 adds results that came after the ICML submission: BASIC-L in Appendix C, and a comparison with CoCa. The abstract's claim of a new best ImageNet score (90.94%) is the ViT-G result. Footnote 1 gives 90.98% for BASIC-L.

**Status.** Planned. #4782 would use it in its strand on editing, merging and task vectors. #5298 would build merge rules for patches. No code, skill or protocol here depends on it. The fix campaign (#5220) is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the version 3 PDF with `curl` and converted it with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own text the abstract, Sections 1 to 7 and the acknowledgements. We also read Appendices A, C, D, E, F, I, J and L. We read Tables 1 to 5, C.1, J.1 and L.1 to L.3 as extracted text. From Appendices B, G, H and K we read the prose and the figure captions. We did not check the derivation in K.2 and K.3 step by step. We did not see any figure as an image, so this page gives no number that only a figure holds. We did not open the code, and we read the references only to check names.

## Summary

**Claim.** Fine-tuned models that start from one pre-trained model often lie in one low-error basin. So the average of their weights is often more accurate than the best single model. Unlike an ensemble, the average costs nothing extra at inference.

**Method.**

- **Recipes.** Each ingredient θᵢ is a fine-tune of one pre-trained θ₀ with its own hyperparameters. A soup is the plain mean of a subset of them. The *uniform soup* averages all of them.
- **Greedy soup.** Sort the models by held-out accuracy. Add each one in turn, and keep it if held-out accuracy does not fall (Recipe 1). So the greedy soup is never worse than the best single model on the held-out set.
- **Learned soup.** Gradient steps on the held-out set fit one mixing weight per model, or per layer (Appendix I). It needs every model in memory.
- **Held-out set.** For ImageNet, 2% of the training set. The official validation set serves as the test set.
- **Models.** CLIP ViT-B/32 by default. Also ALIGN EfficientNet-L2, a JFT-3B pre-trained ViT-G/14, BASIC-L and CLIP ViT-L. For text, BERT-base and BERT-large, and T5-small, T5-base and T5-large. Fine-tuning changes all parameters.
- **Sweeps.** CLIP: 72 models from a random search over learning rate, weight decay, epochs, label smoothing and augmentation. ALIGN: a grid of 12 models. ViT-G: 58 models. Text: 32 models per task, from a random search over learning rate, batch size, epochs and seed.
- **Data and metrics.** ImageNet top-1, and the mean over five shifted test sets: ImageNet-V2, ImageNet-R, ImageNet-Sketch, ObjectNet and ImageNet-A. Also WILDS, CIFAR-10, and four GLUE tasks (MRPC, RTE, CoLA, SST-2) on their validation sets.
- **Baselines.** The best single model on the held-out set, a logit ensemble and a greedy ensemble. Appendix L adds distillation from an ensemble, EMA, SWA and SAM.
- **Analysis.** Section 4 relates the loss of a two-model soup to that of the logit ensemble (Equation 1). One term depends on the curvature of the loss along the line between the models. The other depends on how confident the soup is.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| ViT-G/14, ImageNet top-1, 58 models (Table 4) | Best on held-out 90.72, greedy ensemble 90.93, greedy soup 90.94. CoAtNet-7's reported 90.88 |
| ViT-G/14, mean of five shifts (Table 4) | Best on held-out 84.38, best on each test set (oracle) 84.68, greedy soup 85.02 |
| ViT-G/14 greedy soup | Keeps 14 of 58 models. Significantly better than the best held-out model on every test set but ObjectNet (exact McNemar or permutation test, α = 0.05) |
| CLIP ViT-B/32, ImageNet and shifts (Table 3) | Best model 80.38 and 47.83. Uniform soup 79.97 and 51.45. Greedy soup 81.03 and 50.75. Ensemble 81.19 and 50.77 |
| Same, other recipes (Table 3) | Greedy soup in random order 80.79 (0.05) and 51.30 (0.16), over three orders. Learned soup by layer 81.37 and 50.87 |
| CLIP and ALIGN greedy soups (Section 3.3.1) | Each keeps 5 models. They beat the best model of the sweep by 0.7 and 0.5 points |
| BASIC-L, ImageNet top-1 (Table C.1) | Best on held-out 90.83, greedy soup 90.98, greedy ensemble 91.02 |
| GLUE, greedy soup minus best single model (Table J.1) | Better on 10 of 20 model-task pairs. Gains run from +0.0 to +2.1 (T5-large RTE, 82.7 to 84.8) |
| GLUE, uniform soup (Table J.1) | Worse than the best single model on all 20. BERT-large: MRPC 15.8 against 88.8, SST-2 50.8 against 92.2. T5-large CoLA 0.00 against 61.7 |
| Cross-dataset soup, zero-shot CIFAR-100 (Appendix E) | Zero-shot CLIP plus six models fine-tuned on other datasets: +6.4 points over the CLIP baseline |
| Calibration (Section 5) | Soups do not improve calibration, unlike ensembles. Test: 20 models that differ only in seed |
| ImageNet-22k pre-trained ViT-B/32 (Section 5, Appendix G) | The greedy soup still helps, but less than with CLIP or ALIGN |

Section 3.2 reports that the gain from averaging two models grows with the angle between θ₁ − θ₀ and θ₂ − θ₀. Varying seed, learning rate or augmentation makes the updates more orthogonal. Figure 3 holds those values, and we did not read them. At a high learning rate (10⁻⁴), error rises on the path between two fine-tunes, and the greedy soup leaves those models out (Appendix J.1).

## Where the claims go beyond the evidence

- **The state of the art is a 0.06-point margin over a reported number.** 90.94% against CoAtNet-7's 90.88% compares one run with a number from another paper. The paper tests the soup against its own models, not against CoAtNet.
- **The ViT-G pool is not a sweep.** Appendix J.2.3 says the authors tuned hyperparameters by hand for the best single model. They then put every later run in the pool. Each run's checkpoint is also chosen on the held-out set, every 1,000 steps.
- **Most comparisons have no interval.** Table 4 and Table C.1 use significance tests. Table 3 gives a spread over three model orders only. The CLIP and ALIGN gains of 0.7 and 0.5 points, and the GLUE gains, come from one sweep each. No sweep is repeated with new seeds.
- **The text gains sit on small sets.** RTE's validation set has 277 examples, so its +0.7 for BERT-base is about two examples. The text we read names no separate held-out split for the GLUE tasks. If the greedy pass also read the validation set, then selection and score share data.
- **Table 5 disagrees with Table J.1.** Table 5 gives T5-base CoLA as 60.2 (+0.4) over a best model of 58.8. That difference is 1.4.
- **"One basin" fails often in the text runs.** A uniform soup is worse than the best model on all 20 model-task pairs, sometimes at chance. The authors suggest the broad learning-rate range as the cause. So the basin premise holds only inside a hyperparameter range found by trial.
- **The basin evidence is mostly one model.** The 2D slices and the angle study use CLIP ViT-B/32 on ImageNet. The approximation in Section 4 rests on 21 pairs, and its fit statement leaves out the 10⁻⁴ learning rate. The text gives a reason for that exclusion in footnote 3.
- **No study of what the average computes.** The paper measures accuracy and calibration only. It does not compare the features or units of a soup with those of its ingredients.

The authors state several limits themselves. Gains are smaller after pre-training on ImageNet-22k. Soups do not fix calibration. The text results are "preliminary". Tied input and output embeddings need care when code averages weights in place.

## Use in this repo

- **Nothing here averages models.** `git grep` finds no soup, weight average or interpolation code. The patch format's only built merge rule is `additive` (`RESOLUTIONS` in `openinterpretability/patches/history.py`). [`docs/patches.md`](../patches.md) describes it, and #5298 tracks TIES and the task-vector rules.
- **An additive merge is a sum, not a soup.** This is our reading of the patch code and of Table 2. Two additive weight ops on one parameter both apply, so the merge gives θ₀ + Δ₁ + Δ₂. A uniform soup of the same two fine-tunes is θ₀ + (Δ₁ + Δ₂)/2. A soup is therefore an additive merge with every delta scaled by 1/k. The patch format can hold one as `dense_add` ops. If #5298 adds a mean rule, the paper's evidence is about that rule, not about `additive`.
- **The paper's ingredients are not fixes.** A soup averages fine-tunes for one task that differ only in hyperparameters. A patch merge joins fixes for different failures (#5220 §6). Appendix E is the only case with different tasks, and there "the choice of which fine-tuned models to include" changes accuracy a lot. So the paper gives no evidence that averaging two fixes keeps both.
- **The greedy soup is a held-out selection rule.** It keeps an ingredient only if the held-out score does not fall. That rule guards one score, not every unit. The interference report in [`docs/analyses/interference.md`](../analyses/interference.md) tests whether a changed model keeps each VERIFIED pole, against two noise floors. The paper has no measure at that level.
- **A whole fine-tune moves nearly every neuron.** On Qwen2.5-1.5B against its instruct model, the footprint holds 250,820 of 250,880 neurons (`interference.md`). 50 of 59 VERIFIED poles keep, and 8 flips count above both floors. A soup's ingredients are whole fine-tunes too. So a footprint test of a soup would likely say only that the change is everywhere.
- **Interpolation gives a cheap test with no training.** WiSE-FT (Appendix D) moves along the line from θ₀ to a fine-tune. The registry already has one such pair, Qwen2.5-1.5B and its instruct model. We assume, as `base-instruct.md` does, that neuron i is the same unit in both. Points between them, θ₀ + α·Δ, test whether the 8 flips appear at once or in proportion to α. The basin picture of Section 3.2 predicts a smooth path for a small enough change. The follow-up issue below asks this.
- **The interpolation null is not the random floor.** The interference report's random change has the instruct change's size but no structure. A halfway point has half the size and the full structure. Compared together, they would split the effect of size from that of direction. #5154 asks the size question for the keep test.
- **Weight distance is not the whole story here either.** Section 3.2 links gain to the angle between updates, a weight statistic. The repo's evidence warns against reading function from weight statistics alone. In [`papers/neuron-development/`](../../papers/neuron-development/README.md), Pythia-70m's 86 VERIFIED poles correlate with their final form at a median of 0.62 at step 16,000. Adjacent checkpoints correlate at a median of 0.78 to 0.82. The loss settled before that step.
- **The tied-weights caution applies to two registry models.** The cached configs give `tie_word_embeddings: true` for Qwen2.5-0.5B and Qwen2.5-1.5B, and `false` for pythia-70m. Any averaging code for the Qwen models must average the shared matrix once.
- **Related pages, not repeated here.** [`lorahub.md`](lorahub.md) shows that a composed adapter has cross terms that no task trained, which a plain mean does not have. [`memit.md`](memit.md) and [`editing-at-scale-forgetting.md`](editing-at-scale-forgetting.md) cover edits that add up over many facts. [`wise-lifelong-editing.md`](wise-lifelong-editing.md) merges edit shards with TIES (it is WISE, not WiSE-FT). [`fine-tuning-distorts-features.md`](fine-tuning-distorts-features.md) is the source of the paper's LP initialization. [`fixing-models.md`](fixing-models.md) sums up the fix reviews. #4872, #4873, #4874, #4877 and #4878 review task arithmetic and its merge rules.
- **Proposed follow-up issue.** "Do the VERIFIED poles that instruction tuning flips come back on the weight average of Qwen2.5-1.5B and its instruct model?" Run `open-interpretability interference` on θ₀ + α·Δ for α in {0.25, 0.5, 0.75}. Record the flips at each α against the 8 at α = 1. Area: research questions. Priority: medium.
- **Question for the maintainer.** Should #5298's merge rules include a mean (a soup) beside `additive` and TIES? The paper supports a mean only for one task's fine-tunes, which a fix merge is not. This page records the question and opens no issue.
