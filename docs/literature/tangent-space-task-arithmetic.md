# Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models

**Citation.** Guillermo Ortiz-Jiménez, Alessandro Favero and Pascal Frossard, all at EPFL, Lausanne. The first two are marked as equal contributors. "Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models". [arXiv:2305.12827](https://arxiv.org/abs/2305.12827), version 3 of 21 November 2023. Version 1 is of 22 May 2023, and version 2 of 30 May 2023. Venue: NeurIPS 2023 (Advances in Neural Information Processing Systems 36), as the arXiv journal-reference field states. The paper's text we read doesn't name the venue. The code is at [github.com/gortizji/tangent_task_arithmetic](https://github.com/gortizji/tangent_task_arithmetic). The paper names no funding. Its acknowledgements thank Gabriel Ilharco for help with the code of [task arithmetic](task-arithmetic.md).

**Mismatch.** The main text calls linearised fine-tuning "an O(1) increase in computational costs". Appendix B gives the factors: inference "around two or three times" slower, and twice the gradient cost and memory.

**Status.** Planned. #5298 holds the task-vector merge rules for the patch format. The paper's disentanglement error is a check that a merge's evidence could report (comment on #5298). No code, skill or protocol here depends on it. The fix campaign (#5220) is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). It read every section in the source's own text, by offset: the abstract, Sections 1 to 8, the acknowledgements and Appendices A to D. About 86 characters near the end of one reference were skipped. We didn't see any figure as an image. So the disentanglement heatmaps, the per-task accuracies and the localisation plots appear here only as the text describes them. We didn't read versions 1 and 2.

## Summary

**Claim.** Task arithmetic works because a pre-trained model is *weight disentangled*: each task vector changes the model's output only on its own task's inputs. It is not because fine-tuning stays in the linear (neural tangent kernel) regime. Fine-tuning the model's linearisation makes it more disentangled, so task arithmetic works better.

**Method.**

- **Task vector.** $\tau_t = \theta^*_t - \theta_0$, as in Ilharco et al. (2023).
- **Weight disentanglement** (Property 3). The model is weight disentangled when $f(x, \theta_0 + \sum_t \alpha_t \tau_t) = \sum_t g_t(x, \alpha_t \tau_t) + g_0(x)$. Here each $g_t$ is zero outside task $t$'s inputs $D_t$, and $g_0$ is zero on all of them. The authors note that "this additive decomposition does not imply linearity".
- **Disentanglement error.** For two tasks, $\xi(\alpha_1, \alpha_2) = \sum_{t} \mathbb{E}_{x \sim \mu_t}[\mathrm{dist}(f(x, \theta_0 + \alpha_t \tau_t), f(x, \theta_0 + \alpha_1 \tau_1 + \alpha_2 \tau_2))]$. On a task's own inputs, it asks how often the other task's vector changes the prediction. For classification, the distance is 1 where the two predicted labels differ. It is measured on a 20 by 20 grid over $[-3, 3]^2$, with 2,048 test points for each dataset.
- **Linearisation.** The first-order Taylor expansion around $\theta_0$: $f_{\mathrm{lin}}(x, \theta) = f(x, \theta_0) + (\theta - \theta_0)^\top \nabla_\theta f(x, \theta_0)$. *Post-hoc linearisation* puts an ordinary task vector into it. *Linearised fine-tuning* trains $f_{\mathrm{lin}}$ itself, which "is essentially the same as training a kernel predictor" with the neural tangent kernel. The authors use Jacobian-vector products (`functorch.jvp`).
- **Models and tasks.** CLIP ViT-B/32, ViT-B/16 and ViT-L/14, with only the image encoder linearised. The eight image tasks of Ilharco et al. (2023). Extras: a CLIP ConvNeXt, an ImageNet-1k supervised ViT-B/16, T5-Base on two text tasks, and randomly initialised ViTs.
- **Protocol.** 2,000 steps, batch 128, learning rate $10^{-5}$, no hyperparameter search. One shared scale $\alpha$ from 0 to 1 in steps of 0.05, chosen on a held-out part of the training set. For negation, the scale must keep at least 95% of the control accuracy.
- **The mechanism.** A kernel predictor is a sum of the kernel's eigenfunctions. If each task's eigenfunctions are supported only on that task's inputs, task arithmetic holds (Proposition 1). The authors measure each input's *local energy*, the summed squares of the eigenfunctions, on 200 training points per dataset.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Task addition, 8 tasks, absolute and normalised accuracy (Table 1) | ViT-B/32: non-linear 71.4 and 76.5, post-hoc linearised 57.1 and 81.9, linearised fine-tuning 76.5 and 85.4 |
| Same, ViT-B/16 and ViT-L/14 (Table 1) | Non-linear 75.5 and 80.0, 85.1 and 88.8. Linearised fine-tuning 81.3 and 86.0, 88.5 and 93.5 |
| Task negation, mean target accuracy and control accuracy (Table 2) | ViT-B/32: non-linear 24.0 and 60.7, linearised fine-tuning 10.9 and 60.8. ViT-L/14: 18.0 and 72.5, 7.9 and 72.5 |
| Single-task accuracy (Section 5) | "significantly lower" for the linearised model than the non-linear one, for every ViT size. The values are only in figures |
| Random initialisation, single-task and multi-task accuracy (Table 3) | Chance is about 5. Non-linear fine-tuning reaches 48.5, 40.6 and 18.0 on single tasks, and 5.5, 4.5 and 4.8 for the sum |
| CLIP ConvNeXt, task addition (Table 4) | Non-linear 79.1 and 83.6, linearised 81.1 and 85.7 |
| ImageNet-1k supervised ViT-B/16, task addition (Table 5) | Non-linear 54.9 and 52.2, linearised 41.8 and 66.1 |
| Cost (Appendix B) | Linearised inference takes about two to three times as long. Gradients take about twice the compute and memory |

The authors conclude that "the sole requirement for task arithmetic is actually weight disentanglement", and that "task arithmetic is a property acquired during pre-training". By the text, the post-hoc linearised models show "more extensive regions with low disentanglement errors" than the non-linear ones. By the text, local energy is "significantly higher" on a task's own training points than on another task's.

## Where the claims go beyond the evidence

- **"Not the linear regime" rests on an accuracy gap.** The evidence that CLIP doesn't fine-tune in the linear regime is that the linearised model's single-task accuracy is lower. No measure of linearisation error or of how far the weights move is given.
- **"Pre-training" is inferred, not isolated.** The contrast is a pre-trained CLIP against random initialisation. No model is trained part way, and only CLIP-style pre-training is tested. The paper first says the origin "is still unclear", and then concludes it is pre-training.
- **Localisation is sufficient, not shown necessary.** The authors state that localised eigenfunctions "are not always necessary". Their necessity result (Proposition 3) holds only under an added independence assumption. The localisation evidence is one pair of datasets in the main text, with no number.
- **The size trend holds for one metric only.** The text says the gain from linearisation "progressively diminishes" with model size. That is true of normalised accuracy (+8.9, +6.0, +4.7 points). In absolute accuracy, ViT-B/16 gains most (+5.8, against +5.1 for ViT-B/32).
- **Single runs.** No table has seeds, intervals or tests. The heatmaps and localisation results have no numbers in the text.
- **Linearisation trades single-task accuracy for arithmetic.** On the supervised ViT, the linearised sum is worse in absolute accuracy (41.8 against 54.9), because each linearised task is weaker. The cost is two to three times the inference.
- **No units.** The paper studies weight directions, input regions and kernel eigenfunctions. Its related work separates weight disentanglement from feature disentanglement, and it measures no feature.

## Use in this repo

- **Weight disentanglement is a keep test, on outputs.** Property 3 asks that a task's change move the model only on that task's inputs. The interference report in [`docs/analyses/interference.md`](../analyses/interference.md) asks the unit-level form. It asks whether a change keeps each VERIFIED pole and answer route, against a self floor and a random floor. The paper's error $\xi$ is the output-level form for two changes together. Neither measure is in the other.
- **A merge check for #5298.** A merge in the patch format must bring its own evidence ([`docs/patches.md`](../patches.md)). $\xi$ fits that evidence directly. On each parent's own evidence prompts, compare the parent alone with the merge. Do it at the merge's scale and over a small grid of scales. A high error means the parents aren't disentangled, so the merge damages one of them. We added this to #5298 as a comment, not as a new issue.
- **The registry's one task vector isn't disentangled at the unit level.** Instruction tuning of Qwen2.5-1.5B moves the activation of 250,820 of its 250,880 neurons beyond the random floor (`interference.md`). That is a whole fine-tune on broad data, so its $D_t$ is nearly every input, and the paper's property doesn't forbid it. A narrow patch is where the property can be tested: its footprint should sit on its own prompts.
- **Localised eigenfunctions and localised units.** The paper's mechanism is that a task lives in kernel eigenfunctions that are large only on that task's inputs. Our reading, not the paper's claim: a VERIFIED pole is a unit that fires on a narrow set of inputs. The paper measures no units. So it gives no evidence that localised eigenfunctions line up with localised neurons. The registry could test this. Pick two patches whose prompts fall in different regions of the shared concept map ([`papers/shared-concept-space/`](../../papers/shared-concept-space/README.md)). Then compare their $\xi$ with that of two patches in one region.
- **Linearised patches aren't possible here yet.** The patch format applies weight ops through forward hooks. A linearised patch needs a Jacobian-vector product at $\theta_0$ on every forward pass, at two to three times the inference cost. No patch op does that, and #5298 doesn't list one. Nothing in the fix plan needs it yet, so this page proposes no issue.
- **Related pages.** [`task-arithmetic.md`](task-arithmetic.md) is the method this paper explains. [`model-soups.md`](model-soups.md) averages fine-tunes of one task. #4874 (TIES), #4877 and #4878 review later merge rules. #3403 decomposes task vectors into components.
- **Question for the maintainer.** Should a merge's evidence in the patch format report the paper's disentanglement error beside the interference report's keep test? It would add one model run per parent per scale. The page records the question and opens no issue.
