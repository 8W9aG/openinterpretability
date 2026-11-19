# Spurious Forgetting in Continual Learning of Language Models

**Citation.** Junhao Zheng, Xidi Cai, Shengjie Qiu and Qianli Ma (School of Computer Science and Engineering, South China University of Technology). "Spurious Forgetting in Continual Learning of Language Models". arXiv 2501.13453, version 1 of 23 January 2025, the only version. [Paper](https://arxiv.org/abs/2501.13453).

- **Venue.** The PDF header says "Published as a conference paper at ICLR 2025", and the arXiv comment says "ICLR2025". We did not check the proceedings page.
- **Code.** [github.com/zzz47zzz/spurious-forgetting](https://github.com/zzz47zzz/spurious-forgetting), which we did not open.
- **Mismatch about the code.** The abstract says the code "is publicly available". The reproducibility statement (Appendix J.3) says the code and the Biography dataset "will be made publicly available".
- **Funding and tools.** The paper gives no funding statement and no acknowledgements. GPT-4o wrote the sentence templates of the synthetic dataset (Appendix B.1.2).

**Status.** Planned. Open #5126 asks whether the chat template or the tuning causes the fall in `docs/analyses/base-instruct.md`. That is this paper's split of a loss into format and knowledge, at the grain of single poles. Open #5125 (a sequence of narrow fine-tunes) could add the paper's recovery test. This page gives evidence only, because the fix campaign is the maintainer's.

**Read.** On 2026-11-19 a model (claude-opus-5-5) read the paper for this page, not a person. It downloaded the arXiv v1 PDF with `curl`, made text of it with `pdftotext` and read the text directly. It read these parts in the source's own words: the abstract, Sections 1 to 7, and Appendices A to J. That includes the proofs of Appendix F and every table, Tables 1 to 36. It read the reference list. The figures came through as broken text, so it saw their captions but not their curves. No number below comes from a plot, except where the paper's own text states the number. It did not open the code or the OpenReview page.

## Summary

**Claim.** When a language model learns a new task, its score on an old task can fall sharply while the knowledge behind that task stays. The authors call this *spurious forgetting*. They write task performance as "task alignment" plus "underlying knowledge", and say that the early steps of a new task undo the old alignment. They trace the undoing to near-orthogonal weight updates in the bottom layers. Freezing the bottom layers, a method that they call *Freeze*, reduces the loss in their tests.

**Method.**

- **Synthetic setup (Section 3, Appendices B to D).** The Biography dataset of Allen-Zhu and Li (2024) has 200,000 synthetic people with six attributes each. A model with random initial weights pretrains on biographies of 100,000 people, for 80,000 steps at learning rate 1e-3.
  - *Task 0* fine-tunes on question-and-answer (QA) pairs of people that it saw in pretraining. So it learns the QA format and no new facts. Section 3.1 says these are "the same individuals". Appendix C says the first 50,000.
  - *Task 1* fine-tunes on the QA pairs of 20,000 new people, so it learns new facts and a format at once.
  - Each fine-tune runs 62,500 steps at learning rate 5e-6 with AdamW, on all parameters.
  - The paper does not name the architecture. Appendix G.3 gives a model width of 768, an MLP width of 3,072, and GPT-NeoX names for the matrices. Figure 7a freezes up to 9 layers.
- **Recovery test (Section 3.2, Appendix E).** For a checkpoint, fine-tune for one epoch on half of the Task 0 data, and test on the other half. The two halves share no people, so a high score means the knowledge was still in the model. Appendix E gives a formal definition: the old loss rises, but some function of the model brings it back with no shared knowledge.
- **Four views of the first steps of Task 1 (Sections 3.3 to 3.5).**
  - *Loss landscape*: test loss in the plane of two update directions, the first 150 steps and the rest.
  - *Weight updates*: the mean angle between the column or row spaces of two updates, from SVD at 99% of variance.
  - *Features*: hidden states projected on the mean shift and on the leading principal component, layer by layer.
- **Theory (Section 4, Appendix F).** A stack of L linear layers with residual connections, X_l = (W_l + I)X_(l−1). The weights are small (‖W_l‖ ≤ δ). Each update ΔW_l lies in the left null space of W_l (W_lᵀΔW_l = 0). Under these assumptions the shift of a layer's output is nearly orthogonal to its principal component, and the final shift is at most Lε(1 + δ)^(L−1)‖X_0‖.
- **Baselines on Biography (Section 5.1, Appendix H).** Sequential fine-tuning (SEQ) as the lower bound, and replay of 20% or 50% of the old data. Also EWC, LAMOL, Task Vector and Gradient Projection, each with a grid of hyperparameters.
- **Four real scenarios (Section 5.3, Appendix I).** Each has three seeds.
  - *Safety alignment*: LLaMa-2-7B-Chat, fine-tuned on the 10 "identity shifting" examples of Qi et al. (2024). The measure is the jailbreak rate on AdvBench.
  - *Continual instruction tuning*: LLaMa-3-8B-Instruct on the 8 TRACE tasks in one order.
  - *Continual knowledge editing*: LLaMa-3-8B-Instruct on zsRE in 10 tasks of 1,000 records. The measures are efficacy and paraphrase.
  - *Instance incremental learning*: Pythia-410M on Concept-1K in 10 tasks, with memorisation and generalisation accuracy.
- **Supervised fine-tuning (Appendix I.5).** LLaMa-3-8B-Instruct, Qwen2.5-7B-Instruct and Mistral-8B-Instruct-2410 learn math (MetaMathQA) or code (Magicoder) for one epoch. The bottom layer is frozen or not. Five general benchmarks measure what the models keep.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Biography, Task 0 score in the first 150 steps of Task 1 (Section 3.2) | From nearly 100% to about 10%, then up to about 20% |
| Biography, recovered Task 0 score over the same steps (Section 3.2) | Nearly 100% for the first 150 steps, and 96% at the end of Task 1 |
| Biography, update angles (Section 3.4) | The first 150 steps of Task 1 update in a space close to Task 0's. Later steps update in a different space, most of all in the bottom layers and the input embedding |
| Biography, Task 0 / Task 1 accuracy, SEQ (Table 1) | 11.18±.16 / 99.91±.05 |
| The same, EWC at λ = 1e6 and LAMOL at λ = 0.10 (Table 1) | 13.48 / 99.88, and 18.91 / 99.87 |
| The same, Task Vector, best average and best with Task 1 above 99% (Table 1, Section 5.1) | 30.75 / 95.76, and 22.60 / 99.41 |
| The same, Gradient Projection on attention layers and on all layers (Table 1) | 13.34 / 99.88, and 9.52 / 99.94 |
| The same, Freeze 8 layers, and 7 layers with early stopping (Table 1) | 39.68 / 99.91, and 44.22 / 99.93 |
| The same, replay of 20% and 50% of the old data (Table 1) | 76.93 / 99.87, and 80.62 / 99.88 |
| Mean cosine of the undo direction over 10 trials (Section 5.1) | 0.4 |
| LLaMa-2-7B-Chat, AdvBench jailbreak rate before and after 10 steps on the 10 examples (Appendix I.1.1) | 0% to 99.80% |
| The same, after 10 epochs on 10 refusals that the model wrote before the shift (Appendix I.1.2) | 99.80% to 0.77% |
| The same shift, with the bottom 3 or 6 layers frozen (Table 2) | 79.61±6.53% and 1.15±0.16% |
| TRACE, SEQ, FOMC and C-STANCE after task 5, then after task 6 (Table 4) | 0.00 and 0.00, then 62.41 and 41.79 |
| TRACE, mean score after the last task: SEQ, Freeze 3 layers, Freeze 6 layers (Table 2) | 47.38, 53.20, 51.91 |
| zsRE efficacy / paraphrase: SEQ, Freeze 1 layer after task 1, Freeze 6 layers (Table 2) | 62.47 / 58.24, 70.88 / 64.19, 51.49 / 42.74 |
| Concept-1K memorisation / generalisation: SEQ, Freeze 2 layers after task 1, Freeze 6 layers (Table 2) | 35.98 / 12.61, 42.18 / 14.19, 30.27 / 7.18 |
| LLaMa-3-8B-Instruct, mean general score after math tuning at lr 5e-6, without and with Freeze of 1 layer (Table 35) | 64.15 and 66.11. The untuned model scores 68.89 |
| Mistral-8B-Instruct-2410, the same (Table 35) | 57.42 and 63.96, from 71.97 |
| LLaMa-3-8B-Instruct, mean general score after code tuning at lr 5e-6, without and with Freeze (Table 36) | 71.47 and 71.78, from 68.89 |

The authors draw two rules from Section 5.3. Sometimes the new task is like what the model saw after pretraining (SA and CIT). Then spurious forgetting starts at the first task, and more layers can be frozen. When the new task has a new format and new knowledge (CKE and IIL), freeze one or two layers after the first task.

## Where the claims go beyond the evidence

**Only one synthetic setup shows the split into alignment and knowledge.** Only Biography has a recovery test on held-out people. In the four real scenarios, the paper infers spurious forgetting from the gain that Freeze gives. It does not test whether the knowledge stays. A gain from freezing fits other readings too, such as a smaller step size in effect.

**The recovery test is a large intervention.** It trains a full epoch on half of Task 0. Such a run would also teach the QA format to a model that never had it, as long as the facts are in its weights. So the test shows that the facts stay. It doesn't show that the old alignment is still in the weights and only needs a small push.

**The safety result may show only that the new task was not learnt.** With six frozen layers the jailbreak rate stays at 1.15%. The paper reports no measure of whether the model learnt the identity-shift task. The authors state that Freeze is not a defence against jailbreaks.

**Freezing more layers hurts in two of the four real settings.** Freezing three or six layers from the start lowers zsRE efficacy below SEQ, 53.75 and 51.49 against 62.47. The same holds for Concept-1K memorisation at six layers, 30.27 against 35.98. The rules for when to freeze come from the same four scenarios that they explain, so they are chosen after the data.

**The theory does not test its own key assumption.** Assumption 4.4 needs each update to lie in the left null space of its weight matrix. Figure 4 measures the angle between two updates, not between an update and the weights. Remark 4.5 says the two "align", but the paper never measures WᵀΔW. The model is also linear, with no attention, MLP nonlinearity or normalisation.

**Two proofs state bounds that hold only to first order.** Lemma F.1 claims ‖P_L − I‖ ≤ Lδ. Its own derivation gives (1 + δ)^L − 1, which is at least Lδ, so the stated inequality runs the wrong way. Lemma F.2 and Corollary F.5 drop higher-order terms with "≈" inside an induction that claims a bound.

**Few seeds, and none stated for the synthetic curves.** Table 1 gives ± values but doesn't say how many runs. The curves of Figures 2 to 5, which carry the main argument, show one run each with no interval. The real scenarios use three seeds. The task-wise tables of Appendix I give means only.

**Inconsistencies within the source.**

- Section 3.1 and Appendix C disagree on the Task 0 people (all 100,000 or the first 50,000).
- The text of Appendix I.5 gives LLaMa-3's general score before tuning as 66.89, but Table 35 gives 68.89.
- Mistral's mean of the same five benchmark scores is 71.97 in Table 35 and 73.75 in Table 36.
- The caption of Figure 9 says both columns are for 20 people, but the panels say 20 and 200.
- Appendix G.3 says Section 3.3 showed the angles of the MLP output matrix, but the angles are in Section 3.4.
- Appendix H trains "using the Freeze dataset", which seems to mean Biography.

**"The first to identify" omits close work.** Kotha, Springer and Raghunathan (ICLR 2024) argued that fine-tuning moves task inference and leaves capabilities in place (see [implicit-inference-forgetting.md](implicit-inference-forgetting.md)). This paper does not cite them.

**Credit.** The authors state that Freeze leaves a large gap to replay, and that freezing too many layers costs plasticity (Appendix J.1). They test four further settings on Biography: more tasks, task sizes, a new task type, and other optimisers and learning rates. They report the methods that fail, with their grids. They also report that code tuning raises general scores, against their own story of forgetting.

## Use in this repo

**#5126 asks the paper's question at the grain of a pole.** In `base-instruct.md`, probes inside the chat template lower the keep share on Qwen2.5-1.5B from 88% to 70%. On Qwen2.5-0.5B it falls from 81% to 72%. The fall is mostly in sensitivity. Activations still correlate with the base model at a median r of 0.983 and 0.974. In the paper's terms, the neurons keep their "knowledge" while the format moves what we measure. But that run reads only the instruct model inside the template. #5126 reads the base model inside the template too. That is the control that tells a format effect from a lost function. The paper predicts a format effect, which would show the same fall in the base model.

**Our template losses don't gather in the bottom layers.** The paper puts the alignment change in the bottom layers. We counted the poles that keep only with plain probes. The rows are in `base-instruct-0.5b.jsonl`, `base-instruct.jsonl` and their `-chat` files. On Qwen2.5-0.5B, 13 of 112 are in the first third of the layers. The middle third has 13 of 92, and the last third 14 of 107. On Qwen2.5-1.5B they are 2 of 16, 4 of 13 and 3 of 13. These are our counts from one run, with no interval. They are not a strict test of the paper. A template reads every layer, and a change in the bottom layers would reach the poles above them.

**The instruct tune moved the bottom third least.** [anatomy-of-catastrophic-forgetting.md](anatomy-of-catastrophic-forgetting.md) gives the median base-to-instruct correlation by third on Qwen2.5-0.5B. It is 0.982, 0.962 and 0.973. The paper's theory says that orthogonal updates in the bottom layers move features off their main direction and keep them otherwise. A high correlation in the first third fits that, but it can't tell orthogonal updates from small ones.

**The paper's recovery test would fit #5125.** #5125 fine-tunes Qwen2.5-0.5B on a sequence of narrow tasks and reads the VERIFIED poles again after each stage. The paper shows that a score can fall to near zero and come back with the next task (Table 4, FOMC 0.00 and then 62.41). So a pole that fails after one stage may not stay failed. A short re-tune on held-out data of the earlier format would show whether the pole's function returns. This belongs on #5125 as a comment, not a new issue.

**#5133 could measure the assumption that the paper did not.** #5133 reads Δ = instruct − base for each MLP matrix of the Qwen2.5 pairs. The same rows give ‖WᵀΔ‖ / (‖W‖‖Δ‖) for each layer. That reads Assumption 4.4 directly: is the update in the left null space, and more so in the bottom layers? It trains nothing. This belongs on #5133 as a comment.

**For a fix by training, the paper is evidence that small fine-tunes undo each other.** Ten steps on ten examples take LLaMa-2-7B-Chat from 0% to 99.80% jailbreaks. Ten refusals for ten epochs bring it back to 0.77%. That fits the *Suppression comes back* rows of [fixing-models.md](fixing-models.md). A fix of the #4781 " days" error by a short fine-tune could be undone as fast by later training. That is evidence only. The maintainer designs the fix campaign.

**Freeze, as evidence for that campaign.** Freeze helped most when the new task had the old format, and hurt on knowledge editing when it froze three or more layers. Pythia-70m has six layers, and the copy heads of the #4781 route are in layers 3 and 4. A frozen bottom half would not hold those heads still. The paper says nothing about a model this small, and we draw no rule from it.

**What does not carry over.** The paper studies no single neuron, explanation or probe. Its "features" are the principal components of whole hidden states. It has no random or untrained baseline for the recovery test. Its controlled model trains from random weights on one synthetic corpus, and its real models are 0.4B to 8B.

**Questions for the maintainer.** These are questions, not issues.

- Should a report of a fix by training include a later short tune on unrelated data, to show whether the fix survives?
- Should the base-instruct analyses count a pole that fails only inside the template as a format effect and not a lost explanation, once #5126 reports?
