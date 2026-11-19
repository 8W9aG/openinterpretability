# LoRA+: Efficient Low Rank Adaptation of Large Models

**Citation.** Soufiane Hayou (Simons Institute, UC Berkeley), Nikhil Ghosh (Department of Statistics, UC Berkeley) and Bin Yu (Department of Statistics, UC Berkeley). Hayou and Ghosh are equal contributors. "LoRA+: Efficient Low Rank Adaptation of Large Models". [arXiv:2402.12354](https://arxiv.org/abs/2402.12354), version 2 of 4 July 2024. Version 1 is of 19 February 2024. Venue: Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235. The code is at [github.com/nikhil-ghosh-berkeley/loraplus](https://github.com/nikhil-ghosh-berkeley/loraplus). The arXiv page says "27 pages", but the v2 PDF has 24. Funding: AWS cloud credits under an Amazon Research Award, and NSF and Simons Foundation grants that the acknowledgement lists.

**Status.** Planned. #4782 would cite it in its strand on LoRA and low-rank fine-tuning. #5125 fine-tunes registry models, and a LoRA run there would have to choose this paper's learning-rate ratio. No code, skill or protocol in this repo depends on it.

**Read.** On 2026-11-19 we downloaded the v2 PDF with `curl` and converted it with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own text the abstract, Sections 1 to 6, the acknowledgement, the impact statement and Appendices A to C. That includes the proofs and every figure caption. We did not see the figures as images. The heatmap cell values came out of `pdftotext` in a jumbled order, so this page reports no cell value from Figures 2 to 19. The conversion also lost superscripts. We read "24" as 2⁴, from the context of Figure 1 and the log₂ axis of Figure 6. We did not read version 1, the ICML proceedings page, the code or the references in full.

## Summary

**Claim.** LoRA trains its two adapter matrices, A and B, at one learning rate. In wide networks, this stops one of the two from learning efficiently. A fixed ratio λ = η_B/η_A > 1 between the two learning rates fixes this, at no extra cost. The paper calls this method LoRA+.

**Method.**

- **LoRA.** Each adapted matrix becomes W = W\* + (α/r)·BA. Only B (n₁ × r) and A (r × n₂) train. One of the two starts at zero, so tuning starts from the pretrained model. Init[1] sets B to zero and A to random values, as Hu et al. (2021) do. Init[2] sets A to zero and B to random values.
- **Theory.** The paper takes the width n to infinity with the rank r fixed. It calls tuning *efficient* when the feature update from A alone and from B alone are both of order one. A toy linear model, trained by gradient descent, gives Propositions 1 and 2. A general network, trained by Adam, gives Theorem 1. The authors call the proofs "of a physics level of rigor". The analysis has one training sample and one trainable LoRA layer.
- **Toy test.** A ReLU network f(x) = W_out ϕ(BA ϕ(W_in x)), with d = 5, n = 100 and r = 4. It learns a synthetic sine target from 1,000 training samples, with 100 for test. Only A and B train, over a grid of the two learning rates, with 3 seeds.
- **GLUE.** RoBERTa-base and GPT-2 on MNLI, QQP, SST2 and QNLI. α = r = 8, sequence length 128, 3 epochs (10 for SST2), batch size 32 and 3 seeds. AdamW, a linear schedule and FP16. A grid of 6 × 7 learning-rate pairs. RoBERTa adapts the query and value weights. GPT-2 adapts `c_attn`, `c_proj` and `c_fc`. Appendix C repeats the runs in full precision and at r = 4 and r = 16.
- **Llama-7b.** MNLI for one epoch at r = 8 and α = 16. A 100,000-sample subset of flan-v2 for one epoch at r = 64 and α = 16, scored by 5-shot MMLU. LoRA adapts all seven projections of each block. Two seeds near the optimum and where η_A = η_B, and one seed elsewhere. The grid keeps η_B ≥ η_A.
- **Baseline.** Standard LoRA, η_B = η_A, tuned over the same grid. There is no comparison with full fine-tuning or with other adapter methods.

**Findings, as the source reports them.** All numbers are from the text or from figure captions.

| Setting | Result |
| --- | --- |
| Theory (Proposition 1, Theorem 1) | With η_A = η_B, efficient tuning is "impossible" in the infinite-width limit, under both initialisations |
| Theory (Proposition 2, Theorem 1) | η_A = Θ(n⁻¹) and η_B = Θ(1) give efficient tuning. So η_B/η_A must grow in proportion to the width n |
| Toy model (Section 3.2) | The best train and test losses come with η_B ≫ η_A. The best pairs lie close to the edge of stability |
| GLUE, RoBERTa-base and GPT-2 (Section 5.1) | The best test accuracy comes with some η_B ≫ η_A. The gap to the best η_A ≈ η_B pair is larger on MNLI and QQP than on SST2 and QNLI |
| GLUE, seed noise (Figure 3 caption) | "Fluctuations are of order 0.1%" |
| Llama-7b, flan-v2, MMLU (Section 5.2) | η_B ≫ η_A gives "a roughly 1.3% gain" over the best η_B = η_A |
| Llama-7b, flan-v2, Init[1] (Figure 18b caption) | The best MMLU accuracy is 0.6% higher than with Init[2], and the best ratio is twice as large |
| Llama-7b, MNLI (Section 5.2) | η_B = η_A is "nearly optimal" for all η_B ≥ η_A |
| Default ratio (Section 5.3) | With Init[2], λ ≈ 2⁴ helps RoBERTa. With Init[1], the best ratio is of order 2² to 2³. For Llama, 2¹ to 2² |
| Speed (Figure 7) | RoBERTa-base on MNLI, λ = 2⁴ against λ = 1, each with a tuned η_A. The figure marks "2X SpeedUp" |
| Abstract | "1% − 2% improvements" and "up to ∼ 2X SpeedUp", at the cost of standard LoRA |

## Where the claims go beyond the evidence

- **The width argument is not tested across widths.** The theory says that the best ratio grows with n. RoBERTa-base and GPT-2 have about the same width, and the paper names n ≈ 700 for GPT-2. No experiment changes the width of one model family and tracks the best ratio.
- **The headline gains are thin in the text.** The abstract gives 1% to 2%. The text states one gain, 1.3% on MMLU, at one or two seeds. On Llama MNLI the ratio does not help. The "2X SpeedUp" rests on one curve for one task in Figure 7.
- **The learning rates are chosen on test data.** The GLUE heatmaps report test accuracy, and the authors read the best pair from them. The Llama runs report the best checkpoint by test accuracy. No held-out split chooses the rates before the test.
- **Figure 6 states two rules for its top runs.** The text says the top 4 runs "in terms of test accuracy". The caption says that "the test loss" selects them.
- **The learning-rate labels disagree.** Appendix C.1.2 gives η_A the grid from 1e-4 to 4e-3 and η_B the grid from 1e-5 to 8e-4. The figures put the larger grid on the η_B axis. The Figure 2 caption gives the best toy pair as (η_A, η_B) = (2.78, 1.29e−4), which has η_A ≫ η_B. Both disagree with the claim η_B ≫ η_A, so we take them as swapped labels.
- **The default ratio comes from the less usual initialisation.** Footnote 9 says that Hu et al. (2021) start B at zero, which is Init[1]. The suggested λ ≈ 2⁴ is for Init[2]. With Init[1] the paper finds 2² to 2³.
- **Few seeds and no intervals in the LLM runs.** GLUE uses 3 seeds, and the heatmaps show no intervals. Llama uses 2 seeds near the optimum and 1 elsewhere.
- **The theory rests on simplifications.** It has one training sample, one trainable LoRA layer and an infinite width. Theorem 1 also needs Assumption 1 on Adam's gradients, which Appendix A.5 justifies only for SignSGD. The authors state each of these.
- **The paper measures task scores only.** It does not measure forgetting, the weight change ΔW, or any feature or unit of the model.

The authors state some limits themselves. They report the Llama MNLI case where the ratio does not help. They say that the best ratio depends on the model and the task, and that their analysis "lacks this dimension".

## Use in this repo

- **The repo has no LoRA code.** No code, skill or analysis in this repo trains or merges an adapter. [`fixing-models.md`](fixing-models.md) notes that the repo has no fine-tuning or weight-edit code. So this paper's method cannot run here now.
- **The paper says nothing about units.** It measures GLUE and MMLU accuracy. It does not test neurons, explanations, probes or a random baseline. So it cannot tell us whether a VERIFIED unit keeps its explanation after a merge of an adapter. That is #4782's question.
- **The ratio decides which side of the adapter moves.** This point is our inference from Section 4, not a claim of the paper. Take an adapter on Wup, the matrix that feeds the MLP neurons. B then has one row per neuron, and A holds r shared directions in the residual stream. The paper argues that at η_B = η_A, one of the two matrices does not learn efficiently. At λ ≫ 1, B learns at full speed. So LoRA+ moves each neuron's own weights on the shared directions more, and the directions themselves less.
- **So the ratio is a hidden variable in any merged-adapter test.** Two adapters with one task score can move the registry's units in different ways. That holds if their learning-rate ratios differ. A test of #4782's question must record η_A, η_B and the initialisation. Otherwise a keep share from one run cannot be compared with another.
- **The repo's base-instruct numbers are the reference for such a test.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests VERIFIED poles of Qwen2.5 base models on their instruct models at the same index. On Qwen2.5-1.5B, 38 of 43 hypotheses keep (88%, 95% CI 76–95%), with median correlation 0.983. On Qwen2.5-0.5B, 252 of 311 keep (81%, 95% CI 76–85%), with median correlation 0.974. The analysis does not record how Qwen tuned the instruct models, so it gives no ratio to compare.
- **The open follow-ups read the full-tuning change.** #5133 reads the rank and per-pole norm of Δ = instruct − base. #5154 adds a random Δ of the same norm. #5118 tests diagonal Fisher as a predictor. None of these trains an adapter, so none needs a ratio. #5125 fine-tunes Qwen2.5-0.5B and its instruct model on narrow tasks. If #5125 uses LoRA, it must choose the ratio, and the ratio could change its keep share.
- **Fast feature learning may cost more units.** This is a question, not a finding. The authors design LoRA+ to make the adapter's features move more per step. [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md) reviews Biderman et al. (2024), where LoRA forgets less than full fine-tuning because it learns less. If LoRA+ learns more per step, it may also forget more. Neither paper tests this, and the follow-up below would test it on registry units.
- **Weights keep moving after the score settles.** In [`papers/neuron-development/`](../../papers/neuron-development/README.md), Pythia-70m's 86 VERIFIED poles correlate with their final form at a median of 0.62 at step 16,000. Adjacent checkpoints correlate at a median of 0.78 to 0.82. LoRA+ reports a score at the end of tuning. A unit-level test should therefore read the units at more than one step, not only at the best score.
- **The interference strand.** [`complementary-learning-systems.md`](complementary-learning-systems.md) reviews McClelland et al. (1995), where focused learning damages related knowledge. A higher B rate is a way to learn a narrow task faster. The paper gives no evidence on whether this damages more of what the model knew.
- **Evidence for the fix campaign, not a design.** A fix of the #4781 " days" error by an adapter would have to choose λ. This paper shows that the choice changes the task score. It gives no evidence on how the choice changes other behaviour or the registry's units. The maintainer designs the campaign, so this page records the point only.
- **A question for the maintainer.** If the repo trains adapters, should every record of a merged adapter store η_A, η_B, the initialisation and the rank? That is a choice of record schema, so this page opens no issue for it.
- **A follow-up that needs fine-tuning code.** Train two LoRA adapters on one narrow task on Qwen2.5-0.5B, one at λ = 1 and one at λ = 2⁴, with matched seeds. Merge each one, and test the VERIFIED poles again as `scripts/base_instruct.py` does. Compare the two keep shares, and each pole's activation correlation with the base model. It depends on the fine-tuning work of #5125, and it could share that issue's tasks.
