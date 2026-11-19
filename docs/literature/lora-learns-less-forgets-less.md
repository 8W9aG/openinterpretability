# LoRA Learns Less and Forgets Less

**Citation.** Dan Biderman (Columbia University and Databricks Mosaic Research), Jacob Portes, Jose Javier Gonzalez Ortiz and Mansheej Paul. Then Philip Greengard (Columbia), Connor Jennings, Daniel King, Sam Havens and Vitaliy Chiley. Then Jonathan Frankle, Cody Blakeney and John P. Cunningham (Columbia). The authors without a Columbia mark are at Databricks Mosaic Research. "LoRA Learns Less and Forgets Less". [arXiv:2405.09673](https://arxiv.org/abs/2405.09673), version 2 of 20 September 2024. Version 1 is of 15 May 2024. Venue: Transactions on Machine Learning Research, August 2024. The arXiv comment adds "Featured Certification", and the review is on [OpenReview](https://openreview.net/forum?id=aloEru2qCG). The authors release model checkpoints and LoRA adapters at [github.com/danbider/lora-tradeoffs](https://github.com/danbider/lora-tradeoffs). The text names no funding. Training used the Databricks MosaicML libraries and HuggingFace `peft` on 32 H100 GPUs.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it in its synthesis page. This page proposes one follow-up that reads weights only. The maintainer designs the campaign to fix models, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv v2 PDF (39 pages) and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own words the abstract, sections 1 to 7, the acknowledgements and author contributions, and Appendices A to I. We read Tables 1, S1 to S8 and S10 to S16 as text. We did not see the figures as images, so this page gives only their captions and the numbers that the text states. That includes Figures 1 to 7 and S1 to S10. We did not read version 1, the OpenReview discussion or the references in full.

## Summary

**Claim.** In the usual low-rank settings, LoRA learns code and mathematics less well than full fine-tuning. It also keeps more of the base model's skill on other tasks. Full fine-tuning finds weight changes of a rank 10 to 100 times higher than usual LoRA ranks.

**Method.**

- **Model and LoRA.** Llama-2-7B. LoRA trains a low-rank change Δ = γ_r·AB to each frozen matrix. Here A has r columns, B has r rows, and γ_r = α/r. The main runs adapt all seven matrices of each block: Wq, Wk, Wv, Wo, Wgate, Wup and Wdown. They use r = 16, 64 and 256, α = 2r and LoRA dropout 0.05.
- **Four settings.** Continued pretraining (CPT) on StarCoder-Python and on OpenWebMath, up to 20 billion tokens each. Instruction fine-tuning (IFT) on Magicoder-Evol-Instruct-110K (72.97M tokens) and on MetaMathQA (103M tokens). IFT runs last 1, 2, 4, 8 or 16 epochs. CPT runs cool down from shared checkpoints at 0.25 to 20 billion tokens.
- **Learning.** HumanEval pass@1 for code (temperature 0.2, 50 samples), and GSM8K for mathematics (5-shot, greedy).
- **Forgetting.** The mean of HellaSwag, WinoGrande and ARC-Challenge accuracy. The paper calls the base model's domain the source domain.
- **Baselines.** Full fine-tuning, with a sweep of learning rates. On Magicoder, also weight decay (5e-5 and 1e-4) and attention dropout (0.05 and 0.1).
- **Other analyses.** The diversity of 50 HumanEval samples, as the count of unique strings. A singular value decomposition (SVD) of Δ for full fine-tuning on code CPT. The SVD gives the rank that explains 90% of a matrix's variance, at each layer and module. A mixed chat dataset, Tülu-v2-mix, for up to 6 epochs. Sweeps of α, learning rate, optimizer and target modules.
- **Runs.** The paper trains "one full finetuning model and three LoRA models" for each condition. It gives no seeds and no intervals, except one standard error on MT-bench.

**Findings, as the source reports them.** All numbers are from the text or from Tables S1 to S16.

| Setting | Result |
| --- | --- |
| Code CPT, HumanEval (Table S1) | LoRA r = 256 peaks at 0.224 at 20B tokens. Full fine-tuning has 0.218 at 4B tokens and 0.263 at 20B |
| Code IFT, HumanEval (Table S5) | At epoch 4: r = 16 0.358, r = 64 0.417, r = 256 0.498. Full fine-tuning peaks at 0.497 at epoch 8 |
| Math CPT, GSM8K (Table S3) | LoRA r = 256 peaks at 0.203 at 16B tokens. Full fine-tuning has 0.224 at 4B and 0.293 at 20B |
| Math IFT, GSM8K (Table S7) | LoRA r = 256 peaks at 0.634 at epoch 8, and r = 64 has 0.624 at epoch 4. Full fine-tuning has 0.641 at epoch 2 and 0.642 at epoch 4 |
| Forgetting, code CPT at 20B (Table S2) | Full fine-tuning 0.545, LoRA r = 256 0.617, r = 16 0.635 |
| Forgetting, code IFT at epoch 16 (Table S6) | Full fine-tuning 0.414, LoRA r = 64 0.510 (0.509 in the text), r = 16 0.609 |
| Forgetting, code IFT at epoch 1 (Table S6) | Full fine-tuning 0.595, LoRA 0.652 to 0.655 |
| Forgetting, math CPT (Table S4) | LoRA r = 256 0.616 at 20B tokens, full fine-tuning 0.613 at 16B |
| Forgetting, math IFT at epoch 16 (Table S8) | LoRA r = 256 0.567, full fine-tuning 0.559 |
| Order of forgetting (section 4.2) | IFT forgets more than CPT, code more than mathematics, and longer training more. OpenWebMath, mostly English, forgets least |
| Weight decay and dropout (section 4.5) | Both "learn and forget as much as full finetuning". LoRA r = 256 learns as much while it forgets less. No numbers in the text |
| SVD of Wq, layer 26 (section 4.6) | Base, tuned and Δ all need about half of the 4,096 singular vectors for 90% of the variance |
| Rank of Δ over training (Figure 6, text) | 10 to 100 times typical LoRA ranks at 0.25B tokens. It grows with data, is higher in MLP than attention, and lower in the first and last layers |
| Target modules (section 4.7) | MLP alone or all modules beat attention alone, and "most gains" come from the MLP modules |
| α and learning rate (section 4.7, B) | For r = 256, α = 512 beats 256 and 32. LoRA's best learning rate is about 10 times that of full fine-tuning |
| Tülu-v2-mix, forgetting at epoch 6 (Table S13) | Full fine-tuning 0.621, r = 256 0.629, r = 64 0.647, r = 16 0.657. At epoch 2, full fine-tuning has 0.660, the highest |
| Tülu-v2-mix, MT-bench (C.2, Table S10) | Base model 2.74. Every run is 5.6 to 6.0, and all LoRA runs are within one standard error of full fine-tuning |
| Cost (Appendix I) | LoRA lowers throughput by about 15%. It lowers peak memory by about 40% at small batches and about 15% at large ones |

## Where the claims go beyond the evidence

- **The title is wider than the trade-off result.** Section 4.3 says that the four datasets make it "difficult to conclude" whether the two methods trade off differently. On math IFT, full fine-tuning offers "preferable tradeoffs". So "forgets less" holds at a matched step, not at a matched amount learned. The authors say this themselves.
- **One model, one run per condition.** Every result is on Llama-2-7B, with one run and no intervals. Many gaps are small, for example 0.567 against 0.559 on math IFT. The authors leave model size to future work.
- **The learning rates differ from the paper's own advice.** Section 4.1 describes "an exhaustive learning rate sweep for each method". But code CPT uses 1e-5 for both LoRA and full fine-tuning (Appendix A). Section 4.7 says LoRA's best rate is about ten times higher. So LoRA on code CPT may be under-tuned. The Appendix B sweep also used α = 32 and left out Wgate, which differs from the main runs.
- **Two statements on pass@k disagree.** Section 4.1 says full fine-tuning beats r = 256 for k < 64, "after which the two are equal". Appendix F says full fine-tuning "is superior across all values of k".
- **The regularization baselines do not target forgetting.** Weight decay and dropout fight overfitting. The paper names replay as the usual method against forgetting. It does not test replay, a pull toward the old weights, or a KL term.
- **Forgetting is three multiple-choice scores.** The text does not state the base model's own score, so the absolute loss is only in the figures. The mean of three benchmarks can hide a large loss on one.
- **The diversity measure is coarse.** It counts unique strings, and the authors say that it "is not a sensitive metric".
- **The SVD covers code CPT only.** The authors state this. They also state that a high-rank Δ does not rule out low-rank solutions, and that a task may need less rank than the weights.
- **Tülu checkpoints are not cooled down.** Appendix C says the Tülu checkpoints are "hot", unlike the main IFT runs. So the Tülu curves and the main curves are not fully comparable.

The authors give a limits section on the SVD and on model size. They also report the cases where full fine-tuning wins on both axes, such as Tülu at epoch 2.

## Use in this repo

- **The repo has no LoRA and no fine-tuning code.** No code, skill or analysis uses LoRA, and no registry model is a Llama model. [`fixing-models.md`](fixing-models.md) notes that the repo has no fine-tuning or weight-edit code. So the paper's runs cannot be repeated here as they are.
- **The paper measures behaviour, not units.** Its forgetting score is benchmark accuracy, and it says nothing about neurons, explanations, probes or random baselines. A low forgetting score does not show that a VERIFIED explanation keeps. #4782 asks that question for a merged adapter.
- **Low rank does not mean few neurons.** This point is our inference from section 2, not a claim of the paper. A LoRA change to Wup adds AB to every row of the matrix. So each MLP neuron's input weights move inside one shared r-dimensional subspace. A merged adapter can thus move every neuron of a layer a little, in the same few directions. That is closer to an added shared direction than to a change of a few units. It bears on interference between units and on the superposition directions of `openinterpretability/explain_groups.py`.
- **The repo's base-instruct numbers are on the paper's milder side.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) compares each VERIFIED pole of Qwen2.5 base and instruct models at one index. On Qwen2.5-1.5B, 38 of 43 hypotheses keep, and every correlation is 0.953 or more. On Qwen2.5-0.5B, 252 of 311 keep at a median correlation of 0.974. The paper's Tülu-v2-mix run uses a general chat mixture, and it shows "overall mild forgetting". Narrow code IFT forgets most. The analysis does not record whether the Qwen2.5 instruct models used LoRA or full fine-tuning.
- **The depth pattern is similar, but the evidence is weak.** The paper's Figure 6 puts the highest-rank Δ in the middle layers, for code CPT on Llama-2-7B. On Qwen2.5-0.5B, the middle third keeps least: 63 of 92, against 104 of 112 and 85 of 107. The model, the training and the measure all differ. So this is a question to test, not a match. The follow-up below tests it on the same pairs.
- **The MLP result points at this repo's units.** The paper finds that "most gains" come from the MLP modules, and that MLP Δ has a higher rank than attention Δ. The repo's units are MLP neuron poles. So the weights that a fine-tune changes most are the weights that the registry explains.
- **Pretraining shows a similar gap between score and weights.** In [`papers/neuron-development/`](../../papers/neuron-development/README.md), Pythia-70m's loss settles by step 16,000. Its 86 VERIFIED poles still correlate with their final form at a median of only 0.62 there. Adjacent checkpoints correlate at a median of 0.78 to 0.82. The paper likewise finds that Δ's rank keeps growing with more data. In both, a settled score does not show that the weights stop moving.
- **Interleaving fits the Tülu result.** [`complementary-learning-systems.md`](complementary-learning-systems.md) reviews McClelland et al. (1995), where interleaved learning avoids damage to old knowledge. Here the mixed Tülu data forgets less than narrow code IFT. The paper does not test replay, so this is our reading, not its claim.
- **Evidence for the fix campaign, not a design.** A fix for the #4781 " days" error would be a narrow fine-tune. In the paper, narrow code IFT gives the largest loss on other tasks. LoRA at r = 16 forgets least, but it also learns least. [`fixing-models.md`](fixing-models.md) holds the repo's synthesis on fixes.
- **A question for the maintainer.** Should a weight fix report each VERIFIED pole's keep share as a measure of forgetting, next to benchmark scores? That is a choice in the fix campaign, so this page records it as a question and opens no issue.
- **A follow-up that reads weights only.** For each Qwen2.5 base-instruct pair, compute Δ = instruct minus base for each MLP matrix. Read Δ's rank for 90% of its variance in each layer, as in the paper's section 4.6. Also read the norm of Δ in each pole's own Wgate, Wup and Wdown rows and columns. Then test whether these predict which poles keep in the existing rows, against a null of shuffled poles. It trains nothing, and it can share its rows with #5118, which tests diagonal Fisher as a predictor.
- **What belongs to the synthesis page.** A test of LoRA against full fine-tuning on a registry model would answer #4782's question for merged adapters directly. It needs fine-tuning code that the repo does not have. So this page leaves that choice to the #4782 synthesis page.
