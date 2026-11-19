# Text-to-LoRA: Instant Transformer Adaption

**Citation.** Rujikorn Charakorn, Edoardo Cetin, Yujin Tang and Robert T. Lange, all of Sakana AI. "Text-to-LoRA: Instant Transformer Adaption". [arXiv:2506.06105](https://arxiv.org/abs/2506.06105), version 2 of 9 June 2025. Version 1 is of 6 June 2025. Venue: Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267. The code is at [github.com/SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora). The training data are the 500 Super Natural Instructions (SNI) tasks at `huggingface.co/Lots-of-LoRAs`. The paper gives no funding statement. It says that GPT-4o mini wrote all task descriptions, and that David Ha suggested the title.

**Status.** Planned. #4782 would use it in its strand on weight-space programming and fast weights, and #3381 (weight-space programming) would cite it. No code, skill or protocol in this repo depends on it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the v2 PDF (30 pages) with `curl` and converted it with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own text the abstract, Sections 1 to 7, the acknowledgement and the impact statement. We also read Appendices A to M and Listing 1. We read Tables 1 to 11 as extracted text. We did not see the figures as images. For Figures 6 and 11, the PDF text holds the Pearson coefficients that the plots print, and we report those. We have only the captions and the text for Figures 1 to 5. We did not read the references in full, version 1, the ICML proceedings page or the code.

## Summary

**Claim.** A hypernetwork, Text-to-LoRA (T2L), can make a LoRA adapter for a large language model from a plain-language description of a task, in one forward pass. Trained on hundreds of tasks, it compresses their adapters. It also makes useful adapters for benchmark tasks that it did not see, from their descriptions alone.

**Method.**

- **Base models.** Mistral-7B-Instruct for most runs. Appendix A repeats the main test on Llama-3.1-8B-Instruct and Gemma-2-2B-Instruct.
- **The adapters.** Every LoRA has rank 8 and changes only the query and value projections of each attention block. That is 3.4 M parameters (Listing 1 and Appendix F). No adapter touches an MLP.
- **The hypernetwork.** A text encoder, `gte-large-en-v1.5`, turns the description into a vector. A linear task encoder, learned module and layer embeddings, and residual MLP blocks follow. Three output heads give three sizes: L makes A and B together, M makes A or B, and S makes one rank at a time. They have 55 M, 34 M and 5 M parameters.
- **Two ways to train.**
  - *Reconstruction* fits the hypernetwork's output to a library of trained LoRAs by an L1 loss (Equation 6).
  - *SFT* trains the hypernetwork end to end through the frozen base model on the tasks' own data (Equation 5).
- **Data.** 479 SNI tasks for training, 11 for validation, and 10 removed because they overlap the benchmarks. Each task has up to 200 descriptions from GPT-4o mini, and SFT training uses 128 per task. All samples are English.
- **Benchmarks.** ARC-Challenge, ARC-Easy, BoolQ, GSM8K, HellaSwag, OpenBookQA, PIQA, Winogrande, HumanEval and MBPP. Each T2L score is the mean over three descriptions.
- **Baselines.** The base model, the base model with the description prepended, 3-shot in-context learning, an element-wise average of LoRAs and a multi-task LoRA. Also Hyperdecoders, which the authors built on their own architecture, and Arrow Routing numbers copied from Ostapenko et al. (2024). The "oracle" is one LoRA trained on each benchmark's training split.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Reconstruction of the 9 benchmark LoRAs, mean of 9 tasks (Table 1) | Base model 55.8. T2L 73.0 to 73.5 over the three sizes and two embedding types. The task-specific LoRAs 73.3 |
| Reconstruction of more SNI LoRAs (Figure 3 text) | Full oracle performance below 10⁻⁴ reconstruction loss. About 65% of oracle performance at more than 8 × 10⁻⁴ per-element L1 error |
| Zero-shot, SFT on 479 tasks, Mistral, mean of 10 tasks (Table 2) | Base 55.8, description prepended 60.6, 3-shot ICL 61.0, average LoRA 60.9, multi-task LoRA 66.3, Hyperdecoders 67.3, T2L S 65.9, M 67.5, L 67.7 |
| Same, mean of 8 tasks (Table 2) | Oracle LoRAs 75.8, T2L L 73.9, Hyperdecoders 73.6, Arrow Routing 70.7, multi-task LoRA 71.9 |
| Description embedding, 128 tasks (Table 4) | Mean 65.9 with `gte`, 65.6 with Mistral's last-token activation |
| Descriptions, reconstruction-trained T2L on 9 tasks (Table 5) | Training descriptions 73.3, unseen descriptions 72.2, another task's descriptions 51.4, random strings 63.5 |
| Reconstruction against SFT, about 10 hours each, zero-shot (Table 6) | Mean 61.8 against 66.3 |
| Llama-3.1-8B-Instruct, mean of 10 tasks (Table 7) | Base 73.2, multi-task LoRA 76.5, T2L L 76.9 |
| Gemma-2-2B-Instruct with ICL, mean of 10 tasks (Table 8) | Multi-task LoRA 65.2, T2L L 66.4 |
| SNI's own task definitions instead of GPT-4o mini's (Table 9) | 66.5 against 67.7 |
| Cosine of concatenated A and B against description similarity, 479 SNI LoRAs against one benchmark LoRA (Figure 6) | Pearson −0.00 (ARC-Challenge), 0.01 (BoolQ), −0.01 (OpenBookQA) |
| Same, relative benchmark performance against description similarity (Figure 6) | Pearson 0.14, 0.27, 0.18 |
| Same, cosine of the full ΔW = BᵀA (Figure 11) | Pearson 0.38, 0.17, 0.31 |
| FLOPs per question (Appendix I) | 0.856 TFLOPs with T2L M, against 4.177 for 3-shot ICL |

The authors explain the Figure 6 result by the many pairs of A and B that give one ΔW. A reconstruction loss on ΔW did not improve zero-shot results, they say, but they give no number for it.

## Where the claims go beyond the evidence

- **No seeds, intervals or tests.** Each score is the mean of three descriptions, with no spread. The ranks rest on small gaps. T2L L leads Hyperdecoders by 0.4 points and the multi-task LoRA by 1.4 points over 10 tasks. One run per setting can't rank methods this close.
- **The scaling is not monotone.** In Table 3, T2L S with 479 tasks scores 65.2, below its 65.9 with 256 tasks. In Table 10, at fixed compute, T2L L scores 66.6 with 479 tasks and 67.5 with 256. The text says that T2L "generally benefits" from more tasks. The authors do give a reason for S, its small capacity.
- **"Unseen" tasks are near the training tasks.** The authors say so in a footnote. Most benchmarks are multiple-choice question answering, as are most SNI tasks, and ARC resembles SNI task #47. They removed 10 contaminated datasets. They also say that code tasks are far from the training data. T2L gains least on HumanEval: 39.2 for L against 37.2 for the base model.
- **The random-description null is only in the easy setting.** Table 5 runs random strings and other tasks' descriptions on a reconstruction-trained T2L with 9 targets. Random strings still score 63.5, against 55.8 for the base model. So much of that gain does not need the meaning of the description. The main zero-shot result in Table 2 has no such null.
- **Some oracles are weak.** The text says that on PIQA and Winogrande the oracle LoRA overfits and scores below the base model. That holds for PIQA, 69.9 against 72.8. On Winogrande the oracle scores 51.1 against the base model's 45.0, so the statement is wrong there. "Beats the oracle" on these tasks means little.
- **Qualitative steering rests on one problem.** Figure 4 shows one GSM8K question and four outputs. The claim that descriptions steer the reasoning style has no measure.
- **The clusters are a picture.** Figure 5 is a t-SNE plot at perplexity 4, with no score of cluster separation.
- **The weight-similarity claim is partly withdrawn.** Appendix D says that adapters with similar descriptions "are not similar in the parameter space". Appendix K then finds positive correlations in ΔW space. The authors give Appendix K themselves, which is to their credit. The Appendix D claim is true only of the factor coordinates.
- **Arrow Routing is not a like-for-like baseline.** Its numbers come from another paper, with other adapters and prompts. The authors state this.

The authors state other limits too. Users may write worse descriptions than GPT-4o mini. The output space is LoRA only, and T2L does not reach task-specific LoRAs zero-shot.

## Use in this repo

- **Nothing here runs it.** The repo has no LoRA or fine-tuning code, as [`fixing-models.md`](fixing-models.md) notes. None of the four explored models is in the paper. `meta-llama/Llama-3.1-8B-Instruct`, the base model of Table 7, is in `signed_models` in `pyproject.toml`. But the repository holds no record for it, so no VERIFIED pole of that model exists to test.
- **T2L's adapters do not touch the registry's units.** This is our inference from Listing 1, not a claim of the paper. The adapters change only `q_proj` and `v_proj`. The adapter leaves a pole's own Wgate, Wup and Wdown as they were, and changes only the residual stream that the pole reads. A weight-only reading at a pole, as #5133 plans, would see zero change under such an adapter. Whether the pole keeps its explanation is then a question for forward passes, as in [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md).
- **The base-instruct numbers are for a different change.** There, 38 of 43 Qwen2.5-1.5B hypotheses and 252 of 311 Qwen2.5-0.5B hypotheses keep under instruction tuning. That tuning changed the MLP weights too. An attention-only adapter is a narrower change, and the repo has no evidence on it.
- **Factor coordinates hide shared function.** Appendices D and K are the paper's clearest evidence for this repo. In A and B coordinates, adapters with similar tasks look unrelated (Pearson about 0). In ΔW they correlate weakly (0.17 to 0.38). #5133 already reads the full Δ = instruct − base, not factors, so it avoids this trap. The same lesson applies to any weight comparison with a symmetry, such as a permutation of neurons.
- **The repo matches units by activity, not by weights.** [`docs/analyses/sister-neurons.md`](../analyses/sister-neurons.md) pairs poles across models by their correlation over `corpus.jsonl`. The paper's Figure 6 shows why that choice is safer. Two adapters with similar function can have weight vectors that look unrelated.
- **Can weights check an explanation without forward passes?** The paper does not try. It goes from text to weights, the reverse of this repo's path from units to text. Every check of a generated adapter is a benchmark forward pass, or a t-SNE of the hypernetwork's own activations. The repo's evidence agrees. In [`docs/analyses/weight-kurtosis.md`](../analyses/weight-kurtosis.md), ROTATE's kurtosis signals score AUCs of 0.51 (0.45–0.57) and 0.54 (0.48–0.60) on 518 test poles. Both intervals include 0.5. ROTATE itself is still open as #3446.
- **A text-to-weights map with no unit content.** Like [`hypernetworks.md`](hypernetworks.md), the paper names no neuron, probe or explanation of a unit. Its descriptions explain tasks, not units. #4471 (Mehta et al. 2026) writes weights from code that an agent can read. T2L writes them from a learned map that no one reads. So T2L adds nothing to the question of whether written weights line up with units.
- **Evidence for the fix campaign, not a design.** A description-made adapter raised Mistral's 10-task mean from 55.8 to 67.7. The paper measures only those benchmarks. It reports no score on other skills and no measure of forgetting. [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md) shows why that gap matters for a narrow fix. The random-string row of Table 5 is the kind of matched null that `fixing-models.md` asks of an edit.
- **No follow-up issue.** The weight-only test is #5133, and the keep test under fine-tunes is #5125. The Appendix K result supports #5133's choice to read full Δ, and belongs there as a comment.
- **Question for the maintainer.** Should #4782 count an adapter that leaves a pole's own weights unchanged as a test of that pole's stability? T2L's adapters are of this kind. If so, an attention-only fine-tune would test whether a pole keeps when only its inputs move. The page records this and opens no issue.
