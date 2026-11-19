# Titans: Learning to Memorize at Test Time

**Citation.** Ali Behrouz, Peilin Zhong and Vahab Mirrokni, all at Google Research. "Titans: Learning to Memorize at Test Time". arXiv 2501.00663 [cs.LG], v1 31 December 2024. [Paper](https://arxiv.org/abs/2501.00663). Preprint, no venue given. The arXiv page lists only v1, and we read v1. The paper links no code and no data. The conclusion says that the authors "intend to make the code we used to train and evaluate our models available soon". The paper has no funding section, no acknowledgements and no statement about tools or help from an LLM. Issue #4846 gives the year as 2024, which agrees with the arXiv date.

**Status.** Planned. #4782 would use it in its strand on weight-space programming and fast weights, and #5144 asks the question that it bears on most. No code, skill or protocol in this repo depends on it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** We read the arXiv v1 PDF on 2026-11-19. We extracted its text with `pdftotext` and read all of it. That covers the abstract, Sections 1 to 6, Tables 1 to 5, the references and Appendices A to C. The model that wrote this page (Claude Opus 5.5) read the text itself, with no subagent. We did not see the figures as images. So the results of Figures 6 to 9 are here only as the body text and the captions give them. That includes BABILong (Figure 6), memory depth (Figures 7 and 8) and throughput (Figure 9). Some equations came out of the extraction with their layout broken. We read them against the text around them. We also read the arXiv abstract page for the version history.

## Summary

**Claim.** A small MLP can act as a long-term memory that learns while the model reads. Gradient steps on a key–value loss write each new token into its weights, with momentum and a forget gate. Combined with attention over a limited window, this memory gives "Titans". The authors report that Titans beat Transformers and recent linear recurrent models on language modelling, recall, time series and DNA tasks.

**Method.**

- **The memory and its write (§3.1).** The memory M is an MLP with L_M ≥ 1 layers. Linear maps make a key k_t = x_t W_K and a value v_t = x_t W_V. The inner loss is ℓ(M_{t−1}, x_t) = ‖M_{t−1}(k_t) − v_t‖² (Eq. 12).
- **"Surprise" and forgetting (Eqs. 13–14).** S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}, x_t), and M_t = (1 − α_t) M_{t−1} + S_t. The gradient is the "momentary surprise", and S is a momentum term, the "past surprise". The gate α_t ∈ [0, 1] decays the memory. All three of α_t, θ_t and η_t depend on the token.
- **Two loops.** The inner loop trains M's weights at test time. The outer loop trains every other parameter, W_K and W_V included, by ordinary training. The paper reads the memory with a forward pass and no weight update: y_t = M*(q_t) with q_t = x_t W_Q (Eq. 15).
- **Parallel training (§3.2).** The sequence splits into chunks of size b. Within a chunk, all gradients use the weights from the start of the chunk, so the work is matrix products. The momentum is a linear recurrence, computed with a parallel scan.
- **Persistent memory (§3.3).** N_p learned vectors, the same for every input, go before the sequence. The paper links them to Sukhbaatar et al. (2019), where an FFN is attention with keys and values that do not depend on the input (Eq. 20).
- **Three ways to add the memory (§4).** Memory as a Context (MAC) reads the memory for each segment and puts the result before the segment, under full causal attention. Memory as a Gate (MAG) runs sliding-window attention and the memory side by side, with a gate. Memory as a Layer (MAL) puts the memory before sliding-window attention. "LMM" is the memory alone, with no attention.
- **Theory (§4.4).** Theorem 4.1 states that Titans can solve problems outside TC⁰, unlike Transformers, diagonal linear recurrent models and DeltaNet.
- **Language models (§5.1, §5.2).** 170M, 340M, 400M and 760M parameters. The first three train on 15B tokens of FineWeb-Edu, and the 760M model on 30B. The setup follows Yang, Kautz and Hatamizadeh (2024): Llama 2 tokenizer (32K), 4K training length, AdamW at 4e-4, 0.5M-token batches. Appendix B says that the 400M baseline results come from that paper.
- **Baselines.** Transformer++, RetNet, GLA, HGRN2, Mamba, Mamba2, DeltaNet, TTT, Gated DeltaNet, and the hybrids Samba and Gated DeltaNet-H2.
- **Other tasks.** S-NIAH from RULER at 2K to 16K (§5.3). BABILong, few-shot and fine-tuned, with baseline results from Kuratov et al. (2024) (§5.4). Memory depth L_M = 1 to 4 on a subset of the Pile (§5.5). Time-series forecasting in the Simba framework (§5.6). GenomicsBenchmarks, with baseline results from Arora et al. (2024) (§5.7).

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| 340M / 15B tokens, Wiki ppl and mean accuracy of 8 tasks (Table 1) | Transformer++ 31.52, 42.92. Gated DeltaNet 27.01, 45.42. LMM 26.18, 46.17. MAC 25.43, 47.36. MAG 25.07, 47.54. MAL 24.69, 46.55 |
| 400M / 15B tokens (Table 1) | Transformer++ 30.63, 45.64. Gated DeltaNet-H2 24.19, 47.69. LMM 25.03, 47.83. MAC 25.61, 48.65. MAG 23.59, 48.60. MAL 23.93, 47.87 |
| 760M / 30B tokens (Table 1) | Transformer++ 25.21, 48.69. Gated DeltaNet-H2 19.88, 51.49. LMM 20.04, 51.56. MAC 19.93, 52.51. MAG 18.61, 52.50. MAL 19.07, 50.97 |
| S-NIAH-W at 16K, accuracy (Table 2) | TTT 0.0, Mamba2 0.0, DeltaNet 0.0, LMM 80.6, MAC 95.2, MAG 88.2, MAL 90.4 |
| S-NIAH-PK at 16K (Table 2) | TTT 88.4, Mamba2 5.4, DeltaNet 71.4, LMM 96.2, MAC 98.4 |
| Ablation from LMM, ppl / reasoning / long context (Table 5) | LMM 27.01 / 47.83 / 92.68. Linear memory 28.49 / 46.97 / 85.34. No momentum 28.98 / 45.49 / 87.12. No weight decay 29.04 / 45.11 / 85.60 |
| Ablation, other rows (Table 5) | No convolution 28.73 / 45.82 / 90.28. No persistent memory 27.63 / 46.35 / 92.49 |
| Time series, ETTm1 MSE (Table 3) | Neural memory 0.358, Simba 0.383, PatchTST 0.387 |
| Genomics, top-1 accuracy (Table 4) | Neural memory highest on Enhancer Cohn (75.2) and Enhancer Ens (89.6). HyenaDNA higher on Human Reg. (93.8 against 89.3) and Human OCR Ens. (80.9 against 79.9) |
| BABILong (Figure 6, text only) | "Titans (MAC) outperforms all baselines, including extremely large models, e.g., GPT4." No numbers in the text |
| Memory depth (§5.5, text only) | Deeper memory gives lower perplexity at all lengths, and slower training. No numbers in the text |
| Context length (abstract, §1, §6) | Titans "scale to larger than 2M context window size". No table in the text gives a length above 16K |

## Where the claims go beyond the evidence

- **The 2M-token claim has no number in the text.** The abstract and the conclusion claim more than 2M tokens of context. Table 2 stops at 16K. The only other long-context result is Figure 6, which we could not read. So we can't check the claim from the text.
- **No seeds, intervals or repeats.** Every table gives one number per cell. Many gaps in Table 1 are under one point of accuracy, for example MAC 48.65 against MAG 48.60 at 400M.
- **Baselines come from other papers.** Appendix B says that the 400M baselines come from Yang, Kautz and Hatamizadeh (2024). The BABILong and genomics baselines also come from earlier papers. So the paper did not train every row with the same code.
- **"Best in all three tasks" is not true of every cell.** §5.3 says that the neural memory "achieves the best results compare to baselines in all three tasks". In Table 2, S-NIAH-PK at 4K gives TTT and DeltaNet 98.8 against LMM's 98.4. At 8K, DeltaNet scores 98.6 against 98.2.
- **"Outperforming all baselines" in time series has an exception.** On Traffic, iTransformer's MAE is 0.282 against the neural memory's 0.289 (Table 3). In genomics, the text says "competitive", which fits Table 4 better.
- **One average in Table 1 does not match its row.** For TTT at 760M, the mean of the eight accuracy columns is 49.37, but the table gives 47.32. The other rows agree with their columns within 0.11. MAL's SIQA at 760M is 30.98, against 38 to 42 for every other model. We can't tell from the text whether these are typos.
- **Theorem 4.1 has no proof.** The paper states it in §4.4 with no proof, sketch or appendix. No experiment tests state tracking.
- **The ablation is on one scale.** Table 5's LMM row matches the 400M row of Table 1 (reasoning 47.83). Its perplexity, 27.01, is the mean of Wiki and LMB perplexity, 25.03 and 28.99. We infer this, because the paper does not say it. No ablation is at another scale.
- **Inconsistencies within the text.** §2.1 calls f "the read" and g "the write", but Eqs. 6–7 label f as the write and g as the read. Figure 7's captions give 170M, 360M and 760M models, and §5.1 lists 170M, 340M, 400M and 760M. In Table 1, the 760M Titans rows lack the ∗ that marks hybrids. Eq. 32 has M_t on both sides.
- **No code at the time of v1.** The authors promise code "soon", so no reader can run the experiments from v1.
- **The memory analogy is an argument, not a result.** The paper takes "surprise" and "long-term memory" from human memory research. No experiment tests whether the memory stores surprising tokens more, or whether it acts as a long-term store.

The authors state some limits themselves. A footnote says that v1 aims at "insights/evidences" and that larger models will come in a later version. §5.5 says that deeper memory trains more slowly. §4.3 names MAL's main drawback.

## Use in this repo

- **The memory is an MLP whose weights the text writes.** In [`ffn-key-value-memories.md`](ffn-key-value-memories.md), an MLP neuron's input weights are a key and its output weights a value, and training writes them. Titans' memory has the same form, but an inner gradient loop writes it during each forward pass (Eq. 12). The paper itself links the persistent memory to that FFN view (Eq. 20). So a Titans model has two kinds of MLP: fixed ones, as in this repo's models, and one that changes with every token.
- **A weight-only reading could not see what the memory holds.** The trained weights of a Titans model give only W_K, W_V, W_Q, the gates and M_0. The memory's state M_t depends on the text read so far. This is the limit of [`fast-weights-recent-past.md`](fast-weights-recent-past.md) and [`linear-transformers-fast-weight-programmers.md`](linear-transformers-fast-weight-programmers.md), now for a deep memory. A weight-only check, such as ROTATE (#3446), can't test a rule held in M_t. That is an inference from the paper's equations, not a result that it measures.
- **So the paper supports the ceiling that #5144 asks to count.** #5144 asks how many VERIFIED explanations are token sets and how many need context. Titans moves recall from attention into weights that the text writes, and those weights are still not in the checkpoint. [`docs/analyses/weight-kurtosis.md`](../analyses/weight-kurtosis.md) finds that ROTATE's weight-only kurtosis does not predict VERIFIED. On 518 test poles, the AUC is 0.51 (0.45–0.57) for the output side and 0.54 (0.48–0.60) for the input side.
- **A unit in the memory would have no fixed explanation.** The registry explains a unit by one rule over all text. A neuron inside M_t has weights that differ from one text to the next. Its explanation would depend on the context before it, not only on the token. The protocol has no form for that claim, and this repo has no such model.
- **The forget gate is evidence on interference, inside one sequence.** In Table 5, removing weight decay costs the most: perplexity rises from 27.01 to 29.04, and long-context accuracy falls from 92.68 to 85.60. Removing momentum costs nearly as much. So, in this memory, old writes damage new ones unless the gate decays them. That is the interference question of #4782, in the inner loop of one model. It is not a test of fine-tuning, and it gives no number for the repo's models.
- **The write size has a parallel in #5118 and #5133, but only as an analogy.** Titans writes each token in proportion to the gradient of its key–value loss. #5118 asks whether a pole's diagonal Fisher, a squared-gradient measure, predicts whether it keeps under instruction tuning. #5133 asks the same of the weight change at the pole. [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) has the rows: on Qwen2.5-0.5B, 252 of 311 poles keep (81%). Titans is evidence that gradient size can steer what a memory keeps. It does not test that a large gradient predicts which units change.
- **Weight-space programming (#3381, #4471).** In Titans, gradient descent writes the memory at run time, and nobody reads the written weights. #4471 has an agent write weights as code, to be read. Titans uses a fixed learning rule with learned rates, not a program. The paper does not test whether anyone can read the memory.
- **A model with this memory is a possible case, but not now.** Appendix C calls the LMM a generalised Gated DeltaNet. [`models-rewrite-own-weights.md`](models-rewrite-own-weights.md) reports that Qwen3.5 mixes Gated DeltaNet linear attention with full attention, and #3331 proposes a Qwen3.5-0.8B fine-tune. In such a layer, `openinterpretability/harness/heads.py` finds no N×N attention pattern to score. The four configured models (`pyproject.toml`) use softmax attention. The registry adds no new model before every model meets its coverage target, so this is not work for now.
- **No single-neuron content.** The paper gives no unit explanations, no probes and no null. Nothing in it bears on any pole's verdict.
- **No new follow-up.** #5144 already asks for the count that this paper's argument bears on. #5118 and #5133 already ask the gradient questions. The paper is evidence for those issues, not a reason for a new one.
- **Question for the maintainer.** If a model with a memory that the text writes joins the registry, what is a unit in it? The memory's neurons change with each text, so a VERIFIED claim would need a form that names the context. That would change the decision rule, so it would need a `PROTOCOL_VERSION` bump (`docs/PROTOCOL.md` §6). We record it as a question only.
