# QLoRA: Efficient Finetuning of Quantized LLMs

**Citation.** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman and Luke Zettlemoyer, all of the University of Washington. The paper marks Dettmers and Pagnoni as equal contributors. "QLoRA: Efficient Finetuning of Quantized LLMs". [arXiv:2305.14314](https://arxiv.org/abs/2305.14314), version 1 of 23 May 2023, the only version on arXiv. Venue: Advances in Neural Information Processing Systems 36 (NeurIPS 2023). The NeurIPS proceedings site lists the paper under that title. The arXiv version still says "Preprint. Under review.", so it is the text before the venue's review. The code is at [github.com/artidoro/qlora](https://github.com/artidoro/qlora), and the 4-bit kernels are in [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes). The authors say that they release their models, adapters and generations with human and GPT-4 labels. The text names no funding. The acknowledgements thank the Hyak supercomputer of the University of Washington.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it in its LoRA strand. This page proposes one follow-up that rounds weights only and trains nothing. No code, skill or protocol in this repo depends on the paper. The maintainer designs the campaign to fix models, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv v1 PDF (26 pages) with `curl` and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own text the abstract, Sections 1 to 9 and the acknowledgements. We also read Appendices A to G and the references. We read Tables 1 to 13 as extracted text, except Table 11's layout, which we read as rows of numbers. We did not see the figures as images. So this page gives only their captions and the numbers that the text states, for Figures 1 to 6. We did not read the NeurIPS camera-ready text, only the proceedings page's title and abstract. We did not read the released code.

## Summary

**Claim.** A language model's frozen weights can be stored in 4 bits while LoRA adapters train on top of them. The authors claim that this matches 16-bit full fine-tuning and 16-bit LoRA on their benchmarks. It lets one 48 GB GPU fine-tune a 65B model.

**Method.**

- **Storage and compute.** QLoRA stores each base weight in 4-bit NormalFloat (NF4). Before each product, it turns the weight back into BFloat16, and the product runs in 16 bits. Only the LoRA weights have gradients, in BFloat16. Equation 5 gives one layer's output as X·doubleDequant(W) plus X·L1·L2.
- **NF4.** A data type with 16 values. Each value is a quantile of a standard normal distribution, so each bin holds the same expected share of normal weights. It has an exact zero. Appendix E lists the 16 values, from −1.0 to 1.0.
- **Blocks.** QLoRA scales each block of 64 weights by its largest absolute value. Each block keeps its own scale.
- **Double Quantization (DQ).** The block scales are themselves stored in 8-bit floats, in blocks of 256.
- **Paged Optimizers.** NVIDIA unified memory moves optimizer state to CPU memory when the GPU runs out.
- **LoRA settings.** Adapters on all linear layers of the base model. For the chatbot runs: r = 64, α = 16, dropout 0.1 up to 13B and 0.05 above, constant learning rate (Appendix B.2).
- **Comparison with 16 bits (Section 4).** RoBERTa-large on GLUE and T5 (80M to 11B) on Super-NaturalInstructions, against 16-bit full fine-tuning and 16-bit LoRA. LLaMA 7B to 65B on Alpaca and FLAN v2, scored by 5-shot MMLU, against 16-bit LoRA. Data types after quantization alone, on OPT, BLOOM, Pythia and LLaMA from 125M to 13B, by perplexity and zero-shot accuracy.
- **Chatbots (Section 5).** LLaMA 7B to 65B tuned on eight instruction datasets. The paper names the OASST1 runs Guanaco. It scores them by MMLU, by GPT-4 scores relative to ChatGPT on 80 Vicuna prompts, and by Elo ratings. GPT-4 and Amazon Mechanical Turk workers judge the Elo matches, on Vicuna and on 953 OASST1 queries.
- **Weight normality (Appendix F).** A Shapiro-Wilk test on the weights of each hidden unit of LLaMA 7B.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Memory (Section 1) | Fine-tuning a 65B model drops from more than 780 GB to less than 48 GB of GPU memory |
| Double Quantization (Section 3) | Scale overhead falls from 0.5 to 0.127 bits per weight, about 3 GB for a 65B model |
| Data types after quantization only, mean Pile Common Crawl perplexity (Table 2) | Int4 34.34, Float4 E2M1 31.07, Float4 E3M0 29.48, NF4 with DQ 27.41. The table has no 16-bit row |
| LLaMA 7B to 65B, mean 5-shot MMLU over 8 cells (Table 4) | BFloat16 LoRA 53.0, Float4 52.2, NF4 with DQ 53.1 |
| Same, single cells (Table 4) | 7B on FLAN v2: BFloat16 45.6, NF4 44.5. 33B on FLAN v2: 60.5 against 59.2. 65B on FLAN v2: 62.5 against 63.9 |
| T5-3B, Super-NaturalInstructions RougeL (Table 3) | BFloat16 full fine-tuning 54.3, BFloat16 LoRA 55.4, NF4 with DQ 55.3 |
| RoBERTa-large, GLUE (Table 3) | BFloat16 88.6, LoRA BFloat16 88.8, Int8 88.8, FP4 88.6. No NF4 entry |
| Where LoRA goes (Section 4, Figure 2) | LoRA on the query and value matrices alone does not match full fine-tuning for LLaMA 7B. LoRA on all linear layers does |
| LoRA rank (Appendix A, Figure 4) | With LoRA on all layers, r from 8 to 64 is "unrelated to final performance". Three seeds per setting |
| Vicuna benchmark, share of ChatGPT's score (Table 6) | Guanaco 65B 99.3% (95% CI 4.4%), 33B 97.8%, 13B 90.4%, 7B 87.0% |
| MMLU before and after tuning, 65B (Table 5) | Untuned LLaMA 63.4. After tuning: from 56.7 (Self-Instruct) to 63.9 (FLAN v2). Seven of the eight datasets score below the untuned model |
| MMLU before and after tuning, 13B (Table 5) | Untuned 46.9. Self-Instruct 33.3, FLAN v2 51.4 |
| Judges (Section 5.3, 6.2) | GPT-4 against humans at system level: Kendall τ 0.43, Spearman 0.55. At example level: Fleiss κ 0.25. Humans among themselves: Fleiss κ 0.42 |
| GPT-4's own Elo on Vicuna (Table 7) | 1348 when GPT-4 judges, 1176 when humans judge |
| Data size against data choice (Appendix B.4) | More data or epochs adds 0.0 to 0.5 MMLU points. The choice of dataset moves it by 1.5 to 8.0 |
| Bias, CrowS average (Table 8) | LLaMA-65B 66.6, Guanaco-65B 43.5. Lower is less biased |
| Weight normality (Appendix F) | At a 5% threshold, 7.5% of LLaMA 7B's hidden units have weights that are not normal |

## Where the claims go beyond the evidence

- **"Without any performance degradation" is wider than the tests.** Section 1 says the paper shows this "for the first time". Section 8 says the authors "did not establish" that QLoRA matches 16-bit full fine-tuning at 33B and 65B. At those sizes the comparison is with 16-bit LoRA only. The authors state this limit themselves.
- **Matching rests on means, with no intervals.** Tables 3 and 4 give no seeds or intervals. The mean in Table 4 matches, 53.1 against 53.0. But single cells differ by up to 1.4 points in both directions, for example 59.2 against 60.5 on 33B FLAN v2. So "matches" means the mean only.
- **Quantization alone has no 16-bit reference.** Table 2 compares 4-bit types with one another. It does not give the 16-bit perplexity, so the loss from rounding before tuning is not stated in the text.
- **NF4 is missing where the text claims it.** Table 3 has no NF4 entry for RoBERTa-large, and no 16-bit replication for T5-11B. The Section 4 summary still says NF4 matches 16-bit full fine-tuning.
- **Tuning lowered MMLU in most cells.** Table 5 shows that seven of eight 65B runs score below untuned LLaMA. The paper reads these numbers as dataset suitability, not as forgetting. It does not compare forgetting under QLoRA with forgetting under 16-bit tuning.
- **Paged optimizers are not measured.** Section 4 says that the authors "do not provide hard measurements" for them, apart from one speed comparison.
- **The tables disagree in places.** In Table 6, the Open Assistant 33B and Vicuna 13B rows are identical in all four numbers. Guanaco 13B has an Elo of 916 in Table 1 and 913 in Table 7. Table 8's caption says Guanaco "follows the biased pattern" of LLaMA, but the text says that tuning lowers the bias.
- **The text has gaps.** Appendix B.4 ends with "as discussed in ." with no target. Appendix F ends in a sentence that stops after "neural network weights".
- **The chatbot judges are weak.** The authors report the GPT-4 order effect, its preference for its own outputs and κ = 0.25 against humans. They call the qualitative analysis "by no means comprehensive". These are the authors' own caveats.
- **The normality test is loose.** Appendix F finds 7.5% non-normal units against 5% expected. The authors say the p-value may not be accurate at these sample sizes. NF4's optimality depends on this normality.

## Use in this repo

- **The repo has no quantization, LoRA or fine-tuning code.** `pyproject.toml` lists neither `bitsandbytes` nor `peft`. [`docs/analyses/cpu-acceleration.md`](../analyses/cpu-acceleration.md) left int8 dynamic quantization untested, because GPT-2's MLP uses `Conv1D` and would not be converted. [`fixing-models.md`](fixing-models.md) notes that the repo has no fine-tuning or weight-edit code.
- **The paper measures behaviour, not units.** Its scores are benchmarks and judges. The only unit-level result is Appendix F's normality test. It says nothing about neurons, explanations, probes or random baselines.
- **4-bit rounding is a weight change that the repo has not measured.** NF4 rounds every weight to one of 16 values per block of 64. Activations are then computed in BFloat16. So an NF4 model's activations carry the BFloat16 rounding and a coarser weight rounding on top. This is our reading of Section 3, not the paper's claim.
- **The repo's precision numbers set the scale for that question.** [`docs/analyses/half-precision-reproduce.md`](../analyses/half-precision-reproduce.md) reproduced 24 float32 records in BFloat16. Most numbers moved by about 1% of the scale, and no peak moved by more than 7%. In 2 of the 24 records, a holdout crossed its threshold and its result flipped. [`docs/analyses/neuron-float32-floor.md`](../analyses/neuron-float32-floor.md) found float32 moves of up to 0.005 on Pythia-70m. NF4 rounds weights, not activations, and far more coarsely. So it can move poles more than BFloat16 does. No run here measures by how much.
- **Does a VERIFIED explanation survive NF4?** The repo cannot say yet. The paper's Table 2 suggests that NF4 loses less than other 4-bit types. It does not show the loss against 16 bits, and it measures perplexity, not units. The follow-up below asks the question directly.
- **A merged QLoRA adapter carries two changes.** The adapter trains against the rounded weights. Section 4 says adapter tuning "fully" recovers what rounding loses. So part of the adapter may undo rounding. A merge into the 16-bit weights then adds that part to weights that were never rounded. This is our inference, and the paper does not discuss merges. For #4782's question on merged adapters, the NF4 change must be measured first, or a QLoRA result mixes the two.
- **The base-instruct test is the measure to reuse.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) re-tests VERIFIED poles on Qwen2.5 instruct models at the same index. On 1.5B, 38 of 43 keep (88%), at a median correlation of 0.983. On 0.5B, 252 of 311 keep (81%), at a median of 0.974. `scripts/base_instruct.py` scales each probe by the ratio of the pole's maxima in the two models. The same script could compare a base model with its NF4 copy.
- **Bearing on the open follow-ups.** #5125 fine-tunes Qwen2.5-0.5B on narrow tasks. QLoRA would make that cheap, but its NF4 base would add a rounding change to each stage. Qwen2.5-0.5B fits a 64 GB machine in 16 bits, so the run need not use QLoRA. #5154 adds a random weight change of the instruct change's size. NF4 rounding is another weight change of known size, so it is a second null for #5154. #5133 reads the rank of Δ. The paper finds that LoRA rank barely matters once all layers carry adapters, which fits a low-rank Δ being enough.
- **All linear layers means the MLP.** The paper needs adapters on all linear layers, MLP included, to match full fine-tuning. The repo's units are MLP neuron poles. So the adapters that work best write into the weights that the registry explains. [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md) reports the same for Llama-2-7B, where "most gains" come from the MLP modules.
- **Table 5 is weak evidence of forgetting.** At 65B, seven of eight QLoRA runs score below the untuned model on MMLU, by up to 6.7 points. The narrow, distilled datasets score lowest. That fits [`complementary-learning-systems.md`](complementary-learning-systems.md), where focused learning damages related knowledge. The paper has one run per cell and no intervals, so this is a hint only.
- **Evidence for the fix campaign, not a design.** QLoRA makes a narrow fine-tune cheap, for example one aimed at the #4781 " days" error. The paper also shows that such a tune can lower an unrelated score. [`fixing-models.md`](fixing-models.md) holds the repo's synthesis.
- **A question for the maintainer.** Should a record ever be made on a quantized model, for example a large signed model such as `meta-llama/Llama-3.1-8B-Instruct`? A record pins its dtype, and `reproduce` has no rule for a 4-bit weight type. That is a protocol choice, so this page records it as a question and opens no issue.
- **A follow-up that trains nothing.** Round each linear weight of Qwen2.5-0.5B and Qwen2.5-1.5B to NF4 in a script. Use Appendix E's 16 values, blocks of 64 and the largest-value scale. This needs no `bitsandbytes`, so it runs on the CPU or MPS. Then test each VERIFIED pole again as `scripts/base_instruct.py` does. Compare the keep share with 81% and 88%, and the drift with the BFloat16 numbers above. Also record whether poles whose weight rows fail the Appendix F normality test lose their explanation more often. The keep share without rounding is the baseline for any later QLoRA run in #4782.
