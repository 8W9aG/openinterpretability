# Mechanistic Insights into LoRA: Layer Sparsity via Path Patching

**Citation.** Fei Zuo (East China Normal University), Yizhou Huang and Kezhi Wang (Brunel University of London). "Mechanistic Insights into LoRA: Layer Sparsity for Adaptive Fine-Tuning via Path Patching". arXiv:2610.34756 [cs.LG], v1, 26 October 2026. [Paper](https://arxiv.org/abs/2610.34756), licence CC BY 4.0. The arXiv comment says "Accepted at NeurIPS 2026 (Main Track)". The paper gives no code link. Its checklist says that a code release is "planned under the MIT license". The tools are PyTorch, Hugging Face Transformers, PEFT and TransformerLens. Funding is from UKRI, Horizon Europe, the MSCA, the Royal Society and the NVIDIA Academic Grant Program. **Mismatch:** the title and the introduction name path patching, but the method that the body describes is a zero-ablation of whole blocks (see below).

**Status.** Planned. #5125 asks whether VERIFIED explanations keep through narrow fine-tunes, and a LoRA arm of it could use this paper's layer choice as one arm. No code, skill or protocol in this repo depends on it. The maintainer designs the campaign to fix models (#4784, #5220), so this page reports evidence and records questions only.

**Read.** On 2026-11-21 a `web-fetch` subagent (Claude) read the arXiv abstract page and the full HTML text in four consecutive parts. We have every section, from the abstract to the NeurIPS checklist, in the source's own words. That includes the appendices and Tables 1–18. We saw Figures 1–6 only as captions. We did not read the PDF or the TeX source. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** Standard LoRA adapts every layer, but layers contribute unequally. A cheap causal score on the frozen base model picks the layers that matter for a task. LoRA on about 30% of the layers then matches or beats LoRA on all of them. The important layers depend on the task, and two tasks share few of them.

**Method.**

- **The analysis model.** Qwen2.5-0.5B (24 layers), with rank-8 LoRA on Q, K and V. The training data is Alpaca-ZH, Alpaca-EN, Code-Alpaca and Dolly.
- **The score.** For each component c, S_c(x) = |p_LoRA(t*|x) − p_ablate(c)(t*|x)|, where t* is the top token of the full LoRA model. The importance I_c is the mean of S_c over a calibration set. The paper calls this activation patching in Section 3.1.
- **The layer choice (Adaptive-LoRA).** On the frozen base model, with no adapter, it zeroes each layer's attention output, and then its MLP output. Each layer scores the mean of |p − p_attention zeroed| + |p − p_MLP zeroed| over N = 30 inputs. It keeps the top k ≈ 0.3L layers. LoRA then trains on those layers only. An "efficiency mode" keeps only layers at or below a depth d*.
- **Training.** Rank 16, α = 32, AdamW at 1e-4, 3 epochs, effective batch 64, on 10,000 samples of commonsense_170k or 5,000 of math_10k.
- **Evaluation.** 8 commonsense tasks (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-E, ARC-C, OBQA) and 3 math tasks (GSM8K, SVAMP, AQUA), greedy and zero-shot.
- **Models.** Qwen2.5-1.5B is the primary model. Others are Qwen2.5-0.5B and 3B, a LLaMA-3 3B model, Qwen3-8B and Qwen3-32B, and in the appendices Qwen-7B, LLaMA-7B, LLaMA-13B and Llama-3.1-8B.
- **Baselines.** LoRA at ranks 4 to 32, LoRA+ and DoRA, on Q, K and V in every layer. Random, sequential and reverse layer choices appear only in a perplexity test. InfLoRA, GainLoRA and TreeLoRA appear in one appendix table.

**Findings, as the source reports them.** All numbers are from the paper's text or tables.

| Setting | Result |
| --- | --- |
| Qwen2.5-0.5B, four datasets, rank-8 LoRA on Q/K/V | The MLP has 58.4% of the measured effect, attention 41.6% (range 57.3–59.5% for the MLP) |
| Same | V has about 46% of the attention effect, Q and K about 27% each |
| Same, top 7 of 24 layers | Mean Jaccard overlap of the chosen layers across six task pairs: 39.5% (27.3–75.0%) |
| Same, 100 inputs, five adapter settings | The base model's layer ranking agrees with the trained adapter's: Spearman 0.630–0.893 |
| Qwen2.5-1.5B, rank 16, 8 of 28 layers, seed 42 (Table 1) | Commonsense average 59.1, against 56.9 for LoRA, 57.9 for LoRA+ and 55.5 for DoRA |
| Same, rank 32 | Commonsense 62.2 against 60.4 for LoRA. Math 47.8, against 45.5 for LoRA and 47.3 for LoRA+. A 3.5× smaller adapter |
| Table 2, rank 16, seed 42, commonsense, Adaptive against LoRA | Qwen2.5-0.5B +1.5, Qwen2.5-1.5B +2.2, Qwen2.5-3B +1.9, LLaMA-3 3B +1.2, Qwen3-8B +0.0, Qwen3-32B −0.2 |
| Alpaca-ZH perplexity, 12 of 36 layers (Table 9) | Adaptive 12.32, sequential 14.06, reverse 15.00, random 16.42 |
| Calibration size, Qwen2.5-1.5B (Table 12) | N = 10 gives 59.4, N = 30 gives 59.8, N = 100 gives 59.7 |
| Five calibration draws of N = 30 (Table 14) | Jaccard overlap of the chosen layers 0.617–0.747 across six models |
| Three seeds, eight settings (Table 10) | Mean paired gains of +0.17 to +1.57 points |
| Calibration cost | O(NL) forward passes with no gradient, 20–62 s on one A100 for N = 30 |

## Where the claims go beyond the evidence

- **No path patching in the method.** The title and the introduction say path patching. The body zeroes whole block outputs on single inputs and reads one token's probability. It has no clean and corrupted pairs and no sender or receiver paths. So the scores are zero-ablation effects, with the usual caveats of zero-ablation (below).
- **Zero-ablation is off-distribution.** A zeroed block output is a value that the model never sees. A large score can measure that shock, not the block's normal role. The paper has no mean or resample ablation to compare with.
- **The metric reads one token's probability.** |Δp| of the top token saturates near 0 and 1, and it ignores the other tokens. The paper gives no logit-difference or KL check.
- **The main results have one seed.** Tables 1 and 2 use seed 42 only. The primary model, Qwen2.5-1.5B, has no multi-seed result. The gains are 1–2 points on about 1,000 items per task. At 60% accuracy, one task's standard error is about 1.5 points (our arithmetic), so several single gains are within noise.
- **The gain disappears on the larger models.** Qwen3-8B gives +0.0 and Qwen3-32B gives −0.2 at seed 42. Their three-seed means are +0.17 and +0.23.
- **The layer controls are weak and partial.** Random, sequential and reverse choices appear only for perplexity, with one run each and no spread over random draws. In Table 9, sequential beats Adaptive at 28 and 32 layers, and random beats it at 32. The paper does not discuss those rows.
- **No matched-parameter baseline.** The comparisons are at equal rank, so Adaptive-LoRA has fewer parameters. That favours the paper's point, but it leaves open whether a smaller all-layer adapter would do as well.
- **Inconsistent numbers.** Table 1 says N = 250 for a score of 59.1, and Table 2 gives the same 59.1 with N = 30. Table 12 gives 59.8 at N = 30 for the same setting. The text puts calibration at "1.5–2 hours", but Table 13 gives 4.1–6.3% of a 713–1,240 s run. Table 6 reports Spearman ρ where the text says Pearson. Model names vary between tables.
- **No code yet.** The checklist says that a release is planned, with no link.

The authors give some limits themselves. They select whole layers, not heads or neurons. They test only commonsense and math tasks. They say that they "lack a formal theoretical framework" for why early layers matter most. They fix the chosen layers before training.

## Use in this repo

- **The repo has no LoRA code.** As [`lora-without-regret.md`](lora-without-regret.md) notes, `openinterpretability/` and `scripts/` train no adapters. So this page changes no code now.
- **The paper's unit is a block, not a neuron.** It scores whole attention and MLP outputs per layer. The registry's units are single neuron poles and heads. So its finding that the MLP carries 58% of the effect says nothing about which poles carry it.
- **The same model, a different ablation.** The analysis model is Qwen2.5-0.5B, a configured model here. `openinterpretability/coverage.py` mean-ablates every pole of a layer and asks how much of the loss rise the VERIFIED poles bring back. The paper zero-ablates the whole block. The two are not comparable without a mean-ablation rerun of the paper's score.
- **A layer arm for #5125.** If #5125 gets a LoRA arm, this paper gives a cheap rule for which layers to adapt. The arm could then compare how many VERIFIED poles keep in adapted layers against untouched ones. The base model's own score picks the layers, so the pick is fixed before training. Any change to the poles in the untouched layers would then come only through their inputs.
- **The metric warning holds here too.** The `activation-patching-practice` skill prefers logit difference to probability, after Zhang and Nanda (2023). This paper's |Δp| of one token is the metric that the skill warns against. A rerun of its layer scores should use logit difference or KL.
- **No follow-up issue.** The work that the paper suggests sits inside #5125, which the maintainer has not designed yet. So we record it here, not as a new issue.
- **A question for the maintainer.** If #5125 gets a LoRA arm, should it adapt all layers, or only the layers that a base-model score picks? The second keeps some layers' weights fixed, which makes a clean control for the keep test. But it also changes which units the fine-tune can move.
