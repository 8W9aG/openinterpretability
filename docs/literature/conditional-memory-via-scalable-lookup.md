# Conditional Memory via Scalable Lookup (Engram)

**Citation.** Xin Cheng, Rui Tian and 19 others (Peking University, DeepSeek-AI). "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models". arXiv 2601.07372v2, 12 July 2026 (v1 12 January 2026). [Paper](https://arxiv.org/abs/2601.07372), [code](https://github.com/deepseek-ai/Engram).

**Status.** Not applicable. It is a preprint on model architecture, with no venue given. It trains new models with an added memory module, and its interpretability content is a short analysis section (logit lens, CKA, ablation, gate heatmaps). The repository has a demo of the module with the backbone stubbed out. It has no trained weights.

**Read.** A text extraction of the v2 PDF, including all appendices, on 2026-11-13. The figures are images, so numbers below come from the text, the tables and the figure labels that the extraction kept. We read the code's README but not the code.

## Summary

**Claim.** Transformers have no primitive for looking up static knowledge, so they rebuild it with computation in their early layers. A hashed N-gram embedding table ("Engram"), added beside Mixture-of-Experts (MoE), takes over that work. Total parameters and per-token FLOPs stay fixed. Moving part of the MoE's inactive parameters into Engram then lowers loss and raises benchmark scores, including reasoning, code and maths.

**Method.**

- *Retrieval.* Token IDs are first collapsed by a fixed map (NFKC, lowercasing and so on), which cuts the 128k vocabulary by 23% (Appendix C). For each position, the suffix 2-gram and 3-gram are hashed by K = 8 multiplicative-XOR heads per order into prime-sized tables. The retrieved rows are concatenated into one memory vector. Addresses depend only on token IDs, so they are known before the forward pass.
- *Fusion.* The hidden state is the query and the memory vector gives key and value. A scalar gate α = σ(RMSNorm(h)·RMSNorm(W_K e)/√d) scales the value. A depthwise causal convolution (kernel 4, SiLU) follows, and the result is added to the residual stream. With manifold-constrained hyper-connections (M = 4 branches), each branch has its own key projection and gate, and they share the table and value projection.
- *Placement.* Not every layer. The large models use layers 2 and 15 of 30.
- *Sparsity allocation.* With total and activated parameters fixed, ρ is the share of the inactive parameters given to routed experts. The rest goes to Engram slots.
- *Systems.* Training shards the tables across GPUs. At inference the tables sit in host memory, and rows are prefetched over PCIe while earlier layers run.

**Findings, as the paper reports them.** One run per configuration throughout. The paper reports no seeds, error bars or significance tests.

| Result | Number |
| --- | --- |
| Allocation sweep, 2e20 and 6e20 FLOPs (Figure 3) | U-shaped validation loss in ρ, best near ρ ≈ 75–80%. At 6e20, 1.7248 (pure MoE) → 1.7109 |
| Memory scaling, 2.58e5 to 1.0e7 slots on a 3B MoE | Loss falls linearly in log slots ("a strict power law"), more steeply than OverEncoding |
| Engram-27B vs MoE-27B, 262B tokens, 3.8B activated (Table 1) | Validation loss 1.634 → 1.622. MMLU 57.4 → 60.4, CMMLU 57.9 → 61.9, BBH 50.9 → 55.9, ARC-Challenge 70.1 → 73.8, HumanEval 37.8 → 40.8, MATH 28.3 → 30.7 |
| Engram-40B (18.5B Engram parameters) | Validation loss 1.610. Better than Engram-27B on most benchmarks, worse on HumanEval (38.4), MBPP (46.2) and C3 (61.8) |
| Long context, RULER at 32k (Table 2) | Multi-query NIAH 84.2 (MoE, 50k steps) → 97.0 (Engram, 46k steps, same pre-training loss). Variable tracking 77.0 → 87.2 |
| Logit lens (Figure 4a) | KL from each layer's readout to the final output is lower for Engram, most in early layers |
| CKA (Figure 4b–c) | Engram-27B layer 5 is most similar to about layer 12 of MoE-27B |
| Layer sweep, one 1.6B module on a 12-layer 3B MoE (Figure 5) | Layer 2 best (1.770). Layers 2 and 6 together 1.768, against 1.808 for the baseline |
| Component ablations (Figure 5) | Branch-specific gating, the gate and tokenizer compression matter most. Removing the convolution costs little. Adding 4-grams is slightly worse at 1.6B |
| Engram output zeroed at inference (Figure 6) | Factual QA keeps 29–44% of its score (TriviaQA 29%). Reading comprehension keeps 81–93% (C3 93%) |
| 100B-parameter table in host memory (Table 4) | Throughput falls 1.9% on a dense 4B and 2.8% on a dense 8B (H800, nano-vLLM) |

## Where the claims go beyond the evidence

These are our observations, not the paper's.

- **Benchmark gaps from one run each.** Most Table 1 differences between Engram-27B and MoE-27B are 1–5 points, one checkpoint per model. Appendix B shows the last 10k steps of each benchmark, which helps, but they are still single runs. On BBH (about 6,500 items), one accuracy near 0.5 has a standard error of about 0.6 points (our calculation). On HumanEval (164 problems) it is about 3.8 points, so the +3.0 there is within noise.
- **The abstract's MMLU number differs from the table.** The abstract and introduction say MMLU +3.4. Table 1 gives 57.4 → 60.4, +3.0, and Section 4.2 says +3.0. +3.4 is the MMLU-Redux gap.
- **"Iso-FLOPs" isn't itemised.** Activated parameters are given as 3.8B for every model, "w/o token embed". Engram's key and value projections, four gates and convolution add FLOPs per token. The paper doesn't say whether they are inside that count, or how many FLOPs they add. The embedding rows also train with Adam at 5× the learning rate, while the backbone uses Muon.
- **A "law" from two budgets.** The U-shape comes from two compute budgets, single runs and a loss gain of 0.0139 at its best. It shows a stable optimum over the two scales tried, not a scaling law. The "strict power law" for slots covers about 1.6 orders of magnitude.
- **One table caption overstates.** Table 2 says Engram-27B at iso-loss and iso-FLOPs "substantially outperforms the baseline across all metrics". At iso-loss (46k), single-needle NIAH is 97.6 against 100.0, and common-words extraction 4.3 against 4.5. At 50k, single-needle is 99.3 against 100.0. The largest gain is frequent-words extraction (73.0 → 98.6).
- **Iso-loss on a step schedule.** The 46k checkpoint is chosen because its pre-training loss matches MoE-27B at 50k. Under step decay, a mid-training checkpoint and a fully decayed one at the same loss need not be in the same state. The paper shows one such pair.
- **"Effective depth" rests on two indirect measures.**
  - The logit-lens KL is lower in early layers. But Engram adds hashed bigram and trigram embeddings at layer 2, which carry next-token statistics directly. A readout can be closer to the final distribution without the network having done "deeper" work. The logit lens is also unreliable in early layers (`vocab-projection-lookahead`).
  - CKA is computed only on the last tokens of named entities from Few-NERD, the inputs where an N-gram lookup should help most. No CKA on general text is reported. The conclusion "functionally equivalent to increasing the model's effective depth" goes beyond both.
- **The ablation shows dependence, not storage.** Zeroing Engram's output at inference puts the model off its training distribution, as the paper notes. Factual QA falling to 29–44% shows that the model depends on Engram there. It doesn't show that Engram is "the primary repository for parametric knowledge". The paper reports no control, such as zeroing an MoE component of matched size, or mean ablation.
- **The gate heatmaps are selected.** Each token has 8 gate values (2 layers × 4 branches). Footnote 2 says "not every branch encodes interpretable activation patterns", and Figure 7 shows "the gating values most strongly correlated with semantic pattern matching". Five sentences are shown, and none is quantified. Figure 7 is an illustration, not evidence that "Engram successfully identifies and handles stereotyped linguistic dependencies".
- **The motivation is borrowed.** The claim that standard models spend early layers rebuilding multi-token entities comes from Ghandeharioun et al. (2024), shown as Table 3. The paper doesn't measure it in its own MoE baseline.
- **The throughput test uses dense backbones.** The offload test inserts one table into dense 4B and 8B models with no expert parallelism, and the paper says so. The MoE models of the main results aren't tested.

## Use in this repo

**Not applicable as a method or a model.** The registry explains native units of pretrained models at pinned revisions (`docs/PROTOCOL.md`). The configured models are Qwen2.5-0.5B, Qwen2.5-1.5B and GPT-2 (`pyproject.toml`, `[tool.openinterp] models`). None has an Engram module, and no Engram weights are released, so there are no gates or memory rows to audit.

**Its premise is a claim about our models' early layers, which the registry could check.** The paper argues that standard transformers spend early layers on local, static patterns: multi-token entities and set phrases. The registry's random draws make this testable without an Engram model. Take the VERIFIED poles in the first few layers. What share are explained as completing a local N-gram or multi-token name? And how does that share change with depth? `neuron-family-triage` already lists token and N-gram detectors (Voita et al., Gurnee et al.). The N2G baseline item (`TODO.md`, *Protocol and science*) would give the floor for that question: how much of a VERIFIED explanation a token-pattern matcher already predicts. No `TODO.md` item asks the depth question itself.

**Two habits of the analysis section are illusions our reviewer looks for.**

- *Selection.* Figure 7 shows the gate that looks most interpretable, out of 8. This is the case in the *Selection* check of `interpretability-illusions-review`, and the reason `provenance.selection` records how a unit was chosen.
- *An early logit lens.* Lower early-layer KL is read as earlier computation. `vocab-projection-lookahead` prefers the Jacobian lens in early layers for this reason.

**What doesn't carry over.** Hashing, gating, sparsity allocation and host-memory offload are training and serving details. The repo doesn't train or serve models.
