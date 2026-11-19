# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

**Citation.** Yu Sun (Stanford University), Xinhao Li (UC San Diego) and Karan Dalal (UC Berkeley) are the core contributors. The other authors are Jiarui Xu (UC San Diego), Arjun Vikram, Genghan Zhang and Yann Dubois (Stanford). The advisers are Xinlei Chen (Meta AI), Xiaolong Wang (UC San Diego), Sanmi Koyejo, Tatsunori Hashimoto and Carlos Guestrin (Stanford). "Learning to (Learn at Test Time): RNNs with Expressive Hidden States". arXiv 2407.04620 [cs.LG]. v1 is from 5 July 2024, v2 from 11 August 2024, v3 from 3 April 2025 and v4 from 31 August 2025. We read v4. [Paper](https://arxiv.org/abs/2407.04620). Code: [JAX](https://github.com/test-time-training/ttt-lm-jax) and [PyTorch](https://github.com/test-time-training/ttt-lm-pytorch). The data is the Pile and its Books3 subset. Preprint, no venue given: neither the arXiv page nor the PDF names one. The v4 comment says that v4 changes only the related work and the limitations, and that the authors ran all experiments for v1. Compute came in part from the Google TPU Research Cloud and Hyperbolic Labs. The authors also name grants from Amazon, Cisco, Qualcomm, NSF, NIFA, the Sloan Foundation, Google, Sony, Panasonic, AFOSR, Stanford HAI and IBM.

**Status.** Planned. #4782 would use it in its synthesis on weight-space programming, as evidence for the ceiling that #5144 counts. Its Theorem 2 shows softmax attention as a learner that stores the tokens and has no weights to read. The paper has no method for a transformer's units, and the page proposes no new issue.

**Read.** We read the arXiv v4 PDF on 2026-11-19, as text that `pdftotext` took from it. We have the source's own text for the abstract, §1 to §5, the author contributions, the acknowledgements and the references. We also have Appendix A (the dual form), Appendix B (the Nadaraya–Watson estimator) and Appendix C (experiment details). Tables 1 and 3 came through as text. We also read the arXiv abstract page for the version history. We did not see the figures as images. Figures 2, 4, 7, 10 to 12 and 14 to 16 are plots, and only their axis ticks came through. So the page gives no number read from a plot, only what the text and captions say. A model (Claude Opus 5.5) did the reading.

## Summary

**Claim.** A sequence layer can keep its hidden state as the weights of a small model. Its update rule is then one gradient step of a self-supervised loss on each token. These "Test-Time Training" (TTT) layers have linear cost in the length of the text. In the paper's runs, they keep reducing perplexity up to 32k tokens, while Mamba stops after 16k.

**Method.**

- **The layer (§2.1).** The hidden state is the weights W of a model f. The output rule is `z_t = f(x_t; W_t)`. The update rule is `W_t = W_{t−1} − η∇ℓ(W_{t−1}; x_t)`, where ℓ is a reconstruction loss. The paper says that W "remembers inputs that produce large gradients".
- **Two loops (§2.2, §2.3).** The inner loop trains W on the tokens of one sequence, during the forward pass. The outer loop trains the rest of the network on next-token prediction. It also trains three projections θ_K, θ_V and θ_Q. The loss is `ℓ(W; x_t) = ‖f(θ_K x_t; W) − θ_V x_t‖²`, and the output is `f(θ_Q x_t; W_t)`. So the outer loop learns which task the inner loop solves.
- **Two forms of f (§2.7).** In TTT-Linear, f is a square linear map. In TTT-MLP, f is a two-layer MLP with a hidden width 4× its input, and GELU. Both have a LayerNorm and a residual connection: f(x) = x + LN(f_res(x)). The outer loop learns the start W_0, and all sequences share it. The inner learning rate is η(x) = η_base σ(θ_lr · x), with η_base = 1 for TTT-Linear and 0.1 for TTT-MLP.
- **Speed (§2.4, §2.5).** Mini-batch TTT takes the gradients in a block of b = 16 tokens from the W at the end of the last block. The "dual form" then computes the block's outputs and its last W with matrix products, and never builds the per-token W.
- **Equivalences (§2.6).** Theorem 1: with a linear f, batch gradient descent, η = 1/2 and W_0 = 0, the layer is linear attention. Theorem 2: with the Nadaraya–Watson estimator as the learner, the layer is softmax attention. Then the hidden state is the list of tokens, not a set of weights.
- **Experiments (§3).** Four sizes: 125M, 350M, 760M and 1.3B parameters (Mamba: 130M, 370M, 790M and 1.4B). The data is the Pile at 2k and 8k context, and Books3 at 1k to 32k. The baselines are a Llama-style Transformer and Mamba, with the Chinchilla recipe of the Mamba paper. TTT layers use the Mamba backbone, with temporal convolutions, unless the paper says otherwise. All models use the Llama tokenizer. For long context, a "TF finetune" baseline trains a Transformer at 2k and then fine-tunes it with 20% more tokens.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Table 1, 125M, linear attention as in Katharopoulos et al. | Perplexity 15.91 |
| Table 1, linear attention improved, and its TTT equivalent | 15.23 and 15.23 |
| Table 1, then learnable W_0 | 15.27 (+0.04). The rows after it "cannot train stably without it" |
| Table 1, then LayerNorm and residual in f | 14.05 (−1.22) |
| Table 1, then mini-batch TTT (b = 2048 → 16) | 12.35 (−1.70), the largest single gain |
| Table 1, then learnable η | 11.99 (−0.36) |
| Table 1, then Mamba backbone (this is TTT-Linear) | 11.09 (−0.90) |
| Pile, 2k context (Figure 10, from the text) | TTT-Linear (M), Mamba and Transformer are "comparable". TTT-MLP (M) is "slightly worse" at large FLOP budgets |
| Pile, 8k context (Figure 10, from the text) | Both TTT layers (M) are "significantly better" than Mamba |
| Books, 32k context (Figure 11, from the text) | Both TTT layers (M) are better than Mamba. At 2k, Mamba is "slightly better" than TTT-Linear |
| Books, by model size (Figure 16 caption) | For every method trained from scratch, perplexity gets worse once the context is too long |
| Training time, 2k context, v5e-256 TPU pod (§3.3) | Transformer 0.30 s per iteration, TTT-Linear 0.27 s |
| Dual form against primal form, JAX (§2.5) | Training is "more than 5× faster" |
| Inference FLOPs in Figure 2, right (Appendix C) | TF finetune has 2.8× the inference FLOPs of the TTT layers and Mamba |
| Inner loss over 2,048 tokens, 125M (Figures 4 and 14, from the captions) | The gap between `ℓ(W_0; x_t)` and `ℓ(W_t; x_t)` grows with t, in all 12 layers |

## Where the claims go beyond the evidence

- **The framing is wider than the tests.** The paper says that it reformulates supervised learning and gives "a framework for building" architectures. It tests two forms of f on language modelling only, at 125M to 1.3B parameters.
- **One run per point, with no spread.** No result gives seeds, intervals or repeats. Most comparisons are lines on a plot, and the text calls them "comparable" or "slightly better". Table 1 differences of 0.04 have no error.
- **The backbone is part of the gain.** TTT layers use Mamba's backbone, and the backbone alone gives −0.90 in Table 1. With the Transformer backbone, TTT-Linear does worse in the paper's runs. The authors report this ablation. On Books, they ran it only at 2k and 32k.
- **The recipe is the Mamba paper's for every method.** Footnote 11 says that a new recipe for each method would cost "orders of magnitude more resources". So the comparison could favour the method closest to the recipe's source. The authors also find no clean linear fit, so they draw no scaling law.
- **A citation number points to two works.** §1 and §4.2 cite Gated DeltaNet as [82]. §4.2 also cites Gated Linear Attention as [82]. Reference [82] is the Gated Linear Attention paper (Yang, Wang, Shen, Panda and Kim 2023).
- **The compression claim is not measured.** The paper says that self-supervised learning finds "the underlying structures and relationships" in the context. It shows only the inner loss and perplexity, not what W stores.
- **Wall-clock time is behind FLOPs.** The authors say that TTT-MLP has "challenges in memory I/O". Its wall-clock time grows "much more relative to FLOPs", and they wrote no GPU training kernel.

The authors state several limits themselves: the wall-clock cost, the shared recipe, the missing linear fit, the backbone effect and the small scale.

## Use in this repo

- **Theorem 2 puts softmax attention among learners with no weights to read.** The four configured models (`pyproject.toml`) use softmax attention. In the paper's terms, their hidden state is the list of past tokens, and the output rule scans it with a kernel. No matrix W_t exists to read. This adds a third route to the limit in [`fast-weights-recent-past.md`](fast-weights-recent-past.md) and [`linear-transformers-fast-weight-programmers.md`](linear-transformers-fast-weight-programmers.md). A rule that depends on context lives in the text, not in static weights.
- **In TTT-MLP, even an MLP neuron's weights change with the text.** [`linear-transformers-fast-weight-programmers.md`](linear-transformers-fast-weight-programmers.md) says that the argument against a weight-only reading applies to attention, not to an MLP neuron. TTT-MLP is a counter-case. Its inner MLP has GELU units, like the registry's units, but their input and output weights are W_t. Only W_0 is static, and Figures 4 and 14 show the inner loss moving away from W_0 along the text. The paper never reads these units, so this is an inference from its equations.
- **So the ceiling of #5144 depends on the architecture.** In the registry's models, MLP weights are static, so a token-set rule on a neuron's input side can be tested from weights. #5144 counts how many VERIFIED explanations are of that kind. In a TTT model, the same count would cap a weight-only check for the outer MLPs only. The paper adds no number to #5144.
- **The repo's evidence agrees with the limit.** In `docs/analyses/weight-kurtosis.md`, ROTATE's weight-only kurtosis (#3446) does not predict VERIFIED on 518 test poles. The AUC is 0.51 (0.45–0.57) on the output side. In `docs/analyses/ground-truth-calibration.md`, plants written as input weights passed for 7 of 12 rules, and for none that needs context. "Inside quotes" reached precision 0.59 and recall 0.50. Neither result tests attention as such.
- **The inner loop is a small case of #4782's interference question.** Each token's gradient step writes over the same W. The paper says that W keeps the inputs with large gradients, so other inputs fade. Figure 16 shows that perplexity gets worse past some context length for models trained from scratch. The paper never tests recall of early tokens, so it gives no forgetting number. [`complementary-learning-systems.md`](complementary-learning-systems.md) gives the general case: focused learning of one item damages related items.
- **Dynamic evaluation is the registry's version of the inner loop.** §4.1 names dynamic evaluation: fine-tune a language model on the test text itself. In a registry model, that would be one very narrow fine-tune. #5125 asks whether VERIFIED explanations keep through a sequence of narrow fine-tunes. This paper gives no data for that question. It is only a pointer for #5125's design.
- **Training changes poles, and so does TTT, on a smaller scale.** `papers/neuron-development/README.md` finds that Pythia-70m's 86 VERIFIED poles keep changing after the loss settles. At step 16,000 the median correlation with the final form is 0.62. Adjacent checkpoints correlate at a median of 0.78 to 0.82. A TTT layer adds a further change within each text. An explanation of a TTT-MLP inner unit would describe W_0 plus the updates from one text. The paper measures neither drift for single units.
- **Weight-space programming (#3381, #4471).** Here gradient descent writes the weights at run time. The outer loop picks the task through θ_K and θ_V, and nobody reads the written W. #4471 has an agent write weights as code, to be read. The two are opposites on one axis: the paper writes weights for speed, and #4471 writes them for reading.
- **A model with these layers is a possible case, but not now.** The paper says that DeltaNet is TTT-Linear with b = 1 and no LayerNorm or residual. [`models-rewrite-own-weights.md`](models-rewrite-own-weights.md) reports that Qwen3.5 mixes Gated DeltaNet layers with full attention. #3331 proposes a Qwen3.5-0.8B fine-tune. We did not read the Gated DeltaNet paper, so we don't claim that its layer is a TTT layer. The head harness (`openinterpretability/harness/heads.py`) would find no attention pattern in such a layer. No new model joins the registry before every model meets its coverage target.
- **Related reviews in the queue.** #4846 (Titans) and #4849 (end-to-end test-time training for long context) review two later works in the same line.
- **No single-neuron content.** The paper gives no unit explanations, no probes and no null. Nothing in it bears on any pole's verdict.
- **No new follow-up.** #5144 already asks for the count that Theorem 2 bears on, and #5125 covers narrow fine-tunes. The paper is evidence for those issues, not a reason for a new one.
- **Question for the maintainer.** If a model with fast-weight layers joins, should its explanations name the text span that sets the weights? That would change what an explanation claims, so it may need a `PROTOCOL_VERSION` bump (`docs/PROTOCOL.md` §6). We record it as a question only.
