# GLU Variants Improve Transformer

**Citation.** Noam Shazeer, of Google. "GLU Variants Improve Transformer". [arXiv:2002.05202](https://arxiv.org/abs/2002.05202), version 1 of 12 February 2020, the only version. Preprint, no venue given. The paper links no code and states no funding.

**Mismatch.** None found. Table 1 gives a spread in parentheses for the short runs without saying what it is. The text says the four short runs measure inter-run variability, so it is most likely a standard deviation.

**Status.** Used. Qwen2.5's MLP is SwiGLU, so each Qwen neuron in the registry is the product of a gated half and a linear half. `papers/default-on-poles` relies on that form when it compares Qwen's two sides with GPT-2's.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words, from the abstract to the references. The table values came through the HTML conversion, and we didn't check them against the PDF.

## Summary

**Claim.** In a Transformer's feed-forward layer, replacing the first linear map and its activation with a gated linear unit (GLU) lowers perplexity. The variants with GELU or Swish as the gate (GEGLU, SwiGLU) do best.

**Method.**

- **Definitions.** GLU(x) = σ(xW) ⊗ xV, the product of a gated half and a linear half. ReGLU, GEGLU and SwiGLU swap σ for ReLU, GELU and Swish. Swish_β(x) = x·σ(βx), and every SwiGLU layer in the experiments uses Swish_1. The feed-forward layer is then FFN_SwiGLU(x) = (Swish_1(xW) ⊗ xV)W₂. Biases are left out, following the T5 code.
- **Equal cost.** A GLU layer has three weight matrices, not two. To keep parameters and compute the same, the hidden size drops by 2/3, from 3,072 to 2,048.
- **Setup.** The T5 base encoder-decoder (12 layers each, width 768), pretrained for 524,288 steps on C4's span-filling task, with no dropout. Four short runs of 65,536 steps measure the spread between runs. Fine-tuning is one run on a mixture of GLUE, SuperGLUE and SQuAD, reporting each task's best checkpoint on the dev set.

**Findings, as the source reports them.**

| Setting | ReLU | GELU | GEGLU | SwiGLU | ReGLU |
| --- | --- | --- | --- | --- | --- |
| Log-perplexity, 65,536 steps (Table 1) | 1.997 (0.005) | 1.983 (0.005) | 1.942 (0.004) | 1.944 (0.010) | 1.953 (0.003) |
| Log-perplexity, 524,288 steps (Table 1) | 1.677 | 1.679 | 1.633 | 1.636 | 1.645 |
| GLUE dev average (Table 2) | 83.80 | 83.86 | 84.12 | 84.36 | 84.67 |
| SuperGLUE dev average (Table 3) | 72.76 | 72.98 | 73.96 | 74.56 | 73.66 |

## Where the claims go beyond the evidence

- **One fine-tuning run, best checkpoint.** Each fine-tuning number is one run, and the best of its checkpoints. The author says so, and calls the results "noisy". The GLUE gaps are under 1 point.
- **The long runs have no spread.** Only the short runs were repeated. The 524,288-step numbers are single runs.
- **One model and one objective.** All results are on T5 base with span filling on C4. The paper doesn't test a decoder-only model, such as those in the registry.
- **No explanation.** The author says it outright: "We offer no explanation as to why these architectures seem to work".

## Use in this repo

- **Our citation checks out.** `papers/default-on-poles` writes a Qwen neuron as $\mathrm{silu}(\mathrm{gate}) \times \mathrm{up}$ and cites this paper for SwiGLU. The paper's SwiGLU layer uses Swish_1, which is SiLU, times a linear half. The reference's author and year match. No change is needed.
- **A SwiGLU neuron is not one direction.** A GELU neuron reads one input direction through a fixed curve. A SwiGLU neuron multiplies two readings of the input, the gate and the up projection. So its sign can come from either half, and a reading of "what the neuron responds to" has two input directions to explain. Our reading, not the paper's. It fits `papers/default-on-poles`, which finds Qwen's two sides symmetric, unlike GPT-2's.
- **The width ratio is a convention, not Qwen's.** The paper sets the hidden size to 2/3 of the usual 4× width to match cost. Qwen2.5's MLP widths are its own, so nothing in the registry depends on the 2/3.
- **No question for the maintainer.**
- **Related pages.** [`gelus.md`](gelus.md) reviews GELU, GPT-2's activation, which GEGLU uses as its gate.
