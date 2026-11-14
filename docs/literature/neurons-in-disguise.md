# Many Transcoder Features Are Neurons in Disguise

**Citation.** Stepan Shabalin (Anthropic Fellows and Georgia Institute of Technology) and Mateusz Piotrowski (Anthropic), equal contribution. "Comparing Sparsified MLPs and Transcoders: Many Transcoder Features Are Neurons in Disguise". arXiv 2610.24521v1, 17 October 2026. [Paper](https://arxiv.org/abs/2610.24521). NeurIPS 2026 workshop *Interpretability as a Science*. The paper says that its TopK transcoders are released under the MIT licence, but it gives no link. Funding: the Anthropic Fellows Program, a BlueDot rapid grant and compute from White Circle. The authors say that they used Claude Code to help write, code and review, and that they checked its output.

**Status.** Planned. Issue #3608 counts how many of the transcoder latents in our learned-basis baseline are neurons in disguise.

**Read.** The abstract page and the v1 HTML, on 2026-11-14. A model read the text in seven passes and reported it to us, with short quotes. All of it came back as the paper's own text, except two gaps of about 40 characters at the joins. One gap cuts a row of Table 4, so that row is not here. We did not see the figures.

## Summary

**Claim.** Recent work recovers sparse circuits from raw MLP neurons. So do circuits need transcoders? Many transcoder features are *neuron features*, whose encoder and decoder both align with one MLP neuron. Only those features match the MLP the same way across training runs. The other features vary between runs, don't match the MLP's Jacobian, and move less from where training started. So MLP neurons are "a stronger basis for interpretability than commonly assumed".

**Method.**

- **Models.** Gated MLPs only: Gemma 2 and 3 (1B–27B), Qwen 3 (1.7B–14B), Llama 3.2 1B and Llama 3.1 8B. There are two MoE models in a preliminary part.
- **Neuron as two units.** Each neuron σ(g)·u is split by the sign of its activation into two units, each with its own effective encoder and decoder. The paper recommends treating the two signs as different units.
- **Sparsified MLP.** It keeps the top k of those signed units per layer (k ≈ 64 recommended), with no training. Some variants add small trained linear corrections.
- **Neuron feature.** A transcoder feature whose encoder and decoder both have cosine above a threshold with one neuron's. The threshold is between 0.6 and 0.9, depending on the transcoder.
- **Transcoders.** 13 open sets: Gemma Scope 2, mwhanna's Qwen 3 transcoders, LXTC for Llama 3.1 8B, and two cross-layer sets used per layer. Plus the authors' own TopK and JumpReLU transcoders on Gemma 3 4B layer 22.
- **Tests.**
  - sort evaluations judged by Claude Haiku 4.5
  - sparse probing on 113 tasks
  - bigram probing
  - attribution-graph circuits
  - Jacobian alignment
  - stability across training seeds
  - ablations on arithmetic and on one distress circuit

**Findings, as the text reports them.**

| Setting | Result |
| --- | --- |
| Neuron features among active features, late layers, threshold 0.7 (13 transcoders) | 14% to 59%. 7 of the 13 are above 50% |
| The same, as a share of each dictionary | 3.6% to 31% |
| Gemma family, late layers, threshold 0.6 | 58% to 83% of activations come from neuron features |
| Neuron features in attribution graphs, late layers | At least 40% for every transcoder but one (about 38%) |
| Jacobian alignment, Gemma 3 4B | About 0.3 for neuron features, about 0.05–0.1 for the others |
| Training from a neuron start, Gemma 3 4B layer 22 | Neuron-aligned features carry 95% of the activation mass, against 65% from a random start |
| Single bigrams, Gemma 3 4B | A transcoder feature matches the bigram in 75.6% of cases. A single neuron matches in 0.0% |
| Arithmetic, ablating the neuron features | Accuracy falls from 94.3% to 30.6% |

## Where the claims go beyond the evidence

- **Gated models only.** Every model has a gated MLP (GeGLU or SiLU-gated). The paper doesn't test a GELU MLP such as GPT-2's, and it doesn't use Dunefsky et al.'s GPT-2 transcoders.
- **The threshold moves.** Section 3.1 says 0.6 to 0.9 depending on the transcoder. Table 4 uses 0.6, and Section 3.2 and Table 9 use 0.7. The share of neuron features depends on the choice, and the paper does not report it at one threshold throughout.
- **"Over half" is not true of every transcoder.** Figure 1 says that over 50% of late-layer activations come from neuron features. Table 9 has 7 of 13 transcoders above 50%, and the lowest is 14% (Qwen 3 1.7B).
- **13 or 14 transcoders.** The abstract and Table 9 count 13, and Table 4 has 14 rows. The text does not say why one is left out.
- **The sparse-probing tables may disagree.** The text and Table 5 say that neurons match or beat the transcoders. Table 6, as the model read it, puts the dictionary ahead on every row. A column swap in the HTML could explain it. We could not check the PDF.
- **Selection on test data.** Sparse probing picks each task's best layer by its test AUC.
- **Automated judges only.** The authors say that they ran no human evaluation.
- **Weight-based matching is fragile.** The authors say that their definition of a neuron feature, by weights alone, may not hold up.

## Use in this repo

**It reads the registry's unit.** The paper splits each gated neuron by the sign of its activation and treats the two signs as different units. That is the protocol's pole (`docs/PROTOCOL.md` §1), and the paper finds the two signs' encoders "approximately orthogonal". So it supports reading a neuron's two poles apart, as `neurons-as-an-interpretable-basis.md` does too.

**It changes how to read our learned-basis baseline.** On GPT-2 small, 37 of 40 random transcoder latents were VERIFIED, against 20 of 40 random positive neuron poles (`docs/analyses/transcoder-baseline.md`). The paper's result suggests that part of that comparison may be neurons against neurons: a share of the latents may be neuron features. We don't know that share for GPT-2. The paper's models are all gated, and GPT-2's MLP is GELU. Issue #3608 measures it, and splits the VERIFIED rate between neuron features and other latents.

**It matches the circuits result.** The paper starts from Arora et al.'s finding that circuits are sparse in the neuron basis (`neuron-basis-circuits.md`). It adds that transcoder features which don't align with a neuron are less stable across training runs. That supports the registry's choice to audit native neurons first, with dictionaries as a baseline.

**One measure carries over.** Our cross-reference (`open-interpretability crossref`) already finds the latents whose decoder aligns with a pole's write. The paper's test also needs the encoder to align. Issue #3608 adds that side.
