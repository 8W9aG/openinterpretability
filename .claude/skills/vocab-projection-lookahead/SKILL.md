---
name: vocab-projection-lookahead
description: Read a neuron's input and output weights through the vocabulary (Jacobian lens, logit lens, FFN key-value view) to tell retrospective neurons from lookahead / next-token-prediction neurons. Use when deciding the mechanism claim, when the top examples peak just before a predictable token, or when `measure_lookahead` and the claim disagree.
---

# Vocabulary projection and lookahead

## Papers

- Geva, Schuster, Berant & Levy, 2021, *Transformer Feed-Forward Layers Are Key-Value Memories*, arXiv:2012.14913. Each neuron's input weights act as a *key* that detects a pattern, and its output weights are a *value* that pushes the next-token distribution.
- Geva, Caciularu, Wang & Goldberg, 2022, *Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space*, arXiv:2203.14680. FFN updates are interpretable when projected through the unembedding, and many neurons *promote* or *suppress* coherent token sets.
- nostalgebraist, 2020, *interpreting GPT: the logit lens* (LessWrong). Decode intermediate residual states with the final unembedding.
- Gurnee, Sofroniew, Lindsey et al., 2026, *Verbalizable Representations Form a Global Workspace in Language Models*, arXiv:2607.15495. The **Jacobian lens** transports a layer-ℓ vector to the final layer with the context-averaged Jacobian J_ℓ = E[∂h_final,t′/∂h_ℓ,t] (t′ ≥ t), then decodes it with W_U·norm(·). It agrees with the logit lens in late layers and stays reliable in earlier ones, where the logit lens does not.
- Belrose et al., 2023, *Eliciting Latent Predictions from Transformers with the Tuned Lens*, arXiv:2303.08112. Trains a per-layer affine translator. Gurnee et al. find it tends to "skip ahead" to the model's output, which makes it less useful for inspecting internal computation than the logit or Jacobian lens.
- Voita, Ferrando & Nalmpantis, 2023, *Neurons in Large Language Models: Dead, N-gram, Positional*, arXiv:2309.04827. Many neurons' output updates both promote next-token candidates and **remove information about the tokens that triggered them**.

## Procedure

1. **Output side (what it promotes): use the Jacobian lens.**
   ```python
   from openinterp.harness.jlens import vocabulary_lens

   vl = vocabulary_lens(runner, corpus_snippets[:100], layer, idx, pole)
   print(vl.jacobian_promoted, vl.jacobian_suppressed)  # read these
   print(vl.logit_promoted, vl.overlap, vl.direction_cosine, vl.jacobian_kurtosis)
   ```
   `openinterp audit` records this as `submission.vocabulary_lens`. On Qwen2.5-0.5B, the two lenses share 0 of their top-10 tokens up to the middle layers and converge only near the output: overlap 0.5–0.7 at layer 20, 1.0 at layer 23, while cos(w, J·w) rises from ~0.2 to 1.0. **Don't trust logit-lens tokens before the last few layers.**
   - **Kurtosis ≈ 3** (Gaussian) means the pole promotes no specific token set, under either lens. A high Jacobian kurtosis together with coherent tokens is the signature of a prediction or suppression neuron.
   - Also check the **suppressed** list. Voita et al. found that suppressed tokens are often the trigger itself.
2. **Input side (what it detects).** Project the neuron's input row (`up_proj` and `gate_proj` rows for SwiGLU, `c_fc` column for GPT-2) through the *embedding* matrix. This mostly helps in early layers.
3. **Behavioural check.** `measure_lookahead(runner, prompts, layer, idx, pole)` zero-ablates the pole and reports KL(clean ‖ ablated) and the tokens that lose the most probability. A lookahead neuron shows:
   - top examples peaking one token *before* a predictable token (print the 3 tokens after each peak);
   - an overlap between the promoted-token list and the tokens that actually come next after the peaks;
   - concentrated probability loss under ablation, on those same tokens.
4. **Decide the claim.** Claim lookahead only if all three agree. If the *Jacobian-lens* list is incoherent with kurtosis near 3, the neuron's effect isn't about specific tokens, so claim retrospective. That was the case for `L11/N4255`: its logit-lens list was noise, and the Jacobian lens (kurtosis 3.0) confirmed it rather than revealing hidden structure.

## Pitfalls

- The logit lens is unreliable outside the last few layers (Gurnee et al.; see the numbers above). Use the Jacobian lens, and weight the behavioural check more in early layers.
- The Jacobian lens is an *average* linearisation. A neuron whose effect is strongly context-dependent can have a flat Jacobian-lens reading and still matter in specific contexts, so check steering and ablation too.
- Tied embeddings (e.g. Qwen2.5-0.5B) make the input and output projections share a matrix. That is fine, but don't read symmetry into it.
- A tool that only shows context *before* the peak will miss lookahead neurons entirely (Goodfire 2026, the "boat" example).
- The protocol's `lookahead_consistent` gate uses `measure_lookahead` thresholds (KL ≥ 1e-3, concentration ≥ 0.5). For deep-layer neurons with small direct effects, expect "retrospective" from the measurement even when the weights promote coherent tokens. Note any such discrepancy in the explanation.
