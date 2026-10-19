---
name: vocab-projection-lookahead
description: Read a neuron's input and output weights through the vocabulary (logit lens, tuned lens, FFN key-value view) to tell retrospective neurons from lookahead / next-token-prediction neurons. Use when deciding the mechanism claim, when the top examples peak just before a predictable token, or when `measure_lookahead` and the claim disagree.
---

# Vocabulary projection and lookahead

## Papers

- Geva, Schuster, Berant & Levy, 2021, *Transformer Feed-Forward Layers Are Key-Value Memories*, arXiv:2012.14913. Each neuron's input weights act as a *key* that detects a pattern, and its output weights are a *value* that pushes the next-token distribution.
- Geva, Caciularu, Wang & Goldberg, 2022, *Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space*, arXiv:2203.14680. FFN updates are interpretable when projected through the unembedding, and many neurons *promote* or *suppress* coherent token sets.
- nostalgebraist, 2020, *interpreting GPT: the logit lens* (LessWrong). Decode intermediate residual states with the final unembedding.
- Belrose et al., 2023, *Eliciting Latent Predictions from Transformers with the Tuned Lens*, arXiv:2303.08112. Train a per-layer affine translator; it is more faithful than the logit lens in early layers.
- Voita, Ferrando & Nalmpantis, 2023, *Neurons in Large Language Models: Dead, N-gram, Positional*, arXiv:2309.04827. Many neurons' output updates both promote next-token candidates and **remove information about the tokens that triggered them**.

## Procedure

1. **Output side (what it promotes).**
   ```python
   from openinterp.harness.lookahead import output_token_projection

   top = output_token_projection(runner, layer, idx, pole, k=20)
   print([runner.tokenizer.decode([t]) for t, _ in top])
   ```
   This skips the final norm. For RMSNorm models, a closer approximation scales the direction by the norm gain first:
   ```python
   w = runner.neuron_output_direction(layer, idx) * runner.model.model.norm.weight
   scores = runner.model.get_output_embeddings().weight @ w  # negate for the negative pole
   ```
   Also check the **bottom** of the list: tokens the pole suppresses. Voita et al. found that suppressed tokens are often the trigger itself.
2. **Input side (what it detects).** Project the neuron's input row (`up_proj` and `gate_proj` rows for SwiGLU, `c_fc` column for GPT-2) through the *embedding* matrix. This mostly helps in early layers.
3. **Behavioural check.** `measure_lookahead(runner, prompts, layer, idx, pole)` zero-ablates the pole and reports KL(clean ‖ ablated) and the tokens that lose the most probability. A lookahead neuron shows:
   - top examples peaking one token *before* a predictable token (print the 3 tokens after each peak);
   - an overlap between the promoted-token list and the tokens that actually come next after the peaks;
   - concentrated probability loss under ablation, on those same tokens.
4. **Decide the claim.** Claim lookahead only if all three agree. If the promoted-token list is incoherent (as it was for `L11/N4255`), the neuron's direct effect is not about specific tokens, so claim retrospective and let later layers carry the effect.

## Pitfalls

- The logit lens is unreliable in early layers (Belrose et al.); weight the behavioural check more there.
- Tied embeddings (e.g. Qwen2.5-0.5B) make the input and output projections share a matrix. That is fine, but don't read symmetry into it.
- A tool that only shows context *before* the peak will miss lookahead neurons entirely (Goodfire 2026, the "boat" example).
- The protocol's `lookahead_consistent` gate uses `measure_lookahead` thresholds (KL ≥ 1e-3, concentration ≥ 0.5). For deep-layer neurons with small direct effects, expect "retrospective" from the measurement even when the weights promote coherent tokens. Note any such discrepancy in the explanation.
