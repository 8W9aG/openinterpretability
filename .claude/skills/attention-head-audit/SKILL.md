---
name: attention-head-audit
description: Plan and run the extension of openinterp from MLP neuron poles to attention heads (and later transcoder features), using the circuits literature. Use when the user wants to expand the registry's scope beyond MLP neurons, audit an attention head, or compare native neurons against SAE/transcoder baselines.
---

# Extending the registry to attention heads and learned bases

## Papers

- Olsson et al., 2022, *In-context Learning and Induction Heads*, arXiv:2209.11895. The canonical head type: attends to the token after an earlier copy of the current token, then copies it.
- Wang et al., 2022, *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*, arXiv:2211.00593. Name-mover, S-inhibition and duplicate-token heads, found by path patching.
- Conmy et al., 2023, *Towards Automated Circuit Discovery for Mechanistic Interpretability* (ACDC), arXiv:2304.14997.
- Syed, Rager & Conmy, 2023, arXiv:2310.10348. Edge attribution patching is a fast alternative to ACDC.
- Dunefsky, Chlenski & Nanda, 2024, *Transcoders Find Interpretable LLM Feature Circuits*, arXiv:2406.11944, and Paulo, Shabalin & Belrose, 2025, *Transcoders Beat Sparse Autoencoders for Interpretability*, arXiv:2501.18823. These learned-basis baselines are the natural comparison for the project's native-neuron thesis.

## Harness design for heads

A head's output is its slice of the `o_proj` input (Llama/Qwen) or `c_proj` input (GPT-2 attention), so the existing pre-hook approach carries over:

```python
attn = runner.layers[layer].self_attn  # Qwen2/Llama; GPT-2: .attn
head_dim = runner.model.config.hidden_size // runner.model.config.num_attention_heads


def head_slice(act, head):  # act: [b, seq, n_heads * head_dim]
    return act[..., head * head_dim : (head + 1) * head_dim]
```

What heads need in addition:
1. **Attention patterns.** Load the model with `attn_implementation="eager"` and call it with `output_attentions=True`. SDPA and flash attention don't return weights.
2. **A schema generalisation.** Introduce `ComponentIdentifier` with `kind: Literal["mlp_neuron_pole", "attn_head"]`, `layer`, `index`, and `pole: Pole | None`. Keep `NeuronIdentifier` as the `mlp_neuron_pole` case so existing records still validate. This is a schema change, so bump `PROTOCOL_VERSION`.
3. **A different "active" definition.** Heads have no scalar activation. Use the norm of the head's output, or the attention-pattern score that the hypothesis predicts (e.g. induction score: attention to the token after an earlier copy of the current one).
4. **Probe semantics.** Hard negatives for an induction head are repeated sequences with the copy broken; holdouts are repeated random-token sequences (Olsson et al.).

## Learned-basis baselines

To test the project's premise, compare native neurons with transcoder or SAE features on the same layer, model and corpus, scoring with this protocol: the share VERIFIED among randomly selected units, and balanced accuracy. Learned features go in as another `kind`, never mixed into native-neuron statistics.

## Suggested order

1. Add the schema generalisation and a head hook to `NeuronHookRunner`, with tests on `tiny_gpt2`.
2. Audit known induction heads in GPT-2 small as a positive control. The protocol should VERIFY them; if it doesn't, fix the protocol before going further.
3. Only then open head audits on Qwen and Llama.
