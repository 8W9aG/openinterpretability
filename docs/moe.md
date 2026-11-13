# Mixture-of-experts models

This page records how openinterp will audit neurons in mixture-of-experts (MoE) models. It is the decision that the Models section of `TODO.md` asked for. The experts forward of section 4 and the streaming of section 8 are built. The expert neuron as a unit is not. The implementation steps are in `TODO.md`, Models, and adding the unit kind changes the protocol (`docs/PROTOCOL.md` §6).

## How an MoE layer computes

In an MoE layer the MLP is replaced by many smaller MLPs, the experts, and a router. For each token, the router scores every expert and keeps the top k. Each kept expert e computes its hidden activation a_e = act(W_gate,e x) ⊙ (W_up,e x) from the token's residual x. It then writes `w_e · W_down,e a_e`, where w_e is the token's routing weight for e. The other experts do not run on that token. Some models also have shared experts, which run on every token, and some keep ordinary dense MLPs in their first layers. Mistral Large 4 does both.

| Model | Experts × neurons per layer | Per token | Shared experts | Parameters |
| --- | --- | --- | --- | --- |
| Granite 3.0 1B-A400M | 32 × 512 | 8 | none | 1.4B |
| OLMoE-1B-7B | 64 × 1,024 | 8 | none | 6.9B |
| Qwen1.5-MoE-A2.7B | 60 × 1,408 | 4 | 1 × 5,632, gated by a sigmoid | 14.3B |
| Mistral Large 4 | 128 × 6,144 | 4 | 1 × 6,144 | about 1T |

## The decision

### 1. Units

- **A routed expert neuron pole is a new unit kind.** It is `ExpertNeuronIdentifier`, with `kind = "moe_expert_neuron_pole"`, `layer`, `expert`, `neuron_idx` and `pole`. Its slug is `<model>/L<layer>/E<expert>/N<idx>-<pole>`.
- **A shared expert's neurons use the same kind, with `expert = "shared"`.** The slug is `…/L<layer>/Eshared/N<idx>-<pole>`. Where a model fuses several shared experts into one MLP (Mistral 4), the index runs over the fused MLP.
- **A dense layer in an MoE model keeps `NeuronIdentifier`.** Its neurons are ordinary MLP neurons.

We don't add an optional `expert` field to `NeuronIdentifier`. 48 places in `openinterp/` branch on a unit's kind. With an optional field, each one that ignores the field would measure neuron `neuron_idx` of a dense MLP that isn't there, or of the wrong expert. With a new kind, each place either supports expert neurons or refuses them, as places that don't support heads do today.

### 2. Strength

The strength of an expert neuron pole on a token is the pole of `w_e · a_e[i]`, the routing weight times the neuron's hidden activation. On a token the router doesn't send to expert e, the strength is 0.

- **Why the routing weight is in it.** A dense neuron writes `a[i] · w_out`, and its strength is the pole of `a[i]`, the coefficient on its output direction. An expert neuron writes `w_e · a_e[i] · W_down,e[:, i]`. So `w_e · a_e[i]` is the same quantity: how much of its direction the unit writes. A neuron with a large activation and a small routing weight writes little.
- **Why 0 when not routed.** The model doesn't compute the neuron on that token, and nothing is written. So a hypothesis has to explain the routing as well as the neuron. It explains what the model does.
- **Poles.** The routing weights of every architecture checked are non-negative (softmax or sigmoid, then normalised or scaled). So the sign of `w_e · a_e[i]` is the sign of `a_e[i]`, and the two poles keep their meaning. A model whose routing weights can be negative is refused.
- **A shared expert** uses its gate as the weight: the sigmoid gate in Qwen2-MoE, and 1 where there is no gate (Mistral 4).

We considered the activation without the weight, and the activation the expert would compute on every token (the expert run densely). Neither is what the model writes. The second is still useful to tell the router's selectivity from the neuron's. A scan can record it later as evidence for the proposer, not as the strength.

### 3. Interventions

Every intervention edits `a_e[i]` inside the expert, on the (token, expert) pairs that the run being intervened on routes. The routing is never forced.

- **Ablation** zeroes the pole of `a_e[i]` on every routed token. The router of the same layer reads the residual before the experts, so the ablation can't change that layer's routing. Later layers may route differently because of it. That is part of the ablation's downstream effect, and it is kept.
- **Patching** sets `a_e[i]` on a routed token to the source run's value. It patches the hidden activation, so the target run's own routing weight still scales it. If the source run doesn't route that token to e, the source's strength is 0, so the value is 0. A token that the target run doesn't route can't be patched. The harness records how many there were.
- **Steering** clamps `a_e[i]` on routed tokens only, and records the share of tokens that were routed.

Edits are made inside the expert, not as a change to the layer's output. An edit to the output would be exact only while the expert's write is linear in `a_e`. Some architectures normalise each expert's output (`has_post_expert_norm` in transformers), and then it isn't.

### 4. Where the harness reads and edits expert neurons

transformers 5 runs the experts of about 60 MoE architectures through one registry, `ExpertsInterface`. Each experts class decorated with `use_experts_implementation` looks its forward up by name (`eager`, `grouped_mm`, `batched_mm`, …). openinterp registers its own forward, `openinterp` (`openinterp/harness/moe.py`), and selects it when it loads an MoE model.

That forward loops over the experts the router chose. It computes each one's hidden activation with the model's own gating function (`_apply_gate`). It then passes the activation, the expert's index, its tokens and their routing weights to the harness's hooks, before the down projection. So one function covers each weight layout that the decorator describes. Gate and up can be concatenated or interleaved, the weights transposed, with biases, and with a norm after each expert.

`scripts/moe_experts_check.py` checks this on tiny random models of seven architectures:

| Architecture | Largest difference from `eager` in the logits | Zeroing one expert neuron changes them |
| --- | --- | --- |
| Mixtral | 9.1e-7 | yes |
| OLMoE | 9.5e-7 | yes |
| Qwen2-MoE | 0 | yes |
| Qwen3-MoE | 2.4e-7 | yes |
| GPT-OSS (interleaved, transposed, with biases) | 8.3e-7 | yes |
| GraniteMoE | 0 | yes |
| Mistral 4 (shared experts, a dense first layer) | 9.5e-7 | yes |

Each difference is float32 rounding, and no routing weight was negative. Every measurement on an MoE model, of a dense unit too, runs through this forward. The `eager` and `grouped_mm` forwards round differently, so a record pins the forward in its run environment. An MoE architecture that doesn't use the registry is refused.

A shared expert is an ordinary MLP module with a `down_proj`. Its neurons are read by the hook that dense neurons use, scaled by its gate.

### 5. What an expert neuron's experiment measures

| Measurement | For an expert neuron |
| --- | --- |
| Scan, top examples, histogram, probes, counterfactuals | Yes, with the strength above. |
| Routing profile (new) | Yes. It records the share of scan tokens routed to the expert and the mean routing weight when routed. |
| Ablation KL, patching, steering | Yes, by the rules in section 3. |
| Vocabulary lens, Patchscope, downstream wiring | Yes. The output direction is the neuron's column of the expert's down projection. |
| Upstream wiring | Yes, from the neuron's rows of the expert's gate and up projections. |
| Partner search | Within the same expert and the shared expert. A whole layer is too wide: 64 × 1,024 neurons in OLMoE make the reservoir about 1 GB. |
| Input attribution | No. The router's top-k choice changes in steps along the integrated-gradients path, and the gradient doesn't see those steps. So the attributions need not add up to the activation, which the convergence check requires. |
| Weight-only dead proof, sister search | No. |

A routed expert sees about k/E of the scan's tokens. That is a quarter in Granite, an eighth in OLMoE, and 4 of 128 (3%) in Mistral Large 4. In Mistral Large 4, the default 200,000-token scan gives each expert about 6,000 routed tokens. That is enough for top examples. A draw whose neuron the scan never sees active is an unauditable draw, as a dead feature is.

### 6. Random selection and statistics

Routed expert neurons, shared-expert neurons and dense neurons are three arms, never pooled, as neurons and heads aren't today. A random draw of a routed expert neuron is uniform over MoE layers, then experts, then neurons, then poles. A model's share of verified units is reported per arm.

### 7. Models, in order

1. **Granite 3.0 1B-A400M** (`ibm-granite/granite-3.0-1b-a400m-base`, Apache-2.0). It is 5.5 GB in float32, so it runs on CPU and fits CI's runners. It is the model to build and test the harness on.
2. **OLMoE-1B-7B** (`allenai/OLMoE-1B-7B-0924`, Apache-2.0). Its training data is open, which the training-data links issue ([issue #3377](https://github.com/8W9aG/openinterpretability/issues/3377)) needs. It is 28 GB in float32, so it needs bfloat16 or streaming (`--memory stream`).
3. **Qwen1.5-MoE-A2.7B**, for a gated shared expert. It is under the Tongyi Qianwen licence, which must be checked first.
4. **GPT-OSS**, after its own issue ([issue #3362](https://github.com/8W9aG/openinterpretability/issues/3362)). Its gating function (`_apply_gate`) is a clamped SwiGLU, which the forward already calls.
5. **Mistral Large 4**, last. Its weights are stored in FP8, and the experts of one layer take about 14.5 GB. So it needs the streaming below. transformers 5.20.1, which openinterp uses today, has no `mistral_large4` model type, so it also needs a newer transformers. Check then that its experts use the registry.

### 8. Streaming only the routed experts

The registered forward owns the loop over experts, so in streaming mode it can read only the experts that the router chose. safetensors reads one expert's slice of a stored tensor (`get_slice`), through the open files of `OpenCheckpoint` (`openinterp/harness/memory.py`). `accelerate` must then leave the experts' tensors on the meta device, not load them whole before each pass. In Mistral Large 4 one expert is about 113 MB in FP8, against 14.5 GB for all 128. A pass then reads only the experts its tokens route to, and FP8 is converted per slice. This is built (`openinterp/harness/moe.py`, `OpenCheckpoint.slice`). On Granite 3.0 1B-A400M a streamed one-token pass read 1.41 GB, against 4.78 GB with whole layers. A checkpoint that stores each expert apart is stacked by transformers as it loads, and streaming writes the stacks to the offload folder. Its slices are read from there.
