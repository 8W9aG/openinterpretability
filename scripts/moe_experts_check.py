"""Check the hook point that docs/moe.md chooses for expert neurons, on tiny random models.

transformers 5 runs the experts of about 60 mixture-of-experts architectures through
``ExpertsInterface``: a class decorated with ``use_experts_implementation`` looks its
forward up by name. This script registers one forward, ``openinterp``, that loops over
the routed experts and passes each expert's hidden activation to a hook. For each
architecture it prints:

* the largest difference between its logits and those of the ``eager`` forward
* whether zeroing one neuron inside one expert changes the logits
* whether every routing weight was non-negative, which the poles rely on.

    uv run python scripts/moe_experts_check.py

It needs no download: each model is a random two- or three-layer configuration.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import transformers as tf
from transformers.integrations.moe import ExpertsInterface

Hook = Callable[[torch.Tensor, int, torch.Tensor, torch.Tensor], torch.Tensor]
HOOKS: dict[int, Hook] = {}  # id(experts module) -> fn(act, expert, tokens, weights) -> act


def _linear(x, weight, bias, transposed: bool) -> torch.Tensor:
    y = x @ weight if transposed else x @ weight.T
    return y if bias is None else y + bias


def openinterp_experts_forward(self, hidden, top_k_index, top_k_weights) -> torch.Tensor:
    """The eager experts forward, with each expert's hidden activation passed to a hook."""
    out = torch.zeros_like(hidden)
    for e in torch.unique(top_k_index).tolist():
        if e >= self.num_experts:  # an expert-parallel sentinel
            continue
        tokens, slot = torch.where(top_k_index == e)
        name = "gate_up_proj" if self.has_gate else "up_proj"
        bias = getattr(self, f"{name}_bias")[e] if self.has_bias else None
        pre = _linear(hidden[tokens], getattr(self, name)[e], bias, self.is_transposed)
        act = self._apply_gate(pre) if self.has_gate else self.act_fn(pre)
        weights = top_k_weights[tokens, slot]
        if (hook := HOOKS.get(id(self))) is not None:
            act = hook(act, e, tokens, weights)
        bias = self.down_proj_bias[e] if self.has_bias else None
        y = _linear(act, self.down_proj[e], bias, self.is_transposed)
        if self.has_post_expert_norm:
            y = self.post_expert_norm(y)
        out.index_add_(0, tokens, (y * weights[:, None]).to(out.dtype))
    return out


ExpertsInterface.register("openinterp", openinterp_experts_forward)

SMALL = dict(
    vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, max_position_embeddings=64,
)  # fmt: skip
MODELS = {
    "Mixtral": lambda: tf.MixtralForCausalLM(
        tf.MixtralConfig(**SMALL, intermediate_size=48, num_local_experts=4, num_experts_per_tok=2)
    ),
    "OLMoE": lambda: tf.OlmoeForCausalLM(
        tf.OlmoeConfig(**SMALL, intermediate_size=48, num_experts=8, num_experts_per_tok=2)
    ),
    "Qwen2-MoE": lambda: tf.Qwen2MoeForCausalLM(
        tf.Qwen2MoeConfig(
            **SMALL,
            intermediate_size=64,
            moe_intermediate_size=24,
            shared_expert_intermediate_size=40,
            num_experts=6,
            num_experts_per_tok=2,
        )
    ),
    "Qwen3-MoE": lambda: tf.Qwen3MoeForCausalLM(
        tf.Qwen3MoeConfig(
            **SMALL,
            intermediate_size=64,
            moe_intermediate_size=24,
            num_experts=6,
            num_experts_per_tok=2,
            head_dim=8,
        )
    ),
    "GPT-OSS": lambda: tf.GptOssForCausalLM(
        tf.GptOssConfig(
            **SMALL,
            intermediate_size=24,
            num_local_experts=4,
            num_experts_per_tok=2,
            head_dim=8,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
        )
    ),
    "GraniteMoE": lambda: tf.GraniteMoeForCausalLM(
        tf.GraniteMoeConfig(
            **SMALL, intermediate_size=24, num_local_experts=4, num_experts_per_tok=2
        )
    ),
    "Mistral 4": lambda: tf.Mistral4ForCausalLM(
        tf.Mistral4Config(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            intermediate_size=64,
            moe_intermediate_size=24,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=1,
            kv_lora_rank=16,
            q_lora_rank=16,
            qk_rope_head_dim=4,
            qk_nope_head_dim=4,
            v_head_dim=8,
            pad_token_id=0,
        )
    ),
}


def check(name: str, build) -> str:
    """One line: the difference from eager, and the effect of zeroing one expert neuron."""
    torch.manual_seed(0)
    model = build().eval()
    for p in model.parameters():  # larger weights, so routing and outputs vary
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.2)
    ids = torch.randint(0, 128, (2, 12))
    experts = [m for m in model.modules() if hasattr(m, "has_gate") and hasattr(m, "down_proj")]
    with torch.no_grad():
        model.set_experts_implementation("eager")
        eager = model(ids).logits
        model.set_experts_implementation("openinterp")
        ours = model(ids).logits
    seen = {"negative": False}

    def zero(act, expert, tokens, weights):
        seen["negative"] |= bool((weights < 0).any())
        if expert != 1:
            return act
        act = act.clone()
        act[:, 3] = 0
        return act

    HOOKS[id(experts[0])] = zero
    with torch.no_grad():
        ablated = model(ids).logits
    HOOKS.clear()
    diff = (ours - eager).abs().max().item()
    changed = not torch.equal(ablated, ours)
    return (
        f"{name:10s} max |openinterp − eager| = {diff:.1e}, zeroing E1/N3 changes the "
        f"logits: {changed}, negative routing weights: {seen['negative']}"
    )


if __name__ == "__main__":
    for model_name, build in MODELS.items():
        print(check(model_name, build))
