"""Check the hook point that docs/moe.md chooses for expert neurons, on tiny random models.

transformers 5 runs the experts of about 60 mixture-of-experts architectures through
``ExpertsInterface``: a class decorated with ``use_experts_implementation`` looks its
forward up by name. openinterp registers its own, ``openinterp``
(``openinterp/harness/moe.py``). It loops over the routed experts and passes each
expert's hidden activation to a hook. For each architecture this script prints:

* the largest difference between its logits and those of the ``eager`` forward
* whether zeroing one neuron inside one expert changes the logits
* whether every routing weight was non-negative, which the poles rely on.

    uv run python scripts/moe_experts_check.py

It needs no download: each model is a random two- or three-layer configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import transformers as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openinterp.harness.moe import expert_hook, experts_modules, use_openinterp_experts  # noqa: E402

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


def tiny(build) -> torch.nn.Module:
    """A random model with larger weights than the default, so routing and outputs vary."""
    torch.manual_seed(0)
    model = build().eval()
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.2)
    return model


def check(name: str, build) -> str:
    """One line: the difference from eager, and the effect of zeroing one expert neuron."""
    model = tiny(build)
    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        model.set_experts_implementation("eager")
        eager = model(ids).logits
        use_openinterp_experts(model)
        ours = model(ids).logits
    seen = {"negative": False}

    def zero(act, expert, tokens, weights):
        seen["negative"] |= bool((weights < 0).any())
        if expert != 1:
            return act
        act = act.clone()
        act[:, 3] = 0
        return act

    with torch.no_grad(), expert_hook(experts_modules(model)[0], zero):
        ablated = model(ids).logits
    diff = (ours - eager).abs().max().item()
    changed = not torch.equal(ablated, ours)
    return (
        f"{name:10s} max |openinterp − eager| = {diff:.1e}, zeroing E1/N3 changes the "
        f"logits: {changed}, negative routing weights: {seen['negative']}"
    )


if __name__ == "__main__":
    for model_name, build in MODELS.items():
        print(check(model_name, build))
