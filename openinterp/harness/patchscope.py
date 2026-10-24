"""Verbalise a unit's output direction with a Patchscope (Ghandeharioun et al., 2024).

The residual stream after layer ℓ's block, at a placeholder token of an inspection
prompt, is replaced by the unit's output direction scaled to a multiple of the norm it
replaces. The model then continues greedily, and so describes the direction without
seeing a single example. A random direction injected the same way is the control: any
text the two share comes from the prompt, not the unit.

Two inspection prompts are used, a few-shot "word: category" list and a definition
frame, at two layers: the unit's own layer ℓ (source layer = target layer), and an
early layer, as SelfIE (Chen et al., 2024) does. On Qwen2.5-0.5B the few-shot list
injected at layer 2 names the concept of several VERIFIED poles ("crop", "newspaper",
"official", "culture") where the control gives "animal", "emotion" or "period".
Injected at their own layer, poles from layer 17 up mostly read back the control's
text. The decoding is greedy, so a reading is deterministic given the weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from openinterp.core.exceptions import ArchitectureNotSupportedError
from openinterp.core.patchscope import PatchscopeReading, PatchscopeReadout
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.units import FeatureUnit, NeuronUnit, Unit

PROMPTS = {
    "category": "banana: fruit\nhammer: tool\nviolin: music\nsenator: politics\n{x}:",
    "meaning": 'The meaning of the word "{x}" is',
}
PLACEHOLDER = "?"
SCALES = (1.0, 2.0, 4.0)
EARLY_LAYER = 2
MAX_NEW_TOKENS = 12
CONTROL_SEED = 0


def write_direction(runner: NeuronHookRunner, unit: Unit) -> torch.Tensor | None:
    """The vector one unit of the unit's strength adds to the residual stream: a pole's
    signed ``w_out``, or a dictionary latent's decoder row. A head writes a subspace,
    not one direction, so it has none."""
    if isinstance(unit, NeuronUnit):
        ident = unit.ident
        direction = runner.neuron_output_direction(ident.layer, ident.neuron_idx).float()
        return -direction if ident.pole == "negative" else direction
    if isinstance(unit, FeatureUnit):
        return unit.dictionary(runner).w_dec[unit.ident.feature_idx].float()
    return None


def _prompt_ids(runner: NeuronHookRunner, template: str, placeholder: str) -> tuple[list[int], int]:
    """Token ids of ``template`` with the placeholder as exactly one token, and its index."""
    tok = runner.tokenizer
    before, after = template.split("{x}")
    marker = tok.encode(placeholder, add_special_tokens=False)
    if len(marker) != 1:
        raise ValueError(f"the placeholder {placeholder!r} must be one token, not {len(marker)}")
    prefix = tok.encode(before, add_special_tokens=True) if before else []
    return prefix + marker + tok.encode(after, add_special_tokens=False), len(prefix)


def _block_output(output) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def _residual_norm(runner: NeuronHookRunner, ids: list[int], layer: int, pos: int) -> float:
    captured: list[float] = []

    def capture(_module, _args, output):
        captured.append(float(_block_output(output)[0, pos].float().norm()))

    handle = runner.layers[layer].register_forward_hook(capture)
    try:
        runner.forward({"input_ids": torch.tensor([ids], device=runner.device)})
    finally:
        handle.remove()
    if not captured:
        raise ArchitectureNotSupportedError(f"layer {layer}'s block never ran as a module")
    return captured[-1]


@torch.no_grad()
def _continue(
    runner: NeuronHookRunner,
    ids: list[int],
    pos: int,
    layer: int,
    vectors: torch.Tensor,
    max_new_tokens: int,
) -> list[str]:
    """Greedy continuations of ``ids``, one per row of ``vectors``, each with the residual
    after ``layer``'s block at ``pos`` replaced by that row. Each stops at a line break
    or the end-of-text token."""
    tok = runner.tokenizer
    eos = getattr(tok, "eos_token_id", None)

    def patch(_module, _args, output):
        hidden = _block_output(output).clone()
        hidden[:, pos] = vectors.to(hidden.device, hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    seq = torch.tensor([ids] * len(vectors), device=runner.device)
    done = [False] * len(vectors)
    handle = runner.layers[layer].register_forward_hook(patch)
    try:
        for _ in range(max_new_tokens):
            nxt = runner.forward({"input_ids": seq})[:, -1].argmax(-1)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            for row, token in enumerate(nxt.tolist()):
                done[row] |= token == eos or "\n" in tok.decode([token])
            if all(done):
                break
    finally:
        handle.remove()
    texts = []
    for row in seq[:, len(ids) :].tolist():
        if eos in row:
            row = row[: row.index(eos)]
        texts.append(tok.decode(row).split("\n")[0].strip())
    return texts


def patchscope(
    runner: NeuronHookRunner,
    layer: int,
    direction: torch.Tensor,
    *,
    prompts: Mapping[str, str] = PROMPTS,
    placeholder: str = PLACEHOLDER,
    scales: Sequence[float] = SCALES,
    early_layer: int = EARLY_LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
    control_seed: int = CONTROL_SEED,
) -> PatchscopeReadout:
    """Read ``direction``, written after ``layer``'s MLP, through every inspection prompt
    at ``layer`` and at ``early_layer`` (when that is earlier), at each scale."""
    if runner.tokenizer is None:
        raise ValueError("a Patchscope needs a tokenizer")
    direction = direction.float().cpu()
    generator = torch.Generator().manual_seed(control_seed)
    control = torch.randn(direction.shape, generator=generator)
    units = torch.stack([direction / direction.norm(), control / control.norm()])
    readings = []
    for name, template in prompts.items():
        ids, pos = _prompt_ids(runner, template, placeholder)
        for target in sorted({min(early_layer, layer), layer}):
            norm = _residual_norm(runner, ids, target, pos)
            # Rows: the unit at each scale, then the control at each scale.
            vectors = torch.stack([units[i] * norm * s for i in (0, 1) for s in scales])
            texts = _continue(runner, ids, pos, target, vectors, max_new_tokens)
            for i, scale in enumerate(scales):
                readings.append(
                    PatchscopeReading(
                        prompt=name,
                        layer=target,
                        scale=float(scale),
                        text=texts[i],
                        control_text=texts[len(scales) + i],
                    )
                )
    return PatchscopeReadout(
        prompts=dict(prompts),
        placeholder=placeholder,
        max_new_tokens=max_new_tokens,
        control_seed=control_seed,
        readings=readings,
    )
