"""A corpus sample inside a chat template, read on each pole (#5126).

The keep test of ``docs/analyses/base-instruct.md`` reads probes inside the chat template.
This module gives the matching corpus sample: each document of ``corpus.jsonl`` as one user
turn of a template. Each token is either part of the document's own text or part of the
template. A pole's activations are then read over the two parts apart:

* over the document's tokens, for the pole's correlation between two models on chat text;
* over the template's tokens, for how strongly the pole fires on the template itself.

One rendering tokenizer serves both models, so both read the same text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

DOC_CHARS = 2_000  # the longest document text that one turn holds
SAMPLE_TOKENS = 20_000  # the document tokens of a sample, as in keep.SAMPLE_TOKENS


def chat_sample(render, tokenizer, corpus: Path | str = "corpus.jsonl",
                tokens: int = SAMPLE_TOKENS) -> list[dict]:  # fmt: skip
    """Documents of ``corpus`` as user turns, until they hold ``tokens`` document tokens.

    Args:
        render: The tokenizer whose chat template renders each turn.
        tokenizer: The tokenizer that the models read with.
        corpus: A JSON Lines file with a ``text`` field on each line.
        tokens: The document tokens to collect.

    Returns:
        One dict for each document: ``ids``, a ``[1, length]`` tensor, and ``text``, a
        boolean ``[length]`` tensor that is True on the document's own tokens.

    Raises:
        ValueError: The corpus gives no document tokens.
    """
    out, held = [], 0
    with open(corpus, encoding="utf-8") as lines:
        for line in lines:
            doc = json.loads(line)["text"][:DOC_CHARS].strip()
            if not doc:
                continue
            turn = [{"role": "user", "content": doc}]
            chat = render.apply_chat_template(turn, tokenize=False, add_generation_prompt=False)
            start = chat.rindex(doc)
            enc = tokenizer(chat, return_offsets_mapping=True)
            mask = torch.tensor([a < start + len(doc) and b > start
                                 for a, b in enc["offset_mapping"]])  # fmt: skip
            out.append({"ids": torch.tensor([enc["input_ids"]]), "text": mask})
            held += int(mask.sum())
            if held >= tokens:
                break
    if not held:
        raise ValueError(f"{corpus} gives no document tokens")
    return out


def chat_columns(runner, sample: Sequence[dict], wanted: dict[int, list[int]]) -> dict:
    """Each wanted neuron's activations on the sample, over the text and the template apart.

    Args:
        runner: The model's ``NeuronHookRunner``.
        sample: The documents of :func:`chat_sample`.
        wanted: The neurons to read, by layer.

    Returns:
        ``{layer: {"text": tensor [text tokens, k], "template": tensor [template tokens, k]}}``,
        with the columns in the order of ``wanted[layer]``.
    """
    layers = sorted(wanted)
    parts: dict[int, dict[str, list]] = {layer: {"text": [], "template": []} for layer in layers}
    with torch.no_grad():
        for doc in sample:
            acts = runner.layer_activations({"input_ids": doc["ids"].to(runner.device)}, layers)
            for layer in layers:
                cols = acts[layer][0, :, wanted[layer]].float().cpu()
                parts[layer]["text"].append(cols[doc["text"]])
                parts[layer]["template"].append(cols[~doc["text"]])
    return {layer: {k: torch.cat(v) for k, v in p.items()} for layer, p in parts.items()}


def pole_mean(acts: torch.Tensor, pole: str) -> float:
    """The pole's mean strength over ``acts``: the activation, or its negative for a negative pole.

    Args:
        acts: One neuron's activations.
        pole: ``positive`` or ``negative``.

    Returns:
        The mean of the pole's strength, clamped at 0 from below.

    Examples:
        >>> pole_mean(torch.tensor([1.0, -1.0, 3.0]), "positive")
        1.3333333730697632
        >>> pole_mean(torch.tensor([1.0, -1.0, 3.0]), "negative")
        0.3333333432674408
    """
    signed = acts if pole == "positive" else -acts
    return float(signed.clamp(min=0).mean())
