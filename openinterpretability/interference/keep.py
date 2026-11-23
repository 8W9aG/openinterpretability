"""Unit, input side: each VERIFIED pole's keep test on a changed model (issue #5230).

This is the keep test of ``docs/analyses/base-instruct.md`` (#3375). It moved here from
``scripts/base_instruct.py``, so the interference report and the script share it. Each
neuron pole that stands VERIFIED has its standing record's own probes measured on the base
model and on the changed model. On the changed model, the scale is the record's scale times
a ratio. The ratio is the pole's maximum there over its maximum on the base, over one
corpus sample. A hypothesis *keeps* when the changed readings pass the record's
specificity and sensitivity gates.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

SAMPLE_TOKENS = 20_000
WINDOW = 512
BATCH = 4
GATES = ("specificity", "sensitivity")


def verified_records(wiki_dir: Path, model: str) -> list:
    """The standing record of each neuron pole of ``model`` that stands VERIFIED."""
    from openinterpretability.builder.tree import build_tree
    from openinterpretability.builder.wiki import load_archive
    from openinterpretability.core.errata import load_errata

    out = []
    for node in build_tree(load_archive(wiki_dir), load_errata(wiki_dir)):
        if node.name != model:
            continue
        for layer in node.layers:
            for unit in layer.units:
                for standing in unit.standings.values():
                    if unit.is_neuron and standing.verified:
                        out.append(standing.lead.record)
    return out


def sample_windows(tokenizer, tokens: int = SAMPLE_TOKENS,
                   corpus: Path | str = "corpus.jsonl") -> list[torch.Tensor]:  # fmt: skip
    """``tokens`` tokens of ``corpus`` as rows of :data:`WINDOW` tokens, in batches.

    A corpus shorter than ``tokens`` gives the whole windows that it holds (#5909).

    Raises:
        ValueError: The corpus holds less than one window.
    """
    ids: list[int] = []
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            ids += tokenizer(json.loads(line)["text"])["input_ids"]
            if len(ids) >= tokens:
                break
    whole = min(len(ids), tokens) // WINDOW * WINDOW
    if not whole:
        raise ValueError(f"{corpus} holds {len(ids)} tokens, less than one window of {WINDOW}")
    rows = torch.tensor(ids[:whole]).view(-1, WINDOW)
    return [rows[i : i + BATCH] for i in range(0, len(rows), BATCH)]


def layer_columns(runner, batches, layer: int, neurons: list[int]) -> torch.Tensor:
    """``[tokens, len(neurons)]``: the neurons' activations over every window.

    Each window's first token is left out.
    """
    parts = []
    for batch in batches:
        acts = runner.layer_activations({"input_ids": batch.to(runner.device)}, [layer])[layer]
        parts.append(acts[:, 1:, neurons].float().cpu().reshape(-1, len(neurons)))
    return torch.cat(parts)


def pole_max(acts: torch.Tensor, pole: str) -> float:
    """The pole's largest strength in ``acts``."""
    return float((acts.clamp(min=0) if pole == "positive" else (-acts).clamp(min=0)).max())


def pearson(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """Pearson's r of two equal-length vectors, or ``None`` when one is constant."""
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    denom = float(a.norm() * b.norm())
    return round(float((a * b).sum()) / denom, 4) if denom > 0 else None


def sample_stats(base, instruct, batches, records) -> dict[str, dict]:
    """``{slug: {"ratio", "pearson"}}`` from one pass of each model per layer."""
    by_layer: dict[int, list] = defaultdict(list)
    for r in records:
        by_layer[r.submission.identifier.layer].append(r.submission.identifier)
    out = {}
    for layer, idents in sorted(by_layer.items()):
        neurons = sorted({i.neuron_idx for i in idents})
        a = layer_columns(base, batches, layer, neurons)
        b = layer_columns(instruct, batches, layer, neurons)
        for ident in idents:
            col = neurons.index(ident.neuron_idx)
            top = pole_max(a[:, col], ident.pole)
            ratio = pole_max(b[:, col], ident.pole) / top if top > 0 else None
            out[ident.slug] = {"ratio": ratio, "pearson": pearson(a[:, col], b[:, col])}
        print(f"layer {layer}: {len(idents)} poles", file=sys.stderr)
    return out


def chat_strength(runner, unit, text: str, template=None) -> float | None:
    """The unit's peak over ``text``'s tokens, with ``text`` as a user turn of the template.

    ``template`` is the tokenizer whose chat template renders the turn, by default the
    runner's own. A base model and its instruct model can then read one rendered text,
    since their templates' default system prompts differ (#5126). ``None`` when no token
    overlaps the text.
    """
    tokenizer = runner.tokenizer
    turn = [{"role": "user", "content": text}]
    chat = (template or tokenizer).apply_chat_template(
        turn, tokenize=False, add_generation_prompt=False
    )
    start = chat.rindex(text)  # the user turn, after any system turn
    offsets = tokenizer(chat, return_offsets_mapping=True)["offset_mapping"]
    strengths = unit.strengths(runner, chat)[0]
    held = [
        float(strengths[i])
        for i, (a, b) in enumerate(offsets)
        if a < start + len(text) and b > start and i < len(strengths)
    ]
    return max(held) if held else None


def chat_probes(runner, unit, probes, scale: float, threshold: float, template=None):
    """:func:`measure_probes`'s results, with each probe read by :func:`chat_strength`.

    ``template`` renders the chat turn, as in :func:`chat_strength`.
    """
    from openinterpretability.core.schemas import AdversarialTestResult

    results = []
    for probe in probes:
        strength = chat_strength(runner, unit, probe.text, template)
        active = strength is not None and scale > 0 and strength >= threshold
        results.append(
            AdversarialTestResult(
                kind="positive_holdout" if probe.expected_active else "hard_negative",
                prompt=probe.text, passed=strength is not None and active == probe.expected_active,
                expected_active=probe.expected_active,
                activation=None if strength is None else round(strength, 6),
                detail=probe.rationale, measured=strength is not None,
            )
        )  # fmt: skip
    return results, None, None


def readings(runner, record, scale_factor: float = 1.0, chat: bool = False,
             template=None) -> dict:  # fmt: skip
    """The record's probes measured on ``runner``: rates, and the two gates.

    With ``chat``, each probe is read inside the chat template (:func:`chat_strength`),
    rendered by ``template`` when it is given.
    """
    from openinterpretability.agents.review_measurements import (
        measure_probes,
        settle_first_token_floor,
    )
    from openinterpretability.agents.reviewer import AdversarialProbe, review_metrics
    from openinterpretability.core.schemas import ReviewPolicy

    policy = record.verdict.policy or ReviewPolicy()
    probes = [
        AdversarialProbe(text=r.prompt, expected_active=r.expected_active, rationale="")
        for r in record.verdict.adversarial_test_results
        if r.kind in ("hard_negative", "positive_holdout")
    ]
    _, unit, scale = settle_first_token_floor(record.submission, runner, policy)
    scale *= scale_factor
    threshold = policy.activation_threshold * scale
    if chat:
        results, _, _ = chat_probes(runner, unit, probes, scale, threshold, template)
    else:
        results, _, _ = measure_probes(runner, unit, probes, scale, threshold)
    metrics = review_metrics(results, policy, independent=True)
    return {
        "specificity": metrics.specificity, "sensitivity": metrics.sensitivity,
        "balanced_accuracy": metrics.balanced_accuracy,
        "passes": all(metrics.gates[g] for g in GATES),
        "failed": [g for g in GATES if not metrics.gates[g]],
    }  # fmt: skip


def wilson(k: int, n: int) -> list[float]:
    """The 95% Wilson interval of ``k`` successes in ``n``."""
    if n == 0:
        return [0.0, 1.0]
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]
