"""Which VERIFIED neurons keep their hypothesis after instruction tuning (#3375, #3762).

The pair is a base model (``--base``, Qwen2.5-1.5B by default) and its ``-Instruct``
model. Both have one architecture, so neuron i of layer l is the same position in both.
For each neuron pole of the base model that stands VERIFIED, the script takes its
standing record's own probes and measures them:

* on the base model, with the record's scale. This checks that the record reproduces.
* on the instruct model, at the same layer, neuron and pole. Its scale is the record's
  scale times a ratio: the pole's maximum in the instruct model over its maximum in the
  base model. Both maxima are over one sample of ``corpus.jsonl``.

A hypothesis *keeps* when the instruct readings pass the record's specificity and
sensitivity gates. The script also gives the Pearson correlation of the neuron's
activation between the two models on the sample. It writes no record.

With ``--chat``, each probe goes to the instruct model as one user turn of its chat
template, with no generation prompt. The pole is read only over the tokens that overlap
the probe's own characters, so the template's tokens can't fire it. The base readings
stay plain text.

    uv run python scripts/base_instruct.py --device mps --out docs/analyses/base-instruct.jsonl
    uv run python scripts/base_instruct.py --base Qwen/Qwen2.5-0.5B --chat --device mps \\
        --out docs/analyses/base-instruct-0.5b-chat.jsonl
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import click
import torch

BASE = "Qwen/Qwen2.5-1.5B"
SAMPLE_TOKENS = 20_000
WINDOW = 512
BATCH = 4
GATES = ("specificity", "sensitivity")


def verified_records(wiki_dir: Path, model: str) -> list:
    """The standing record of each neuron pole of ``model`` that stands VERIFIED."""
    from openinterp.builder.tree import build_tree
    from openinterp.builder.wiki import load_archive
    from openinterp.core.errata import load_errata

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


def sample_windows(tokenizer, tokens: int = SAMPLE_TOKENS) -> list[torch.Tensor]:
    """``tokens`` tokens of ``corpus.jsonl`` as rows of :data:`WINDOW` tokens, in batches."""
    ids: list[int] = []
    with open("corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            ids += tokenizer(json.loads(line)["text"])["input_ids"]
            if len(ids) >= tokens:
                break
    rows = torch.tensor(ids[: tokens // WINDOW * WINDOW]).view(-1, WINDOW)
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
        click.echo(f"layer {layer}: {len(idents)} poles", err=True)
    return out


def chat_strength(runner, unit, text: str) -> float | None:
    """The unit's peak over ``text``'s tokens, with ``text`` as a user turn of the template.

    ``None`` when no token overlaps the text.
    """
    tokenizer = runner.tokenizer
    turn = [{"role": "user", "content": text}]
    chat = tokenizer.apply_chat_template(turn, tokenize=False, add_generation_prompt=False)
    start = chat.rindex(text)  # the user turn, after any system turn
    offsets = tokenizer(chat, return_offsets_mapping=True)["offset_mapping"]
    strengths = unit.strengths(runner, chat)[0]
    held = [
        float(strengths[i])
        for i, (a, b) in enumerate(offsets)
        if a < start + len(text) and b > start and i < len(strengths)
    ]
    return max(held) if held else None


def chat_probes(runner, unit, probes, scale: float, threshold: float):
    """:func:`measure_probes`'s results, with each probe read by :func:`chat_strength`."""
    from openinterp.core.schemas import AdversarialTestResult

    results = []
    for probe in probes:
        strength = chat_strength(runner, unit, probe.text)
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


def readings(runner, record, scale_factor: float = 1.0, chat: bool = False) -> dict:
    """The record's probes measured on ``runner``: rates, and the two gates.

    With ``chat``, each probe is read inside the chat template (:func:`chat_strength`).
    """
    from openinterp.agents.review_measurements import measure_probes, settle_first_token_floor
    from openinterp.agents.reviewer import AdversarialProbe, review_metrics
    from openinterp.core.schemas import ReviewPolicy

    policy = record.verdict.policy or ReviewPolicy()
    probes = [
        AdversarialProbe(text=r.prompt, expected_active=r.expected_active, rationale="")
        for r in record.verdict.adversarial_test_results
        if r.kind in ("hard_negative", "positive_holdout")
    ]
    _, unit, scale = settle_first_token_floor(record.submission, runner, policy)
    scale *= scale_factor
    measure = chat_probes if chat else measure_probes
    results, _, _ = measure(runner, unit, probes, scale, policy.activation_threshold * scale)
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


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--base", "base_name", default=BASE, show_default=True)
@click.option("--chat", is_flag=True, help="Read the instruct model inside its chat template.")
def main(wiki_dir: Path, out: Path, device: str, base_name: str, chat: bool) -> None:
    """Measure each VERIFIED pole's probes on both models, and write the rows."""
    from openinterp.cli import _load_runner

    instruct_name = f"{base_name}-Instruct"
    records = verified_records(wiki_dir, base_name)
    base = _load_runner(base_name, device, "float32", None)
    instruct = _load_runner(instruct_name, device, "float32", None)
    batches = sample_windows(base.tokenizer)
    stats = sample_stats(base, instruct, batches, records)
    rows = []
    for record in records:
        ident = record.submission.identifier
        s = stats[ident.slug]
        on_base = readings(base, record)
        on_instruct = readings(instruct, record, s["ratio"] or 1.0, chat)
        rows.append({
            "table": "pole", "slug": ident.slug, "layer": ident.layer,
            "depth": round(ident.layer / (base.n_layers - 1), 4), "pole": ident.pole,
            "explanation": record.submission.natural_language_explanation,
            "recorded": {g: getattr(record.verdict.metrics, g) for g in GATES},
            "base": on_base, "instruct": on_instruct, **s,
            "keeps": on_base["passes"] and on_instruct["passes"],
        })  # fmt: skip
    reproduced = [r for r in rows if r["base"]["passes"]]
    kept = sum(r["keeps"] for r in reproduced)
    summary = {
        "table": "summary", "base": base_name, "instruct": instruct_name, "chat": chat,
        "poles": len(rows),
        "reproduced_on_base": len(reproduced), "keeps": kept,
        "keep_share": round(kept / len(reproduced), 4) if reproduced else None,
        "keep_ci95": wilson(kept, len(reproduced)), "sample_tokens": SAMPLE_TOKENS,
    }  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in [summary, *rows]))
    click.echo(json.dumps(summary))


if __name__ == "__main__":
    main()
