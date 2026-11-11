"""Follow up the bracket results of papers/syntax-positive-controls/README.md.

Three tests, all on written text. The text comes from the templates here and in
``syntax_probes.py``, not from a corpus.

- ``close_later``: when GPT-2 closes fewer round brackets than are open, does it close
  the rest with its next tokens? Each step adds the closing token that the model finds
  most probable, and reads the probability of one more closer.
- ``blocks``: the round-depth poles of GPT-2 under curly brackets that hold code blocks,
  against an object literal, round brackets and indentation alone.
- ``qwen``: the best round-depth pole of each Qwen2.5 model on the probes of
  ``syntax_probes.py``. Its effect on a closing bracket, by ablation and through the
  unembedding.

    uv run python scripts/syntax_followups.py --out papers/syntax-positive-controls/followups.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import click
import torch
from syntax_probes import (
    CALLEES,
    CONTROLS,
    DEPTHS,
    ROUND_DEPTH_POLES,
    TARGETS,
    _rounded,
    ablate_poles,
    calls,
    closer_counts,
    contexts,
    extra_contexts,
)

from openinterp.cli import _load, resolve_device
from openinterp.harness.hooks import pole_strength

GPT2 = "openai-community/gpt2"
# The best `round depth` pole of each Qwen2.5 model, in its sweep over whole documents.
QWEN_POLES = {
    "Qwen/Qwen2.5-0.5B": (21, 953, "negative"),
    "Qwen/Qwen2.5-1.5B": (26, 7101, "positive"),
}
# GPT-2's best `curly depth` pole, which the blocks test reads with the round-depth poles.
CURLY_POLE = (4, 3038, "negative")
CHAIN_STEPS = 6  # more than the most closers that a chain at depth 5 needs
GREEDY_TOKENS = 16
SEED = 0
TOP_TOKENS = 15
# Conditions under which the chain runs: the clean model, and two sets of ablated poles.
CHAIN_SETS = {
    "clean": (),
    "L2 N676+ ablated": ROUND_DEPTH_POLES[:1],
    "the other four ablated": ROUND_DEPTH_POLES[1:],
}
HEADERS = ("if (ready)", "while (busy)", "if (n > 0)", "for (i = 0; i < n; i++)")
KEYS = ("a", "b", "c", "d", "e")


def counts_for(runner, size: int) -> torch.Tensor:
    """Return :func:`closer_counts` padded with zeros to ``size``, the width of the logits.

    Qwen2.5's logits have more entries than its tokenizer has tokens. No token is
    behind the extra entries, so each closes no bracket.
    """
    counts = closer_counts(runner)
    return torch.cat([counts, counts.new_zeros(max(size - len(counts), 0))])[:size]


def logits_at_end(runner, ids: torch.Tensor, poles=()) -> torch.Tensor:
    """Return the logits of the next token after ``ids``, with ``poles`` ablated."""
    changes = ablate_poles(poles) if poles else None
    return runner.run_with_interventions(ids.unsqueeze(0), changes)[0, -1].float()


def closer_chain(runner, ids: torch.Tensor, depth: int, poles=()) -> dict:
    """Add the most probable closing token until no bracket is open, and record each step.

    At each step, ``p_closer`` is the probability that the next token starts with ``)``,
    and ``open`` is the number of brackets still open before it. The chain stops when no
    bracket is open, after one more reading, so the last step shows the probability of a
    closer that would close too many. ``tokens`` holds the closers that the chain added.
    """
    steps, tokens, remaining = [], [], depth
    for _ in range(CHAIN_STEPS):
        probs = logits_at_end(runner, ids, poles).softmax(-1).cpu()
        counts = counts_for(runner, len(probs))
        closing = (counts > 0).nonzero().squeeze(1)
        total = float(probs[closing].sum())
        mean_count = float((probs * counts).sum()) / max(total, 1e-9)
        steps.append({"open": remaining, "p_closer": total, "count": mean_count})
        if remaining <= 0:
            break
        token = int(closing[probs[closing].argmax()])
        remaining -= int(counts[token])
        tokens.append(token)
        ids = torch.cat([ids, torch.tensor([token], device=ids.device)])
    return {"steps": steps, "tokens": tokens, "closed_too_many": remaining < 0}


def greedy_balance(runner, ids: torch.Tensor, poles=()) -> int | None:
    """Return the net number of round brackets that greedy text closes on its line.

    Each token of the text is the most probable next token. If the line does not end
    within ``GREEDY_TOKENS`` tokens, return ``None``: such text is usually a loop that
    repeats an argument, and its count says nothing about closing.
    """
    out = []
    for _ in range(GREEDY_TOKENS):
        token = int(logits_at_end(runner, ids, poles).argmax())
        word = runner.tokenizer.decode([token])
        if "\n" in word:
            text = "".join(out) + word.split("\n")[0]
            return text.count(")") - text.count("(")
        out.append(word)
        ids = torch.cat([ids, torch.tensor([token], device=ids.device)])
    return None


def close_later_rows(runner) -> list[dict]:
    """Return how GPT-2 closes its open brackets over several tokens, per depth and set.

    For each depth, condition and target, a chain adds the most probable closer at each
    step (see :func:`closer_chain`). Each row gives the mean over targets, per step.
    It also counts the lines of greedy text that end, and how many of those close every
    bracket, too few or too many (see :func:`greedy_balance`).
    """
    rows = []
    for depth in DEPTHS:
        texts = [f"out = {calls(depth, k)}first,{t}" for k, t in enumerate(TARGETS)]
        encoded = [runner.encode([text])["input_ids"][0] for text in texts]
        for name, poles in CHAIN_SETS.items():
            chains = [closer_chain(runner, ids, depth, poles) for ids in encoded]
            first = [c["steps"][0] for c in chains]
            # The reading just after the first closer, when brackets are still open.
            after = [c["steps"][1] for c in chains if c["steps"][1]["open"] > 0]
            # The reading after the chain closed every bracket, and no more.
            done = [c["steps"][-1] for c in chains if c["steps"][-1]["open"] == 0]
            mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
            balance = [greedy_balance(runner, ids, poles) for ids in encoded]
            rows.append(
                {
                    "kind": "close_later",
                    "depth": depth,
                    "set": name,
                    "first_count": mean([s["count"] for s in first]),
                    "first_p_closer": mean([s["p_closer"] for s in first]),
                    "still_open_items": len(after),
                    "still_open_p_closer": mean([s["p_closer"] for s in after]),
                    "still_open_left": mean([s["open"] for s in after]),
                    "all_closed_items": len(done),
                    "all_closed_p_closer": mean([s["p_closer"] for s in done]),
                    "closed_too_many": sum(c["closed_too_many"] for c in chains),
                    "steps_to_close": mean([len(c["steps"]) - 1 for c in chains]),
                    "greedy_lines_ended": sum(b is not None for b in balance),
                    "greedy_balanced": sum(b == depth for b in balance),
                    "greedy_too_few": sum(b is not None and b < depth for b in balance),
                    "greedy_too_many": sum(b is not None and b > depth for b in balance),
                    "items": len(texts),
                }
            )
    return [_rounded(row) for row in rows]


def block_contexts(depth: int, offset: int) -> dict[str, str]:
    """Return the text before ``first +`` per condition, at ``depth`` levels of nesting.

    ``block`` opens ``depth`` code blocks, each header on its own line. Each header holds
    a round bracket that is already closed. ``block_closed`` closes the same blocks
    first. ``block_one_line`` opens them on one line. ``object`` opens ``depth`` object
    literals. ``indent`` nests Python blocks, which have no brackets. ``round`` opens
    ``depth`` calls, as in ``syntax_probes.py``.
    """
    headers = [HEADERS[(offset + i) % len(HEADERS)] for i in range(depth)]
    opened = "".join(f"{'    ' * i}{h} {{\n" for i, h in enumerate(headers))
    closed = opened + "".join(f"{'    ' * i}}}\n" for i in reversed(range(depth)))
    python = "".join(
        f"{'    ' * i}{h.split(' (')[0]} {CALLEES[(offset + i) % len(CALLEES)]}:\n"
        for i, h in enumerate(headers)
    )
    return {
        "block": f"{opened}{'    ' * depth}x = ",
        "block_closed": f"{closed}x = ",
        "block_one_line": "".join(f"{h} {{ " for h in headers) + "x = ",
        "object": "x = " + "".join(f"{{{KEYS[i]}: " for i in range(depth)),
        "indent": f"{python}{'    ' * depth}x = ",
        "round": f"x = {calls(depth, offset)}",
    }


def read_last(runner, text: str, poles) -> list[float]:
    """Return each pole's strength at the last token of ``text``."""
    layers = sorted({layer for layer, _, _ in poles})
    _, cache = runner.run_with_cache([text], layers)
    return [float(pole_strength(cache[layer][0, -1, n], pole)) for layer, n, pole in poles]


def mean_rows(kind: str, cells: dict[tuple, list[float]], key: str) -> list[dict]:
    """Return the mean and standard error of each cell, keyed by pole, condition and ``key``."""
    rows = []
    for ((layer, neuron, pole), condition, value), found in cells.items():
        spread = torch.tensor(found)
        rows.append(
            {
                "kind": kind,
                "layer": layer,
                "neuron": neuron,
                "pole": pole,
                "condition": condition,
                key: value,
                "mean": round(float(spread.mean()), 4),
                "sem": round(float(spread.std() / len(found) ** 0.5), 4),
                "items": len(found),
            }
        )
    return rows


def block_rows(runner) -> list[dict]:
    """Return the strength of each round-depth pole, and the curly pole, under blocks."""
    poles = (*ROUND_DEPTH_POLES, CURLY_POLE)
    cells: dict[tuple, list[float]] = {}
    for depth in DEPTHS:
        for offset, target in enumerate(TARGETS):
            for condition, before in block_contexts(depth, offset).items():
                values = read_last(runner, f"{before}first +{target}", poles)
                for pole, value in zip(poles, values, strict=True):
                    cells.setdefault((pole, condition, depth), []).append(value)
    return mean_rows("blocks", cells, "depth")


def qwen_probe_rows(runner, model: str, pole) -> list[dict]:
    """Return the pole's strength on the probes of ``syntax_probes.py``, and with distance."""
    cells: dict[tuple, list[float]] = {}
    for depth in DEPTHS:
        for offset, target in enumerate(TARGETS):
            every = contexts(depth, offset) | extra_contexts(depth, offset)
            for condition, before in every.items():
                value = read_last(runner, f"{before}first,{target}", (pole,))[0]
                cells.setdefault((pole, condition, depth), []).append(value)
    distance: dict[tuple, list[float]] = {}
    for filler in (0, 5, 10, 20, 40, 80):
        for condition, start in (("open", "out = load("), ("closed", "out = load(x)\nfirst = ")):
            for target in TARGETS:
                middle = "".join(f"a{i}, " for i in range(filler))
                value = read_last(runner, f"{start}{middle}first,{target}", (pole,))[0]
                distance.setdefault((pole, condition, filler), []).append(value)
    rows = [
        *mean_rows("qwen_probe", cells, "depth"),
        *mean_rows("qwen_distance", distance, "filler_items"),
    ]
    return [row | {"model": model} for row in rows]


def qwen_ablation_rows(runner, model: str, pole) -> list[dict]:
    """Return how ablating the pole, or random poles of its layer, changes the next closer.

    Each measure is a mean over ``TARGETS``. ``any`` is the probability of a token that
    starts with ``)``. ``count`` is the mean number of brackets that a closing token
    closes. The controls are ``CONTROLS`` random poles of the same layer and sign.
    """
    layer, neuron, sign = pole
    picker = random.Random(SEED)
    others = [n for n in picker.sample(range(runner.mlp_width(layer)), CONTROLS + 1) if n != neuron]
    controls = [(layer, n, sign) for n in others[:CONTROLS]]
    rows = []
    for depth in DEPTHS:
        texts = [f"out = {calls(depth, k)}first,{t}" for k, t in enumerate(TARGETS)]
        encoded = [runner.encode([text])["input_ids"][0] for text in texts]

        def measure(poles, encoded=encoded) -> dict:
            found = []
            for ids in encoded:
                probs = logits_at_end(runner, ids, poles).softmax(-1).cpu()
                counts = counts_for(runner, len(probs))
                total = float(probs[counts > 0].sum())
                found.append(
                    {"any": total, "count": float((probs * counts).sum()) / max(total, 1e-9)}
                )
            return {k: sum(f[k] for f in found) / len(found) for k in found[0]}

        found = [measure((c,)) for c in controls]
        rows.append(
            {
                "kind": "qwen_ablation",
                "model": model,
                "depth": depth,
                "clean": measure(()),
                "ablated": measure((pole,)),
                "controls_mean": {k: sum(f[k] for f in found) / len(found) for k in found[0]},
                "controls_range": {
                    k: [min(f[k] for f in found), max(f[k] for f in found)] for k in found[0]
                },
            }
        )
    return [_rounded(row) for row in rows]


def unembed_row(runner, model: str, pole) -> dict:
    """Return the tokens that the pole's output raises most through the unembedding.

    This is the direct path only: the pole's output direction, through the final norm's
    scale, times the unembedding. It leaves out every later layer. The sign follows the
    pole, so a negative pole's direction is the negative of its neuron's.
    """
    layer, neuron, sign = pole
    direction = runner.neuron_output_direction(layer, neuron).float()
    direction = direction if sign == "positive" else -direction
    norm = runner.final_norm()
    if norm is not None and getattr(norm, "weight", None) is not None:
        direction = direction * norm.weight.detach().float()
    unembed = runner.model.get_output_embeddings().weight.detach().float()
    logits = (unembed @ direction.to(unembed.device)).cpu()
    order = logits.argsort(descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    counts = counts_for(runner, len(logits))
    closing = (counts > 0).nonzero().squeeze(1)
    single = runner.tokenizer.encode(")", add_special_tokens=False)
    return {
        "kind": "qwen_unembed",
        "model": model,
        "layer": layer,
        "neuron": neuron,
        "pole": sign,
        "top": [runner.tokenizer.decode([int(i)]) for i in order[:TOP_TOKENS]],
        "bottom": [runner.tokenizer.decode([int(i)]) for i in order[-TOP_TOKENS:].flip(0)],
        "close_rank": int(rank[single[0]]) if len(single) == 1 else None,
        "closer_median_rank": int(rank[closing].median()),
        "closers_in_top_100": int((rank[closing] < 100).sum()),
        "closers": int(len(closing)),
        "vocab": int(len(logits)),
    }


@click.command()
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cpu")
def main(out, device):
    """Write the rows of the three follow-up tests to ``out``."""
    with torch.no_grad(), out.open("w", encoding="utf-8") as sink:

        def write(rows: list[dict]) -> None:
            sink.write("".join(json.dumps(row) + "\n" for row in rows))
            sink.flush()

        runner = _load(GPT2, resolve_device(device), torch.float32, None, None)
        write(close_later_rows(runner))
        write(block_rows(runner))
        for model, pole in QWEN_POLES.items():
            runner = _load(model, resolve_device(device), torch.float32, None, None)
            write(qwen_probe_rows(runner, model, pole))
            write(qwen_ablation_rows(runner, model, pole))
            write([unembed_row(runner, model, pole)])


if __name__ == "__main__":
    main()
