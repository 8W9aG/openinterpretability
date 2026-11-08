"""Test the sweep's candidates on written text, where the depth is the only change.

The numbers behind papers/syntax-positive-controls/README.md. A correlation over a corpus
can't tell depth from what goes with it. So this script puts one token after one context
token, and changes what comes before them. Brackets are open, or they are already closed,
or they are in a comment or a string. It reads each candidate pole at that token. All the
text comes from the templates here, not from a corpus.

    uv run python scripts/syntax_probes.py --results papers/syntax-positive-controls/results.jsonl
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path

import click
import torch

from openinterp.cli import _load, resolve_device
from openinterp.harness.hooks import ablate_pole, pole_strength
from openinterp.syntax import bracket_depths, indent_depths, parse_labels

MODEL = "openai-community/gpt2"
# (layer, neuron, pole): each is a sweep winner that a depth or construct label set picked.
CANDIDATES = (
    (2, 676, "positive"),
    (4, 1330, "positive"),
    (4, 3038, "negative"),
    (5, 1743, "positive"),
    (6, 1416, "negative"),
    # The other best poles of `round depth`: second in L2, and first in L3, L8 and L9.
    (2, 1195, "positive"),
    (3, 695, "positive"),
    (8, 689, "positive"),
    (9, 340, "positive"),
)
# The poles that `round depth` ranks highest across layers, L2 N676+ first. Each gets the
# distance test, and the joint ablation zeroes them alone and together.
ROUND_DEPTH_POLES = (CANDIDATES[0], *CANDIDATES[5:])
TARGETS = (" value", " count", " name", " size", " index", " total", " key", " item")
CALLEES = ("load", "parse", "read", "check", "build", "send", "open", "find")
DEPTHS = (1, 2, 3, 4, 5)
ABLATE_LAYER, ABLATE_NEURON = 2, 676
CONTROLS = 20
ABLATE_SEED = 0
WRITES = ("probe", "trace", "distance", "ablation", "joint_ablation")
TRACE = """def report(rows, limit):
    # sum the totals (per row)
    total = 0
    for row in rows:
        if check(row, limit):
            total += sum(scale(x, size(row[0])) for x in row)
    text = "done (all rows)"
    return total, text
"""


def typed(depth: int, offset: int, opener: str) -> str:
    """Return ``depth`` nested open calls that use the bracket ``opener``."""
    names = (CALLEES[(offset + i) % len(CALLEES)] for i in range(depth))
    return "".join(f"{name}{opener}" for name in names)


def extra_contexts(depth: int, offset: int) -> dict[str, str]:
    """Return the text before ``first,`` per further condition, at ``depth`` open brackets.

    ``square`` and ``curly`` change the bracket. ``lines`` puts each open call on its own
    line. ``partly_closed`` opens one more call than ``depth``, then closes one, so a
    pole that counts has the strength of ``open`` at ``depth``.
    """
    closed_one = f"{calls(depth, offset)}{CALLEES[(offset + depth) % len(CALLEES)]}(x), "
    lines = "".join(
        f"{call}(\n{'    ' * (i + 1)}"
        for i, call in enumerate(CALLEES[(offset + i) % len(CALLEES)] for i in range(depth))
    )
    return {
        "square": f"out = {typed(depth, offset, '[')}",
        "curly": f"out = {typed(depth, offset, '{')}",
        "lines": f"out = {lines}",
        "partly_closed": f"out = {closed_one}",
    }


def distance_rows(runner, candidate) -> list[dict]:
    """Return a pole's strength at a token some tokens after one open bracket, or none."""
    rows = []
    for filler in (0, 5, 10, 20, 40, 80, 160):
        for condition, start in (("open", "out = load("), ("closed", "out = load(x)\nfirst = ")):
            values = []
            for target in TARGETS:
                middle = "".join(f"a{i}, " for i in range(filler))
                text = f"{start}{middle}first,{target}"
                values.append(read_token(runner, text, (candidate,))[0])
            spread = torch.tensor(values)
            layer, neuron, pole = candidate
            rows.append(
                {
                    "kind": "distance",
                    "layer": layer,
                    "neuron": neuron,
                    "pole": pole,
                    "condition": condition,
                    "filler_items": filler,
                    "mean": round(float(spread.mean()), 4),
                    "sem": round(float(spread.std() / len(values) ** 0.5), 4),
                }
            )
    return rows


def closer_probability(runner, text: str, intervention=None, interventions=None) -> float:
    """Return the next token's probability of starting with a closing round bracket.

    ``intervention`` changes ``ABLATE_LAYER``. ``interventions`` maps layers to changes.
    """
    if intervention is not None:
        interventions = {ABLATE_LAYER: intervention}
    if not interventions:
        logits = runner.run_with_cache([text], [ABLATE_LAYER])[0]
    else:
        logits = runner.run_with_interventions([text], interventions)
    probs = logits[0, -1].softmax(-1)
    vocab = closers(runner)
    return float(probs[vocab].sum())


@cache
def closers(runner) -> torch.Tensor:
    """Return the ids of the tokens that start with a closing round bracket."""
    tok = runner.tokenizer
    words = (tok.decode([i]) for i in range(len(tok)))
    return torch.tensor([i for i, word in enumerate(words) if word.lstrip().startswith(")")])


def ablation_rows(runner) -> list[dict]:
    """Return how ablating the pole, or control poles, changes P(closing bracket)."""
    width = runner.mlp_width(ABLATE_LAYER)
    picker = random.Random(ABLATE_SEED)
    controls = [i for i in picker.sample(range(width), CONTROLS + 1) if i != ABLATE_NEURON][
        :CONTROLS
    ]
    rows = []
    for depth in DEPTHS:
        texts = [f"out = {calls(depth, k)}first,{t}" for k, t in enumerate(TARGETS)]
        clean = [closer_probability(runner, text) for text in texts]
        changed = {
            neuron: [
                closer_probability(runner, text, ablate_pole(neuron, "positive")) for text in texts
            ]
            for neuron in (ABLATE_NEURON, *controls)
        }
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        rows.append(
            {
                "kind": "ablation",
                "depth": depth,
                "clean": round(mean(clean), 4),
                "ablated": round(mean(changed[ABLATE_NEURON]), 4),
                "controls_mean": round(mean([mean(changed[n]) for n in controls]), 4),
                "controls_min": round(min(mean(changed[n]) for n in controls), 4),
            }
        )
    return rows


def ablate_poles(poles) -> dict:
    """Return the interventions that zero every pole of ``poles``, by layer."""
    by_layer: dict[int, list] = {}
    for layer, neuron, pole in poles:
        by_layer.setdefault(layer, []).append(ablate_pole(neuron, pole))

    def chain(changes):
        def intervene(act: torch.Tensor) -> torch.Tensor:
            for change in changes:
                act = change(act)
            return act

        return intervene

    return {layer: chain(changes) for layer, changes in by_layer.items()}


def control_sets(runner) -> list[tuple]:
    """Return ``CONTROLS`` sets of random positive poles, each in the layers of the round set.

    Each set has one pole in each layer that ``ROUND_DEPTH_POLES`` has a pole in, and as
    many as it has there. No control pole is one of the round set's neurons.
    """
    picker = random.Random(ABLATE_SEED)
    taken = {(layer, neuron) for layer, neuron, _ in ROUND_DEPTH_POLES}
    out = []
    for _ in range(CONTROLS):
        chosen = []
        for layer, _, _ in ROUND_DEPTH_POLES:
            free = [n for n in range(runner.mlp_width(layer)) if (layer, n) not in taken]
            free = [n for n in free if (layer, n, "positive") not in chosen]
            chosen.append((layer, picker.choice(free), "positive"))
        out.append(tuple(chosen))
    return out


def next_probs(runner, text: str, interventions=None) -> torch.Tensor:
    """Return the next token's probabilities, with ``interventions`` (layer to change)."""
    if not interventions:
        logits = runner.run_with_cache([text], [ABLATE_LAYER])[0]
    else:
        logits = runner.run_with_interventions([text], interventions)
    return logits[0, -1].softmax(-1)


@cache
def closer_counts(runner) -> torch.Tensor:
    """Return each token's count of leading closing round brackets, after any spaces."""
    tok = runner.tokenizer
    words = (tok.decode([i]).lstrip() for i in range(len(tok)))
    return torch.tensor([len(word) - len(word.lstrip(")")) for word in words])


def closing_measures(probs: torch.Tensor, counts: torch.Tensor, depth: int) -> dict:
    """Return three measures of how the next token closes ``depth`` open round brackets.

    ``any`` is the probability of a token that starts with ``)``. ``exact`` is the
    probability of a token that closes all ``depth`` brackets and no more. ``count`` is
    the mean number of brackets that a closing token closes, weighted by probability.
    """
    closing = counts > 0
    total = float(probs[closing].sum())
    return {
        "any": total,
        "exact": float(probs[counts == depth].sum()),
        "count": float((probs * counts).sum()) / max(total, 1e-9),
    }


def joint_ablation_rows(runner) -> list[dict]:
    """Return how the next token closes brackets with each round-depth pole ablated.

    Each pole is ablated alone, all five together, and the four that are not L2 N676+.
    The controls ablate random sets of poles in the same layers, as many as together.
    Each measure is a mean over ``TARGETS`` (see :func:`closing_measures`).
    """
    sets = {
        f"L{layer} N{neuron}+": ((layer, neuron, pole),)
        for layer, neuron, pole in ROUND_DEPTH_POLES
    }
    sets["all five"] = ROUND_DEPTH_POLES
    sets["all but L2 N676+"] = ROUND_DEPTH_POLES[1:]
    controls = control_sets(runner)
    counts = closer_counts(runner)
    rows = []
    for depth in DEPTHS:
        texts = [f"out = {calls(depth, k)}first,{t}" for k, t in enumerate(TARGETS)]

        def measure(poles, depth=depth, texts=texts) -> dict:
            changes = ablate_poles(poles) if poles else None
            found = [closing_measures(next_probs(runner, t, changes), counts, depth) for t in texts]
            return {k: sum(f[k] for f in found) / len(found) for k in found[0]}

        row = {"kind": "joint_ablation", "depth": depth, "clean": measure(())}
        for name, poles in sets.items():
            row[name] = measure(poles)
        found = [measure(c) for c in controls]
        row["controls_mean"] = {k: sum(f[k] for f in found) / len(found) for k in found[0]}
        row["controls_range"] = {
            k: [min(f[k] for f in found), max(f[k] for f in found)] for k in found[0]
        }
        rows.append(_rounded(row))
    return rows


def _rounded(value):
    """Round every float in ``value`` to 4 places, through dicts and lists."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def calls(depth: int, offset: int) -> str:
    """Return ``depth`` nested open calls, with names that start at ``CALLEES[offset]``."""
    return "".join(f"{CALLEES[(offset + i) % len(CALLEES)]}(" for i in range(depth))


def contexts(depth: int, offset: int) -> dict[str, str]:
    """Return the text before ``first,`` per condition, each with ``depth`` calls in it.

    In ``open``, the calls are still open. In ``closed``, the same calls have closed. In
    ``comment`` and ``string``, the open calls are in a comment or a string, so they are
    not brackets to the parser.
    """
    opened = calls(depth, offset)
    closed = opened + "x" + ")" * depth
    return {
        "open": f"out = {opened}",
        "closed": f"out = {closed}\nfirst = ",
        "comment": f"# {opened}\nfirst = ",
        "string": f'out = "{opened}',
    }


def read_token(runner, text: str, candidates) -> list[float]:
    """Return each candidate's strength at the last token of ``text``."""
    layers = sorted({layer for layer, _, _ in candidates})
    _, cache = runner.run_with_cache([text], layers)
    return [
        float(pole_strength(cache[layer][0, -1, neuron], pole))
        for layer, neuron, pole in candidates
    ]


def probe_rows(runner) -> list[dict]:
    """Return each candidate's mean strength per condition and depth, with its spread."""
    cells: dict[tuple, list[float]] = {}
    for depth in DEPTHS:
        for offset, target in enumerate(TARGETS):
            every = contexts(depth, offset) | extra_contexts(depth, offset)
            for condition, before in every.items():
                text = f"{before}first,{target}"
                values = read_token(runner, text, CANDIDATES)
                for candidate, value in zip(CANDIDATES, values, strict=True):
                    cells.setdefault((candidate, condition, depth), []).append(value)
    rows = []
    for ((layer, neuron, pole), condition, depth), values in cells.items():
        spread = torch.tensor(values)
        rows.append(
            {
                "kind": "probe",
                "layer": layer,
                "neuron": neuron,
                "pole": pole,
                "condition": condition,
                "depth": depth,
                "mean": round(float(spread.mean()), 4),
                "sem": round(float(spread.std() / len(values) ** 0.5), 4),
                "items": len(values),
            }
        )
    return rows


def trace_row(runner) -> dict:
    """Return one written snippet's depth and strengths, token by token."""
    tokens = runner.tokenizer(TRACE, return_offsets_mapping=True)
    nodes, constructs = parse_labels(TRACE, "Python")
    brackets, indents = bracket_depths(TRACE, nodes), indent_depths(TRACE)
    layers = sorted({layer for layer, _, _ in CANDIDATES})
    _, cache = runner.run_with_cache([TRACE], layers)
    rows = []
    for i, (start, end) in enumerate(tokens["offset_mapping"]):
        at = max(start, end - 1)
        rows.append(
            {
                "token": TRACE[start:end],
                "node": nodes[at],
                "construct": constructs[at],
                "bracket_depth": brackets[at],
                "indent_depth": indents[at],
                "strength": [
                    round(float(pole_strength(cache[layer][0, i, neuron], pole)), 4)
                    for layer, neuron, pole in CANDIDATES
                ],
            }
        )
    return {"kind": "trace", "candidates": [list(c) for c in CANDIDATES], "tokens": rows}


@click.command()
@click.option("--results", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cpu")
def main(results, device):
    """Replace the probe and trace rows of ``results``."""
    runner = _load(MODEL, resolve_device(device), torch.float32, None, None)
    with torch.no_grad():
        rows = [
            *probe_rows(runner),
            trace_row(runner),
            *(row for pole in ROUND_DEPTH_POLES for row in distance_rows(runner, pole)),
            *ablation_rows(runner),
            *joint_ablation_rows(runner),
        ]
    lines = results.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if json.loads(line)["kind"] not in WRITES]
    text = "\n".join([*kept, *map(json.dumps, rows)]) + "\n"
    results.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
