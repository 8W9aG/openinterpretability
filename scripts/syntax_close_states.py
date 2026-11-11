"""Find why GPT-2 tells open brackets from closed ones less well at depths 4 and 5.

The numbers behind *Why the push survives closing* in
papers/syntax-positive-controls/README.md. Each item is a written call with ``depth``
round brackets open, as in ``syntax_followups.py``. The model's most probable closers
follow it. In the ``open`` state, the first closer leaves brackets open. In the ``closed``
state, the closers have closed every bracket.

- ``close_ways``: close every bracket with the model's closers, with one ``)`` per token,
  with ``))`` tokens, or with the fewest tokens. Read the log-odds of one more closer.
- ``close_states``: in each state, read three things at the last token. The first is the
  attention of L3H5 to open brackets and to closers. The second is the push that L3H5
  carries: the drop in log-odds when L3H5 takes its output from the run with L2 N676+
  ablated. The third is the strength of L8 N689+.

    P=papers/syntax-positive-controls
    uv run python scripts/syntax_close_states.py --results $P/results.followups.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from syntax_close_paths import (
    MODEL,
    READER,
    chain_items,
    log_odds,
    patch,
    run_cached,
)
from syntax_followups import closer_chain
from syntax_probes import TARGETS, _rounded, calls

from openinterp.cli import _load, resolve_device
from openinterp.harness.hooks import pole_strength

STATE_DEPTHS = (2, 3, 4, 5)
WRITES = ("close_ways", "close_states")
L8_POLE = (8, 689, "positive")


def closer_ids(runner, count: int) -> int:
    """Return the id of the one token that is ``count`` closing round brackets."""
    ids = runner.tokenizer.encode(")" * count)
    if len(ids) != 1:
        raise ValueError(f"{')' * count!r} is not one token")
    return ids[0]


def ways_to_close(runner, depth: int, chain: list[int]) -> dict[str, list[int]]:
    """Return the closer tokens of each way to close ``depth`` brackets."""
    pairs = [closer_ids(runner, 2)] * (depth // 2) + [closer_ids(runner, 1)] * (depth % 2)
    fewest = [closer_ids(runner, 4)] * (depth // 4)
    fewest += [closer_ids(runner, depth % 4)] if depth % 4 else []
    return {
        "model": chain,
        "singles": [closer_ids(runner, 1)] * depth,
        "pairs": pairs,
        "fewest": fewest,
    }


def way_rows(runner) -> list[dict]:
    """Return the mean log-odds of one more closer after each way to close every bracket.

    The ``model`` way uses the model's own closers. It leaves out items where they close
    too many brackets.
    """
    rows = []
    for depth in STATE_DEPTHS:
        found: dict[str, list[float]] = {}
        for k, target in enumerate(TARGETS):
            base = runner.encode([f"out = {calls(depth, k)}first,{target}"])["input_ids"][0]
            chain = closer_chain(runner, base, depth)
            done = chain["steps"][-1]["open"] == 0 and not chain["closed_too_many"]
            for name, extra in ways_to_close(runner, depth, chain["tokens"]).items():
                if name == "model" and not done:
                    continue
                ids = torch.cat([base, torch.tensor(extra, device=base.device)])
                found.setdefault(name, []).append(log_odds(runner, ids))
        row = {"kind": "close_ways", "depth": depth}
        row |= {name: sum(v) / len(v) for name, v in found.items()}
        rows.append(_rounded(row | {"items": {name: len(v) for name, v in found.items()}}))
    return rows


def reader_attention(runner, ids: torch.Tensor) -> torch.Tensor:
    """Return the attention of L3H5 from the last token to each token of ``ids``."""
    _, layer, head = READER
    block = runner.model.transformer.h[layer]
    store: dict = {}
    hook = block.ln_1.register_forward_hook(lambda _m, _a, out: store.__setitem__("x", out))
    try:
        runner.model(ids.unsqueeze(0).to(runner.device))
    finally:
        hook.remove()
    width = runner.head_dim(layer)
    query, key, _ = block.attn.c_attn(store["x"])[0].split(block.attn.embed_dim, dim=-1)
    cut = slice(head * width, (head + 1) * width)
    scores = query[-1, cut] @ key[:, cut].T / width**0.5
    return scores.softmax(-1).cpu()


def state_row(runner, depth: int, state: str, items: list[torch.Tensor]) -> dict:
    """Return the attention, carried push and L8 N689+ strength of one state, as means."""
    width = runner.head_dim(0)
    _, layer, head = READER
    to_open, to_close, carried, l8 = 0.0, 0.0, 0.0, 0.0
    for ids in items:
        words = [runner.tokenizer.decode([int(i)]) for i in ids]
        attention = reader_attention(runner, ids)
        to_open += sum(float(a) for a, w in zip(attention, words, strict=True) if "(" in w)
        to_close += sum(
            float(a) for a, w in zip(attention, words, strict=True) if w.lstrip().startswith(")")
        )
        ablated = run_cached(runner, ids, True)
        noised = log_odds(runner, ids, **patch(ablated, "head", layer, head, width))
        carried += log_odds(runner, ids) - noised
        _, cache = runner.run_with_cache(ids.unsqueeze(0), [L8_POLE[0]])
        l8 += float(pole_strength(cache[L8_POLE[0]][0, -1, L8_POLE[1]], L8_POLE[2]))
    n = len(items)
    row = {"kind": "close_states", "depth": depth, "state": state, "items": n}
    row |= {"attention_open": to_open / n, "attention_close": to_close / n}
    return _rounded(row | {"carried_push": carried / n, "l8_n689": l8 / n})


@click.command()
@click.option("--results", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cpu")
def main(results, device):
    """Replace this script's rows of ``results``, and keep the others."""
    runner = _load(MODEL, resolve_device(device), torch.float32, None, None)
    with torch.no_grad():
        rows = way_rows(runner)
        for depth in STATE_DEPTHS:
            for state, closed in (("open", False), ("closed", True)):
                items = chain_items(runner, depth, closed=closed)
                if items:
                    rows.append(state_row(runner, depth, state, items))
    lines = results.read_text(encoding="utf-8").splitlines() if results.exists() else []
    kept = [line for line in lines if json.loads(line)["kind"] not in WRITES]
    results.write_text("".join(f"{line}\n" for line in [*kept, *map(json.dumps, rows)]))


if __name__ == "__main__":
    main()
