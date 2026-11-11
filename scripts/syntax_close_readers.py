"""Find the heads that read closing brackets and cancel GPT-2's push to close more.

The numbers behind *What reads the closers* in papers/syntax-positive-controls/README.md.
Each item is a written call with ``depth`` round brackets open. The model's most probable
closers follow it, until no bracket is open (as in ``syntax_close_paths.py``). There the
model should stop closing. A head reads a token through that token's value vector. So
zeroing one head's values at the closer tokens stops that head from reading them.

- ``close_readers``: for each head of layers 0 to 8, zero its values at the closers. Read
  the log-odds of one more closer, and the strength of L8 N689+ at the last token. The
  ``earlier`` variant leaves the last token's own value. The ``all`` variant does not.
- ``close_reader_attention``: for the heads with the largest effect, the attention from
  the last token to the closers and to the open brackets, at each depth.

    P=papers/syntax-positive-controls
    uv run python scripts/syntax_close_readers.py --results $P/results.followups.jsonl
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import click
import torch
from syntax_close_paths import MODEL, chain_items, log_odds
from syntax_probes import TARGETS, _rounded, calls

from openinterp.cli import _load, resolve_device
from openinterp.harness.hooks import pole_strength

DEPTHS = (2, 3, 4, 5)
LAST_LAYER = 8  # L8 N689+ reads the heads of this layer and the ones before it
L8_POLE = (8, 689, "positive")
TOP_HEADS = 8
WRITES = ("close_readers", "close_reader_attention", "close_counts", "close_stretch")
READERS = ((0, 10), (1, 3), (2, 10), (3, 10))  # the heads with the largest effect
COUNT_DEPTHS = (3, 4, 5)
# Mean strength of each depth pole under 1 to 5 open brackets, on the written probes of
# ``syntax_probes.py`` (results.jsonl, condition ``open``). The steps shrink with depth.
DEPTH_CODE = {
    (2, 676): (4.19, 6.75, 8.63, 9.99, 11.02),
    (2, 1195): (3.43, 5.88, 7.68, 8.97, 9.94),
}


def closer_positions(runner, ids: torch.Tensor, earlier: bool) -> list[int]:
    """Return the positions of the closing tokens after the open call, as a list.

    With ``earlier``, leave out the last token.
    """
    words = [runner.tokenizer.decode([int(i)]) for i in ids]
    found = [k for k, w in enumerate(words) if w.lstrip().startswith(")")]
    return [k for k in found if k < len(words) - 1] if earlier else found


@contextmanager
def values_zeroed(runner, heads: list[tuple[int, int]], positions: list[int]):
    """Zero the value vectors of ``heads`` (layer, head) at ``positions``, in the block."""
    width = runner.head_dim(0)
    model_width = runner.model.config.n_embd
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    handles = []

    def hook_for(chosen: list[int]):
        def hook(_module, _args, out: torch.Tensor) -> torch.Tensor:
            out = out.clone()
            for head in chosen:
                start = 2 * model_width + head * width
                out[0, positions, start : start + width] = 0.0
            return out

        return hook

    try:
        for layer, chosen in by_layer.items():
            attn = runner.model.transformer.h[layer].attn.c_attn
            handles.append(attn.register_forward_hook(hook_for(chosen)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def measure(runner, ids: torch.Tensor) -> tuple[float, float]:
    """Return the log-odds of a closer, and the strength of L8 N689+, at the last token."""
    odds = log_odds(runner, ids)
    _, cache = runner.run_with_cache(ids.unsqueeze(0), [L8_POLE[0]])
    return odds, float(pole_strength(cache[L8_POLE[0]][0, -1, L8_POLE[1]], L8_POLE[2]))


def mean_measure(runner, items, heads, earlier: bool) -> tuple[float, float]:
    """Return :func:`measure` as means over ``items``, with ``heads`` zeroed at closers."""
    total_odds, total_pole = 0.0, 0.0
    for ids in items:
        with values_zeroed(runner, heads, closer_positions(runner, ids, earlier)):
            odds, pole = measure(runner, ids)
        total_odds, total_pole = total_odds + odds, total_pole + pole
    return total_odds / len(items), total_pole / len(items)


def reader_rows(runner, items_by_depth: dict) -> list[dict]:
    """Return each head's effect on the closed state when it can't read the closers."""
    heads = [(layer, h) for layer in range(LAST_LAYER + 1) for h in range(runner.n_heads)]
    rows = []
    for depth, items in items_by_depth.items():
        clean = mean_measure(runner, items, [], earlier=False)
        for variant, earlier in (("earlier", True), ("all", False)):
            found = []
            for layer, head in heads:
                odds, pole = mean_measure(runner, items, [(layer, head)], earlier)
                found.append({"head": f"L{layer}H{head}", "odds": odds, "l8_n689": pole})
            found.sort(key=lambda f: -f["odds"])
            every = mean_measure(runner, items, heads, earlier)
            row = {"kind": "close_readers", "depth": depth, "variant": variant}
            row |= {"items": len(items), "clean_odds": clean[0], "clean_l8_n689": clean[1]}
            row |= {"every_head_odds": every[0], "every_head_l8_n689": every[1]}
            rows.append(_rounded(row | {"heads": found}))
    return rows


def head_attention(runner, ids: torch.Tensor, layer: int, head: int) -> torch.Tensor:
    """Return the attention of one head from the last token to each token of ``ids``."""
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
    return (query[-1, cut] @ key[:, cut].T / width**0.5).softmax(-1).cpu()


def attention_rows(runner, items_by_depth: dict, readers: list[str]) -> list[dict]:
    """Return each reader head's attention to closers and to open brackets, by depth."""
    rows = []
    for name in readers:
        layer, head = (int(x) for x in name[1:].split("H"))
        for depth, items in items_by_depth.items():
            to_close, to_open = 0.0, 0.0
            for ids in items:
                words = [runner.tokenizer.decode([int(i)]) for i in ids]
                attention = head_attention(runner, ids, layer, head)
                for a, w in zip(attention, words, strict=True):
                    to_close += float(a) * w.lstrip().startswith(")") / len(items)
                    to_open += float(a) * ("(" in w) / len(items)
            row = {"kind": "close_reader_attention", "head": name, "depth": depth}
            rows.append(_rounded(row | {"to_closers": to_close, "to_openers": to_open}))
    return rows


def count_items(runner, depth: int, first: int) -> list[torch.Tensor]:
    """Return the items at ``depth`` that end with a closer of ``first`` brackets, then ``)``."""
    closer = {k: runner.tokenizer.encode(")" * k)[0] for k in (1, 2, 3, 4)}
    out = []
    for k, target in enumerate(TARGETS):
        base = runner.encode([f"out = {calls(depth, k)}first,{target}"])["input_ids"][0]
        extra = torch.tensor([closer[first], closer[1]], device=base.device)
        out.append(torch.cat([base, extra]))
    return out


def count_rows(runner) -> list[dict]:
    """Return the measures for each count of brackets left open, with the readers on or off."""
    heads = [(layer, h) for layer in range(LAST_LAYER + 1) for h in range(runner.n_heads)]
    rows = []
    for depth in COUNT_DEPTHS:
        for first in range(1, min(depth - 1, 4) + 1):
            items = count_items(runner, depth, first)
            clean = mean_measure(runner, items, [], earlier=True)
            off = mean_measure(runner, items, list(READERS), earlier=True)
            every = mean_measure(runner, items, heads, earlier=True)
            row = {"kind": "close_counts", "depth": depth, "first": first}
            row |= {"left_open": depth - first - 1, "odds": clean[0], "l8_n689": clean[1]}
            row |= {"readers_off_odds": off[0], "readers_off_l8_n689": off[1]}
            rows.append(_rounded(row | {"every_head_off_odds": every[0]}))
    return rows


def stretch(neuron: int, code: tuple[float, ...], uniform: float | None):
    """Return an intervention that remaps one pole's strength at every token.

    With ``uniform``, it multiplies the strength by that factor. Otherwise it maps the
    strength at depth d (``code``) to an even code, with the step of the first bracket.
    It interpolates between the points, and it extends the last segment.
    """
    step = code[1] - code[0]
    xs = torch.tensor([0.0, *code])
    ys = torch.tensor([0.0, *(code[0] + step * d for d in range(len(code)))])

    def remap(x: torch.Tensor) -> torch.Tensor:
        if uniform is not None:
            return x * uniform
        k = torch.bucketize(x.clamp(min=0), xs[1:-1].to(x.device))
        x0, x1 = xs.to(x.device)[k], xs.to(x.device)[k + 1]
        y0, y1 = ys.to(x.device)[k], ys.to(x.device)[k + 1]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    def intervene(act: torch.Tensor) -> torch.Tensor:
        act = act.clone()
        part = act[..., neuron].clamp(min=0)
        act[..., neuron] = act[..., neuron] - part + remap(part)
        return act

    return intervene


def stretch_rows(runner) -> list[dict]:
    """Return the measures with the L2 depth code made even, or scaled up as a control."""
    top = {key: (code[0] + (code[1] - code[0]) * 4) / code[-1] for key, code in DEPTH_CODE.items()}
    rows = []
    for name, uniform in (("even", False), ("uniform", True)):
        changes = [
            stretch(n, code, top[(lay, n)] if uniform else None)
            for (lay, n), code in DEPTH_CODE.items()
        ]

        def both(act, changes=changes):
            for change in changes:
                act = change(act)
            return act

        for depth in COUNT_DEPTHS:
            for first in range(1, min(depth - 1, 4) + 1):
                items = count_items(runner, depth, first)
                with runner.hooks(interventions={2: both}):
                    odds, pole = mean_measure(runner, items, [], earlier=True)
                row = {"kind": "close_stretch", "code": name, "depth": depth, "first": first}
                rows.append(
                    _rounded(row | {"left_open": depth - first - 1, "odds": odds, "l8_n689": pole})
                )
    return rows


@click.command()
@click.option("--results", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cpu")
@click.option("--part", type=click.Choice(["all", "counts"]), default="all")
def main(results, device, part):
    """Replace this script's rows of ``results``, and keep the others.

    With ``--part counts``, write only the ``close_counts`` and ``close_stretch`` rows.
    """
    runner = _load(MODEL, resolve_device(device), torch.float32, None, None)
    writes = WRITES if part == "all" else ("close_counts", "close_stretch")
    with torch.no_grad():
        if part == "counts":
            rows = [*count_rows(runner), *stretch_rows(runner)]
        else:
            rows = all_rows(runner)
    lines = results.read_text(encoding="utf-8").splitlines() if results.exists() else []
    kept = [line for line in lines if json.loads(line)["kind"] not in writes]
    results.write_text("".join(f"{line}\n" for line in [*kept, *map(json.dumps, rows)]))


def all_rows(runner) -> list[dict]:
    """Return every row of this script."""
    items = {d: chain_items(runner, d, closed=True) for d in DEPTHS}
    rows = reader_rows(runner, items)
    best: dict[str, float] = {}
    for row in rows:
        if row["variant"] == "earlier":
            for f in row["heads"]:
                gain = f["odds"] - row["clean_odds"]
                best[f["head"]] = max(best.get(f["head"], -99.0), gain)
    readers = sorted(best, key=lambda h: -best[h])[:TOP_HEADS]
    rows += attention_rows(runner, items, readers)
    return [*rows, *count_rows(runner), *stretch_rows(runner)]


if __name__ == "__main__":
    main()
