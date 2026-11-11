"""Split what tells L8 N689+ that every bracket is closed into parts, at each depth.

The numbers behind *Where the match weakens* in papers/syntax-positive-controls/README.md.
The items are those of ``close_counts`` in ``syntax_close_readers.py``. At ``depth``, a
first closer leaves one bracket open (``off_by_one``) or none (``matched``), and ``)``
follows it. L8 N689+ is near zero when no bracket is open at depth 3, and not at depth 5.

The input of the neuron is a layer norm of the residual stream, then a weight vector.
Fix the norm's scale at its value in the run. Then the input is a sum over the parts
that write to the residual stream at the last token. The parts are the embeddings, each
head of layers 0 to 8 and each MLP of layers 0 to 7. So the gap between ``off_by_one`` and
``matched`` splits into one share for each part. ``check`` is the sum of the parts minus
the input that the model computes. It is near zero when the split is exact.

- ``close_gate``: the attention of L7H4 from the last token to the open brackets, in each
  state. Then the same items with that attention set to a fixed share, moved to the first
  token.
- ``close_packing``: every bracket closed, with the closers packed into tokens in
  several ways. The last token is ``)``, ``))``, `` )`` or ``);``. There are 64 items
  for each way: 8 callee orders times 8 targets.
- ``close_count_by_token``: one closer of 1 to 4 brackets, then ``)``, after calls of depth
  2 to 7. The depth where L8 N689+ falls shows how many brackets the model counts. With
  ``--corpus``, also count each closer token in that corpus.
- ``close_natural``: the chains of ``syntax_followups.py``, split by whether every step
  of the chain had a real chance of a closer. A forced step is one where P(closer) was
  under ``FORCED_BELOW`` and the chain still added the most probable closer.

    P=papers/syntax-positive-controls
    uv run python scripts/syntax_close_match.py --results $P/results.followups.jsonl
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import click
import torch
from syntax_close_paths import MODEL
from syntax_close_readers import count_items, head_attention, measure
from syntax_followups import closer_chain
from syntax_probes import CALLEES, TARGETS, _rounded, calls

from openinterp.cli import _load, resolve_device

LAYER, NEURON = 8, 689  # L8 N689+
DEPTHS = (3, 4, 5)
TOP_PARTS = 12
WRITES = ("close_match", "close_gate", "close_packing", "close_count_by_token", "close_natural")
FORCED_BELOW = 0.05
COUNT_DEPTHS = range(2, 8)
GATE = (7, 4)  # L7H4
GATE_SHARES = (0.17, 0.0)  # its share at depth 3 with no bracket open, and none
PACKINGS = {
    4: (
        (")", ")", ")", ")"),
        ("))", ")", ")"),
        ("))", "))"),
        (")))", ")"),
        (")))", ");"),
        ("))", "),", " )"),
        ("))", ")", " )"),
    ),
    5: (
        (")",) * 5,
        ("))", ")", ")", ")"),
        ("))", "))", ")"),
        (")))", ")", ")"),
        (")))", "))"),
        ("))))", ")"),
        (")", ")))", ");"),
    ),
    6: (
        (")",) * 6,
        ("))", ")", ")", ")", ")"),
        ("))", "))", "))"),
        (")))", ")", ")", ")"),
        ("))))", ")", ")"),
    ),
}  # each closes every bracket; the model's own chains use "),", " )" and ");"


def record_parts(model, store: dict) -> list:
    """Register the hooks that store each part's output in ``store``, and return them."""
    hooks = []
    for layer in range(LAYER + 1):
        block = model.h[layer]
        hooks.append(
            block.attn.c_proj.register_forward_pre_hook(
                lambda _m, args, layer=layer: store.__setitem__(("z", layer), args[0])
            )
        )
        if layer < LAYER:
            hooks.append(
                block.mlp.register_forward_hook(
                    lambda _m, _a, out, layer=layer: store.__setitem__(("mlp", layer), out)
                )
            )
    hooks.append(
        model.h[LAYER].ln_2.register_forward_pre_hook(
            lambda _m, args: store.__setitem__("resid", args[0])
        )
    )
    hooks.append(
        model.h[LAYER].mlp.c_fc.register_forward_hook(
            lambda _m, _a, out: store.__setitem__("pre", out)
        )
    )
    return hooks


def parts_at_last(runner, ids: torch.Tensor) -> tuple[dict[str, torch.Tensor], float, float]:
    """Return each part's write at the last token, the norm's scale and the true input.

    The scale is the standard deviation of the residual stream that L8's MLP norm reads.
    """
    model = runner.model.transformer
    store: dict = {}
    hooks = record_parts(model, store)
    try:
        runner.model(ids.unsqueeze(0).to(runner.device))
    finally:
        for hook in hooks:
            hook.remove()
    width = runner.head_dim(0)
    parts: dict[str, torch.Tensor] = {}
    position = ids.shape[0] - 1
    parts["embeddings"] = (model.wte.weight[ids[-1]] + model.wpe.weight[position]).detach()
    for layer in range(LAYER + 1):
        proj = model.h[layer].attn.c_proj
        z = store[("z", layer)][0, -1]
        for head in range(runner.n_heads):
            cut = slice(head * width, (head + 1) * width)
            parts[f"L{layer}H{head}"] = z[cut] @ proj.weight[cut]
        parts[f"L{layer} attention bias"] = proj.bias.detach()
        if layer < LAYER:
            parts[f"L{layer} MLP"] = store[("mlp", layer)][0, -1]
    resid = store["resid"][0, -1]
    scale = float(resid.var(unbiased=False).add(model.h[LAYER].ln_2.eps).sqrt())
    return parts, scale, float(store["pre"][0, -1, NEURON])


def contributions(runner, ids: torch.Tensor) -> tuple[dict[str, float], float]:
    """Return each part's share of the neuron's input, and the check of the split."""
    norm = runner.model.transformer.h[LAYER].ln_2
    weight = runner.model.transformer.h[LAYER].mlp.c_fc.weight[:, NEURON]
    bias = float(runner.model.transformer.h[LAYER].mlp.c_fc.bias[NEURON])
    parts, scale, true = parts_at_last(runner, ids)
    reader = norm.weight * weight / scale
    out = {name: float(((v - v.mean()) * reader).sum()) for name, v in parts.items()}
    constant = float((norm.bias * weight).sum()) + bias
    return out, sum(out.values()) + constant - true


def match_rows(runner) -> list[dict]:
    """Return the parts that give the gap between one bracket open and none, by depth."""
    rows = []
    for depth in DEPTHS:
        found = {}
        checks = []
        for state, first in (("off_by_one", depth - 2), ("matched", depth - 1)):
            if first > 4:
                break
            totals: dict[str, float] = {}
            items = count_items(runner, depth, first)
            for ids in items:
                shares, check = contributions(runner, ids)
                checks.append(abs(check))
                for name, value in shares.items():
                    totals[name] = totals.get(name, 0.0) + value / len(items)
            found[state] = totals
        if len(found) < 2:
            continue
        gap = {n: found["off_by_one"][n] - found["matched"][n] for n in found["matched"]}
        top = sorted(gap.items(), key=lambda kv: -abs(kv[1]))[:TOP_PARTS]
        row = {"kind": "close_match", "depth": depth, "gap_total": sum(gap.values())}
        row |= {"check_max": max(checks), "top": [{"part": n, "gap": g} for n, g in top]}
        rows.append(_rounded(row | {"all": gap}))
    return rows


def one_token(runner, word: str) -> int:
    """Return the id of ``word``, which must be one token."""
    ids = runner.tokenizer.encode(word)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is not one token")
    return ids[0]


def mean_pair(pairs: list[tuple[float, float]]) -> list[float]:
    """Return the mean of each half of ``pairs``: the log-odds, then L8 N689+."""
    return [sum(p[0] for p in pairs) / len(pairs), sum(p[1] for p in pairs) / len(pairs)]


def opener_positions(runner, ids: torch.Tensor) -> list[int]:
    """Return the positions of the tokens that hold an open round bracket."""
    words = [runner.tokenizer.decode([int(i)]) for i in ids]
    return [k for k, word in enumerate(words) if "(" in word]


@contextmanager
def gate_fixed(runner, ids: torch.Tensor, share: float):
    """Set the attention of L7H4 from the last token to the open brackets to ``share``.

    The rest of that attention moves to the first token. Only the last token changes.
    """
    layer, head = GATE
    attn = runner.model.transformer.h[layer].attn
    width, store = runner.head_dim(layer), {}
    openers = opener_positions(runner, ids)

    def grab(_module, _args, out):
        store["qkv"] = out

    def edit(_module, args):
        z = args[0].clone()
        query, key, value = store["qkv"][0].split(attn.embed_dim, dim=-1)
        cut = slice(head * width, (head + 1) * width)
        weights = (query[-1, cut] @ key[:, cut].T / width**0.5).softmax(-1)
        current = weights[openers].sum()
        weights[openers] = weights[openers] * (share / current)
        weights[0] = weights[0] + current - share
        z[0, -1, cut] = weights @ value[:, cut]
        return (z, *args[1:])

    hooks = [attn.c_attn.register_forward_hook(grab), attn.c_proj.register_forward_pre_hook(edit)]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def gate_rows(runner) -> list[dict]:
    """Return the gate's attention to open brackets, and the effect of fixing it."""
    rows = []
    for depth in DEPTHS:
        for state, first in (("off_by_one", depth - 2), ("matched", depth - 1)):
            items = count_items(runner, depth, first)
            share = sum(
                float(head_attention(runner, ids, *GATE)[opener_positions(runner, ids)].sum())
                for ids in items
            ) / len(items)
            row = {"kind": "close_gate", "depth": depth, "state": state, "opener_share": share}
            row |= {"model": mean_pair([measure(runner, i) for i in items])}
            for fixed in GATE_SHARES:
                found = []
                for ids in items:
                    with gate_fixed(runner, ids, fixed):
                        found.append(measure(runner, ids))
                row[f"share {fixed}"] = mean_pair(found)
            rows.append(_rounded(row))
    return rows


def packing_rows(runner) -> list[dict]:
    """Return the log-odds of one more closer and L8 N689+ for each packing of closers."""
    rows = []
    for depth, packings in PACKINGS.items():
        for packing in packings:
            found = []
            for offset in range(len(CALLEES)):
                calls = "".join(f"{CALLEES[(offset + i) % len(CALLEES)]}(" for i in range(depth))
                for target in TARGETS:
                    base = runner.encode([f"out = {calls}first,{target}"])["input_ids"][0]
                    extra = [one_token(runner, word) for word in packing]
                    extra = torch.tensor(extra, device=base.device)
                    found.append(measure(runner, torch.cat([base, extra])))
            odds, pole = mean_pair(found)
            row = {"kind": "close_packing", "depth": depth, "closers": list(packing)}
            rows.append(_rounded(row | {"items": len(found), "odds": odds, "l8_n689": pole}))
    return rows


def all_items(runner, depth: int, closers: list[int]) -> list[torch.Tensor]:
    """Return the 64 items at ``depth`` (8 callee orders times 8 targets), then ``closers``."""
    out = []
    for offset in range(len(CALLEES)):
        opened = "".join(f"{CALLEES[(offset + i) % len(CALLEES)]}(" for i in range(depth))
        for target in TARGETS:
            base = runner.encode([f"out = {opened}first,{target}"])["input_ids"][0]
            out.append(torch.cat([base, torch.tensor(closers, device=base.device)]))
    return out


def count_by_token_rows(runner, corpus: Path | None) -> list[dict]:
    """Return L8 N689+ for each closer token of 1 to 4 brackets, then ``)``, by depth."""
    ids = {k: one_token(runner, ")" * k) for k in (1, 2, 3, 4)}
    seen = {k: 0 for k in ids}
    if corpus is not None:
        for line in corpus.read_text(encoding="utf-8").splitlines():
            tokens = runner.tokenizer.encode(json.loads(line)["text"])
            for k, token in ids.items():
                seen[k] += tokens.count(token)
    rows = []
    for k, token in ids.items():
        by_depth = {}
        for depth in COUNT_DEPTHS:
            items = all_items(runner, depth, [token, ids[1]])
            by_depth[depth] = mean_pair([measure(runner, i) for i in items])
        row = {"kind": "close_count_by_token", "brackets": k, "matched_depth": k + 1}
        rows.append(
            _rounded(row | {"in_corpus": seen[k] if corpus else None, "by_depth": by_depth})
        )
    return rows


def natural_rows(runner) -> list[dict]:
    """Return the chance of one more closer after each chain, natural or forced."""
    rows = []
    for depth in (2, 3, 4, 5):
        for k, target in enumerate(TARGETS):
            base = runner.encode([f"out = {calls(depth, k)}first,{target}"])["input_ids"][0]
            chain = closer_chain(runner, base, depth)
            steps = chain["steps"]
            if steps[-1]["open"] != 0:
                continue  # the chain closed too many brackets
            lowest = min(step["p_closer"] for step in steps[:-1])
            words = [runner.tokenizer.decode([t]) for t in chain["tokens"]]
            row = {"kind": "close_natural", "depth": depth, "item": k, "closers": words}
            row |= {"forced": lowest < FORCED_BELOW, "lowest_step_p_closer": lowest}
            rows.append(_rounded(row | {"p_one_more": steps[-1]["p_closer"]}))
    return rows


@click.command()
@click.option("--results", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cpu")
@click.option(
    "--corpus",
    type=click.Path(path_type=Path),
    default=None,
    help="A corpus .jsonl in which to count each closer token.",
)
def main(results, device, corpus):
    """Replace this script's rows of ``results``, and keep the others."""
    runner = _load(MODEL, resolve_device(device), torch.float32, None, None)
    with torch.no_grad():
        rows = [*match_rows(runner), *gate_rows(runner), *packing_rows(runner)]
        rows += [*count_by_token_rows(runner, corpus), *natural_rows(runner)]
    lines = results.read_text(encoding="utf-8").splitlines() if results.exists() else []
    kept = [line for line in lines if json.loads(line)["kind"] not in WRITES]
    results.write_text("".join(f"{line}\n" for line in [*kept, *map(json.dumps, rows)]))


if __name__ == "__main__":
    main()
