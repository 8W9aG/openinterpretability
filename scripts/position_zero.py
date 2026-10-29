"""Every neuron pole's first-token floor and active share, layer by layer.

The numbers behind protocol 0.8 (docs/analyses/position-zero.md). One scan of each
layer, read as an audit reads it (corpus sample, windows), and one pass over the
vocabulary at the first position. For each pole:

* *scale*: its maximum over every counted token, and over counted tokens after each
  piece's first;
* *first-token floor*: active (≥ 0.25 × the second scale) at the first position on at
  least half the vocabulary, as ``openinterp.harness.scan.settle`` judges it;
* *active share*: the share of the tokens it keeps (all, or all but first tokens with a
  floor) at or above 0.25 × its scale; default-on at half or more.

Writes one JSON line per layer and pole, with counts over the layer's neurons.

    uv run python scripts/position_zero.py --model openai-community/gpt2 \\
        --window-tokens 128 --out docs/analyses/position-zero.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from openinterp.cli import _load
from openinterp.cli_evidence import read_corpus
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS, sample_tokens
from openinterp.harness.scan import BATCH_TOKENS, batches, cut_pieces

POLES = ("positive", "negative")


def strengths(acts: torch.Tensor, pole: str) -> torch.Tensor:
    return acts.clamp(min=0) if pole == "positive" else (-acts).clamp(min=0)


@torch.no_grad()
def first_position(runner, layer: int, batch_size: int = 8192) -> torch.Tensor:
    """``[vocab, width]``: the layer's activations at the first position, per token."""
    tokenizer = runner.tokenizer
    special = set(tokenizer.all_special_ids or [])
    vocab = [i for i in range(len(tokenizer)) if i not in special]
    rows = []
    for start in range(0, len(vocab), batch_size):
        ids = torch.tensor(vocab[start : start + batch_size], device=runner.device).unsqueeze(1)
        _, cache = runner.run_with_cache({"input_ids": ids}, [layer])
        rows.append(cache[layer][:, 0].float().cpu())
    return torch.cat(rows)


def layer_rows(runner, pieces, layer: int) -> list[dict]:
    """Two passes over ``pieces`` (each pole's maxima, then its shares above its line)
    and one over the vocabulary."""

    def counted():
        for batch in batches(runner, pieces, BATCH_TOKENS):
            _, cache = runner.run_with_cache(batch.encoded, [layer])
            acts = cache[layer].float().cpu()[batch.counted]
            yield acts, batch.first[batch.counted]

    width = runner.mlp_width(layer)
    top = {(p, w): torch.zeros(width) for p in POLES for w in ("all", "rest")}
    for acts, first in counted():
        for pole in POLES:
            s = strengths(acts, pole)
            top[pole, "all"] = torch.maximum(top[pole, "all"], s.max(0).values)
            if (~first).any():
                top[pole, "rest"] = torch.maximum(top[pole, "rest"], s[~first].max(0).values)

    vocab = first_position(runner, layer)
    floor = {
        p: (strengths(vocab, p) >= 0.25 * top[p, "rest"]).float().mean(0) >= 0.5 for p in POLES
    }
    scale = {p: torch.where(floor[p], top[p, "rest"], top[p, "all"]) for p in POLES}
    above = {p: torch.zeros(width) for p in POLES}
    kept = {p: torch.zeros(width) for p in POLES}
    for acts, first in counted():
        for pole in POLES:
            hot = strengths(acts, pole) >= 0.25 * scale[pole]
            keep = ~(first.unsqueeze(1) & floor[pole])  # [tokens, width]
            above[pole] += (hot & keep).sum(0)
            kept[pole] += keep.sum(0)

    rows = []
    for pole in POLES:
        share = above[pole] / kept[pole].clamp(min=1)
        default_on = share >= 0.5
        rows.append(
            {
                "layer": layer,
                "pole": pole,
                "neurons": width,
                "first_token_floor": int(floor[pole].sum()),
                "floor_not_default_on": int((floor[pole] & ~default_on).sum()),
                "scale_set_at_first_token": int((top[pole, "rest"] < top[pole, "all"]).sum()),
                "default_on": int(default_on.sum()),
                "active_share_hist": torch.histc(share, bins=10, min=0, max=1).int().tolist(),
            }
        )
    return rows


@click.command()
@click.option("--model", "model_name", required=True)
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"))
@click.option("--window-tokens", type=int)
@click.option("--max-tokens", type=int, default=DEFAULT_SAMPLE_TOKENS, show_default=True)
@click.option("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED, show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(model_name, corpus, window_tokens, max_tokens, sample_seed, device, out):
    runner = _load(model_name, device, "float32", None, None)
    texts, *_ = read_corpus(corpus)
    texts, sample = sample_tokens(runner.tokenizer, texts, max_tokens, sample_seed)
    pieces = list(cut_pieces(runner.tokenizer, texts, window_tokens, None))
    with out.open("a") as f:
        for layer in range(runner.n_layers):
            for row in layer_rows(runner, pieces, layer):
                row = {"model": model_name, "corpus_sample": sample.sha256[:12], **row}
                f.write(json.dumps(row) + "\n")
                f.flush()
            click.echo(f"layer {layer} done", err=True)


if __name__ == "__main__":
    main()
