"""``openinterp activations`` (a scan's per-token data) and ``openinterp scan-cache``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click

from openinterp.cli import cli, with_model_options
from openinterp.cli_evidence import open_scan_cache, read_corpus, scan_cache_option
from openinterp.core.schemas import ACTIVATION_THRESHOLD
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS

if TYPE_CHECKING:
    import torch

COLUMNS = ("doc", "window_start", "index", "token_id", "token", "strength", "context")


def _tsv(value) -> str:
    text = str(value)
    for raw, escaped in (("\\", "\\\\"), ("\t", "\\t"), ("\n", "\\n"), ("\r", "\\r")):
        text = text.replace(raw, escaped)
    return text


@cli.command()
@click.option("--model", "model_name", required=True, help="Hugging Face repo.")
@click.option("--layer", type=int, required=True)
@click.option("--neuron", type=int, required=True)
@click.option("--pole", type=click.Choice(["positive", "negative"]), default="positive")
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    help="The audit's corpus (same sample and windows options), so its cached scan is reused.",
)
@click.option("--max-tokens", default=DEFAULT_SAMPLE_TOKENS, show_default=True, type=int)
@click.option("--sample-seed", default=DEFAULT_SAMPLE_SEED, show_default=True, type=int)
@click.option("--window-tokens", type=click.IntRange(2))
@click.option("--stride", type=click.IntRange(1))
@click.option(
    "--top",
    default=50,
    show_default=True,
    type=click.IntRange(0),
    help="The strongest N tokens, strongest first. 0: every matching token in corpus order.",
)
@click.option("--token", "tokens", multiple=True, help="Only this token (exact text; repeat).")
@click.option("--min-strength", type=float, help="Only tokens at least this strong.")
@click.option("--context", default=12, show_default=True, help="Tokens of left context shown.")
@click.option("--format", "fmt", type=click.Choice(["tsv", "jsonl"]), default="tsv")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), help="Default: stdout.")
@scan_cache_option
@with_model_options
def activations(
    model_name,
    layer,
    neuron,
    pole,
    corpus,
    max_tokens,
    sample_seed,
    window_tokens,
    stride,
    top,
    tokens,
    min_strength,
    context,
    fmt,
    out,
    use_scan_cache,
    device,
    dtype,
):
    """Per-token pole strengths of one neuron over the scan corpus.

    Reads the layer's cached scan when there is one (an audit of any neuron in the
    layer, with the same corpus options, stores it), else scans and caches the layer.
    Each token counts once, as in the audit's histogram. Use this rather than writing
    a scan loop: e.g. every occurrence of a token with --token ' the' --top 0.
    """
    from openinterp.cli import _load_runner
    from openinterp.harness.scan import cut_pieces

    if stride is not None and (window_tokens is None or stride > window_tokens):
        raise click.UsageError("--stride needs --window-tokens, and can't exceed it.")
    runner = _load_runner(model_name, device, dtype)
    tokenizer = runner.tokenizer
    texts, *_ = read_corpus(corpus)
    if max_tokens:
        from openinterp.corpus import sample_tokens

        texts, _ = sample_tokens(tokenizer, texts, max_tokens, sample_seed)
    pieces = list(cut_pieces(tokenizer, texts, window_tokens, stride))
    scan = _neuron_strengths(runner, pieces, layer, neuron, pole, open_scan_cache(use_scan_cache))
    rows = _select(scan, tokenizer, min_strength, tokens, top)
    windowed = window_tokens is not None
    lines = [_format(scan, pieces, row, tokenizer, context, windowed, fmt) for row in rows.tolist()]
    if fmt == "tsv":
        lines.insert(0, "\t".join(COLUMNS))
    text = "\n".join(lines) + "\n"
    if out is None:
        click.echo(text, nl=False)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        click.echo(f"Wrote {len(rows):,} tokens to {out}", err=True)


@dataclass
class _Strengths:
    """Every scanned token: its id, pole strength, whether it counts, piece and index."""

    ids: torch.Tensor
    values: torch.Tensor
    counted: torch.Tensor
    piece_of: list[int]
    index: list[int]


def _neuron_strengths(runner, pieces, layer: int, neuron: int, pole, cache) -> _Strengths:
    import torch

    from openinterp.harness.hooks import pole_strength
    from openinterp.harness.scan import scan_chunks

    ids, strengths, counted, piece_of, index = [], [], [], [], []
    next_piece = 0
    for chunk in scan_chunks(runner, pieces, layer, cache=cache):
        ids.append(chunk.ids)
        strengths.append(pole_strength(chunk.acts[:, neuron].float(), pole))
        counted.append(chunk.counted)
        for n in chunk.lengths:  # chunks hold consecutive pieces
            piece_of.extend([next_piece] * n)
            index.extend(range(n))
            next_piece += 1
    if cache is not None and cache.last is not None:
        verb = "Replayed" if cache.replayed else "Cached"
        click.echo(f"{verb} layer {layer}'s scan: {cache.last}", err=True)
    return _Strengths(torch.cat(ids), torch.cat(strengths), torch.cat(counted), piece_of, index)


def _select(scan: _Strengths, tokenizer, min_strength, tokens, top: int) -> torch.Tensor:
    """The rows to print, after reporting the scan's size, peak and active count."""
    import torch

    ids_t, values = scan.ids, scan.values
    keep = scan.counted.clone()
    peak = float(values[keep].max()) if keep.any() else 0.0
    active = int((values[keep] >= ACTIVATION_THRESHOLD * peak).sum()) if peak > 0 else 0
    click.echo(
        f"{int(keep.sum()):,} tokens; max {peak:.4g}; {active:,} at or above "
        f"{ACTIVATION_THRESHOLD:g} × max (the protocol's active threshold)",
        err=True,
    )
    if min_strength is not None:
        keep &= values >= min_strength
    if tokens:
        seen = ids_t[keep].unique().tolist()
        wanted = [i for i in seen if tokenizer.decode([i]) in set(tokens)]
        keep &= torch.isin(ids_t, torch.tensor(wanted, dtype=ids_t.dtype))
    rows = keep.nonzero().flatten()
    if top:
        rows = rows[values[rows].argsort(descending=True, stable=True)[:top]]
    return rows


def _format(scan: _Strengths, pieces, row: int, tokenizer, context, windowed, fmt) -> str:
    ids_t, index = scan.ids, scan.index
    piece = pieces[scan.piece_of[row]]
    first = row - index[row]
    left = ids_t[max(first, row - context) : row].tolist() if context else []
    record = {
        "doc": piece.doc,
        "window_start": piece.start if windowed else 0,
        "index": index[row],
        "token_id": int(ids_t[row]),
        "token": tokenizer.decode([int(ids_t[row])]),
        "strength": round(float(scan.values[row]), 5),
        "context": tokenizer.decode(left) if left else "",
    }
    if fmt == "jsonl":
        return json.dumps(record, ensure_ascii=False)
    return "\t".join(_tsv(record[c]) for c in COLUMNS)


@cli.command("scan-cache")
@click.option("--clear", is_flag=True, help="Delete every cached layer scan.")
def scan_cache(clear):
    """List (or clear) the cached layer scans that audits and `activations` replay.

    \b
    Location: $OPENINTERP_SCAN_CACHE (\"off\" disables it), else
    $XDG_CACHE_HOME/openinterp/scans or ~/.cache/openinterp/scans.
    Size cap: $OPENINTERP_SCAN_CACHE_GB (default 40); least recently used go first.
    """
    from openinterp.harness.layer_cache import LayerCache

    cache = LayerCache.default()
    if cache is None:
        click.echo("The scan cache is off ($OPENINTERP_SCAN_CACHE).")
        return
    if clear:
        click.echo(f"Deleted {cache.clear()} cached scans from {cache.root}")
        return
    entries = cache.entries()
    total = sum(e.size for e in entries)
    click.echo(
        f"{cache.root}: {len(entries)} scans, {total / 1e9:.1f} of {cache.max_bytes / 1e9:.0f} GB"
    )
    for entry in reversed(entries):  # most recently used first
        f = entry.meta["fields"]
        click.echo(
            f"  {f['model']} L{f['layer']}  {entry.meta['rows']:,} tokens  "
            f"{entry.size / 1e9:.2f} GB  {f['dtype']} on {f['device']}  "
            f"pieces {f['pieces_sha256'][:12]}  {entry.path.name}"
        )
