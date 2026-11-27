"""A training sample from Pythia's own data, in GPT-2's tokens (#6935).

The 32x batch runs of ``scripts/train_resting.py`` read about 262 million tokens in 2,000
steps. Their old sample is ``tokens.pt`` (pile-10k, 15.3 million tokens). The runs read it
about 17 times, and Pythia reads its data less than once. This script builds a sample large
enough for one pass. Its source is ``EleutherAI/pile-deduped-pythia-preshuffled``, the
deduplicated Pile in Pythia's training order, as GPT-NeoX token ids.

It reads the start of the first shard by HTTP range, so it never downloads the 30 GB shard.
Pythia's data has no end-of-text token between documents, so the sample has none either.
It decodes the ids in chunks. It cuts out each held-out corpus document where its first
200 characters appear, and encodes the rest with GPT-2's tokenizer::

    uv run python scripts/pythia_tokens.py --tokens 300000000 --out tokens_280m.pt
    uv run python scripts/train_resting.py train --cache tokens_280m.pt ...
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

import click
import numpy as np

REPO = "EleutherAI/pile-deduped-pythia-preshuffled"
SHARD = "document-00000-of-00020.bin"
CHUNK_BYTES = 64 * 1024 * 1024  # bytes per range request
CHUNK_IDS = 1_000_000  # GPT-NeoX ids decoded at once
PREFIX = 200  # characters that name a held-out document, as in train_resting.py


def chunks(ids: np.ndarray, size: int = CHUNK_IDS) -> Iterator[np.ndarray]:
    """``ids`` in consecutive chunks of ``size``, the last one shorter.

    Args:
        ids: Token ids.
        size: The chunk's length.

    Yields:
        Each chunk, in order.

    Examples:
        >>> [c.tolist() for c in chunks(np.arange(5), 2)]
        [[0, 1], [2, 3], [4]]
    """
    for start in range(0, len(ids), size):
        yield ids[start : start + size]


def without(text: str, held_out: Sequence[str]) -> tuple[str, int]:
    """``text`` with each held-out document cut out where its first characters appear.

    The cut starts at the match and is as long as the document. When the document runs past
    the end of ``text``, the cut stops at the end.

    Args:
        text: Decoded training text.
        held_out: The held-out documents.

    Returns:
        The text, and how many cuts were made.

    Examples:
        >>> without("abcHELDxyz tail", ["HELDxyz"])
        ('abc tail', 1)
        >>> without("no match here", ["HELDxyz"])
        ('no match here', 0)
    """
    cuts = 0
    for doc in held_out:
        head = doc[:PREFIX]
        at = text.find(head) if head else -1
        while at >= 0:
            text = text[:at] + text[at + len(doc) :]
            cuts += 1
            at = text.find(head, at)
    return text, cuts


def read_ids(count: int, url: str) -> np.ndarray:
    """The first ``count`` token ids of the shard at ``url``, read by HTTP range.

    Args:
        count: How many ids to read.
        url: The shard's URL.

    Returns:
        The ids, as unsigned 16-bit integers.
    """
    out = bytearray()
    while len(out) < count * 2:
        end = min(len(out) + CHUNK_BYTES, count * 2) - 1
        # The URL is the dataset's fixed https URL (hf_hub_url), never a user's.
        headers = {"Range": f"bytes={len(out)}-{end}"}
        request = urllib.request.Request(url, headers=headers)  # noqa: S310
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            out += response.read()
        click.echo(f"read {len(out) / 2:,.0f} of {count:,} ids", err=True)
    return np.frombuffer(bytes(out), dtype=np.uint16)


@click.command()
@click.option("--tokens", type=int, default=300_000_000, show_default=True,
              help="GPT-NeoX ids to read. GPT-2 gives about 1.1 times as many tokens.")  # fmt: skip
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--corpus", type=click.Path(path_type=Path), default=Path("corpus.jsonl"),
              show_default=True, help="The held-out corpus to leave out.")  # fmt: skip
def main(tokens: int, out: Path, corpus: Path) -> None:
    """Build the sample and save it as one vector of GPT-2 token ids.

    Args:
        tokens: How many GPT-NeoX ids to read from the start of the shard.
        out: Where to save the vector.
        corpus: The held-out corpus whose documents the sample leaves out.
    """
    import torch
    from huggingface_hub import hf_hub_url
    from train_resting import TOKENIZER  # first: it loads the CLI in the order it needs
    from transformers import AutoTokenizer

    from openinterpretability.cli_evidence import read_corpus

    ids = read_ids(tokens, hf_hub_url(REPO, SHARD, repo_type="dataset"))
    neox = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    gpt2 = AutoTokenizer.from_pretrained(TOKENIZER)
    held_out, *_ = read_corpus(corpus)
    parts, cuts = [], 0
    for n, chunk in enumerate(chunks(ids), 1):
        text, found = without(neox.decode(chunk.tolist()), held_out)
        cuts += found
        parts.append(np.array(gpt2(text, add_special_tokens=False)["input_ids"], dtype=np.int32))
        click.echo(f"chunk {n}: {sum(map(len, parts)):,} GPT-2 tokens, {cuts} cuts", err=True)
    # Add one end-of-text token at the end. train_resting.py reads the vocabulary from the
    # tokens (their largest id + 1), so this makes it GPT-2's, as with tokens.pt.
    parts.append(np.array([gpt2.eos_token_id], dtype=np.int32))
    result = torch.from_numpy(np.concatenate(parts).astype(np.int64))
    torch.save(result, out)
    click.echo(f"{len(result):,} GPT-2 tokens to {out}, {cuts} held-out cuts", err=True)


if __name__ == "__main__":
    main()
