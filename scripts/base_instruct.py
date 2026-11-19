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
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.interference.keep import (  # noqa: E402, F401  (tests use them)
    GATES,
    SAMPLE_TOKENS,
    chat_strength,
    pearson,
    pole_max,
    readings,
    sample_stats,
    sample_windows,
    verified_records,
    wilson,
)

BASE = "Qwen/Qwen2.5-1.5B"


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--base", "base_name", default=BASE, show_default=True)
@click.option("--chat", is_flag=True, help="Read the instruct model inside its chat template.")
def main(wiki_dir: Path, out: Path, device: str, base_name: str, chat: bool) -> None:
    """Measure each VERIFIED pole's probes on both models, and write the rows."""
    from openinterpretability.cli import _load_runner

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
