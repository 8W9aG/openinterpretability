"""Labelled text for `openinterp probe-compare`, and the rows of its analysis.

Each concept is a regular expression over characters, so its labels never depend on
the model. ``labels`` marks a corpus with one concept; ``report`` gathers the
comparison reports into JSON lines (docs/analyses/probe-comparison.md).

    uv run python scripts/probe_comparison.py labels --concept year \\
        --corpus corpus.jsonl --out exchange/probe-comparison/year.pile.jsonl
    uv run python scripts/probe_comparison.py report exchange/probe-comparison/*.report.json \\
        --out docs/analyses/probe-comparison.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click

FEMININE = (
    "she|her|hers|herself|woman|women|girls?|mothers?|daughters?|sisters?|wife|wives|"
    "queens?|princess(?:es)?|lady|ladies|actress(?:es)?|grandmothers?|aunts?|nieces?|"
    "females?|Mrs|Ms"
)

CONCEPTS = {
    "thousands": (
        "a numeral written with a thousands separator (every character of it)",
        re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?!,?\d)"),
    ),
    "year": (
        "a calendar year 1500–2029 (or its decade, '1960s') not preceded by a currency sign "
        "or separator, or an ordinal century ('19th century')",
        re.compile(
            r"(?<![\d.,$£€])(?:1[5-9]\d\d|20[0-2]\d)s?(?![\d,.]?\d)"
            r"|\b\d{1,2}(?:st|nd|rd|th)[- ]century\b"
        ),
    ),
    "feminine": (
        f"a feminine-marked word, case-insensitive: {FEMININE.replace('|', ', ')}",
        re.compile(rf"\b(?:{FEMININE})\b", re.IGNORECASE),
    ),
    "street": (
        "a capitalised thoroughfare noun after a word: Street, Avenue, Road, Boulevard, "
        "Drive, Lane or Highway",
        re.compile(r"(?<=[A-Za-z] )(?:Street|Avenue|Road|Boulevard|Drive|Lane|Highway)\b"),
    ),
}


@click.group()
def main() -> None:
    pass


@main.command()
@click.option("--concept", required=True, type=click.Choice(list(CONCEPTS)))
@click.option("--corpus", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
def labels(concept, corpus, out):
    """Mark every document of --corpus with the concept's spans."""
    rule = CONCEPTS[concept][1]
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = spans = 0
    with out.open("w", encoding="utf-8") as f:
        for line in corpus.read_text(encoding="utf-8").splitlines():
            text = json.loads(line)["text"]
            found = [[m.start(), m.end()] for m in rule.finditer(text)]
            f.write(json.dumps({"text": text, "spans": found}) + "\n")
            docs, spans = docs + 1, spans + len(found)
    click.echo(f"{concept}: {spans} spans in {docs} documents → {out}")


@main.command()
@click.argument("reports", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
def report(reports, out):
    """One JSON line per comparison, without the explanation (it is in the record)."""
    rows = []
    for path in reports:
        row = json.loads(path.read_text())
        row.pop("explanation")
        rows.append(row)
    rows.sort(key=lambda r: (r["slug"], r["split"]))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"{len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
