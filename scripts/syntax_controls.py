"""Check that the syntax label sets find the code neurons that the literature expects.

The numbers behind papers/syntax-positive-controls/README.md. The sweep reads every neuron
pole of a model over a code corpus. For each pole and label set, it fits per-label means
on one half of the documents. It then reports the variance they explain in the other half.
The null gives each window the labels of a window from another document. So the labels
keep their position and run lengths, and lose their link to the text.

    uv run python scripts/syntax_controls.py sweep --corpus code.jsonl --out results.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from openinterp.cli import _load, resolve_device
from openinterp.core.confounds import (
    BRACKET_DEPTH,
    INDENT_DEPTH,
    PROGRAMMING_LANGUAGE,
    ROUND_DEPTH,
    SYNTAX_CONSTRUCT,
    SYNTAX_NODE,
)
from openinterp.corpus import DEFAULT_SAMPLE_SEED, DEFAULT_SAMPLE_TOKENS, sample_indices
from openinterp.evidence import read_corpus, read_labels
from openinterp.harness.annotations import UNLABELLED, TokenLabels
from openinterp.harness.hooks import pole_strength
from openinterp.harness.scan import cut_pieces, scan_chunks, scan_neuron
from openinterp.planner import configured_windows
from openinterp.syntax import (
    BRACKET,
    COMMENT,
    GRAMMARS,
    STRING,
    TOP_LEVEL,
    SyntaxTagger,
    _depth_label,
    _paint,
    _runs,
    bracket_depths,
    parse_labels,
    syntax_taggers,
)

MAX_LEVELS = 16  # more than any syntax label set has
TOP_POLES = 5
POLES = ("positive", "negative")
CUT_CHARS = 1500  # half of the corpus's max_chars: the stability test's shorter cut
FULL_CHARS = 3000
CODE, INSIDE = "code", "inside"  # the variants of a depth label set


def depth_variants(tokens: TokenLabels, labels: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return ``labels`` with each depth set's two narrower variants.

    The ``code`` variant leaves out comments and strings, so a licence header can't explain
    it. The ``inside`` variant also leaves out depth 0. So only a pole that tells one
    depth from another scores on it, and a pole that tells inside from outside doesn't.
    """
    names = tokens.names[SYNTAX_NODE]
    prose = torch.tensor(
        [names[int(i)] in (COMMENT, STRING) if i >= 0 else True for i in labels[SYNTAX_NODE]]
    )
    out = dict(labels)
    for name in (BRACKET_DEPTH, INDENT_DEPTH):
        zero = tokens.names[name].index("0") if "0" in tokens.names[name] else -2
        code = labels[name].where(~prose, UNLABELLED)
        out[f"{name}, {CODE}"] = code
        out[f"{name}, {CODE}, {INSIDE}"] = code.where(code != zero, UNLABELLED)
    return out


def stability(rows: list[dict], parse) -> list[dict]:
    """Compare ``parse``'s labels of a document's first characters, under two cuts.

    A label that a cut does not change agrees. It also gives the share of characters
    inside brackets that read as top level, which a lost enclosing construct causes.
    """
    totals: dict[str, list[int]] = {}
    for row in rows:
        text, language = row["text"], row["code_language"]
        if language not in GRAMMARS or len(text) < FULL_CHARS:
            continue
        cut = text[:CUT_CHARS]
        nodes, constructs = parse(text, language)
        short_nodes, short_constructs = parse(cut, language)
        depths = bracket_depths(cut, short_nodes)
        inside = [d != "0" for d in depths]
        lost = sum(i and c == TOP_LEVEL for i, c in zip(inside, short_constructs, strict=True))
        counts = (
            CUT_CHARS,
            sum(a == b for a, b in zip(nodes, short_nodes, strict=False)),
            sum(a == b for a, b in zip(constructs, short_constructs, strict=False)),
            sum(inside),
            lost,
        )
        for key in (language, "all"):
            totals[key] = [a + b for a, b in zip(totals.get(key, [0] * 5), counts, strict=True)]
    return [
        {
            "language": key,
            "node_agreement": round(t[1] / t[0], 4),
            "construct_agreement": round(t[2] / t[0], 4),
            "inside_brackets_top_level": round(t[4] / max(t[3], 1), 4),
        }
        for key, t in sorted(totals.items())
    ]


class HeldOutR2:
    """The sums that give per-label means in two folds, for one label set."""

    def __init__(self, width: int) -> None:
        self.sums = torch.zeros(2, MAX_LEVELS, width, dtype=torch.float64)
        self.counts = torch.zeros(2, MAX_LEVELS, dtype=torch.float64)
        self.squares = torch.zeros(2, width, dtype=torch.float64)

    def update(self, fold: int, labels: torch.Tensor, values: torch.Tensor) -> None:
        """Add one window's labelled tokens: ``labels`` is ``[n]``, ``values`` is ``[n, width]``."""
        seen = labels != UNLABELLED
        labels, values = labels[seen], values[seen].double()
        self.sums[fold].index_add_(0, labels, values)
        self.counts[fold] += torch.bincount(labels, minlength=MAX_LEVELS).double()
        self.squares[fold] += (values * values).sum(0)

    def r2(self) -> torch.Tensor:
        """Each pole's held-out R², averaged over the two ways to split the folds."""
        scores = [self._score(train, 1 - train) for train in (0, 1)]
        return (scores[0] + scores[1]) / 2

    def _score(self, train: int, test: int) -> torch.Tensor:
        means = self.sums[train] / self.counts[train].clamp(min=1).unsqueeze(1)
        sums, counts = self.sums[test], self.counts[test].unsqueeze(1)
        errors = self.squares[test] - 2 * (means * sums).sum(0) + (means**2 * counts).sum(0)
        total = counts.sum()
        spread = self.squares[test] - sums.sum(0) ** 2 / total.clamp(min=1)
        return torch.where(spread > 1e-9, 1 - errors / spread.clamp(min=1e-9), 0.0)


def _pad(labels: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.full((size,), UNLABELLED, dtype=torch.long)
    out[: min(size, labels.numel())] = labels[:size]
    return out


def sweep_layer(runner, tokens: TokenLabels, pieces, layer: int, variants: bool = True) -> dict:
    """Return each label set's held-out R² per pole, with and without the link to the text."""
    width = 2 * runner.mlp_width(layer)
    names = [t.name for t in tokens.taggers]
    for depth in (BRACKET_DEPTH, INDENT_DEPTH) if variants else ():
        names += [f"{depth}, {CODE}", f"{depth}, {CODE}, {INSIDE}"]
    real = {n: HeldOutR2(width) for n in names}
    null = {n: HeldOutR2(width) for n in names}
    other: dict | None = None  # the labels of a window from the document before this one
    current: dict | None = None
    current_doc = None
    for chunk in scan_chunks(runner, pieces, layer):
        acts = chunk.acts.float()
        values = torch.cat([pole_strength(acts, pole) for pole in POLES], dim=1)
        for piece, row in chunk.rows():
            labels = tokens.of(piece, chunk.ids[row])
            if variants:
                labels = depth_variants(tokens, labels)
            counted = chunk.counted[row]
            if piece.doc != current_doc:
                other, current, current_doc = current, labels, piece.doc
            for name in names:
                real[name].update(piece.doc % 2, labels[name][counted], values[row][counted])
                if other is not None:
                    shifted = _pad(other[name], counted.numel())[counted]
                    null[name].update(piece.doc % 2, shifted, values[row][counted])
    return {n: (real[n].r2(), null[n].r2()) for n in names}


def load_code(runner, corpus: Path, max_tokens: int):
    """Return a sample of the corpus's documents that have a grammar, and their languages."""
    texts = read_corpus(corpus)[0]
    languages = read_labels(corpus)[PROGRAMMING_LANGUAGE]
    kept = [i for i, language in enumerate(languages) if language in GRAMMARS]
    picked, sample = sample_indices(
        runner.tokenizer, [texts[i] for i in kept], max_tokens, DEFAULT_SAMPLE_SEED
    )
    return [texts[kept[i]] for i in picked], [languages[kept[i]] for i in picked], sample


def layer_row(layer: int, width: int, name: str, real: torch.Tensor, null: torch.Tensor) -> dict:
    """One layer's result for one label set."""
    top = real.topk(TOP_POLES)
    return {
        "kind": "layer",
        "layer": layer,
        "label_set": name,
        "top": [
            {"neuron": int(i) % width, "pole": POLES[int(i) // width], "r2": round(float(r), 4)}
            for r, i in zip(top.values, top.indices, strict=True)
        ],
        "null_max_r2": round(float(null.max()), 4),
        "median_r2": round(float(real.median()), 5),
        "poles_above_null": int((real > null.max()).sum()),
    }


def winner_row(runner, texts, name, best: dict, window, taggers) -> dict:
    """Scan a label set's best pole as an audit would, and return what the scan records."""
    scan = scan_neuron(
        runner, texts, best["layer"], best["neuron"], best["pole"], k=10,
        window_tokens=window, taggers=taggers,
    )  # fmt: skip
    confounds = scan.confounds
    return {
        "kind": "winner",
        "label_set": name,
        **best,
        "token_r2": confounds.token_r2,
        "position_r2": confounds.position_r2,
        "labels": {k: v.r2 for k, v in confounds.labels.items()},
        "levels": {
            k: [s.model_dump() for s in v.levels]
            for k, v in confounds.labels.items()
            if k in (BRACKET_DEPTH, INDENT_DEPTH, SYNTAX_CONSTRUCT, SYNTAX_NODE)
        },
        "examples": [
            {"peak": e.tokens[e.max_token_index], "at": e.peak_labels} for e in scan.examples[:6]
        ],
    }


def literal_depths(text: str, opener: str, closer: str) -> list[str]:
    """Return each character's depth of ``opener``, counted in the text with no parser."""
    out, depth = [], 0
    for char in text:
        depth += char == opener
        out.append(_depth_label(depth))
        depth = max(0, depth - (char == closer))
    return out


def parsed_depths(text: str, language: str, opener: str, closer: str) -> list[str]:
    """Return each character's depth of ``opener``, where the parser calls it a bracket."""
    nodes, _ = parse_labels(text, language)
    out, depth = [], 0
    for char, node in zip(text, nodes, strict=True):
        depth += node == BRACKET and char == opener
        out.append(_depth_label(depth))
        depth = max(0, depth - (node == BRACKET and char == closer))
    return out


def bracket_kind_taggers(languages) -> list[SyntaxTagger]:
    """Return one tagger per way to count depth: by bracket kind, with and without a parser."""

    def parsed(opener: str, closer: str):
        return lambda doc, text: _runs(parsed_depths(text, languages[doc], opener, closer))

    def literal(opener: str, closer: str):
        return lambda doc, text: _runs(literal_depths(text, opener, closer))

    return [
        SyntaxTagger("round depth", parsed("(", ")")),
        SyntaxTagger("square depth", parsed("[", "]")),
        SyntaxTagger("curly depth", parsed("{", "}")),
        SyntaxTagger("literal round depth", literal("(", ")")),
    ]


@click.command("kinds")
@click.option("--corpus", type=click.Path(path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--model", default="openai-community/gpt2")
@click.option("--device", default="cpu")
@click.option("--max-tokens", default=DEFAULT_SAMPLE_TOKENS)
def kinds(corpus, out, model, device, max_tokens):
    """Append the sweep of depth by bracket kind to ``out``, replacing any earlier rows."""
    runner = _load(model, resolve_device(device), torch.float32, None, None)
    texts, languages, _ = load_code(runner, corpus, max_tokens)
    taggers = bracket_kind_taggers(languages)
    tokens = TokenLabels(runner.tokenizer, texts, taggers)
    pieces = list(cut_pieces(runner.tokenizer, texts, configured_windows().get(model), None))
    lines = out.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if json.loads(line)["kind"] != "kinds"]
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with out.open("a", encoding="utf-8") as sink:
        for layer in range(runner.n_layers):
            width = runner.mlp_width(layer)
            found = sweep_layer(runner, tokens, pieces, layer, variants=False)
            for name, (real, null) in found.items():
                row = layer_row(layer, width, name, real, null) | {"kind": "kinds"}
                row["n676_positive"] = round(float(real[676]), 4) if layer == 2 else None
                row["n676_rank"] = int((real > real[676]).sum()) if layer == 2 else None
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                click.echo(f"L{layer} {name}: {row['top'][0]} null {row['null_max_r2']}")


@click.command()
@click.option("--model", default="openai-community/gpt2")
@click.option("--corpus", type=click.Path(path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cpu")
@click.option("--max-tokens", default=DEFAULT_SAMPLE_TOKENS)
def main(model, corpus, out, device, max_tokens):
    """Write each layer's best poles per syntax label set, then scan the best of each."""
    runner = _load(model, resolve_device(device), torch.float32, None, None)
    texts, languages, sample = load_code(runner, corpus, max_tokens)
    # `kinds` measures round depth, with its neighbours and a parser-free count.
    taggers = [t for t in syntax_taggers(languages) if t.name != ROUND_DEPTH]
    window = configured_windows().get(model)
    pieces = list(cut_pieces(runner.tokenizer, texts, window, None))
    tokens = TokenLabels(runner.tokenizer, texts, taggers)
    click.echo(f"{sample.documents} documents, {sample.tokens} tokens, {len(pieces)} pieces")

    best: dict[str, dict] = {}
    with out.open("w", encoding="utf-8") as sink:

        def write(row: dict) -> None:
            sink.write(json.dumps(row) + "\n")
            sink.flush()

        write({"kind": "sample", "model": model, **sample.model_dump()})
        rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
        for label, parse in (("one parse", _paint), ("closers added", parse_labels)):
            for row in stability(rows, parse):
                write({"kind": "stability", "method": label, **row})
        for layer in range(runner.n_layers):
            width = runner.mlp_width(layer)
            for name, (real, null) in sweep_layer(runner, tokens, pieces, layer).items():
                row = layer_row(layer, width, name, real, null)
                write(row)
                click.echo(f"L{layer} {name}: {row['top'][0]} null {row['null_max_r2']}")
                if name not in best or row["top"][0]["r2"] > best[name]["r2"]:
                    best[name] = {"layer": layer, **row["top"][0]}
        for name, pole in best.items():
            if not name.endswith(CODE):  # the `inside` variant is the narrower test of each depth
                write(winner_row(runner, texts, name, pole, window, taggers))


@click.group()
def cli():
    """Run a sweep of the syntax label sets, or of depth by bracket kind."""


cli.add_command(main, "sweep")
cli.add_command(kinds)

if __name__ == "__main__":
    cli()
