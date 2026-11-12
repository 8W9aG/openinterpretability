"""Which detector labels the programming languages of unlabelled corpora.

The results and the decision are in ``docs/analyses/code-detection.md``.

Scores four detectors. Each reads a document's text only, because an unlabelled corpus
has no file names.

- Magika, Google's file-type model (``magika``). It runs three ways: with a confidence
  line, with its own thresholds, and as ``openinterp.codeid`` runs it.
- go-enry, the Go port of GitHub Linguist (``scripts/code_detection_enry``): its shebang
  and modeline rules, then its Bayesian classifier over every language it knows.
- ``philomath-1209/programming-language-identification``, a CodeBERTa classifier of
  26 languages, as a transformer baseline.
- Pygments' ``guess_lexer``, the common cheap baseline.

The tests are:

- github-code files, whose language is exact (their ``code_language``), cut to 50 and
  200 characters and whole.
- fineweb-edu prose, which a detector should not label as code.
- Prose followed by code, as in a Stack Exchange post.
- The registry's corpus, whose pile-10k subset (``pile_set_name``) says which documents
  are GitHub files.

    D=exchange/code-detection
    uv run openinterp corpus --dataset github-code --docs 1000 --out $D/code.jsonl
    uv run openinterp corpus --dataset fineweb-edu --docs 300 --out $D/prose.jsonl
    (cd scripts/code_detection_enry && go build -o ../../$D/enry-helper .)
    uv run python scripts/code_detection.py $D corpus.jsonl \
        --out docs/analyses/code-detection.jsonl
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import click

from openinterp import codeid
from openinterp.corpus_presets import GITHUB_CODE_LANGUAGES
from openinterp.syntax import GRAMMARS

PILE_10K = "NeelNanda/pile-10k"
PILE_10K_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"  # as corpus.source.json pins
PILE_10K_FILE = "data/train-00000-of-00001-4746b8785c874cc7.parquet"
CUTS = (50, 200, None)  # characters; None: the whole document (at most 3,000)
LINES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)  # confidence lines for abstaining
SHARES = (0.25, 0.5, 0.75)  # the code's share of a mixed document's characters
MIXED_CHARS = 1500

# Labels that claim no programming language. A detector that gives one abstains.
ABSTAIN = {"txt", "unknown", "empty", "text", "text only", "plaintext"}
# Detector labels whose github-code name differs from the label in lower case.
ALIASES = {
    "cpp": "C++", "cs": "C#", "csharp": "C#", "go": "GO", "golang": "GO",
    "bash": "Shell", "sh": "Shell", "batch": "Batchfile", "asm": "Assembly",
    "gas": "Assembly", "nasm": "Assembly", "arm assembly": "Assembly", "latex": "TeX",
    "vba": "Visual Basic", "vb.net": "Visual Basic", "visual basic .net": "Visual Basic",
    "make": "Makefile", "docker": "Dockerfile", "js": "JavaScript", "fortran": "FORTRAN",
    "html+php": "PHP",
}  # fmt: skip


def read(path: Path) -> list[dict]:
    """The JSONL rows of ``path``."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def namer(names: Sequence[str]) -> Callable[[str | None], str | None]:
    """Return a function that gives a detector label its github-code name.

    An abstaining label becomes ``None``. A label with no github-code name stays as it is,
    so it counts as wrong on code and as a code label on prose.
    """
    by_lower = {n.lower(): n for n in names}

    def name(label: str | None) -> str | None:
        if label is None or label.lower() in ABSTAIN:
            return None
        key = label.lower()
        return ALIASES.get(key) or by_lower.get(key) or label

    return name


Detector = Callable[[Sequence[str]], list[tuple[str | None, float]]]


def magika_detectors() -> dict[str, Detector]:
    """Magika with its top label and score, and with its own per-type thresholds."""
    from magika import Magika

    magika = Magika()

    def predictions(texts):
        return [magika.identify_bytes(t.encode("utf-8", "surrogatepass")).prediction for t in texts]

    return {
        "magika": lambda texts: [(r.dl.label, float(r.score)) for r in predictions(texts)],
        # Its own thresholds decide when it abstains (`txt` or `unknown`).
        "magika (own thresholds)": lambda texts: [
            (r.output.label, 1.0) for r in predictions(texts)
        ],
    }


def enry_detector(helper: Path) -> Detector:
    """go-enry through its helper binary. It gives no confidence and always names a language."""

    def detect(texts):
        lines = "".join(json.dumps(t) + "\n" for t in texts)
        result = subprocess.run([str(helper)], input=lines, capture_output=True, text=True)
        result.check_returncode()
        return [(line.split("\t")[0] or None, 1.0) for line in result.stdout.splitlines()]

    return detect


def codeberta_detector() -> Detector:
    """The CodeBERTa classifier of 26 languages, on CPU, with its softmax score."""
    from transformers import pipeline

    model = "philomath-1209/programming-language-identification"
    classify = pipeline("text-classification", model=model, device="cpu")

    def detect(texts):
        results = classify(list(texts), truncation=True, batch_size=16)
        return [(r["label"], float(r["score"])) for r in results]

    return detect


def pygments_detect(texts: Sequence[str]) -> list[tuple[str | None, float]]:
    """Pygments' ``guess_lexer``, with the lexer's ``analyse_text`` score."""
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound

    out: list[tuple[str | None, float]] = []
    for text in texts:
        try:
            lexer = guess_lexer(text)
        except ClassNotFound:
            out.append((None, 0.0))
            continue
        out.append((lexer.aliases[0] if lexer.aliases else lexer.name, lexer.analyse_text(text)))
    return out


def detectors(enry: Path) -> dict[str, Detector]:
    """Each detector as ``texts -> [(label, confidence)]``, labels not yet renamed."""
    magika = magika_detectors()
    return {
        "magika": magika["magika"],
        # Labels that are github-code names already, or None.
        "magika (openinterp.codeid)": lambda texts: [codeid.predict(t) for t in texts],
        "magika (own thresholds)": magika["magika (own thresholds)"],
        "go-enry (Linguist)": enry_detector(enry),
        "codeberta language-id": codeberta_detector(),
        "pygments guess_lexer": pygments_detect,
    }


def mixed_documents(prose: list[dict], code: list[dict], seed: int = 0):
    """Return documents of prose followed by code of a language in ``GRAMMARS``.

    The code makes up a quarter, half or three quarters of the characters. Each item is
    ``(text, code language, code share)``.
    """
    rng = random.Random(seed)
    parsed = [r for r in code if r["code_language"] in GRAMMARS]
    out = []
    for i in range(200):
        p, c, share = rng.choice(prose), rng.choice(parsed), SHARES[i % len(SHARES)]
        text = (
            p["text"][: int(MIXED_CHARS * (1 - share))]
            + "\n\n"
            + c["text"][: int(MIXED_CHARS * share)]
        )
        out.append((text, c["code_language"], share))
    return out


def pile_sets() -> list[str]:
    """The pile-10k subset of each document of the registry's corpus (its first 500)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(PILE_10K, PILE_10K_FILE, repo_type="dataset", revision=PILE_10K_REVISION)
    meta = pq.read_table(path, columns=["meta"]).slice(0, 500).column("meta").to_pylist()
    return [m["pile_set_name"] for m in meta]


def ratio(count: int, total: int) -> float:
    """``count / total``, to four places."""
    return round(count / max(total, 1), 4)


def code_scores(detect: Detector, name, code) -> tuple[dict, list]:
    """Accuracies and speed on the code files, and each file's whole-file prediction."""
    row: dict = {}
    truth = [r["code_language"] for r in code]
    start = time.perf_counter()
    whole = [(name(label), p) for label, p in detect([r["text"] for r in code])]
    row["ms_per_document"] = round(1000 * (time.perf_counter() - start) / len(code), 2)
    for cut in CUTS:
        got = (
            whole
            if cut is None
            else [(name(x), p) for x, p in detect([r["text"][:cut] for r in code])]
        )
        right = sum(g == t for (g, _), t in zip(got, truth, strict=True))
        row[f"code_accuracy_{cut or 'whole'}"] = ratio(right, len(code))
    parsed = [(g, t) for (g, _), t in zip(whole, truth, strict=True) if t in GRAMMARS]
    row["grammar_accuracy_whole"] = ratio(sum(g == t for g, t in parsed), len(parsed))
    wrong = (f"{t} -> {g}" for (g, _), t in zip(whole, truth, strict=True) if g != t)
    row["confusions"] = Counter(wrong).most_common(12)
    return row, list(zip(whole, truth, strict=True))


def line_scores(whole: list, on_prose: list) -> dict:
    """At each confidence line: the code files kept, their accuracy, and the prose labelled."""
    row: dict = {}
    for line in LINES:
        kept = [g == t for (g, p), t in whole if g and p >= line]
        row[f"code_kept_{line}"] = ratio(len(kept), len(whole))
        row[f"code_kept_accuracy_{line}"] = ratio(sum(kept), len(kept))
        labelled = [g for g, p in on_prose if g and p >= line]
        row[f"prose_labelled_{line}"] = ratio(len(labelled), len(on_prose))
        row[f"prose_grammar_labelled_{line}"] = ratio(
            sum(g in GRAMMARS for g in labelled), len(on_prose)
        )
    row["prose_labels"] = Counter(g for g, _ in on_prose).most_common(8)
    return row


def mixed_scores(detect: Detector, name, mixed) -> dict:
    """For each code share, how often the code's language is the label, and at what confidence."""
    row: dict = {}
    got = [(name(x), p) for x, p in detect([t for t, _, _ in mixed])]
    for share in SHARES:
        items = [
            (g, p, lang) for (g, p), (_, lang, s) in zip(got, mixed, strict=True) if s == share
        ]
        row[f"mixed_{share}_code_language"] = ratio(
            sum(g == lang for g, _, lang in items), len(items)
        )
        row[f"mixed_{share}_confidence"] = [round(p, 3) for g, p, lang in items if g == lang]
    return row


def score(detect: Detector, name, code, prose, mixed, registry) -> dict:
    """One detector's accuracies, speed and labels over every test set."""
    row, whole = code_scores(detect, name, code)
    on_prose = [(name(x), p) for x, p in detect([r["text"] for r in prose])]
    row |= line_scores(whole, on_prose) | mixed_scores(detect, name, mixed)
    row["registry"] = [(name(x), round(p, 4)) for x, p in detect([r["text"] for r in registry])]
    return row


@click.command()
@click.argument("work", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("registry", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(work: Path, registry: Path, out: Path) -> None:
    """Score every detector and write one row each."""
    code, prose = read(work / "code.jsonl"), read(work / "prose.jsonl")
    corpus, sets = read(registry), pile_sets()
    mixed = mixed_documents(prose, code)
    name = namer(GITHUB_CODE_LANGUAGES)
    with out.open("w") as fh:
        fh.write(json.dumps({"registry_pile_sets": sets}) + "\n")
        for detector, detect in detectors(work / "enry-helper").items():
            row = {"detector": detector} | score(detect, name, code, prose, mixed, corpus)
            short = {k: v for k, v in row.items() if k != "registry" and "confidence" not in k}
            click.echo(f"{detector}: {json.dumps(short)}")
            fh.write(json.dumps(row) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
