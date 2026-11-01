"""Which language detector labels unlabelled corpora (``docs/analyses/language-detection.md``).

Scores fastText's lid.176 (as ``openinterp.langid`` runs it), lingua, langdetect (seeded)
and langid.py on Wikipedia documents whose language is exact (their subset), cut to 50
and 200 characters and whole; on English mixed with another language; on source code,
which a detector should be unsure of; and by agreement on the registry's corpus.

    D=exchange/language-detection
    uv run openinterp corpus --dataset wikipedia-multilingual --docs 1000 --out $D/wiki.jsonl
    uv run openinterp corpus --dataset wikimedia/wikipedia --docs 400 --docs-per-row-group 50 \
        --files "$(for l in es pt it nl uk bg ru ca; do printf "20231101.$l/train-*.parquet,"; \
        done | sed 's/,$//')" --lang-pattern '^\\d+\\.([\\w-]+)/' --out $D/close.jsonl
    uv run openinterp corpus --dataset github-code --docs 200 --out $D/code.jsonl
    uv run --with lingua-language-detector --with langdetect --with langid \
        python scripts/language_detection.py $D corpus.jsonl \
        --out docs/analyses/language-detection.jsonl
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import click

from openinterp import langid

CUTS = (50, 200, None)  # characters; None: the whole document (at most 3,000)
LINES = (0.5, 0.6, 0.7, 0.8, 0.9)  # confidence lines for abstaining


def norm(code: str) -> str:
    return code.lower().replace("_", "-").split("-")[0]


def detectors() -> dict:
    """Each detector as ``text -> (language, confidence)``."""
    import langdetect
    from langid.langid import LanguageIdentifier, model
    from lingua import LanguageDetectorBuilder

    lingua = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

    def lingua_(text):
        values = lingua.compute_language_confidence_values(text)
        top = values[0] if values else None
        return (norm(top.language.iso_code_639_1.name), top.value) if top else (None, 0.0)

    def langdetect_(text):
        langdetect.DetectorFactory.seed = 0  # unseeded, it is not deterministic
        try:
            top = langdetect.detect_langs(text)[0]
        except langdetect.LangDetectException:
            return None, 0.0
        return norm(top.lang), top.prob

    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    def langid_(text):
        lang, p = identifier.classify(text)
        return norm(lang), float(p)

    return {
        "fasttext lid.176": langid.predict,
        "lingua": lingua_,
        "langdetect": langdetect_,
        "langid.py": langid_,
    }


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mixed_documents(wiki: list[dict], n: int = 200, chars: int = 1500, seed: int = 0):
    """English followed by another language, English a quarter, half or three quarters of
    the characters; the right answer is the majority language (either, at half)."""
    rng = random.Random(seed)
    english = [r for r in wiki if r["lang"] == "en"]
    others = [r for r in wiki if r["lang"] != "en"]
    out = []
    for _ in range(n):
        a, b, share = rng.choice(english), rng.choice(others), rng.choice([0.25, 0.5, 0.75])
        text = a["text"][: int(chars * share)] + "\n\n" + b["text"][: int(chars * (1 - share))]
        right = {"en"} if share > 0.5 else {b["lang"]} if share < 0.5 else {"en", b["lang"]}
        out.append((text, right))
    return out


def score(detect, wiki, close, mixed, code, registry) -> dict:
    row: dict = {}
    start, calls = time.perf_counter(), 0
    for name, docs in (("distinct", wiki), ("close", close)):
        for cut in CUTS:
            right = sum(detect(r["text"][:cut])[0] == r["lang"] for r in docs)
            row[f"{name}_accuracy_{cut or 'whole'}"] = round(right / len(docs), 4)
            calls += len(docs)
    row["ms_per_document"] = round(1000 * (time.perf_counter() - start) / calls, 3)
    row["mixed_majority"] = round(sum(detect(t)[0] in r for t, r in mixed) / len(mixed), 4)
    labelled = [(detect(r["text"]), r["lang"]) for r in wiki + close]
    code_conf = [detect(r["text"])[1] for r in code]
    for line in LINES:
        kept = [lang == truth for (lang, p), truth in labelled if p >= line]
        row[f"wiki_kept_{line}"] = round(len(kept) / len(labelled), 4)
        row[f"wiki_kept_accuracy_{line}"] = round(sum(kept) / max(len(kept), 1), 4)
        row[f"code_labelled_{line}"] = round(sum(c >= line for c in code_conf) / len(code), 4)
    row["registry"] = [detect(r["text"])[0] for r in registry]
    return row


@click.command()
@click.argument("work", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("registry", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(work: Path, registry: Path, out: Path) -> None:
    wiki, close, code = (read(work / f) for f in ("wiki.jsonl", "close.jsonl", "code.jsonl"))
    corpus = read(registry)
    mixed = mixed_documents(wiki + close)
    rows = []
    for name, detect in detectors().items():
        row = {"detector": name} | score(detect, wiki, close, mixed, code, corpus)
        mix = Counter(row["registry"]).most_common(6)
        click.echo(f"{name}: {json.dumps({k: v for k, v in row.items() if k != 'registry'})}")
        click.echo(f"  registry corpus: {mix}")
        rows.append(row)
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            agree = sum(x == y for x, y in zip(a["registry"], b["registry"], strict=True))
            click.echo(f"agreement {a['detector']} / {b['detector']}: {agree / len(corpus):.3f}")
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row | {"registry": Counter(row["registry"])}) + "\n")


if __name__ == "__main__":
    main()
