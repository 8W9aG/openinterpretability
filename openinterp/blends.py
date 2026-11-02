"""Fixed, versioned blends of corpus presets (`openinterp corpus --dataset mixed`).

A blend samples each of its presets for its share of the documents, with one seed, and
writes them in order as one corpus, each document recording its preset as ``source``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from openinterp.core.schemas import CorpusSource
from openinterp.corpus import PRESETS, _sample_hub, jsonl_bytes, write_corpus

# Fixed blends of presets, by versioned name: each preset's weight in the blend's documents.
# A blend is never edited once released; a new mix is a new version, so a corpus's name
# says what it holds. `mixed` (the latest version, `openinterp corpus`'s default) spreads
# random-selection statistics over domains instead of one dataset's: web text, the Pile's
# mix of sources (papers, legal text, forums, code), long encyclopedic articles, source code
# and nine languages besides English.
BLENDS = {
    "mixed-v1": {
        "fineweb-edu": 4,
        "c4": 4,
        "pile-uncopyrighted": 4,
        "wikitext": 2,
        "github-code": 3,
        "wikipedia-multilingual": 3,
    },
}
BLEND_ALIASES = {"mixed": "mixed-v1"}
# Presets left out of `mixed`, and why. Every other preset is in its latest version, so
# adding a preset means adding a version of the blend (tests/test_corpus.py checks).
NOT_BLENDED = {
    "pile-10k": "pile-uncopyrighted covers the Pile, without its copyrighted components",
    "tinystories": "synthetic children's stories, a control corpus rather than natural text",
}
DEFAULT_DATASET = "mixed"


def blend_shares(name: str, documents: int) -> dict[str, int]:
    """Each of the blend's presets' documents out of ``documents``, by weight, rounded down,
    with the remainder going one each to the presets listed first."""
    weights = BLENDS[BLEND_ALIASES.get(name, name)]
    total = sum(weights.values())
    shares = {preset: documents * w // total for preset, w in weights.items()}
    for preset in list(shares)[: documents - sum(shares.values())]:
        shares[preset] += 1
    return shares


def build_blend(
    name: str,
    out: Path,
    *,
    documents: int = 500,
    seed: int = 0,
    max_chars: int = 3000,
    ascii: bool = False,
) -> CorpusSource:
    """Sample each of a blend's presets (:data:`BLENDS`) for its share of ``documents``,
    with one seed, and write them in the blend's order as one corpus. Each document records
    its preset as ``source``, and Wikipedia's their language as ``lang``. The sidecar lists
    every preset's own source, as ``openinterp corpus --dataset <preset>`` writes it alone,
    so each part can be rebuilt and checked separately."""
    name = BLEND_ALIASES.get(name, name)
    texts: list[str] = []
    labels: dict[str, list[str | None]] = {"lang": [], "source": []}
    components = []
    for preset, share in blend_shares(name, documents).items():
        p = PRESETS[preset]
        part, part_labels, source = _sample_hub(
            p.dataset,
            files=p.files,
            text_column=p.text_column,
            revision=p.revision,
            documents=share,
            seed=seed,
            max_chars=max_chars,
            docs_per_row_group=p.docs_per_row_group,
            layout=p.layout,
            lang_pattern=p.lang_pattern,
        )
        sha = hashlib.sha256(jsonl_bytes(part, ascii, part_labels)).hexdigest()
        components.append(CorpusSource(documents=len(part), sha256=sha, ascii=ascii, **source))
        texts += part
        labels["lang"] += (part_labels or {}).get("lang", [None] * len(part))
        labels["source"] += [preset] * len(part)
    return write_corpus(
        texts,
        out,
        ascii=ascii,
        labels=labels,
        dataset=name,
        files=",".join(c.dataset for c in components),
        seed=seed,
        max_chars=max_chars,
        docs_per_row_group=None,
        license="; ".join(f"{c.dataset}: {c.license or 'not stated'}" for c in components),
        blend=name,
        components=components,
    )
