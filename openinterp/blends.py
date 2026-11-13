"""Fixed, versioned blends of corpus presets (`openinterp corpus --dataset mixed`).

A blend samples each of its presets for its share of the documents, with one seed. It
writes them in order as one corpus. Each document records its preset as ``source``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from openinterp.core.schemas import CorpusSource
from openinterp.corpus import PRESETS, _sample_hub, jsonl_bytes, write_corpus

# Fixed blends of presets, by versioned name. Each value is a preset's weight in the blend.
# A blend is never edited once released. A new mix is a new version, so a corpus's name
# says what it holds. `mixed` is the latest version and `openinterp corpus`'s default. It
# spreads random-selection statistics over domains, not one dataset's. They are web
# text, the Pile's mix of sources (papers, legal text, forums, code), long encyclopedic
# articles, source code and nine languages besides English. From v3, they include maths,
# from v4, chat transcripts, from v5, web text in 22 languages besides English, and from
# v6, scientific papers. From v7, a part without exact labels gets detected ones: its
# documents' languages (openinterp.langid) and programming languages (openinterp.codeid).
BLENDS = {
    "mixed-v1": {
        "fineweb-edu": 4,
        "c4": 4,
        "pile-uncopyrighted": 4,
        "wikitext": 2,
        "github-code": 3,
        "wikipedia-multilingual": 3,
    },
    # The same mix as v1. Its code documents also record their programming language.
    "mixed-v2": {
        "fineweb-edu": 4,
        "c4": 4,
        "pile-uncopyrighted": 4,
        "wikitext": 2,
        "github-code": 3,
        "wikipedia-multilingual": 3,
    },
    # v2's mix and weights, plus mathematical text at the same weight as wikitext.
    "mixed-v3": {
        "fineweb-edu": 4,
        "c4": 4,
        "pile-uncopyrighted": 4,
        "wikitext": 2,
        "github-code": 3,
        "wikipedia-multilingual": 3,
        "open-web-math": 2,
    },
    # v3's mix and weights, plus chat transcripts at the same weight as wikitext, written
    # with Qwen2.5's chat template, for instruct models.
    "mixed-v4": {
        "fineweb-edu": 4,
        "c4": 4,
        "pile-uncopyrighted": 4,
        "wikitext": 2,
        "github-code": 3,
        "wikipedia-multilingual": 3,
        "open-web-math": 2,
        "ultrachat": 2,
    },
    # v4's mix and weights, plus multilingual web text at the weight of Wikipedia's
    # languages, so non-English text is not only encyclopedic.
    "mixed-v5": {
        "fineweb-edu": 4,
        "c4": 4,
        "pile-uncopyrighted": 4,
        "wikitext": 2,
        "github-code": 3,
        "wikipedia-multilingual": 3,
        "open-web-math": 2,
        "ultrachat": 2,
        "fineweb-2": 3,
    },
    # v5's mix and weights, plus scientific papers (abstracts and full texts) at the weight
    # of wikitext.
    "mixed-v6": {
        "fineweb-edu": 4,
        "c4": 4,
        "pile-uncopyrighted": 4,
        "wikitext": 2,
        "github-code": 3,
        "wikipedia-multilingual": 3,
        "open-web-math": 2,
        "ultrachat": 2,
        "fineweb-2": 3,
        "pes2o": 2,
    },
    # v6's mix and weights. Each part without exact labels gets detected ones, so English
    # web text records its language and the Pile's code and chat answers record theirs.
    "mixed-v7": {
        "fineweb-edu": 4,
        "c4": 4,
        "pile-uncopyrighted": 4,
        "wikitext": 2,
        "github-code": 3,
        "wikipedia-multilingual": 3,
        "open-web-math": 2,
        "ultrachat": 2,
        "fineweb-2": 3,
        "pes2o": 2,
    },
}
BLEND_ALIASES = {"mixed": "mixed-v7"}
# Released before documents recorded a programming language. So their bytes stay without it.
NO_CODE_LANGUAGE = {"mixed-v1"}
# Released before parts without exact labels got detected ones. So their bytes stay without.
NO_DETECTION = {"mixed-v1", "mixed-v2", "mixed-v3", "mixed-v4", "mixed-v5", "mixed-v6"}
# Presets left out of `mixed`, and why. Every other preset is in its latest version, so
# adding a preset means adding a version of the blend (tests/test_corpus.py checks).
NOT_BLENDED = {
    "gsm8k": "one benchmark's short word problems; open-web-math gives the blend its maths",
    "pile-10k": "pile-uncopyrighted covers the Pile, without its copyrighted components",
    "tinystories": "synthetic children's stories, a control corpus rather than natural text",
}
DEFAULT_DATASET = "mixed"


def blend_shares(name: str, documents: int) -> dict[str, int]:
    """Split ``documents`` among the blend's presets by weight, rounded down.

    The remainder goes one each to the presets listed first.
    """
    weights = BLENDS[BLEND_ALIASES.get(name, name)]
    total = sum(weights.values())
    shares = {preset: documents * w // total for preset, w in weights.items()}
    for preset in list(shares)[: documents - sum(shares.values())]:
        shares[preset] += 1
    return shares


def _sample_preset(
    preset: str,
    documents: int,
    seed: int,
    max_chars: int,
    code_language: bool,
    revisions: dict[str, str | None],
    detect: bool = False,
):
    """Return what :func:`openinterp.corpus._sample_hub` reads for a blend's ``preset``.

    ``revisions`` overrides the preset's revision, by dataset, and its chat template's
    model commit, by model. With ``detect``, a preset without exact languages or exact
    programming languages gets detected ones.
    """
    p = PRESETS[preset]
    return _sample_hub(
        p.dataset,
        files=p.files,
        text_column=p.text_column,
        revision=revisions.get(p.dataset, p.revision),
        documents=documents,
        seed=seed,
        max_chars=max_chars,
        docs_per_row_group=p.docs_per_row_group,
        layout=p.layout,
        lang_pattern=p.lang_pattern,
        detect_lang=detect and p.lang_pattern is None,
        code_language_column=p.code_language_column if code_language else None,
        detect_code=detect and code_language and p.code_language_column is None,
        chat_template=p.chat_template,
        chat_template_revision=revisions.get(p.chat_template) if p.chat_template else None,
    )


def build_blend(
    name: str,
    out: Path,
    *,
    documents: int = 500,
    seed: int = 0,
    max_chars: int = 3000,
    ascii: bool = False,
    revisions: dict[str, str | None] | None = None,
) -> CorpusSource:
    """Sample each of a blend's presets (:data:`BLENDS`) and write one corpus.

    Each preset gives its share of ``documents``, with one seed, in the blend's order. Each
    document records its preset as ``source``, and Wikipedia's documents record their
    language as ``lang``. GitHub code's documents record theirs as ``code_language``, except
    in the versions of :data:`NO_CODE_LANGUAGE`. Outside :data:`NO_DETECTION`, the other
    presets' documents record a detected language and programming language, where the
    detector is sure of one. The sidecar lists the source of every preset, as
    ``openinterp corpus --dataset <preset>`` writes it alone. So you can rebuild and check
    each part separately. ``revisions`` pins datasets to commits, by dataset, and chat
    templates' models to commits, by model. Without it, each preset reads its latest commits.
    """
    name = BLEND_ALIASES.get(name, name)
    texts: list[str] = []
    labels: dict[str, list[str | None]] = {"lang": [], "source": []}
    code_language, detect = name not in NO_CODE_LANGUAGE, name not in NO_DETECTION
    if code_language:
        labels["code_language"] = []
    components = []
    for preset, share in blend_shares(name, documents).items():
        part, part_labels, source = _sample_preset(
            preset, share, seed, max_chars, code_language, revisions or {}, detect
        )
        sha = hashlib.sha256(jsonl_bytes(part, ascii, part_labels)).hexdigest()
        components.append(CorpusSource(documents=len(part), sha256=sha, ascii=ascii, **source))
        texts += part
        labels["lang"] += (part_labels or {}).get("lang", [None] * len(part))
        labels["source"] += [preset] * len(part)
        if code_language:
            labels["code_language"] += (part_labels or {}).get("code_language", [None] * len(part))
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
