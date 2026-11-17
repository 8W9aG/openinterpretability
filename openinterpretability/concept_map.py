"""A concept map of a model's studied neurons (issue #4422, from the study of #3290).

Each studied pole is placed by the meaning of the text it fires on, as
``docs/analyses/concept-map-feasibility.md`` did:

- **Texts.** A pole's texts are its newest record's top examples, up to
  :data:`TEXTS_PER_POLE`. Each is cut to :data:`WINDOW` tokens on each side of its marked peak.
- **Vectors.** Each text is embedded with :data:`EMBEDDER`. A pole's vector is its mean
  embedding minus the corpus mean (the mean of every pole's mean), then L2-normalised.
  Without the centring, one-pole neurons clump. The norm before normalising is kept as a
  confidence.
- **Points.** Each studied pole is its own point. Most two-pole neurons have unrelated
  poles, so a neuron's two poles are joined by a line, not merged.
- **Clusters.** Leiden communities (seed 0) of the :data:`NEIGHBOURS`-nearest-neighbour graph.
  A cluster is named only from its VERIFIED members' explanations, by the words that set
  them apart (:func:`name_clusters`). A cluster without one is marked unverified.
- **Layout.** UMAP in two dimensions, cosine, ``random_state=0``, scaled to [0, 1]. Distances
  between clusters mean little: layouts kept only about half of each point's neighbours.

``open-interpretability map <model>`` writes the result to ``data/concept-maps/<slug>.json``.
So the site build needs none of the heavy libraries of the ``map`` extra.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MAPS_DIR = Path("data/concept-maps")
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"  # it beat all-mpnet-base-v2 in the study
WINDOW = 20  # tokens on each side of the peak
TEXTS_PER_POLE = 10
NEIGHBOURS = 10  # the k of the clustering graph


@dataclass(frozen=True)
class Pole:
    """One studied pole: where it is, what its scan's top texts are, and what it stands at."""

    slug: str
    layer: int
    neuron: int
    pole: str
    texts: tuple[str, ...]
    explanation: str
    status: str


def window(example) -> str:
    """The example's text within :data:`WINDOW` tokens of its peak, with the peak marked."""
    tokens, peak = example.tokens, example.max_token_index
    left, right = max(0, peak - WINDOW), min(len(tokens), peak + WINDOW + 1)
    parts = [f"«{t}»" if i == peak else t for i, t in enumerate(tokens[left:right], left)]
    return "".join(parts).strip()


def studied_poles(model: str, wiki_dir: Path) -> list[Pole]:
    """Each studied neuron pole of ``model``, from its newest record, in slug order."""
    from openinterpretability.builder.wiki import load_archive

    newest: dict = {}
    for record in load_archive(Path(wiki_dir)):
        ident = record.submission.identifier
        if ident.model_name != model or ident.kind != "mlp_neuron_pole":
            continue
        old = newest.get(ident.slug)
        if old is None or record.verdict.timestamp > old.verdict.timestamp:
            newest[ident.slug] = record
    out = []
    for slug, record in sorted(newest.items()):
        sub, ident = record.submission, record.submission.identifier
        texts = tuple(window(e) for e in sub.top_examples if e.tokens)[:TEXTS_PER_POLE]
        if texts:
            out.append(Pole(slug, ident.layer, ident.neuron_idx, ident.pole, texts,
                            sub.natural_language_explanation, record.verdict.status))  # fmt: skip
    return out


def pole_vectors(poles: list[Pole], encode: Callable):
    """Each pole's centred, normalised mean embedding, and its norm before normalising."""
    import numpy as np

    means = np.array([np.asarray(encode(list(p.texts))).mean(0) for p in poles])
    centred = means - means.mean(0)
    norms = np.linalg.norm(centred, axis=1)
    return centred / np.where(norms > 0, norms, 1.0)[:, None], norms


def clusters(vectors, k: int = NEIGHBOURS) -> list[int]:
    """Leiden communities (seed 0) of the k-nearest-neighbour graph of ``vectors``."""
    import igraph as ig
    import leidenalg
    import numpy as np

    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    nearest = np.argsort(-sims, axis=1)[:, : min(k, len(vectors) - 1)]
    edges = {tuple(sorted((i, int(j)))) for i in range(len(vectors)) for j in nearest[i]}
    graph = ig.Graph(n=len(vectors), edges=sorted(edges))
    part = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, seed=0)
    return list(part.membership)


def layout(vectors) -> list[tuple[float, float]]:
    """A 2-D UMAP of ``vectors`` (cosine, ``random_state=0``), scaled to [0, 1] on each axis."""
    import numpy as np
    import umap

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=0,
                        n_neighbors=min(15, len(vectors) - 1))  # fmt: skip
    xy = reducer.fit_transform(vectors)
    low, span = xy.min(0), np.ptp(xy, axis=0)
    xy = (xy - low) / np.where(span > 0, span, 1.0)
    return [(round(float(x), 4), round(float(y), 4)) for x, y in xy]


_STOP = frozenset(
    (  # words that name no cluster
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
        "when",
        "pole",
        "poles",
        "fires",
        "fire",
        "firing",
        "neuron",
        "token",
        "tokens",
        "text",
        "current",
        "retrospectively",
        "retrospective",
        "prospectively",
        "prospective",
        "activates",
        "active",
        "which",
        "where",
        "after",
        "before",
        "mostly",
        "mainly",
    )
)
NAME_WORDS = 3  # the words of a cluster's name


def _words(text: str) -> list[str]:
    import re

    return [w for w in re.findall(r"[a-z][a-z-]{2,}", text.lower()) if w not in _STOP]


def name_clusters(poles: list[Pole], member: list[int]) -> list[dict]:
    """Each cluster's size and name, named from its VERIFIED members only.

    The name is the :data:`NAME_WORDS` words that most set the cluster's VERIFIED
    explanations apart from the other clusters' (a class-based TF-IDF). Words that every
    explanation uses, such as "pole" or "fires", are left out. A cluster with no VERIFIED
    member has no name, and ``verified`` is False.
    """
    import math
    from collections import Counter

    counts = {cid: Counter() for cid in sorted(set(member))}
    for p, cid in zip(poles, member, strict=True):
        if p.status == "VERIFIED":
            counts[cid].update(_words(p.explanation))
    holding = Counter(w for c in counts.values() for w in c)  # clusters that use each word
    out = []
    for cid, words in counts.items():
        total = sum(words.values()) or 1
        # a word in every cluster scores 0: it names none of them
        score = {w: n / total * math.log(len(counts) / holding[w]) for w, n in words.items()}
        score = {w: v for w, v in score.items() if v > 0 and words[w] >= 2}
        top = sorted(score, key=lambda w: (-score[w], w))[:NAME_WORDS]
        size = sum(1 for m in member if m == cid)
        out.append({"id": cid, "size": size, "name": " · ".join(top) or None,
                    "verified": bool(top)})  # fmt: skip
    return out


def build(model: str, poles: list[Pole], encode: Callable, *, place=layout, group=clusters) -> dict:
    """The map of ``poles``: points, clusters and what made it (module docstring)."""
    vectors, norms = pole_vectors(poles, encode)
    member, xy = group(vectors), place(vectors)
    points = [
        {"slug": p.slug, "layer": p.layer, "neuron": p.neuron, "pole": p.pole,
         "x": x, "y": y, "cluster": int(c), "norm": round(float(n), 4)}
        for p, (x, y), c, n in zip(poles, xy, member, norms, strict=True)
    ]  # fmt: skip
    return {
        "model": model, "embedder": EMBEDDER, "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "poles": len(poles), "clusters": name_clusters(poles, member), "points": points,
    }  # fmt: skip


def map_path(model: str, folder: Path = MAPS_DIR) -> Path:
    """``data/concept-maps/<model slug>.json``."""
    from openinterpretability.core.identifiers import model_slug

    return Path(folder) / f"{model_slug(model)}.json"


def write(found: dict, path: Path) -> None:
    """Write a map as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=1) + "\n", encoding="utf-8")


def load(path: Path) -> dict | None:
    """A written map, or None when there is none."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
