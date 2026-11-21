"""One concept map shared by every model's studied neurons (#4422, #4586).

Each studied pole of every model is placed by the meaning of the text it fires on, as
``docs/analyses/concept-map-feasibility.md`` did. All models share one space. So a pole has
the same position on every page, and each new model fills in the space that the others
made.

- **Texts.** A pole's texts are its newest record's top examples, up to
  :data:`TEXTS_PER_POLE`. Each is cut to :data:`WINDOW` tokens on each side of its marked peak.
  A line of the package's 40-line sample corpus is left out. A scan on that corpus finds the
  same few lines for every neuron, and they say nothing about what the pole reads. In the
  first shared fit, those poles made one island of their own. A pole whose records have no
  other text isn't placed.
- **Vectors.** Each text is embedded with :data:`EMBEDDER`. A pole's vector is its mean
  embedding minus the *centre*, then L2-normalised. The centre is the mean of every pole's
  mean, pooled over all models when the map was fitted. Without the centring, one-pole
  neurons clump. The norm before normalising is kept as a confidence.
- **Points.** Each studied pole is its own point. A line joins a neuron's two poles.
- **Regions.** Leiden communities of the :data:`NEIGHBOURS`-nearest-neighbour graph. The
  resolution is searched for the count nearest :data:`TARGET`, as the literature map's is.
  ``concept_regions`` names them and measures each model's reach.
- **Layout.** UMAP in two dimensions, cosine, ``random_state=0``, scaled to [0, 1].
  Distances between regions mean little: layouts kept about half of each point's neighbours.
- **Stable.** A later run keeps each old pole's position and region. A new pole goes at
  the similarity-weighted mean of its :data:`NEIGHBOURS` nearest old poles, in their
  majority region (:func:`place_new`). UMAP's own ``transform`` starts a new point in the
  same way. No fitted reducer is kept, because loading a pickle from the repository runs
  code. Only a refit (``--refit``) moves old points. It starts UMAP from the old positions
  and adds 1 to ``version``.

``open-interpretability map`` writes the result to :data:`SHARED`. So the site build needs
none of the heavy libraries of the ``map`` extra.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MAPS_DIR = Path("data/concept-maps")
SHARED = MAPS_DIR / "shared.json"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"  # it beat all-mpnet-base-v2 in the study
WINDOW = 20  # tokens on each side of the peak
TEXTS_PER_POLE = 10
NEIGHBOURS = 10  # the k of the clustering graph, and of placing a new pole
TARGET = 8  # regions to aim at: one for each palette colour (builder/cluster_map.SLOTS)
RESOLUTIONS = [round(0.4 + 0.1 * i, 1) for i in range(17)]  # 0.4 to 2.0, searched in order


@dataclass(frozen=True)
class Pole:
    """One studied pole: whose it is, its top texts, its standing and when it was first studied."""

    model: str
    slug: str
    layer: int
    neuron: int
    pole: str
    texts: tuple[str, ...]
    explanation: str
    status: str
    first: str = ""  # the date of its first verdict, YYYY-MM-DD
    corpus: str = ""  # the family of the scan corpus its texts come from (corpus_family)


def window(example) -> str:
    """The example's text within :data:`WINDOW` tokens of its peak, with the peak marked."""
    tokens, peak = example.tokens, example.max_token_index
    left, right = max(0, peak - WINDOW), min(len(tokens), peak + WINDOW + 1)
    parts = [f"«{t}»" if i == peak else t for i, t in enumerate(tokens[left:right], left)]
    return "".join(parts).strip()


def _sample_lines() -> frozenset[str]:
    """The lines of the package's sample corpus, which say nothing about a pole."""
    from openinterpretability.data import sample_corpus

    return frozenset(sample_corpus())


def _texts(sub, sample: frozenset[str]) -> tuple[str, ...]:
    """The record's top examples as texts, without the sample corpus's lines."""
    kept = [e for e in sub.top_examples if e.tokens and "".join(e.tokens).strip() not in sample]
    return tuple(window(e) for e in kept)[:TEXTS_PER_POLE]


def corpus_family(provenance) -> str:
    """The scan corpus's family: ``mixed`` for every version of the mixed blend, else its name."""
    src = provenance.corpus_source
    name = (src.blend or src.dataset) if src is not None else (provenance.corpus or "")
    return "mixed" if name.startswith("mixed-") else name.split(":")[0]


def studied_poles(models, wiki_dir: Path) -> list[Pole]:
    """Each studied neuron pole of ``models``, from its newest record with texts, in slug order."""
    from openinterpretability.builder.wiki import load_archive

    sample = _sample_lines()
    newest: dict = {}
    first: dict[str, str] = {}
    for record in load_archive(Path(wiki_dir)):
        ident = record.submission.identifier
        if ident.model_name not in models or ident.kind != "mlp_neuron_pole":
            continue
        stamp = record.verdict.timestamp
        first[ident.slug] = min(first.get(ident.slug, stamp), stamp)
        texts = _texts(record.submission, sample)
        old = newest.get(ident.slug)
        if texts and (old is None or stamp > old[0].verdict.timestamp):
            newest[ident.slug] = (record, texts)
    out = []
    for slug, (record, texts) in sorted(newest.items()):
        sub, ident = record.submission, record.submission.identifier
        # newest holds only poles with texts (above), so each one goes on the map.
        out.append(Pole(ident.model_name, slug, ident.layer, ident.neuron_idx, ident.pole,
                        texts, sub.natural_language_explanation, record.verdict.status,
                        first[slug].strftime("%Y-%m-%d"),
                        corpus_family(sub.provenance)))  # fmt: skip
    return out


def means(poles: list[Pole], encode: Callable):
    """Each pole's mean embedding, before centring."""
    import numpy as np

    return np.array([np.asarray(encode(list(p.texts))).mean(0) for p in poles])


def centred(raw, centre):
    """``raw`` minus ``centre``, L2-normalised, and each row's norm before normalising."""
    import numpy as np

    moved = raw - np.asarray(centre)
    norms = np.linalg.norm(moved, axis=1)
    return moved / np.where(norms > 0, norms, 1.0)[:, None], norms


def clusters(vectors, k: int = NEIGHBOURS, target: int = TARGET) -> list[int]:
    """Leiden communities of the k-nearest-neighbour graph, at the resolution nearest ``target``.

    Each of :data:`RESOLUTIONS` is tried in order, and the first with the count nearest
    ``target`` wins. A fixed resolution would let one added pole change the region count.
    """
    import igraph as ig
    import leidenalg
    import numpy as np

    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    nearest = np.argsort(-sims, axis=1)[:, : min(k, len(vectors) - 1)]
    edges = {tuple(sorted((i, int(j)))) for i in range(len(vectors)) for j in nearest[i]}
    graph = ig.Graph(n=len(vectors), edges=sorted(edges))
    best = None
    for resolution in RESOLUTIONS:
        part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                        resolution_parameter=resolution, seed=0)  # fmt: skip
        if best is None or abs(len(part) - target) < abs(len(best) - target):
            best = part
    return list(best.membership)


def layout(vectors, init=None) -> list[tuple[float, float]]:
    """A 2-D UMAP of ``vectors`` (cosine, ``random_state=0``), scaled to [0, 1] on each axis.

    ``init``, when given, is the starting position of each row, as a refit uses.
    """
    import numpy as np
    import umap

    start = "spectral" if init is None else np.asarray(init, dtype=float)
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=0, init=start,
                        n_neighbors=min(15, len(vectors) - 1))  # fmt: skip
    xy = reducer.fit_transform(vectors)
    low, span = xy.min(0), np.ptp(xy, axis=0)
    xy = (xy - low) / np.where(span > 0, span, 1.0)
    return [(round(float(x), 4), round(float(y), 4)) for x, y in xy]


def place_new(old_vectors, old_points: list[dict], new_vectors, k: int = NEIGHBOURS):
    """A position and a region for each new vector, from its ``k`` nearest old poles.

    The position is the mean of their positions, weighted by cosine similarity (a negative
    similarity weighs 0). The region is the one that most of them are in, by weight.
    """
    import numpy as np

    sims = new_vectors @ old_vectors.T
    out = []
    for row in sims:
        near = np.argsort(-row)[: min(k, len(old_points))]
        weights = np.clip(row[near], 0.0, None)
        weights = weights if weights.sum() > 0 else np.ones(len(near))
        x = sum(w * old_points[j]["x"] for w, j in zip(weights, near, strict=True)) / weights.sum()
        y = sum(w * old_points[j]["y"] for w, j in zip(weights, near, strict=True)) / weights.sum()
        votes = Counter()
        for w, j in zip(weights, near, strict=True):
            votes[old_points[j]["cluster"]] += float(w)
        region = min(votes, key=lambda r: (-votes[r], r))
        out.append((round(float(x), 4), round(float(y), 4), int(region)))
    return out


def _point(p: Pole, x: float, y: float, cluster: int, norm: float) -> dict:
    return {"model": p.model, "slug": p.slug, "layer": p.layer, "neuron": p.neuron,
            "pole": p.pole, "x": x, "y": y, "cluster": int(cluster),
            "norm": round(float(norm), 4), "first": p.first, "corpus": p.corpus}  # fmt: skip


def build(poles: list[Pole], encode: Callable, old: dict | None = None, *, refit: bool = False,
          place=layout, group=clusters) -> dict:  # fmt: skip
    """The shared map of ``poles``, keeping ``old``'s points unless ``refit`` (module docstring)."""
    raw = means(poles, encode)
    if old is None or refit:
        centre, points = _fit(poles, raw, old, place, group)
        version = (old["version"] + 1) if old else 1
    else:
        centre, points = _extend(poles, raw, old)
        version = old["version"]
    return _shared(poles, points, centre, version, old if not refit else None)


def _fit(poles: list[Pole], raw, old: dict | None, place, group):
    """A new fit: a pooled centre, a layout (from ``old``'s positions, if any) and regions."""
    centre = raw.mean(0)
    vectors, norms = centred(raw, centre)
    init = _start(poles, vectors, old) if old else None
    xy = place(vectors) if init is None else place(vectors, init=init)
    member = group(vectors)
    rows = zip(poles, xy, member, norms, strict=True)
    return centre, [_point(p, x, y, c, n) for p, (x, y), c, n in rows]


def _extend(poles: list[Pole], raw, old: dict):
    """``old``'s centre and points, with each new pole placed among its nearest old ones."""
    import numpy as np

    centre = np.asarray(old["centre"])
    vectors, norms = centred(raw, centre)
    kept = {p["slug"]: p for p in old["points"]}
    is_old = [p.slug in kept for p in poles]
    olds = [kept[p.slug] for p, o in zip(poles, is_old, strict=True) if o]
    placed = iter(place_new(vectors[is_old], olds, vectors[[not o for o in is_old]]))
    points = []
    for p, o, n in zip(poles, is_old, norms, strict=True):
        was = kept.get(p.slug)
        x, y, c = (was["x"], was["y"], was["cluster"]) if o else next(placed)
        points.append(_point(p, x, y, c, n))
    return centre, points


def _start(poles: list[Pole], vectors, old: dict):
    """A refit's starting positions: old poles where they were, new ones placed among them."""
    import numpy as np

    kept = {p["slug"]: p for p in old["points"]}
    is_old = np.array([p.slug in kept for p in poles])
    if not is_old.any():
        return None
    olds = [kept[p.slug] for p, o in zip(poles, is_old, strict=True) if o]
    new = iter(place_new(vectors[is_old], olds, vectors[~is_old]))
    start = [
        (kept[p.slug]["x"], kept[p.slug]["y"]) if o else next(new)[:2]
        for p, o in zip(poles, is_old, strict=True)
    ]
    return np.array(start, dtype=float)


def _shared(poles: list[Pole], points: list[dict], centre, version: int, old: dict | None) -> dict:
    """The file's fields: the fit, the points, the named regions and each model's reach."""
    from openinterpretability.concept_regions import coverage, name_clusters

    member = [p["cluster"] for p in points]
    regions = name_clusters(poles, member)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return {
        "version": version, "fitted": old["fitted"] if old else today, "date": today,
        "embedder": EMBEDDER, "models": sorted({p.model for p in poles}), "poles": len(poles),
        "clusters": regions, "coverage": coverage(points, [c["id"] for c in regions]),
        "points": points, "centre": [round(float(v), 6) for v in centre],
    }  # fmt: skip


def write(found: dict, path: Path = SHARED) -> None:
    """Write a map as JSON, one point to a line, so that a diff shows each moved or new pole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rest = {k: v for k, v in found.items() if k not in ("points", "centre")}
    head = json.dumps(rest, indent=1, ensure_ascii=False)[:-2]
    points = ",\n  ".join(json.dumps(p, ensure_ascii=False) for p in found["points"])
    centre = json.dumps(found["centre"])
    text = f'{head},\n "points": [\n  {points}\n ],\n "centre": {centre}\n}}\n'
    path.write_text(text, encoding="utf-8")


def load(path: Path = SHARED) -> dict | None:
    """A written map, or None when there is none."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
