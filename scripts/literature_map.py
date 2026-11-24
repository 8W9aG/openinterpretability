"""Lay the literature reviews out on a map by what they are about (issues #3294 and #4577).

Each review in docs/literature/ is embedded from its title and its *Summary* section. Each
queued paper (an open `area: literature` issue) is embedded from its title and the first
paragraph of the issue. The model is all-MiniLM-L6-v2, which did best in the concept-map study
(#3290). UMAP lays every paper out in two dimensions. Close points are
papers about similar things. The axes themselves have no meaning.

The papers' :data:`KNN`-nearest-neighbour graph is split into Leiden communities. This is
the method of Alexander Coulton's map of cancer research, which issue #4577 cites as the
model. A community with fewer than :data:`MIN_REVIEWS` reviews merges into its closest one.
So queued papers join their topic, and there is no "queued" region (#5798). There are at most
:data:`SLOTS` regions, one per palette colour. Each point then takes the most common region
of its :data:`LOCAL` nearest points on the map, so that each region is one cloud. Each of
:data:`SEEDS` gives a layout, and the most compact one is kept (#6529).

Regions are numbered from 1, biggest first. Each is named by the section of
docs/literature/README.md that most of its reviews are under. ``map.json`` also gives the
map's *compactness*, and the regions that each region touches. Each touch has its count of
neighbour links, which chooses the colours (``builder/cluster_map.touching_slots``). Each review
keeps its :data:`NEIGHBOURS` nearest reviews by cosine, which the map links on hover.

The heavy libraries stay out of the project. Run it after adding or changing a review::

    uv run --extra map python scripts/literature_map.py

It writes docs/literature/map.json, which the site's literature index draws. It needs `gh`
to read the queue; without it, the map has no queued papers.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from literature_map_io import MIN_COMPACT, plain_lead, queued, report  # noqa: E402, F401

from openinterpretability.builder.cluster_map import SLOTS  # noqa: E402
from openinterpretability.builder.literature_pages import (  # noqa: E402
    LITERATURE,
    MAP,
    map_text,
    sections,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# UMAP's and Leiden's seeds, tried in order: one page can reshuffle one seed's map (#6529).
SEEDS = (4577, 1, 2, 3, 4, 5, 6, 7)
RESOLUTIONS = [round(0.8 + 0.1 * i, 1) for i in range(15)]  # 0.8 to 2.2, searched in order
KNN = 10  # neighbours per paper in the graph that Leiden clusters
MIN_REVIEWS = 3  # a community with fewer reviews merges into its closest one (#5798)
LOCAL = 8  # neighbours on the 2-D map that smooth a point's region and say which regions touch
NEIGHBOURS = 3


def layout(vectors, seed: int):
    """UMAP coordinates of ``vectors``, scaled into [0, 1] on each axis.

    Args:
        vectors: The papers' embeddings.
        seed: UMAP's seed.

    Returns:
        The points.
    """
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=seed)
    points = reducer.fit_transform(vectors)
    return (points - points.min(0)) / (points.max(0) - points.min(0))


def knn_edges(vectors) -> dict[tuple[int, int], float]:
    """The papers' :data:`KNN`-nearest-neighbour graph: each edge (low, high) and its weight.

    Args:
        vectors: The papers' normalised embeddings.

    Returns:
        The cosine of each edge, at least 0, keyed by the pair of paper indices.
    """
    import numpy as np

    similar = vectors @ vectors.T
    np.fill_diagonal(similar, -1)
    edges: dict[tuple[int, int], float] = {}
    for i, row in enumerate(similar):
        for j in [int(j) for j in np.argsort(-row) if j != i][:KNN]:
            key = (min(i, j), max(i, j))
            edges[key] = max(edges.get(key, 0.0), float(max(row[j], 0.0)))
    return edges


def merge_small(labels: list[int], edges: dict, reviewed: int) -> list[int]:
    """Dissolve each community with fewer than :data:`MIN_REVIEWS` reviews, paper by paper.

    A community of mostly queued papers can't be named from the index (#5798). The smallest
    such community goes first. Each of its papers joins the community that it shares the
    most edge weight with, so the papers spread over their own topics.

    Args:
        labels: Each paper's community.
        edges: The graph of :func:`knn_edges`.
        reviewed: The number of reviews, which come first in ``labels``.

    Returns:
        The labels after merging.
    """
    labels = list(labels)
    while True:
        reviews = Counter(labels[:reviewed])
        small = sorted((k for k in set(labels) if reviews[k] < MIN_REVIEWS),
                       key=lambda k: (labels.count(k), k))  # fmt: skip
        if not small or len(set(labels)) == 1:
            return labels
        labels = _dissolve(labels, edges, small[0])


def _dissolve(labels: list[int], edges: dict, k: int) -> list[int]:
    """Move each paper of community ``k`` to the other community it is most tied to.

    Papers with no tie outside ``k`` wait for their neighbours to move first. If none can
    move, the rest join the biggest other community.

    Args:
        labels: Each paper's community.
        edges: The graph of :func:`knn_edges`.
        k: The community to dissolve.

    Returns:
        The labels with no paper left in ``k``.
    """
    labels = list(labels)
    while k in labels:
        ties: dict[int, Counter] = {}
        for (i, j), w in edges.items():
            for a, b in ((i, j), (j, i)):
                if labels[a] == k and labels[b] != k:
                    ties.setdefault(a, Counter())[labels[b]] += w
        if not ties:
            biggest = Counter(lab for lab in labels if lab != k).most_common(1)[0][0]
            return [biggest if lab == k else lab for lab in labels]
        for i, tie in ties.items():
            labels[i] = tie.most_common(1)[0][0]
    return labels


def communities(vectors, reviewed: int, seed: int) -> list[int]:
    """Leiden communities of the papers' graph, merged until each can be named.

    A fixed resolution gave 10 regions, then 13 once one review was added. So each of
    :data:`RESOLUTIONS` is tried. The first count nearest :data:`SLOTS`, but not over it,
    wins, so that every region gets a palette colour (#5798).

    Args:
        vectors: The papers' normalised embeddings.
        reviewed: The number of reviews, which come first.
        seed: Leiden's seed.

    Returns:
        Each paper's community.
    """
    import igraph
    import leidenalg

    edges = knn_edges(vectors)
    graph = igraph.Graph(n=len(vectors), edges=list(edges))
    graph.es["weight"] = list(edges.values())
    best: list[int] | None = None
    for resolution in RESOLUTIONS:
        part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                        weights="weight", resolution_parameter=resolution,
                                        seed=seed)  # fmt: skip
        labels = merge_small(list(part.membership), edges, reviewed)
        if len(set(labels)) <= SLOTS and (
            best is None or SLOTS - len(set(labels)) < SLOTS - len(set(best))
        ):
            best = labels
    return best or [0] * len(vectors)


def smooth(points, labels: list[int]) -> list[int]:
    """Give each point the most common region of its :data:`LOCAL` nearest points on the map.

    Leiden groups papers by their embeddings, so a region can fall in several places of
    the 2-D layout (#5798). Repeated until no label changes, at most 20 times. Ties keep
    the point's own region.

    Args:
        points: The 2-D layout.
        labels: Each point's region.

    Returns:
        The smoothed labels.
    """
    import numpy as np

    near = np.argsort(((points[:, None] - points[None]) ** 2).sum(-1), axis=1)[:, : LOCAL + 1]
    labels = list(labels)
    for _ in range(20):
        votes = [Counter(labels[j] for j in row) for row in near]
        new = [max(v, key=lambda k: (v[k], k == labels[i])) for i, v in enumerate(votes)]
        if new == labels:
            break
        labels = new
    return labels


def absorb_small(points, labels: list[int], reviewed: int) -> list[int]:
    """Merge each region that :func:`smooth` left with fewer than :data:`MIN_REVIEWS` reviews.

    Smoothing can empty a community of reviews, which leaves it with no name (#5810). Its
    points join the region of their nearest point outside it (:func:`merge_region`).

    Args:
        points: The 2-D layout.
        labels: Each point's region, after smoothing.
        reviewed: The number of reviews, which come first in ``labels``.

    Returns:
        The labels after merging.
    """
    import numpy as np

    order = np.argsort(((points[:, None] - points[None]) ** 2).sum(-1), axis=1)
    labels = list(labels)
    while True:
        reviews = Counter(labels[:reviewed])
        small = sorted((k for k in set(labels) if reviews[k] < MIN_REVIEWS),
                       key=lambda k: (labels.count(k), k))  # fmt: skip
        if not small or len(set(labels)) == 1:
            return labels
        labels = merge_region(order, labels, small[0])


def merge_region(order, labels: list[int], k: int) -> list[int]:
    """Move each point of region ``k`` to the region of its nearest point outside ``k``.

    Args:
        order: Each point's other points, nearest first (an argsort of the distances).
        labels: Each point's region.
        k: The region to merge away.

    Returns:
        The labels with no point left in ``k``.
    """
    return [next(labels[j] for j in order[i] if labels[j] != k) if lab == k else lab
            for i, lab in enumerate(labels)]  # fmt: skip


def named_regions(points, labels: list[int], names: list[str], reviewed: int):
    """:func:`regions`, after merging each region that no free section can name (#6030).

    A small region whose sections bigger regions took has no name, so it merges too.

    Args:
        points: The 2-D layout.
        labels: Each point's region.
        names: The reviews' names, in the order of the points.
        reviewed: The number of reviews, which come first.

    Returns:
        :func:`regions` of the merged labels.
    """
    import numpy as np

    order = np.argsort(((points[:, None] - points[None]) ** 2).sum(-1), axis=1)
    while True:
        region_of, rows = regions(points, labels, names, reviewed)
        unnamed = [r["number"] for r in rows if r["name"] == f"Region {r['number']}"]
        if not unnamed or len(rows) == 1:
            return region_of, rows
        labels = merge_region(order, region_of, unnamed[0])


def compactness(points, labels: list[int]) -> float:
    """The share of points whose nearest region centre is their own region's (#5798).

    Args:
        points: The 2-D layout.
        labels: Each point's region.

    Returns:
        A share from 0 to 1. Every point in one tight cloud per region gives 1.
    """
    import numpy as np

    keys = sorted(set(labels))
    arr = np.asarray(labels)
    centres = np.stack([points[arr == k].mean(0) for k in keys])
    nearest = ((points[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
    own = [keys[n] == lab for n, lab in zip(nearest, labels, strict=True)]
    return round(float(np.mean(own)), 4)


def touching(points, labels: list[int]) -> dict[int, Counter]:
    """The regions that each region touches, and how closely: their shared neighbour links.

    Each point links to its :data:`LOCAL` nearest points on the map. A link between two
    regions counts for both of them.

    Args:
        points: The 2-D layout.
        labels: Each point's region.

    Returns:
        For each region, the count of links to each other region.
    """
    import numpy as np

    near = np.argsort(((points[:, None] - points[None]) ** 2).sum(-1), axis=1)[:, 1 : LOCAL + 1]
    out: dict[int, Counter] = {k: Counter() for k in labels}
    for i, row in enumerate(near):
        for j in row:
            if labels[j] != labels[i]:
                out[labels[i]][labels[j]] += 1
                out[labels[j]][labels[i]] += 1
    return out


TIE = " · "  # joins the sections of a region whose reviews split evenly between them


def region_name(counts: Counter, taken: set[str]) -> str:
    """The free section that most of a region's reviews are under, and add it to ``taken``.

    When free sections tie, the region is named after all of them, in order of name, joined
    by :data:`TIE`. So a region that mixes two topics says so (#5798).

    Args:
        counts: The region's reviews, by README section. None is a review in no section.
        taken: The sections that bigger regions took already.

    Returns:
        The region's name, or "" when no section is free.

    Examples:
        >>> taken = {"Probing"}
        >>> region_name(Counter({"Probing": 6, "B": 5, "A": 5}), taken)
        'A · B'
        >>> sorted(taken)
        ['A', 'B', 'Probing']
    """
    free = [(c, s) for s, c in counts.most_common() if s and s not in taken]
    if not free:
        return ""
    tied = sorted(s for c, s in free if c == free[0][0])
    taken.update(tied)
    return TIE.join(tied)


def regions(points, labels: list[int], names: list[str], reviewed: int):
    """Each point's region number, and each region's number, name, centre, size and neighbours.

    Regions are numbered from 1, biggest first. The first ``reviewed`` points are reviews,
    with ``names``. A region is named by the README section that most of its reviews are
    under (:func:`region_name`), and the biggest region chooses first (#5798).

    Args:
        points: The 2-D layout.
        labels: Each point's region.
        names: The reviews' page names.
        reviewed: The number of reviews, which come first.

    Returns:
        Each point's region number, and one row per region.
    """
    sizes = Counter(labels)
    number = {k: n for n, (k, _) in enumerate(sizes.most_common(), 1)}
    by_review = sections(LITERATURE)
    touches = touching(points, labels)
    taken: set[str] = set()
    rows = []

    def by_number(links: Counter) -> list:
        """``links`` as (region, count) pairs, in the order of the regions' numbers.

        Args:
            links: The count of links to each other region, by Leiden label.

        Returns:
            The pairs, sorted by the regions' numbers.
        """
        return sorted(links.items(), key=lambda tc: number[tc[0]])

    for k, n in sorted(number.items(), key=lambda kv: kv[1]):
        members = [i for i, lab in enumerate(labels) if lab == k]
        counts = Counter(by_review.get(names[i]) for i in members if i < reviewed)
        name = region_name(counts, taken) or f"Region {n}"
        x, y = points[members].mean(0)
        rows.append({"number": n, "name": name, "x": round(float(x), 4),
                     "y": round(float(y), 4), "size": len(members),
                     "touches": {str(number[t]): c for t, c in by_number(touches[k])}})  # fmt: skip
    return [number[k] for k in labels], rows


def neighbours(vectors, names: list[str]) -> list[list[str]]:
    """Each review's :data:`NEIGHBOURS` nearest reviews by cosine, nearest first."""
    import numpy as np

    similar = vectors @ vectors.T
    np.fill_diagonal(similar, -1)
    return [[names[j] for j in np.argsort(-row)[:NEIGHBOURS]] for row in similar]


def arrange(vectors, names: list[str], reviewed: int, seed: int) -> tuple:
    """One layout of the papers with ``seed``.

    Args:
        vectors: The papers' normalised embeddings, the reviews first.
        names: The reviews' names.
        reviewed: The number of reviews.
        seed: The seed of UMAP and Leiden.

    Returns:
        The seed, the points, each paper's region, the regions' rows and the compactness.
    """
    import numpy as np

    points = layout(vectors, seed)
    labels = smooth(points, communities(np.asarray(vectors), reviewed, seed))
    labels = absorb_small(points, labels, reviewed)  # smoothing can empty a region
    region_of, region_rows = named_regions(points, labels, names, reviewed)
    return seed, points, region_of, region_rows, compactness(points, region_of)


def most_compact(vectors, names: list[str], reviewed: int, seeds=SEEDS, build=arrange) -> tuple:
    """The most compact of the layouts that :data:`SEEDS` give; a tie keeps the earlier seed.

    Args:
        vectors: The papers' normalised embeddings, the reviews first.
        names: The reviews' names.
        reviewed: The number of reviews.
        seeds: The seeds, in order.
        build: Makes one layout (:func:`arrange`).

    Returns:
        The kept layout, as :func:`arrange` returns it.
    """
    best = None
    for seed in seeds:
        tried = build(vectors, names, reviewed, seed)
        if best is None or tried[-1] > best[-1]:
            best = tried
    return best


def main() -> None:
    """Embed the reviews and the queue, lay them out, group them, and write the map."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    reviews = [p for p in sorted(LITERATURE.glob("*.md")) if map_text(p) is not None]
    texts = [map_text(p) for p in reviews]
    queue = queued()
    model = SentenceTransformer(MODEL, device="cpu")
    vectors = model.encode(texts + [q["text"] for q in queue], normalize_embeddings=True)
    names = [p.stem for p in reviews]
    seed, points, region_of, region_rows, compact = most_compact(vectors, names, len(reviews))
    near = neighbours(np.asarray(vectors[: len(reviews)]), names)
    rows = [
        {"name": name, "x": round(float(x), 4), "y": round(float(y), 4),
         "sha256": hashlib.sha256(text.encode()).hexdigest(), "region": region,
         "neighbours": close}
        for name, text, (x, y), region, close in
        zip(names, texts, points, region_of, near, strict=False)
    ]  # fmt: skip
    queue_rows = [
        {"number": q["number"], "title": q["title"], "x": round(float(x), 4),
         "y": round(float(y), 4), "region": region}
        for q, (x, y), region in
        zip(queue, points[len(reviews):], region_of[len(reviews):], strict=True)
    ]  # fmt: skip
    out = LITERATURE / MAP
    data = {"model": MODEL, "layout": f"umap seed {seed}",
            "compactness": compact, "regions": region_rows,
            "reviews": rows, "queued": queue_rows}  # fmt: skip
    out.write_text(json.dumps(data, indent=1) + "\n")
    report(data, out)


if __name__ == "__main__":
    main()
