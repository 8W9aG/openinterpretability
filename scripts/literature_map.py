"""Lay the literature reviews out on a map by what they are about (issues #3294 and #4577).

Each review in docs/literature/ is embedded from its title and its *Summary* section. Each
queued paper (an open `area: literature` issue) is embedded from its title and the first
paragraph of the issue. The model is all-MiniLM-L6-v2, which did best in the concept-map study
(#3290). UMAP, with a fixed seed, lays every paper out in two dimensions. Close points are
papers about similar things. The axes themselves have no meaning.

The papers' :data:`KNN`-nearest-neighbour graph is split into Leiden communities. This is
the method of Alexander Coulton's map of cancer research, which issue #4577 cites as the
model. A community with fewer than :data:`MIN_REVIEWS` reviews merges into its closest one.
So queued papers join their topic, and there is no "queued" region (#5798). There are at most
:data:`SLOTS` regions, one per palette colour. Each point then takes the most common region
of its :data:`LOCAL` nearest points on the map, so that each region is one cloud.

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
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.builder.cluster_map import SLOTS  # noqa: E402
from openinterpretability.builder.literature_pages import (  # noqa: E402
    LITERATURE,
    MAP,
    map_text,
    sections,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 4577
RESOLUTIONS = [round(0.8 + 0.1 * i, 1) for i in range(15)]  # 0.8 to 2.2, searched in order
KNN = 10  # neighbours per paper in the graph that Leiden clusters
MIN_REVIEWS = 3  # a community with fewer reviews merges into its closest one (#5798)
LOCAL = 8  # neighbours on the 2-D map that smooth a point's region and say which regions touch
MIN_COMPACT = 0.8  # the least compactness the map may have (tests/test_literature_map.py)
NEIGHBOURS = 3
REPO = "8W9aG/openinterpretability"


# Words that every queued issue shares, which would pull the queue into one cluster (#5798).
BOILERPLATE = re.compile(
    r"\*\*(Off-topic|Bears on):\*\*|Write a short not-applicable page\.", re.IGNORECASE
)


def plain_lead(text: str) -> str:
    """``text`` without the issue template's markers and stock sentence.

    Args:
        text: The issue's lead paragraph.

    Returns:
        The paragraph's own words.

    Examples:
        >>> plain_lead("**Off-topic:** World models. Write a short not-applicable page.")
        'World models.'
    """
    return " ".join(BOILERPLATE.sub("", text).split())


def queued() -> list[dict]:
    """The open `area: literature` issues: number, title and the text to embed."""
    try:
        out = subprocess.run(
            ["gh", "issue", "list", "--repo", REPO, "--label", "area: literature", "--state",
             "open", "--limit", "500", "--json", "number,title,body"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for issue in json.loads(out):
        title = re.sub(r"^Review\s+", "", issue["title"]).strip()
        lead = (issue["body"] or "").split("\n\n")[1:2]  # the paragraph after the source line
        rows.append({"number": issue["number"], "title": title,
                     "text": f"{title}\n\n{plain_lead(' '.join(lead))}".strip()})  # fmt: skip
    return sorted(rows, key=lambda r: r["number"])


def layout(vectors):
    """UMAP coordinates of ``vectors``, scaled into [0, 1] on each axis."""
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=SEED)
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


def communities(vectors, reviewed: int) -> list[int]:
    """Leiden communities of the papers' graph, merged until each can be named.

    A fixed resolution gave 10 regions, then 13 once one review was added. So each of
    :data:`RESOLUTIONS` is tried. The first count nearest :data:`SLOTS`, but not over it,
    wins, so that every region gets a palette colour (#5798). Leiden's seed is fixed.

    Args:
        vectors: The papers' normalised embeddings.
        reviewed: The number of reviews, which come first.

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
                                        seed=SEED)  # fmt: skip
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

    Smoothing can move every review out of a community that :func:`merge_small` kept. A
    region of queued papers only then has no name (#5810). Each point of the smallest such
    region joins the region of its nearest point on the map outside it.

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
        k = small[0]
        labels = [next(labels[j] for j in order[i] if labels[j] != k) if lab == k else lab
                  for i, lab in enumerate(labels)]  # fmt: skip


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
    under (:func:`region_name`), and the biggest region chooses first. :func:`merge_small`
    gave each region at least :data:`MIN_REVIEWS` reviews, so each has a name (#5798).

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


def main() -> None:
    """Embed the reviews and the queue, lay them out, group them, and write the map."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    reviews = [p for p in sorted(LITERATURE.glob("*.md")) if map_text(p) is not None]
    texts = [map_text(p) for p in reviews]
    queue = queued()
    model = SentenceTransformer(MODEL, device="cpu")
    vectors = model.encode(texts + [q["text"] for q in queue], normalize_embeddings=True)
    points = layout(vectors)
    names = [p.stem for p in reviews]
    labels = smooth(points, communities(np.asarray(vectors), len(reviews)))
    labels = absorb_small(points, labels, len(reviews))  # smoothing can empty a region
    region_of, region_rows = regions(points, labels, names, len(reviews))
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
    data = {"model": MODEL, "layout": f"umap seed {SEED}",
            "compactness": compactness(points, region_of), "regions": region_rows,
            "reviews": rows, "queued": queue_rows}  # fmt: skip
    out.write_text(json.dumps(data, indent=1) + "\n")
    report(data, out)


def report(data: dict, out: Path) -> None:
    """Print what the map holds, and warn when its regions are not compact enough.

    Args:
        data: The map, as written.
        out: Where it was written.
    """
    print(
        f"{len(data['reviews'])} reviews, {len(data['queued'])} queued, "
        f"{len(data['regions'])} regions, compactness {data['compactness']} → {out}"
    )
    if data["compactness"] < MIN_COMPACT:
        print(
            f"Compactness is under {MIN_COMPACT}, so tests/test_literature_map.py fails. "
            "Look at the layout before you commit it.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
