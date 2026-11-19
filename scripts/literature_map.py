"""Lay the literature reviews out on a map by what they are about (issues #3294 and #4577).

Each review in docs/literature/ is embedded from its title and its *Summary* section. Each
queued paper (an open `area: literature` issue) is embedded from its title and the first
paragraph of the issue. The model is all-MiniLM-L6-v2, which did best in the concept-map study
(#3290). UMAP, with a fixed seed, lays every paper out in two dimensions. Close points are
papers about similar things. The axes themselves have no meaning.

The papers' :data:`KNN`-nearest-neighbour graph is split into Leiden communities, at most
:data:`RESOLUTION`, numbered from 1, biggest first. This is the method of Alexander Coulton's
map of cancer research, which issue #4577 cites as the model. Each region is named by the
section of docs/literature/README.md that most of its reviews are under. A region with fewer
than :data:`MIN_REVIEWS` reviews is mostly queued papers, and is called :data:`QUEUED`. Each review
also keeps its :data:`NEIGHBOURS` nearest reviews by cosine, which the map links on hover.

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

from openinterpretability.builder.literature_pages import (  # noqa: E402
    LITERATURE,
    MAP,
    map_text,
    sections,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 4577
RESOLUTION = 1.5  # Leiden's: about ten regions for some 300 papers
KNN = 10  # neighbours per paper in the graph that Leiden clusters
MIN_REVIEWS = 3  # a region with fewer reviews is mostly queued papers, and isn't named
NEIGHBOURS = 3
REPO = "8W9aG/openinterpretability"


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
                     "text": f"{title}\n\n{' '.join(lead)}".strip()})  # fmt: skip
    return sorted(rows, key=lambda r: r["number"])


def layout(vectors):
    """UMAP coordinates of ``vectors``, scaled into [0, 1] on each axis."""
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.08, metric="cosine", random_state=SEED)
    points = reducer.fit_transform(vectors)
    return (points - points.min(0)) / (points.max(0) - points.min(0))


def communities(vectors) -> list[int]:
    """Leiden communities of the papers' k-nearest-neighbour graph, at :data:`RESOLUTION`."""
    import igraph
    import leidenalg
    import numpy as np

    similar = vectors @ vectors.T
    np.fill_diagonal(similar, -1)
    edges, weights = [], []
    for i, row in enumerate(similar):
        for j in np.argsort(-row)[:KNN]:
            edges.append((i, int(j)))
            weights.append(float(max(row[j], 0.0)))
    graph = igraph.Graph(n=len(vectors), edges=edges)
    graph.es["weight"] = weights
    graph = graph.simplify(combine_edges="max")
    part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                    weights="weight", resolution_parameter=RESOLUTION,
                                    seed=SEED)  # fmt: skip
    return list(part.membership)


QUEUED = "Queued, not reviewed yet"  # the name of a region with too few reviews to name


def regions(points, labels: list[int], names: list[str], reviewed: int):
    """Each point's region number, and each region's number, name, centre and size.

    Regions are numbered from 1, biggest first. The first ``reviewed`` points are reviews,
    with ``names``. A region is named by the README section that most of its reviews are
    under, and the biggest region chooses first. With fewer than :data:`MIN_REVIEWS` reviews,
    it is mostly queued papers on mixed topics, and is called :data:`QUEUED` (``named`` False).
    """
    sizes = Counter(labels)
    number = {k: n for n, (k, _) in enumerate(sizes.most_common(), 1)}
    by_review = sections(LITERATURE)
    taken: set[str] = set()
    rows = []
    for k, n in sorted(number.items(), key=lambda kv: kv[1]):
        members = [i for i, lab in enumerate(labels) if lab == k]
        counts = Counter(by_review.get(names[i]) for i in members if i < reviewed)
        choices = [s for s, _ in counts.most_common() if s and s not in taken]
        named = counts.total() >= MIN_REVIEWS and bool(choices)
        name = choices[0] if named else QUEUED
        taken.add(name)
        x, y = points[members].mean(0)
        rows.append({"number": n, "name": name, "named": named, "x": round(float(x), 4),
                     "y": round(float(y), 4), "size": len(members)})  # fmt: skip
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
    labels = communities(np.asarray(vectors))
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
    data = {"model": MODEL, "layout": f"umap seed {SEED}", "regions": region_rows,
            "reviews": rows, "queued": queue_rows}  # fmt: skip
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(rows)} reviews, {len(queue_rows)} queued, {len(region_rows)} regions → {out}")


if __name__ == "__main__":
    main()
