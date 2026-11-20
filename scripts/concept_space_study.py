"""Do the models' studied poles spread alike over the shared concept map (#4586)?

It writes papers/shared-concept-space/results.jsonl. It needs the ``map`` extra and the
committed ``data/concept-maps/shared.json``. It embeds every studied pole again, centres it
on the map's stored centre, and measures each model's region counts at several partitions:

- the map's own regions, as the site draws them.
- Leiden partitions of the same graph, at the resolution nearest each of :data:`TARGETS`.

It also records each pole's place on the map and its first-study date, for the figures.
For each partition it records each model's coverage, and each pair's overlap with a
label-shuffled null. It also records each model's split-half overlap, the overlap of two
random halves of one model's poles. The split-half overlap is the ceiling that sampling
noise allows.

The scan corpus is a confound: models were scanned on different corpora over time. So
for :data:`CONTROLLED` partitions it repeats each pair within one corpus family. It also
compares one model's poles on two corpus families.

    uv run --extra map python scripts/concept_space_study.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from openinterpretability import concept_map as cm
from openinterpretability.concept_regions import SEED, SHUFFLES, coverage, intersection
from openinterpretability.planner import configured_models

OUT = Path("papers/shared-concept-space/results.jsonl")
TARGETS = (15, 30, 60)  # finer partitions than the map's
RESOLUTIONS = [round(0.2 * i, 1) for i in range(1, 76)]  # 0.2 to 15.0
FAMILIES = ("mixed", "NeelNanda/pile-10k")  # the scan corpora that most models share
MIN_POLES = 40  # a model's poles on one corpus, for it to enter that corpus's comparison
CONTROLLED = ("map", "leiden-30")  # the partitions that the corpus control repeats


def split_half(points: list[dict], model: str, rng: random.Random) -> float:
    """The mean overlap of two random halves of one model's poles."""
    mine = [p["cluster"] for p in points if p["model"] == model]
    found = []
    for _ in range(SHUFFLES):
        rng.shuffle(mine)
        half = len(mine) // 2
        found.append(intersection(Counter(mine[:half]), Counter(mine[half:])))
    return sum(found) / len(found)


def partition(vectors, target: int) -> list[int]:
    """Leiden communities of the map's k-nearest-neighbour graph, nearest ``target`` in count."""
    old = cm.RESOLUTIONS
    cm.RESOLUTIONS = RESOLUTIONS
    try:
        return cm.clusters(vectors, target=target)
    finally:
        cm.RESOLUTIONS = old


def rows_for(name: str, points: list[dict]) -> list[dict]:
    """The result rows of one partition."""
    regions = sorted({p["cluster"] for p in points})
    found = coverage(points, regions)
    rng = random.Random(SEED)
    out = [{"kind": "partition", "partition": name, "regions": len(regions)}]
    for model, row in found["models"].items():
        out.append({"kind": "model", "partition": name, "model": model, "poles": row["poles"],
                    "coverage": row["coverage"], "reached": len(row["regions"]),
                    "counts": row["regions"],
                    "split_half": round(split_half(points, model, rng), 4)})  # fmt: skip
    out += [{"kind": "pair", "partition": name, **pair} for pair in found["pairs"]]
    return out


def corpus_rows(name: str, points: list[dict]) -> list[dict]:
    """Pairs of models within one corpus family, and one model across two families."""
    out = []
    for family in FAMILIES:
        size = Counter(p["model"] for p in points if p["corpus"] == family)
        mine = [p for p in points if p["corpus"] == family and size[p["model"]] >= MIN_POLES]
        found = coverage(mine, sorted({p["cluster"] for p in points}))
        out += [{"kind": "within", "partition": name, "corpus": family, **pair}
                for pair in found["pairs"]]  # fmt: skip
    for model in sorted({p["model"] for p in points}):
        sides = [[p for p in points if p["model"] == model and p["corpus"] == f] for f in FAMILIES]
        if min(map(len, sides)) < MIN_POLES:
            continue
        mine = [{"model": p["corpus"], "cluster": p["cluster"]} for p in sides[0] + sides[1]]
        (pair,) = coverage(mine, sorted({p["cluster"] for p in points}))["pairs"]
        out.append({"kind": "across", "partition": name, "model": model, **pair})
    return out


def main() -> None:
    """Embed every mapped pole again, measure each partition and write the results."""
    from sentence_transformers import SentenceTransformer

    shared = cm.load()
    poles = cm.studied_poles(set(configured_models()), Path("wiki"))
    kept = {p["slug"]: p for p in shared["points"]}
    poles = [p for p in poles if p.slug in kept]
    embedder = SentenceTransformer(cm.EMBEDDER, device="cpu")
    raw = cm.means(poles, lambda texts: embedder.encode(texts, normalize_embeddings=True))
    vectors, _ = cm.centred(raw, shared["centre"])
    base = [{"model": p.model, "cluster": kept[p.slug]["cluster"], "corpus": p.corpus}
            for p in poles]  # fmt: skip
    rows = [{"kind": "map", "version": shared["version"], "fitted": shared["fitted"],
             "poles": len(poles), "embedder": shared["embedder"]}]  # fmt: skip
    rows += rows_for("map", base) + corpus_rows("map", base)
    rows += [{"kind": "region", "id": c["id"], "name": c["name"], "size": c["size"]}
             for c in shared["clusters"]]  # fmt: skip
    rows += [{"kind": "point", "model": p.model, "x": kept[p.slug]["x"], "y": kept[p.slug]["y"],
              "first": kept[p.slug]["first"]} for p in poles]  # fmt: skip
    for target in TARGETS:
        member = partition(vectors, target)
        cut = [
            {"model": p.model, "cluster": c, "corpus": p.corpus}
            for p, c in zip(poles, member, strict=True)
        ]
        rows += rows_for(f"leiden-{target}", cut)
        if f"leiden-{target}" in CONTROLLED:
            rows += corpus_rows(f"leiden-{target}", cut)
        print(f"target {target}: {len(set(member))} regions", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} rows → {OUT}")


if __name__ == "__main__":
    main()
