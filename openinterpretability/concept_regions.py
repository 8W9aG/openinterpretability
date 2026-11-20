"""The regions of the shared concept map: their names, and which models reach them (#4586).

- **Names.** A region is named only from its VERIFIED members' explanations, from every
  model, by the words that set them apart (:func:`name_clusters`). A region without one
  is marked unverified.
- **Coverage.** For each region, the number of each model's poles in it. A model's
  coverage is the share of regions that hold at least one of its poles. The overlap of two
  models is the histogram intersection of their region shares. It is the share of poles
  that match region for region, and 1 when the two spread alike. The null shuffles the model
  labels across the poles :data:`SHUFFLES` times. It gives the overlap that two models of
  those sizes would show if they drew from one spread. An overlap near the null means the
  map can't tell the two models apart by region. ``p`` is the share of shuffles at or below
  the overlap seen.

The studied poles are not a fixed sample of each model, so coverage measures what the
registry has studied, not what the model holds.
"""

from __future__ import annotations

from collections import Counter

SHUFFLES = 200  # label shuffles for the overlap null
SEED = 4586


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


def name_clusters(poles: list, member: list[int]) -> list[dict]:
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


def counts(points: list[dict]) -> dict[str, Counter]:
    """For each model, its number of poles in each region."""
    out: dict[str, Counter] = {}
    for p in points:
        out.setdefault(p["model"], Counter())[p["cluster"]] += 1
    return out


def intersection(a: Counter, b: Counter) -> float:
    """The histogram intersection of two region counts, each made a share first."""
    ta, tb = sum(a.values()) or 1, sum(b.values()) or 1
    return sum(min(a[r] / ta, b[r] / tb) for r in set(a) | set(b))


def _pair(a: str, b: str, seen: float, null: list[float]) -> dict:
    """A pair's overlap, its null's mean and 2.5th percentile, and the one-sided p.

    ``p`` is the share of shuffles whose overlap is at most the one seen, counting the
    seen one too, so it is never 0.
    """
    low = sorted(null)[int(0.025 * len(null))]
    p = (1 + sum(n <= seen for n in null)) / (1 + len(null))
    return {"a": a, "b": b, "overlap": round(seen, 4), "null": round(sum(null) / len(null), 4),
            "null_low": round(low, 4), "p": round(p, 4)}  # fmt: skip


def coverage(points: list[dict], regions: list[int]) -> dict:
    """Each model's counts and coverage, and each pair's overlap with its shuffled null."""
    import random

    by_model = counts(points)
    models = sorted(by_model)
    rng = random.Random(SEED)
    labels = [p["model"] for p in points]
    nulls: dict[tuple[str, str], list[float]] = {}
    for _ in range(SHUFFLES):
        rng.shuffle(labels)
        shuffled = counts(
            [{"model": m, "cluster": p["cluster"]} for m, p in zip(labels, points, strict=True)]
        )
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                nulls.setdefault((a, b), []).append(intersection(shuffled[a], shuffled[b]))
    pairs = [
        _pair(a, b, intersection(by_model[a], by_model[b]), v)
        for (a, b), v in sorted(nulls.items())
    ]
    per_model = {m: {"poles": sum(c.values()), "regions": {str(r): c[r] for r in regions if c[r]},
                     "coverage": round(sum(1 for r in regions if c[r]) / len(regions), 4)}
                 for m, c in sorted(by_model.items())}  # fmt: skip
    return {"models": per_model, "pairs": pairs, "shuffles": SHUFFLES}
