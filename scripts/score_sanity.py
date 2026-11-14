"""Label sanity checks on the metrics of `openinterp score` (issue #3237).

"Evaluating Neuron Explanations" (Oikarinen et al., arXiv 2506.05774) gives two tests.
An explanation metric must drop when the explanation's labels lose part of the concept
(missing labels). It must also drop when they claim more than the concept (extra labels).
Here the judge's stored answers play the part of the explanation's labels, and the
measured window labels stay fixed. For each stored
detection and fuzzing score, the script perturbs the answers at several rates and records
how each metric changes. Write-up: docs/analyses/score-sanity.md.

    uv run python scripts/score_sanity.py [--out docs/analyses/score-sanity.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from explanation_free_rules import scored_records  # noqa: E402

RATES = (0.1, 0.2, 0.3, 0.5)
DRAWS = 20  # seeded perturbations per record and rate
SEED = 3237

Pairs = Sequence[tuple[bool, bool]]  # (answer, label)


def _rates(pairs: Pairs) -> tuple[int, int, int, int]:
    """(true positives, false positives, false negatives, true negatives)."""
    tp = sum(a and y for a, y in pairs)
    fp = sum(a and not y for a, y in pairs)
    fn = sum(not a and y for a, y in pairs)
    return tp, fp, fn, len(pairs) - tp - fp - fn


def balanced(pairs: Pairs) -> float | None:
    """Mean of recall and specificity: the metric that `openinterp score` reports."""
    tp, fp, fn, tn = _rates(pairs)
    if not tp + fn or not fp + tn:
        return None
    return (tp / (tp + fn) + tn / (tn + fp)) / 2


def recall(pairs: Pairs) -> float | None:
    """The share of active windows that the answers call active."""
    tp, _, fn, _ = _rates(pairs)
    return tp / (tp + fn) if tp + fn else None


def precision(pairs: Pairs) -> float | None:
    """The share of windows called active that are active."""
    tp, fp, _, _ = _rates(pairs)
    return tp / (tp + fp) if tp + fp else None


def f1(pairs: Pairs) -> float | None:
    """The harmonic mean of precision and recall."""
    tp, fp, fn, _ = _rates(pairs)
    return 2 * tp / (2 * tp + fp + fn) if tp + fn else None


METRICS: dict[str, Callable[[Pairs], float | None]] = {
    "balanced_accuracy": balanced,
    "recall": recall,
    "precision": precision,
    "f1": f1,
}


def perturb(pairs: Pairs, test: str, rate: float, rng: random.Random) -> list[tuple[bool, bool]]:
    """Flip each answer that ``test`` names with probability ``rate``.

    ``missing`` turns "active" answers to "inactive". ``extra`` turns "inactive" answers
    to "active". The labels never change. Each answer flips on its own draw, so a score
    with few such answers is still perturbed on average.
    """
    target = test == "missing"  # the answer value that the test flips
    return [(not a if a == target and rng.random() < rate else a, y) for a, y in pairs]


def beats_chance(pairs: Pairs) -> bool:
    """Whether the answers call active windows active more often than inactive ones.

    For balanced accuracy, both tests change the score by −rate·(recall − false-positive
    rate)/2 on average. So the score drops exactly when this holds.
    """
    tp, fp, fn, tn = _rates(pairs)
    return bool(tp + fn and fp + tn) and tp / (tp + fn) > fp / (fp + tn)


def check(scores: Sequence[Pairs]) -> dict:
    """Per test, rate and metric: the mean change and the share of scores that drop.

    A score counts when the metric is defined before and after. A tie counts as no drop.
    """
    rng = random.Random(SEED)
    out: dict = {}
    for test in ("missing", "extra"):
        out[test] = {}
        for rate in RATES:
            row = {}
            for name, metric in METRICS.items():
                changes: dict[bool, list[float]] = {True: [], False: []}
                for pairs in scores:
                    before = metric(pairs)
                    after = [metric(perturb(pairs, test, rate, rng)) for _ in range(DRAWS)]
                    after = [a for a in after if a is not None]
                    if before is not None and after:
                        changes[beats_chance(pairs)].append(sum(after) / len(after) - before)
                row[name] = summarise(changes[True] + changes[False]) | {
                    "beats_chance": summarise(changes[True]),
                    "at_or_below_chance": summarise(changes[False]),
                }
            out[test][str(rate)] = row
    return out


def summarise(changes: list[float]) -> dict:
    """The number of scores, their mean change and the share that drop (a tie is no drop)."""
    n = len(changes)
    return {
        "scores": n,
        "mean_change": round(sum(changes) / n, 4) if n else None,
        "share_dropping": round(sum(c < -1e-12 for c in changes) / n, 4) if n else None,
    }


def pairs_of(windows) -> list[tuple[bool, bool]]:
    """(answer, label) for each judged window."""
    return [(w.judged, w.label) for w in windows if w.judged is not None]


def dump(value, depth: int = 0, inline_at: int = 3) -> str:
    """JSON indented down to ``inline_at`` levels, with each deeper value on one line.

    That keeps one rate of one test, with every metric, on one line of the file.
    """
    if depth >= inline_at or not isinstance(value, dict):
        return json.dumps(value)
    pad = " " * (depth + 1)
    items = [f"{pad}{json.dumps(k)}: {dump(v, depth + 1, inline_at)}" for k, v in value.items()]
    return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"


def main() -> None:
    """Run both tests on every stored detection and fuzzing score, and write the JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=Path("docs/analyses/score-sanity.json"))
    args = ap.parse_args()
    records = scored_records()
    result = {"records": len(records), "rates": RATES, "draws": DRAWS, "seed": SEED}
    for kind, attr in (("detection", "detection_windows"), ("fuzzing", "fuzzing_windows")):
        scores = [p for r in records if (p := pairs_of(getattr(r, attr)))]
        result[kind] = {"scores": len(scores), **check(scores)}
    args.out.write_text(dump(result) + "\n")
    for kind in ("detection", "fuzzing"):
        for test in ("missing", "extra"):
            row = result[kind][test]["0.2"]
            cells = ", ".join(
                f"{m} {v['mean_change']:+.3f} ({v['share_dropping']:.0%} drop; "
                f"{v['beats_chance']['share_dropping']:.0%} of {v['beats_chance']['scores']} "
                f"above chance)"
                for m, v in row.items()
            )
            print(f"{kind} {test} at 0.2: {cells}")


if __name__ == "__main__":
    main()
