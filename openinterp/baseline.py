"""The Patchscope readout as a blind proposer baseline (`openinterp patchscope-baseline`).

Does the model's own, example-free description of a unit's output direction explain the
unit as well as the agent's hypothesis? One reading per unit is turned into an
explanation by a rule fixed before any baseline was scored:

* the ``category`` prompt, at the lowest layer read (layer 2, or the unit's own if
  earlier), at ``BASELINE_SCALE``;
* wrapped as ``TEMPLATE``.

The baseline is then scored on exactly the held-out windows the agent's explanation was
scored on, which the record stores with their labels, so the two are paired and nothing
is re-measured. Detection and fuzzing need a judge; a pair counts only when the same
judge model answered both. The embedding score needs none. The units looked at while
choosing the prompts and the rule (``DEV_SLUGS``) are reported apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import mean

from pydantic import BaseModel

from openinterp.core.explanation import ExplanationScores
from openinterp.core.patchscope import PatchscopeReading, PatchscopeReadout

BASELINE_PROMPT = "category"
BASELINE_SCALE = 2.0
TEMPLATE = 'Fires on tokens related to "{}".'
# Poles whose readouts were inspected while choosing the prompts, layers and scale.
DEV_SLUGS = frozenset(
    f"Qwen--Qwen2.5-0.5B/{unit}"
    for unit in (
        "L0/N788-positive",
        "L3/N3738-negative",
        "L5/N2106-positive",
        "L5/N2131-negative",
        "L5/N4579-negative",
        "L14/N2611-positive",
        "L17/N4749-negative",
        "L22/N1058-negative",
        "L22/N2686-positive",
    )
)
METRICS = ("detection", "fuzzing", "embedding")


def blind_reading(readout: PatchscopeReadout) -> PatchscopeReading | None:
    """The one reading the baseline uses, by the fixed rule; ``None`` if it wasn't read."""
    readings = [r for r in readout.readings if r.prompt == BASELINE_PROMPT]
    if not readings:
        return None
    lowest = min(r.layer for r in readings)
    return next((r for r in readings if r.layer == lowest and r.scale == BASELINE_SCALE), None)


def blind_explanation(readout: PatchscopeReadout) -> str | None:
    reading = blind_reading(readout)
    return TEMPLATE.format(reading.text) if reading and reading.text else None


def rescored(
    scores: ExplanationScores,
    explanation: str,
    *,
    judge_model: str | None = None,
    detection: Sequence[bool] | None = None,
    fuzzing: Sequence[bool] | None = None,
    similarities: Sequence[float] | None = None,
) -> ExplanationScores:
    """``scores``'s windows and labels with another explanation's answers: what that
    explanation scores on the same held-out text. Answers not given are left unjudged."""
    from openinterp.harness.explanation_scores import finalize_scores

    def swap(windows, answers, sims=None):
        return [
            w.model_copy(
                update={
                    "judged": None if answers is None else answers[i],
                    "similarity": None if sims is None else sims[i],
                }
            )
            for i, w in enumerate(windows)
        ]

    if detection is not None and len(detection) != len(scores.detection_windows):
        raise ValueError("one detection answer per window")
    if fuzzing is not None and len(fuzzing) != len(scores.fuzzing_windows):
        raise ValueError("one fuzzing answer per window")
    return finalize_scores(
        scores.model_copy(
            update={
                "explanation": explanation,
                "judge_model": judge_model,
                "detection_windows": swap(scores.detection_windows, detection, similarities),
                "fuzzing_windows": swap(scores.fuzzing_windows, fuzzing),
            }
        )
    )


def metric(scores: ExplanationScores | None, name: str) -> float | None:
    if scores is None:
        return None
    if name == "embedding":
        return scores.embedding_auroc
    judged = getattr(scores, name)
    return None if judged is None else judged.balanced_accuracy


class BaselineRow(BaseModel):
    """One experiment's agent explanation against the baseline, on the same windows."""

    experiment_id: str
    slug: str
    status: str
    selection: str | None
    dev: bool
    reading: str
    agent_explanation: str
    baseline_explanation: str | None
    judge_model: str | None
    baseline_judge_model: str | None = None
    detection_answer: list[int] | None = None
    fuzzing_answer: list[int] | None = None
    agent: dict[str, float | None]
    baseline: dict[str, float | None]


def sign_test(differences: Sequence[float]) -> float | None:
    """Two-sided exact sign test that baseline − agent is as often > 0 as < 0 (ties out)."""
    up = sum(d > 0 for d in differences)
    n = up + sum(d < 0 for d in differences)
    if not n:
        return None
    tail = sum(math.comb(n, k) for k in range(min(up, n - up) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarise(rows: Sequence[BaselineRow], name: str) -> dict | None:
    """Paired comparison on ``name`` over the rows where both explanations have a score."""
    pairs = [(r.agent[name], r.baseline[name]) for r in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return None
    diffs = [b - a for a, b in pairs]
    return {
        "n": len(pairs),
        "agent": mean(a for a, _ in pairs),
        "baseline": mean(b for _, b in pairs),
        "difference": mean(diffs),
        "baseline_at_least_agent": sum(d >= 0 for d in diffs) / len(diffs),
        "sign_test_p": sign_test(diffs),
    }


def report(rows: Sequence[BaselineRow]) -> str:
    """The comparison as a Markdown page: the rule, every metric over held-out
    experiments, randomly selected and VERIFIED units and the dev set, and the readings."""
    from collections import Counter

    groups = _groups(rows)
    readings = Counter(r.reading for r in rows)
    judged = sum(r.baseline_judge_model is not None for r in rows)
    lines = [
        "# Patchscope readout as a blind proposer baseline",
        "",
        "Generated by `openinterp patchscope-baseline`; do not edit. Method: "
        "`openinterp/baseline.py` and docs/PROTOCOL.md §4.",
        "",
        f"Each unit's readout becomes one explanation by a rule fixed before scoring: the "
        f"`{BASELINE_PROMPT}` prompt at the lowest layer read, at ×{BASELINE_SCALE:g} the "
        f"replaced norm, as `{TEMPLATE.format('<reading>')}`. It is scored on the same "
        f"held-out windows as the agent's explanation. Detection and fuzzing pair only "
        f"experiments whose agent explanation was judged by the same model as the baseline's "
        f"({judged} of {len(rows)}); embedding uses the record's own encoder. The sign test "
        f"counts experiments, and a unit with several experiments counts more than once.",
        "",
        "| Group | Metric | n | Agent | Baseline | Baseline − agent | Baseline ≥ agent "
        "| Sign test p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [_metric_row(label, group, name) for label, group in groups for name in METRICS]
    lines = [line for line in lines if line is not None]
    common = ", ".join(f"{text!r} ×{n}" for text, n in readings.most_common(10))
    lines += [
        "",
        f"{len(readings)} distinct readings over {len(rows)} experiments; the most common: "
        f"{common}.",
    ]
    return "\n".join(lines) + "\n"


def _groups(rows: Sequence[BaselineRow]) -> list[tuple[str, list[BaselineRow]]]:
    return [
        ("Held-out experiments", [r for r in rows if not r.dev]),
        (
            "Held-out, randomly selected units",
            [r for r in rows if not r.dev and r.selection == "random"],
        ),
        ("Held-out, VERIFIED", [r for r in rows if not r.dev and r.status == "VERIFIED"]),
        ("Dev set (inspected while choosing the rule)", [r for r in rows if r.dev]),
    ]


def _metric_row(label: str, group: Sequence[BaselineRow], name: str) -> str | None:
    """One table row comparing the agent and the baseline on ``name``; ``None`` if unpaired."""
    s = summarise(group, name)
    if s is None:
        return None
    p = "—" if s["sign_test_p"] is None else f"{s['sign_test_p']:.2g}"
    return (
        f"| {label} | {name} | {s['n']} | {s['agent']:.2f} | {s['baseline']:.2f} | "
        f"{s['difference']:+.2f} | {s['baseline_at_least_agent']:.0%} | {p} |"
    )
