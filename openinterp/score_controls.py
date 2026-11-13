"""Control explanations for ``openinterp score --controls``.

A detection or fuzzing score can be high for reasons that have nothing to do with the
explanation. The held-out windows can have structure that a judge picks up without it,
for example when active windows are longer or have more code. "Scoring Explanations
Without Reading Them" (arXiv 2610.37676) reports such shortcuts. So the same windows
are judged twice more:

- ``empty``: with no explanation.
- ``other_unit``: with the explanation of another unit of the same model, drawn with the
  task's seed. A VERIFIED explanation is drawn when one exists, so the control is a real,
  specific claim about a different unit.

The gap between the real score and each control is the part of the score that the
explanation earns. The controls are reported, not gated (docs/PROTOCOL.md §4).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel

from openinterp.core.explanation import ControlScore, ExplanationScores
from openinterp.core.paths import DEFAULT_WIKI_DIR, experiments_dir

KINDS = ("detection", "fuzzing")


class ControlTask(BaseModel):
    """One control explanation, as ``openinterp score --emit-task`` writes it."""

    kind: str
    explanation: str
    source_slug: str | None = None


def other_unit_explanation(slug: str, seed: int, wiki_dir: Path = DEFAULT_WIKI_DIR):
    """Draw another unit's explanation from the same model: ``(slug, explanation)``.

    Draws among VERIFIED explanations when the model has any, else among all of them.
    Returns ``None`` when no other unit of the model has a record.
    """
    model_dir = experiments_dir(Path(wiki_dir)) / slug.split("/", 1)[0]
    latest: dict[str, tuple[str, bool]] = {}
    for path in sorted(model_dir.rglob("*.json")):  # sorted by time within a unit
        unit = path.parent.relative_to(experiments_dir(Path(wiki_dir))).as_posix()
        if unit == slug:
            continue
        record = json.loads(path.read_text())
        verdict = record.get("verdict") or {}
        explanation = record["submission"]["natural_language_explanation"]
        latest[unit] = (explanation, verdict.get("status") == "VERIFIED")
    pool = sorted(u for u, (_, ok) in latest.items() if ok) or sorted(latest)
    if not pool:
        return None
    # The slug joins the seed, so units scored with one seed don't all draw the same decoy.
    unit = random.Random(f"{seed}:{slug}").choice(pool)
    return unit, latest[unit][0]


def control_tasks(slug: str, seed: int, wiki_dir: Path = DEFAULT_WIKI_DIR) -> list[ControlTask]:
    """The controls for one unit: always ``empty``, and ``other_unit`` when one exists."""
    controls = [ControlTask(kind="empty", explanation="")]
    other = other_unit_explanation(slug, seed, wiki_dir)
    if other is not None:
        controls.append(ControlTask(kind="other_unit", explanation=other[1], source_slug=other[0]))
    return controls


def prompt_path(task_path: Path, control: ControlTask, kind: str) -> Path:
    """Where ``--emit-task`` writes a control's judge prompt, next to the task."""
    return task_path.with_name(
        task_path.name.split(".")[0] + f".control-{control.kind}.{kind}.prompt.md"
    )


def answer_path(task_path: Path, control: ControlTask, kind: str) -> Path:
    """Where the judge writes its answer to a control's prompt."""
    return task_path.with_name(
        task_path.name.split(".")[0] + f".control-{control.kind}.{kind}.json"
    )


def unscored(control: ControlTask, detection: list[bool], fuzzing: list[bool]) -> ControlScore:
    """A control with the judge's answers. ``finalize_scores`` adds its scores."""
    return ControlScore(
        kind=control.kind,
        explanation=control.explanation,
        source_slug=control.source_slug,
        detection_judged=detection,
        fuzzing_judged=fuzzing,
    )


def gap_summary(scores: ExplanationScores) -> str:
    """One line per control: the real score minus the control's, for each kind."""
    lines = []
    for c in scores.controls:
        parts = []
        for kind in KINDS:
            real, ctrl = getattr(scores, kind), getattr(c, kind)
            if real is not None and ctrl is not None:
                gap = real.balanced_accuracy - ctrl.balanced_accuracy
                parts.append(f"{kind} {ctrl.balanced_accuracy:.2f} (gap {gap:+.2f})")
        source = f" from {c.source_slug}" if c.source_slug else ""
        lines.append(f"Control {c.kind}{source}: " + (", ".join(parts) or "not judged"))
    return "\n".join(lines)


def write_prompts(task, task_path: Path, noun: str) -> None:
    """Write a judge prompt for each control and kind, next to the task."""
    from openinterp.agents.judge import DetectionAnswer, ExplanationJudge, FuzzingAnswer

    models = {"detection": DetectionAnswer, "fuzzing": FuzzingAnswer}
    builders = {
        "detection": ExplanationJudge.detection_prompt,
        "fuzzing": ExplanationJudge.fuzzing_prompt,
    }
    for control in task.controls:
        for kind in KINDS:
            windows = getattr(task, kind)
            if not windows:
                continue
            text = builders[kind](control.explanation, windows, noun)
            answer = answer_path(task_path, control, kind)
            rendered = ExplanationJudge.render_prompt(models[kind], text, answer)
            prompt_path(task_path, control, kind).write_text(rendered)


def read_answers(task, task_path: Path) -> list[ControlScore]:
    """Read the judge's answers to each control's prompts (step two of ``--emit-task``)."""
    from openinterp.agents.judge import DetectionAnswer, FuzzingAnswer, answered

    out = []
    for control in task.controls:
        judged = {}
        for kind, model, field in (
            ("detection", DetectionAnswer, "active"),
            ("fuzzing", FuzzingAnswer, "correct"),
        ):
            windows = getattr(task, kind)
            if not windows:
                judged[kind] = []
                continue
            answer = model.model_validate_json(answer_path(task_path, control, kind).read_text())
            judged[kind] = answered(getattr(answer, field), len(windows), f"{control.kind} {kind}")
        out.append(unscored(control, judged["detection"], judged["fuzzing"]))
    return out


def judge_controls(judge, task, noun: str) -> list[ControlScore]:
    """Ask an LLM judge each control's questions (the one-step ``openinterp score``)."""
    from openinterp.agents.judge import answered

    out = []
    for control in task.controls:
        det = judge.detect(control.explanation, task.detection, noun)
        fuz = judge.fuzz(control.explanation, task.fuzzing, noun) if task.fuzzing else None
        out.append(
            unscored(
                control,
                answered(det.active, len(task.detection), "detection"),
                answered(fuz.correct, len(task.fuzzing), "fuzzing") if fuz else [],
            )
        )
    return out
