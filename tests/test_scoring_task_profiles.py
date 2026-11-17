"""A scoring task keeps every unit kind's profile, so `score --task` can measure it (#3767)."""

import ast
from pathlib import Path

from openinterpretability.agents.judge import ScoringTask
from openinterpretability.core.schemas import SuperpositionIdentifier
from openinterpretability.harness.superpositions import SuperpositionUnit
from openinterpretability.harness.units import submission_unit
from tests.test_superpositions import _ident, _profile

ROOT = Path(__file__).parents[1]
PROFILES = ("head_profile", "dictionary_profile", "superposition_profile")


def _task(profile, ident) -> ScoringTask:
    return ScoringTask(
        identifier=ident, superposition_profile=profile, explanation="e", threshold=1.0,
        window_tokens=64, quantiles=2, quantile_edges=[0.0, 1.0], seed=0, corpus="c",
        candidate_windows=0, active_windows=0, detection=[], fuzzing=[],
    )  # fmt: skip


def test_a_superposition_task_keeps_its_profile_and_unit():
    profile = _profile()
    task = ScoringTask.model_validate_json(_task(profile, _ident(profile)).model_dump_json())
    assert isinstance(task.identifier, SuperpositionIdentifier)
    assert task.superposition_profile == profile
    assert isinstance(submission_unit(task), SuperpositionUnit)


def test_every_scoring_task_passes_every_profile():
    """A new unit kind's profile must not be dropped where a task is built."""
    missing = []
    for path in (ROOT / "openinterpretability").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ScoringTask":
                given = {k.arg for k in node.keywords}
                missing += [f"{path.name}:{node.lineno} {p}" for p in PROFILES if p not in given]
    assert not missing, missing
