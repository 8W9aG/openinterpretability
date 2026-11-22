"""Control explanations for `open-interpretability score --controls`.

The code is in openinterpretability/score_controls.py.
"""

import json

from openinterpretability.agents.judge import DetectionAnswer, FuzzingAnswer, ScoringTask
from openinterpretability.core.explanation import ControlScore, ExplanationScores, ScoredWindow
from openinterpretability.harness.explanation_scores import finalize_scores
from openinterpretability.score_controls import gap_summary, other_unit_explanation
from tests.test_explanation_scores import flow, run, toy_record  # noqa: F401, F811  (fixtures)
from tests.test_reproduce import draft  # noqa: F401  (fixture)


def _write(wiki, unit, explanation, status, stamp="20261101T000000Z"):
    path = wiki / "experiments" / unit / f"{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "submission": {"natural_language_explanation": explanation},
        "verdict": {"status": status},
    }
    path.write_text(json.dumps(record))


def test_other_unit_prefers_verified_and_skips_the_unit_itself(tmp_path):
    wiki = tmp_path / "wiki"
    _write(wiki, "toy/L1/N2-positive", "the unit itself", "VERIFIED")
    _write(wiki, "toy/L1/N3-positive", "rejected claim", "REJECTED")
    _write(wiki, "toy/L0/N7-negative", "verified claim", "VERIFIED")
    _write(wiki, "other/L0/N1-positive", "another model", "VERIFIED")
    for seed in range(5):
        assert other_unit_explanation("toy/L1/N2-positive", seed, wiki) == (
            "toy/L0/N7-negative",
            "verified claim",
        )


def test_other_unit_falls_back_to_any_and_to_none(tmp_path):
    wiki = tmp_path / "wiki"
    assert other_unit_explanation("toy/L1/N2-positive", 0, wiki) is None
    _write(wiki, "toy/L1/N3-positive", "rejected claim", "REJECTED")
    assert other_unit_explanation("toy/L1/N2-positive", 0, wiki)[1] == "rejected claim"


def _window(label, judged, quantile=None):
    return ScoredWindow(
        text="t", tokens=["t"], activations=[0.0], label=label, judged=judged, quantile=quantile
    )


def test_control_is_scored_on_the_real_windows_labels():
    windows = [_window(True, True, 0), _window(True, True, 0), _window(False, False)]
    control = ControlScore(kind="empty", explanation="", detection_judged=[False, False, False])
    scores = ExplanationScores(
        slug="toy/L1/N2-positive", explanation="x", threshold=1.0, window_tokens=8,
        quantiles=1, quantile_edges=[], seed=0, corpus="c", candidate_windows=3,
        active_windows=2, detection_windows=windows, controls=[control],
    )  # fmt: skip
    scores = finalize_scores(scores)
    assert scores.detection.balanced_accuracy == 1.0
    assert scores.controls[0].detection.balanced_accuracy == 0.5
    assert scores.controls[0].fuzzing is None
    assert "Control empty: detection 0.50 (gap +0.50)" in gap_summary(scores)


def test_two_step_scoring_with_controls(flow):  # noqa: F811
    _write(flow.path / "wiki", "toy/L0/N7-negative", "fires on vowels", "VERIFIED")
    run("score", "--audit-file", "sub.json", "--corpus", "corpus.txt", "--seed", 3,
        "--window-tokens", 8, "--quantiles", 2, "--per-quantile", 3, "--controls",
        "--emit-task", "score.task.json")  # fmt: skip
    task = ScoringTask.model_validate_json((flow.path / "score.task.json").read_text())
    assert [c.kind for c in task.controls] == ["empty", "other_unit"]
    other = (flow.path / "score.control-other_unit.detection.prompt.md").read_text()
    assert "fires on vowels" in other and flow.sub.natural_language_explanation not in other
    for name in ("score", "score.control-empty", "score.control-other_unit"):
        (flow.path / f"{name}.detection.json").write_text(
            DetectionAnswer(active=[1]).model_dump_json()
        )
        (flow.path / f"{name}.fuzzing.json").write_text(FuzzingAnswer(correct=[]).model_dump_json())
    out = run("score", "--task", "score.task.json", "--judge-model", "external/judge").output
    assert "Control empty" in out and "Control other_unit from toy/L0/N7-negative" in out
    scores = ExplanationScores.model_validate_json((flow.path / "score.scores.json").read_text())
    assert [c.kind for c in scores.controls] == ["empty", "other_unit"]
    assert all(c.detection is not None and c.fuzzing is not None for c in scores.controls)


def test_scoring_without_controls_adds_none(flow):  # noqa: F811
    run("score", "--audit-file", "sub.json", "--corpus", "corpus.txt", "--seed", 3,
        "--window-tokens", 8, "--emit-task", "score.task.json")  # fmt: skip
    assert (
        ScoringTask.model_validate_json((flow.path / "score.task.json").read_text()).controls == []
    )
    assert not list(flow.path.glob("score.control-*"))


def test_units_scored_with_one_seed_draw_different_decoys(tmp_path):
    wiki = tmp_path / "wiki"
    for i in range(20):
        _write(wiki, f"toy/L0/N{i}-positive", f"claim {i}", "VERIFIED")
    drawn = {other_unit_explanation(f"toy/L1/N{j}-positive", 7, wiki)[0] for j in range(10)}
    assert len(drawn) > 1


def _task(fuzzing=True, controls=None):
    from openinterpretability.agents.judge import TaskWindow
    from openinterpretability.core.schemas import NeuronIdentifier
    from openinterpretability.score_controls import ControlTask

    window = TaskWindow(text="ab", tokens=["a", "b"], highlighted=[1])
    return ScoringTask(
        identifier=NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive"),
        explanation="x", threshold=1.0, window_tokens=8, quantiles=1, quantile_edges=[],
        seed=0, corpus="c", candidate_windows=2, active_windows=1,
        detection=[window, window], fuzzing=[window] if fuzzing else [],
        controls=controls or [ControlTask(kind="empty", explanation="")],
    )  # fmt: skip


def test_a_unit_alone_in_its_model_gets_only_the_empty_control(tmp_path):
    from openinterpretability.score_controls import control_tasks

    (control,) = control_tasks("toy/L1/N2-positive", 0, tmp_path / "wiki")
    assert (control.kind, control.explanation, control.source_slug) == ("empty", "", None)


def test_a_task_with_no_fuzzing_windows_writes_and_reads_only_detection(tmp_path):
    from openinterpretability.score_controls import answer_path, read_answers, write_prompts

    task, path = _task(fuzzing=False), tmp_path / "score.task.json"
    write_prompts(task, path, "neuron")
    assert [p.name for p in tmp_path.iterdir()] == ["score.control-empty.detection.prompt.md"]
    (control,) = task.controls
    answer_path(path, control, "detection").write_text(
        DetectionAnswer(active=[2]).model_dump_json()
    )
    (score,) = read_answers(task, path)
    assert score.detection_judged == [False, True] and score.fuzzing_judged == []


class _Judge:
    """An LLM judge that answers every question the same way, and records them."""

    def __init__(self):
        self.asked = []

    def detect(self, explanation, windows, noun):
        self.asked.append(("detect", explanation, len(windows), noun))
        return DetectionAnswer(active=[1])

    def fuzz(self, explanation, windows, noun):
        self.asked.append(("fuzz", explanation, len(windows), noun))
        return FuzzingAnswer(correct=[1])


def test_judge_controls_asks_each_controls_questions():
    from openinterpretability.score_controls import ControlTask, judge_controls

    other = ControlTask(kind="other_unit", explanation="vowels", source_slug="toy/L0/N7")
    controls = [ControlTask(kind="empty", explanation=""), other]
    judge = _Judge()
    scores = judge_controls(judge, _task(controls=controls), "neuron")
    assert [(s.kind, s.source_slug) for s in scores] == [
        ("empty", None),
        ("other_unit", "toy/L0/N7"),
    ]
    assert all(s.detection_judged == [True, False] and s.fuzzing_judged == [True] for s in scores)
    assert judge.asked[2:] == [("detect", "vowels", 2, "neuron"), ("fuzz", "vowels", 1, "neuron")]
    (no_fuzz,) = judge_controls(_Judge(), _task(fuzzing=False), "neuron")
    assert no_fuzz.fuzzing_judged == []
