"""Detection, fuzzing and embedding scores on held-out windows (Delphi-style)."""

import json
import random
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterp.agents.judge import (
    DetectionAnswer,
    ExplanationJudge,
    FuzzingAnswer,
    ScoringTask,
    TaskWindow,
    answered,
)
from openinterp.cli import cli
from openinterp.core.explanation import ExplanationScores, ScoredWindow
from openinterp.core.schemas import AuditRecord, NeuronIdentifier
from openinterp.harness.explanation_scores import (
    Window,
    auroc,
    judge_score,
    label_windows,
    measure_windows,
    remeasure,
    sample_heldout,
)
from openinterp.reproduce import reproduce_record
from tests.test_agents import SMALL, _probes, _submission
from tests.test_reproduce import draft  # noqa: F401  (fixture)

IDENT = NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")
CORPUS = ["".join(random.Random(i).choices("abcdefghijklmnop qrstuvwxyz", k=40)) for i in range(60)]


@pytest.fixture
def toy_record(toy_runner, draft):  # noqa: F811
    from openinterp.agents.reviewer import evaluate_submission

    sub = _submission(IDENT, toy_runner, draft, honest=True)
    probes = _probes(toy_runner, IDENT, sub, honest=True)
    verdict = evaluate_submission(sub, toy_runner, probes, "external/reviewer", SMALL)
    return AuditRecord(submission=sub, verdict=verdict)


def _threshold(runner, quantile=0.5):
    peaks = sorted(w.peak for w in measure_windows(runner, CORPUS, IDENT))
    return peaks[int(quantile * len(peaks))]


def test_auroc_counts_ties_half_and_needs_both_classes():
    assert auroc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert auroc([0.1, 0.9], [True, False]) == 0.0
    assert auroc([0.1, 0.9], [True, True]) is None


def _scored(label, judged, quantile=None):
    return ScoredWindow(
        text="t", tokens=["t"], activations=[0.0], label=label, judged=judged, quantile=quantile
    )


def test_judge_score_is_balanced_and_split_by_quantile():
    windows = [_scored(True, True, 0), _scored(True, False, 1), _scored(True, True, 1)]
    windows += [_scored(False, False)] * 3 + [_scored(False, True)]
    score = judge_score(windows, quantiles=3)
    assert score.true_positive_rate == pytest.approx(2 / 3)
    assert score.false_positive_rate == pytest.approx(1 / 4)
    assert score.balanced_accuracy == pytest.approx((2 / 3 + 3 / 4) / 2)
    assert score.recall_by_quantile == [1.0, 0.5, None]
    assert judge_score([_scored(True, None)], 1) is None


def test_fuzzing_label_needs_exactly_the_active_tokens():
    w = [0.0, 2.0, 0.0, 3.0]
    windows = [Window("x", list("abcd"), w, marks) for marks in [(1, 3), (1,), (0, 2)]]
    windows.append(Window("x", list("abcd"), [0.0] * 4, (0,)))
    labels = [s.label for s in label_windows(windows, 1.0, [2.5], fuzzing=True)]
    assert labels == [True, False, False, False]
    detection = label_windows(windows, 1.0, [2.5], fuzzing=False)
    assert [s.label for s in detection] == [True, True, True, False]
    assert detection[0].quantile == 1  # peak 3.0 is past the 2.5 edge


def test_sample_heldout_is_seeded_and_held_out(toy_runner):
    threshold = _threshold(toy_runner)
    kwargs = dict(seed=7, exclude=CORPUS[:10], window_tokens=8, quantiles=3, per_quantile=2)
    sample = sample_heldout(toy_runner, CORPUS, IDENT, threshold, **kwargs)
    assert sample == sample_heldout(toy_runner, CORPUS, IDENT, threshold, **kwargs)
    assert sample.candidate_windows == 5 * 50  # five 8-token windows per unexcluded text
    fuzzing = label_windows(sample.fuzzing, threshold, sample.quantile_edges, fuzzing=True)
    assert any(w.label for w in fuzzing) and not all(w.label for w in fuzzing)
    assert all(w.highlighted for w in fuzzing)


def test_a_windowed_top_example_holds_out_its_whole_document(toy_runner):
    excerpts = [t[5:20] for t in CORPUS[:10]]  # what a windowed scan shows the proposer
    sample = sample_heldout(toy_runner, CORPUS, IDENT, 1.0, seed=0, exclude=excerpts,
                            window_tokens=8)  # fmt: skip
    assert sample.candidate_windows == 5 * 50


def test_sample_heldout_draws_evenly_from_each_quantile(toy_runner, monkeypatch):
    # Distinct peaks 0.00 … 0.99, first token only; the toy model's own peaks tie.
    def measure(runner, texts, ident):
        return [Window(t, ["x", "y"], [int(t) / 100, 0.0]) for t in texts]

    monkeypatch.setattr("openinterp.harness.explanation_scores.cut_windows", lambda r, t, n: t)
    monkeypatch.setattr("openinterp.harness.explanation_scores.measure_windows", measure)
    texts = [str(i) for i in range(100)]
    sample = sample_heldout(toy_runner, texts, IDENT, 0.4, seed=1, quantiles=3, per_quantile=2)
    assert (sample.active_windows, sample.quantile_edges) == (60, [0.6, 0.8])
    detection = label_windows(sample.detection, 0.4, sample.quantile_edges, fuzzing=False)
    quantiles = sorted(w.quantile for w in detection if w.label)
    assert quantiles == [0, 0, 1, 1, 2, 2] and len(detection) == 12
    fuzzing = label_windows(sample.fuzzing, 0.4, sample.quantile_edges, fuzzing=True)
    assert sum(w.label for w in fuzzing) == 3 and len(fuzzing) == 6 + 3  # 3 right, 3 decoys
    assert not {w.text for w in sample.detection} & {w.text for w in sample.fuzzing}


def test_without_active_windows_detection_scores_false_positives_only(toy_runner):
    sample = sample_heldout(toy_runner, CORPUS, IDENT, 1e9, seed=0, window_tokens=8)
    assert (len(sample.detection), sample.fuzzing, sample.active_windows) == (25, [], 0)
    with pytest.raises(ValueError, match="no held-out windows"):
        sample_heldout(toy_runner, CORPUS, IDENT, 1.0, seed=0, exclude=CORPUS)


def test_judge_prompts_hide_activations_and_mark_tokens(mocker):
    windows = [TaskWindow(text="ab\nc", tokens=["a", "b", "\n", "c"], highlighted=[1])]
    detect = ExplanationJudge.detection_prompt("Fires on b.", windows)
    assert "1. ab⏎c" in detect and "<<" not in detect
    assert "1. a<<b>>⏎c" in ExplanationJudge.fuzzing_prompt("Fires on b.", windows)

    client = mocker.Mock()
    client.chat.completions.create.return_value = DetectionAnswer(active=[1])
    assert ExplanationJudge("openai/judge", client=client).detect("x", windows).active == [1]

    assert answered([2, 2], 3, "detection") == [False, True, False]
    with pytest.raises(ValueError, match=r"\[4\]"):
        answered([4], 3, "detection")


@pytest.fixture
def flow(tmp_path, monkeypatch, toy_runner, toy_record):
    toy_runner.model.config = SimpleNamespace()
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: toy_runner)
    monkeypatch.setattr(
        "openinterp.harness.explanation_scores.embedding_similarities",
        lambda model, query, texts: [float(len(t) % 7) for t in texts],
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("\n".join(CORPUS))
    threshold = _threshold(toy_runner)
    sub = toy_record.submission.model_copy(update={"max_activation": threshold / 0.25})
    (tmp_path / "sub.json").write_text(sub.model_dump_json())
    return SimpleNamespace(path=tmp_path, sub=sub)


def run(*args):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def test_two_step_scoring_attaches_to_a_review(flow):
    run("score", "--audit-file", "sub.json", "--corpus", "corpus.txt", "--seed", 3,
        "--window-tokens", 8, "--quantiles", 2, "--per-quantile", 3,
        "--emit-task", "score.task.json")  # fmt: skip
    task = ScoringTask.model_validate_json((flow.path / "score.task.json").read_text())
    assert "activations" not in (flow.path / "score.task.json").read_text()  # labels hidden
    for kind in ("detection", "fuzzing"):
        prompt = (flow.path / f"score.{kind}.prompt.md").read_text()
        assert f"score.{kind}.json" in prompt and flow.sub.natural_language_explanation in prompt

    (flow.path / "score.detection.json").write_text(
        DetectionAnswer(active=list(range(1, len(task.detection) + 1, 2))).model_dump_json()
    )
    (flow.path / "score.fuzzing.json").write_text(FuzzingAnswer(correct=[1]).model_dump_json())
    out = run("score", "--task", "score.task.json", "--judge-model", "external/judge").output
    assert "Detection" in out and "embedding AUROC" in out
    scores = ExplanationScores.model_validate_json((flow.path / "score.scores.json").read_text())
    assert scores.judge_model == "external/judge" and scores.seed == 3
    assert scores.detection is not None and scores.fuzzing is not None
    assert scores.embedding_auroc is not None
    assert [w.judged for w in scores.detection_windows][:2] == [True, False]
    stages = [s.stage for s in scores.costs.stages]  # drawing, then scoring
    assert stages[:1] == ["heldout_windows"] and {"judge", "embedding", "remeasure"} <= set(stages)
    assert [(c.agent, c.external) for c in scores.costs.llm_calls] == [("judge", True)] * 2

    probes = {"probes": [{"text": t, "expected_active": i < 4, "rationale": "r"}
                         for i, t in enumerate(CORPUS[:10])]}  # fmt: skip
    (flow.path / "probes.json").write_text(json.dumps(probes))
    run("review", "--audit-file", "sub.json", "--probes", "probes.json",
        "--agent-model", "external/reviewer", "--explanation-scores", "score.scores.json",
        "--wiki-dir", "wiki")  # fmt: skip
    (record_path,) = (flow.path / "wiki" / "experiments").rglob("*.json")
    record = AuditRecord.model_validate_json(record_path.read_text())
    # The scores' costs move into the record's provenance (here there is none to hold them).
    assert record.verdict.explanation_scores == scores.model_copy(update={"costs": None})
    page = next((flow.path / "wiki" / "experiments").rglob("2*.md")).read_text()
    assert "Explanation scores on held-out text" in page


def test_scores_for_another_explanation_are_refused(flow):
    run("score", "--audit-file", "sub.json", "--corpus", "corpus.txt", "--window-tokens", 8,
        "--quantiles", 2, "--per-quantile", 2, "--emit-task", "score.task.json")  # fmt: skip
    (flow.path / "score.detection.json").write_text('{"active": []}')
    (flow.path / "score.fuzzing.json").write_text('{"correct": []}')
    run("score", "--task", "score.task.json", "--judge-model", "external/judge",
        "--embedding-model", "")  # fmt: skip
    other = flow.sub.model_copy(update={"natural_language_explanation": "Something else."})
    (flow.path / "other.json").write_text(other.model_dump_json())
    (flow.path / "probes.json").write_text(json.dumps(
        {"probes": [{"text": t, "expected_active": False, "rationale": "r"} for t in CORPUS[:10]]}
    ))  # fmt: skip
    result = CliRunner().invoke(
        cli, ["review", "--audit-file", "other.json", "--probes", "probes.json",
              "--agent-model", "r", "--explanation-scores", "score.scores.json"],
    )  # fmt: skip
    assert result.exit_code != 0 and "different explanation" in result.output


def test_reproduce_rederives_scores_and_warns_on_edits(toy_record, toy_runner):
    threshold = _threshold(toy_runner)
    sample = sample_heldout(toy_runner, CORPUS, IDENT, threshold, seed=1, window_tokens=8,
                            quantiles=2, per_quantile=2)  # fmt: skip
    windows = label_windows(sample.detection, threshold, sample.quantile_edges, fuzzing=False)
    windows = [w.model_copy(update={"judged": w.label, "similarity": 0.1}) for w in windows]
    scores = remeasure(
        ExplanationScores(
            slug=IDENT.slug, explanation="x", threshold=threshold, window_tokens=8, quantiles=2,
            quantile_edges=sample.quantile_edges, seed=1, corpus="c", candidate_windows=1,
            active_windows=1, detection_windows=windows,
        ),
        toy_runner,
        IDENT,
    )  # fmt: skip
    assert scores.detection.balanced_accuracy == 1.0 and scores.embedding_auroc == 0.5

    record = toy_record
    honest = record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"explanation_scores": scores})}
    )
    assert reproduce_record(honest, toy_runner).warnings == []
    edited = scores.model_copy(update={"embedding_auroc": 0.9})
    tampered = record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"explanation_scores": edited})}
    )
    report = reproduce_record(tampered, toy_runner)
    assert report.ok and any("embedding_auroc differs" in w for w in report.warnings)


def test_task_without_active_windows_needs_no_fuzzing_judge(flow):
    sub = flow.sub.model_copy(update={"max_activation": 1e9})
    (flow.path / "dead.json").write_text(sub.model_dump_json())
    run("score", "--audit-file", "dead.json", "--corpus", "corpus.txt", "--window-tokens", 8,
        "--emit-task", "score.task.json")  # fmt: skip
    assert not (flow.path / "score.fuzzing.prompt.md").exists()
    (flow.path / "score.detection.json").write_text('{"active": [1, 2]}')
    run("score", "--task", "score.task.json", "--judge-model", "external/judge")
    scores = ExplanationScores.model_validate_json((flow.path / "score.scores.json").read_text())
    assert scores.fuzzing is None and scores.detection.false_positive_rate == 2 / 25
    assert scores.detection.true_positive_rate is None and scores.embedding_auroc is None


def test_one_step_review_scores_the_explanation_itself(flow, monkeypatch):
    from openinterp.agents.reviewer import AdversarialProbe, ProbeSet

    probes = ProbeSet(probes=[AdversarialProbe(text=t, expected_active=i < 4, rationale="r")
                              for i, t in enumerate(CORPUS[:10])])  # fmt: skip
    monkeypatch.setattr("openinterp.agents.reviewer.ReviewerAgent.generate_probes",
                        lambda self, sub, n: probes)  # fmt: skip
    judged = []

    def detect(self, explanation, windows):
        judged.append(self.model)
        return DetectionAnswer(active=[1])

    monkeypatch.setattr(ExplanationJudge, "detect", detect)
    monkeypatch.setattr(ExplanationJudge, "fuzz", lambda self, e, ws: FuzzingAnswer(correct=[]))
    run("review", "--audit-file", "sub.json", "--agent-model", "openai/reviewer",
        "--corpus", "corpus.txt", "--wiki-dir", "wiki")  # fmt: skip
    (record_path,) = (flow.path / "wiki" / "experiments").rglob("*.json")
    scores = AuditRecord.model_validate_json(record_path.read_text()).verdict.explanation_scores
    assert judged == ["openai/reviewer"] and scores.judge_model == "openai/reviewer"
