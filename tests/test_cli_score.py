"""``open-interpretability score``: its option checks, the one-step judge and refusals."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from openinterpretability.agents.judge import DetectionAnswer, ExplanationJudge, FuzzingAnswer
from openinterpretability.cli import cli
from openinterpretability.cli_score import _threshold, load_scores
from openinterpretability.core.explanation import ExplanationScores
from tests.test_explanation_scores import flow, run, toy_record  # noqa: F401  (fixtures)
from tests.test_reproduce import draft  # noqa: F401  (fixture)


def _invoke(*args):
    return CliRunner().invoke(cli, ["score", *(str(a) for a in args)])


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--task", "t.json", "--audit-file", "t.json"], "drop --audit-file/--emit-task"),
        (["--task", "t.json", "--emit-task", "e.json"], "drop --audit-file/--emit-task"),
        (["--task", "t.json"], "--judge-model is required with --task"),
        ([], "--audit-file is required unless --task is given"),
        (["--audit-file", "t.json"], "--out is required when scoring in one step"),
    ],
)
def test_score_refuses_options_that_do_not_go_together(tmp_path, monkeypatch, args, message):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "t.json").write_text("{}")
    result = _invoke(*args)
    assert result.exit_code == 2 and message in result.output


def test_an_inverse_claim_gets_no_explanation_scores(flow):  # noqa: F811
    sub = flow.sub.model_copy(update={"claim": "inverse"})
    (flow.path / "inverse.json").write_text(sub.model_dump_json())
    result = _invoke("--audit-file", "inverse.json", "--emit-task", "score.task.json")
    assert result.exit_code == 2
    assert "explanation scores are not computed for an inverse claim" in result.output
    assert not (flow.path / "score.task.json").exists()


def test_one_step_scoring_asks_the_llm_judge_and_writes_the_scores(flow, monkeypatch):  # noqa: F811
    asked = []

    def detect(self, explanation, windows, noun="neuron"):
        asked.append(("detect", self.model, len(windows)))
        return DetectionAnswer(active=[1])

    def fuzz(self, explanation, windows, noun="neuron"):
        asked.append(("fuzz", self.model, len(windows)))
        return FuzzingAnswer(correct=[])

    monkeypatch.setattr(ExplanationJudge, "detect", detect)
    monkeypatch.setattr(ExplanationJudge, "fuzz", fuzz)
    out = run("score", "--audit-file", "sub.json", "--corpus", "corpus.txt", "--seed", 3,
              "--window-tokens", 8, "--quantiles", 2, "--per-quantile", 2,
              "--judge-model", "openai/judge", "--embedding-model", "",
              "--out", "scores/s.json").stdout  # fmt: skip
    scores = ExplanationScores.model_validate_json((flow.path / "scores/s.json").read_text())
    assert [a[:2] for a in asked] == [("detect", "openai/judge"), ("fuzz", "openai/judge")]
    assert asked[0][2] == len(scores.detection_windows)
    assert scores.judge_model == "openai/judge" and scores.seed == 3
    assert scores.embedding_model is None and scores.costs is not None
    assert [w.judged for w in scores.detection_windows][0] is True
    assert out.splitlines()[0].startswith("Detection ")
    assert out.splitlines()[-1] == "Wrote scores/s.json"


def test_a_submission_without_a_max_activation_must_be_rescanned(flow):  # noqa: F811
    (flow.path / "bare.json").write_text(
        flow.sub.model_copy(update={"max_activation": None}).model_dump_json()
    )
    result = _invoke("--audit-file", "bare.json", "--corpus", "corpus.txt",
                     "--emit-task", "score.task.json")  # fmt: skip
    assert result.exit_code == 1
    assert "the submission has no max_activation; rescan it first." in result.output


def test_a_corpus_of_only_top_examples_has_no_held_out_windows(flow):  # noqa: F811
    tops = "\n".join(e.text for e in flow.sub.top_examples)
    (flow.path / "tops.txt").write_text(tops)
    result = _invoke("--audit-file", "sub.json", "--corpus", "tops.txt",
                     "--emit-task", "score.task.json")  # fmt: skip
    assert result.exit_code == 1
    assert "no held-out windows: every corpus snippet is a top example" in result.output


def test_a_judge_answer_naming_a_missing_text_is_refused(flow):  # noqa: F811
    run("score", "--audit-file", "sub.json", "--corpus", "corpus.txt", "--window-tokens", 8,
        "--quantiles", 2, "--per-quantile", 2, "--emit-task", "score.task.json")  # fmt: skip
    (flow.path / "score.detection.json").write_text('{"active": [999]}')
    (flow.path / "score.fuzzing.json").write_text('{"correct": []}')
    result = _invoke("--task", "score.task.json", "--judge-model", "external/judge")
    assert result.exit_code == 1
    assert "detection answer names texts [999], but there are only" in result.output
    assert not (flow.path / "score.scores.json").exists()


def _scores_file(tmp_path, sub, **change):
    base = {"slug": sub.identifier.slug, "explanation": sub.natural_language_explanation,
            "threshold": _threshold(sub), "window_tokens": 8, "quantiles": 2,
            "quantile_edges": [], "seed": 1, "corpus": "c", "candidate_windows": 1,
            "active_windows": 0}  # fmt: skip
    path = tmp_path / "scores.json"
    path.write_text(ExplanationScores(**(base | change)).model_dump_json())
    return path


def test_load_scores_checks_the_slug_the_explanation_and_the_threshold(tmp_path, record):
    sub = record.submission
    assert load_scores(None, sub) is None
    good = _scores_file(tmp_path, sub)
    assert load_scores(good, sub).slug == sub.identifier.slug
    for change, message in [
        (
            {"slug": "other/L0/N0-positive"},
            f"scores other/L0/N0-positive, not {sub.identifier.slug}",
        ),
        ({"explanation": "Something else."}, "scores a different explanation"),
        ({"threshold": _threshold(sub) * 2}, "used another activation threshold"),
    ]:
        path = _scores_file(tmp_path, sub, **change)
        with pytest.raises(click.UsageError, match=message):
            load_scores(path, sub)
