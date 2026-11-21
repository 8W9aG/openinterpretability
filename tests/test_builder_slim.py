from openinterpretability.builder.experiment import Experiment
from openinterpretability.builder.significance import significance
from openinterpretability.builder.slim import SlimArchive, full_record, slim
from openinterpretability.builder.wiki import experiment_id
from openinterpretability.core.explanation import ExplanationScores, ScoredWindow

WINDOW = ScoredWindow(text="ab", tokens=["a", "b"], activations=[0.1, 0.9], label=True)


def scored(record):
    scores = ExplanationScores(
        slug=record.submission.identifier.slug, explanation="x", threshold=0.6,
        window_tokens=32, quantiles=1, quantile_edges=[], seed=3, corpus="c",
        candidate_windows=2, active_windows=1,
        detection_windows=[WINDOW], fuzzing_windows=[WINDOW],
    )  # fmt: skip
    verdict = record.verdict.model_copy(update={"explanation_scores": scores})
    return record.model_copy(update={"verdict": verdict})


def test_slim_empties_the_scored_windows_but_keeps_the_scores(record):
    held = slim(scored(record))
    scores = held.verdict.explanation_scores
    assert scores.detection_windows == [] and scores.fuzzing_windows == []
    assert scores.active_windows == 1 and scores.explanation == "x"
    assert held.submission.top_examples == []


def test_a_record_with_no_verdict_is_held_in_full(record, tmp_path):
    unreviewed = record.model_copy(update={"verdict": None})
    archive = SlimArchive([])
    assert archive(tmp_path / "r.json", unreviewed) is unreviewed
    assert archive.paths == {}


def test_full_record_of_an_experiment_held_in_full_is_its_record(record):
    e = Experiment(record, significance(record), experiment_id(record), True)
    assert e.source is None
    assert full_record(e) is record
