"""``patchscope-baseline`` and ``neuronpedia-baseline``: readouts, waiting and refusals."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from click.testing import CliRunner

from openinterpretability.cli import cli
from openinterpretability.core.schemas import NeuronIdentifier
from openinterpretability.harness.explanation_scores import finalize_scores
from tests.test_baseline import readout, scores


def _record(record, neuron=2, hours=0, explanation_scores=None, patchscope=None):
    ident = NeuronIdentifier(model_name="toy", layer=1, neuron_idx=neuron, pole="positive")
    sub = record.submission.model_copy(update={"identifier": ident, "patchscope": patchscope})
    verdict = record.verdict.model_copy(update={
        "explanation_scores": explanation_scores or scores(),
        "timestamp": record.verdict.timestamp + timedelta(hours=hours),
    })  # fmt: skip
    return record.model_copy(update={"submission": sub, "verdict": verdict})


@pytest.fixture
def loads(toy_runner, monkeypatch):
    """The toy runner in place of a model download, and each load it stands in for."""
    calls = []

    def load(*args, **kwargs):
        calls.append(args)
        return toy_runner

    monkeypatch.setattr("openinterpretability.cli._load_runner", load)
    monkeypatch.setattr(
        "openinterpretability.harness.explanation_scores.embedding_similarities",
        lambda _model, query, texts: [float(len(query) % 2)] * len(texts),
    )
    return calls


def _archive(tmp_path, *records):
    from openinterpretability.builder.wiki import archive_record

    for r in records:
        archive_record(r, tmp_path)
    return tmp_path


def _invoke(wiki, judge="external/judge"):
    args = ["patchscope-baseline", "--wiki-dir", wiki, "--work-dir", wiki / "work",
            "--out", wiki / "out" / "baseline", "--judge-model", judge]  # fmt: skip
    return CliRunner().invoke(cli, [str(a) for a in args])


def _rows(wiki):
    jsonl = (wiki / "out" / "baseline").with_suffix(".jsonl")
    return [json.loads(line) for line in jsonl.read_text().splitlines()]


def test_readouts_are_read_once_per_unit_with_one_runner_per_model(tmp_path, record, loads):
    no_encoder = finalize_scores(scores().model_copy(update={"embedding_model": None}))
    stored = readout({("category", 2, 2.0): "crop"})
    wiki = _archive(
        tmp_path,
        _record(record, neuron=2),
        _record(record, neuron=2, hours=1),  # a second experiment on the same unit
        _record(record, neuron=3),  # another unit of the same model
        _record(record, neuron=4, explanation_scores=no_encoder, patchscope=stored),
    )
    result = _invoke(wiki, judge="external/other")  # not the judge of the agent's scores
    assert result.exit_code == 0, result.output
    assert len(loads) == 1
    cached = sorted(p.name for p in (wiki / "work" / "readouts").rglob("*.json"))
    assert cached == ["N2-positive.json", "N3-positive.json"]  # N4 has its own readout
    assert not (wiki / "work" / "judge").exists() and "await" not in result.output
    rows = {(r["slug"], r["experiment_id"]): r for r in _rows(wiki)}
    assert len(rows) == 4
    assert all(r["baseline"]["detection"] is None for r in rows.values())
    by_slug = {slug: r for (slug, _), r in rows.items()}
    assert by_slug["toy/L1/N4-positive"]["reading"] == "crop"
    assert by_slug["toy/L1/N4-positive"]["baseline"]["embedding"] is None
    assert by_slug["toy/L1/N3-positive"]["baseline"]["embedding"] is not None


def test_a_waiting_task_is_kept_and_a_bad_answer_is_refused(tmp_path, record, loads):
    wiki = _archive(tmp_path, _record(record))
    assert _invoke(wiki).exit_code == 0
    (task,) = (wiki / "work" / "judge").rglob("*.task.json")
    task.write_text(task.read_text() + "\n")  # mark it, to see that it is not rewritten
    marked = task.read_text()
    again = _invoke(wiki)
    assert again.exit_code == 0 and "1 experiments await judge answers" in again.output
    assert task.read_text() == marked
    answer = task.with_name(task.name.replace(".task.json", ".detection.json"))
    answer.write_text("not json")
    answer.with_name(answer.name.replace("detection", "fuzzing")).write_text('{"correct": []}')
    bad = _invoke(wiki)
    assert bad.exit_code != 0 and "is not a valid answer; judge it again" in bad.output


def test_without_fuzzing_windows_the_detection_answer_is_enough(tmp_path, record, loads):
    detection_only = finalize_scores(scores().model_copy(update={"fuzzing_windows": []}))
    wiki = _archive(tmp_path, _record(record, explanation_scores=detection_only))
    assert _invoke(wiki).exit_code == 0
    (task,) = (wiki / "work" / "judge").rglob("*.task.json")
    task.with_name(task.name.replace(".task.json", ".detection.json")).write_text(
        '{"active": [1, 2]}'
    )
    done = _invoke(wiki)
    assert done.exit_code == 0, done.output
    assert "await" not in done.output
    (row,) = _rows(wiki)
    assert row["baseline"]["detection"] == 1.0 and row["baseline"]["fuzzing"] is None
    assert row["fuzzing_answer"] is None


def test_neuronpedia_baseline_skips_unhosted_units_and_stops_when_it_cannot_read(
    tmp_path, record, monkeypatch
):
    from openinterpretability.core.identifiers import FeatureIdentifier

    hosted = FeatureIdentifier(model_name="toy", layer=1, dictionary="toy-tc", feature_idx=7)
    unhosted = hosted.model_copy(update={"feature_idx": 9})
    unscored = record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"explanation_scores": None})}
    )

    def with_ident(ident):
        return _record(record).model_copy(
            update={"submission": record.submission.model_copy(update={"identifier": ident})}
        )

    archive = [unscored, _record(record), with_ident(unhosted), with_ident(hosted)]
    monkeypatch.setattr(
        "openinterpretability.builder.wiki.load_archive_paths",
        lambda _w: [(None, r) for r in archive],
    )
    monkeypatch.setattr(
        "openinterpretability.crossref.latent_url",
        lambda d, layer, i, api=False: None if i == 9 else f"https://np.example/{layer}/{i}",
    )
    monkeypatch.setattr("openinterpretability.crossref.latent_explanations", lambda *_a: None)
    out = tmp_path / "out" / "np"
    args = ["neuronpedia-baseline", "--work-dir", str(tmp_path / "work"), "--out", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    assert "1 experiments on latents Neuronpedia hosts." in result.output
    assert "could not read https://np.example/1/7; try again later" in result.output
    assert not out.with_suffix(".jsonl").exists()
