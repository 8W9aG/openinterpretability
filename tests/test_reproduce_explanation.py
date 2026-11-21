"""``reproduce`` measures an explanation record again from its stored inputs (#4793)."""

import json

import pytest
from click.testing import CliRunner

from openinterpretability import explain as ex
from openinterpretability.core.explanation_records import Narration, load_all
from openinterpretability.reproduce import Tolerance
from openinterpretability.reproduce_explanation import (
    means_corpus,
    not_reproducible,
    reproduce_explanation,
    unquoted_numbers,
)

TEXTS = ["a b c d e f g h", "h g f e d c b a", "a c e g b d f h"] * 20
PROMPT = "a b c d"


@pytest.fixture
def recorded(toy_runner, tmp_path, monkeypatch):
    """``explain --record`` on the toy model, with its top unit VERIFIED: (path, record)."""
    from openinterpretability.cli import cli

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    means = ex.unit_means(toy_runner, "toy", TEXTS, 200)
    top = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=1).poles[0]
    labels = {ex._slug("toy", top.key): ("VERIFIED", "fires on a", 1.0)}
    monkeypatch.setattr(ex, "registry_labels", lambda *a, **k: labels)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    wiki = tmp_path / "wiki"
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(wiki), "--top", "5", "--edges", "2", "--record",
            "--no-land"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    (((path, record),),) = load_all(wiki).values()
    return path, record, corpus


def _scaled(unit):
    return unit.model_copy(update={"attribution": unit.attribution * 1.5})


def _again(record, toy_runner):
    return reproduce_explanation(record, toy_runner, TEXTS, Tolerance(), groups=[])


def test_the_record_keeps_its_inputs(recorded):
    _, record, _ = recorded
    inputs = record.inputs
    assert inputs.explained and inputs.means_tokens == 200 and inputs.means_window == 4
    assert inputs.environment.dtype and (inputs.top, inputs.edges) == (5, 2)
    assert record.explained_share > 0 and not_reproducible(record) is None


def test_a_recorded_explanation_reproduces(recorded, toy_runner):
    report = _again(recorded[1], toy_runner)
    assert report.ok, report.mismatches
    assert report.compared > 20


@pytest.mark.parametrize("edit", [
    lambda r: {"explained_share": r.explained_share + 0.2},
    lambda r: {"prompt_coverage": (r.prompt_coverage or 0.0) - 0.3},
    lambda r: {"units": [_scaled(r.units[0]), *r.units[1:]]},
    lambda r: {"figures": r.figures.model_copy(update={"logit": r.figures.logit + 1.0})},
    lambda r: {"answer": " zz"},
    lambda r: {"inputs": r.inputs.model_copy(update={"explained": []})},
])  # fmt: skip
def test_an_edited_record_fails(recorded, toy_runner, edit):
    record = recorded[1]
    report = _again(record.model_copy(update=edit(record)), toy_runner)
    assert not report.ok


def test_an_old_record_is_reported_and_a_new_one_must_have_inputs(recorded, toy_runner,
                                                                  monkeypatch):  # fmt: skip
    from openinterpretability.cli import cli

    path, record, corpus = recorded
    monkeypatch.setattr("openinterpretability.cli._load_runner", lambda *a, **k: toy_runner)
    out = CliRunner().invoke(cli, ["reproduce", str(path), "--corpus", str(corpus)])
    assert out.exit_code == 0, out.output
    assert "REPRODUCED" in out.output and "1/1 record(s) reproduced" in out.output
    path.write_text(record.model_copy(update={"inputs": None}).model_dump_json())
    old = CliRunner().invoke(cli, ["reproduce", str(path)])
    assert old.exit_code == 0 and "NOT REPRODUCIBLE" in old.output and "#4793" in old.output


def test_an_edited_record_fails_the_command(recorded, toy_runner, monkeypatch):
    from openinterpretability.cli import cli

    path, record, corpus = recorded
    monkeypatch.setattr("openinterpretability.cli._load_runner", lambda *a, **k: toy_runner)
    path.write_text(record.model_copy(update={"blocked_share": 0.5}).model_dump_json())
    out = CliRunner().invoke(cli, ["reproduce", str(path), "--corpus", str(corpus)])
    assert out.exit_code == 1 and "blocked share" in out.output


def test_a_corpus_with_no_source_must_be_given(recorded):
    record = recorded[1]
    with pytest.raises(ValueError, match="no source"):
        means_corpus(record, {})
    assert means_corpus(record, {record.inputs.means_corpus_sha256: TEXTS}) == TEXTS


def test_the_narration_may_quote_only_the_records_numbers(recorded):
    record = recorded[1]
    share = f"{record.explained_share:.1%}"
    told = Narration(paragraph=f"Verified units hold {share}; the rest is 12.345 logits.",
                     narrator_model="external/x")  # fmt: skip
    assert unquoted_numbers(record.model_copy(update={"narration": told})) == ["12.345"]


def test_means_read_another_way_are_not_reproducible(recorded):
    record = recorded[1]
    old = record.model_copy(
        update={"inputs": record.inputs.model_copy(update={"means_window": 99})}
    )
    why = not_reproducible(old)
    assert why.startswith("its means were read in windows of 99 tokens") and "reads 4" in why


def test_the_built_in_corpus_is_used_when_its_hash_matches(recorded):
    from openinterpretability.evidence import read_corpus

    record = recorded[1]
    texts, _, sha, _ = read_corpus(None)
    builtin = record.model_copy(
        update={"inputs": record.inputs.model_copy(update={"means_corpus_sha256": sha})}
    )
    assert means_corpus(builtin, {}) == texts


def _sourced(record):
    from openinterpretability.core.corpora import CorpusSource

    source = CorpusSource(dataset="org/a", revision="r1", files="*.parquet", documents=2,
                          seed=7, max_chars=9, sha256="s" * 64)  # fmt: skip
    inputs = record.inputs.model_copy(update={"means_corpus_source": source})
    return record.model_copy(update={"inputs": inputs})


def test_a_corpus_with_a_source_is_built_again_once_into_the_cache(recorded, monkeypatch,
                                                                   tmp_path):  # fmt: skip
    built = []

    def rebuild(source, out):
        built.append(out)
        out.write_text("".join(json.dumps({"text": t}) + "\n" for t in ["one", "two"]))
        return source

    monkeypatch.setenv("OPENINTERPRETABILITY_REBUILT_CORPORA", str(tmp_path / "rebuilt"))
    monkeypatch.setattr("openinterpretability.corpus_rebuild.rebuild", rebuild)
    record = _sourced(recorded[1])
    assert means_corpus(record, {}) == ["one", "two"]
    assert means_corpus(record, {}) == ["one", "two"]
    assert built == [tmp_path / "rebuilt" / f"{'s' * 64}.jsonl"]  # built once, then read


def test_a_corpus_cannot_be_built_again_with_the_cache_off(recorded, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_REBUILT_CORPORA", "off")
    with pytest.raises(ValueError, match="corpus cache is off"):
        means_corpus(_sourced(recorded[1]), {})


def test_a_record_without_figures_reproduces_without_them(recorded, toy_runner):
    record = recorded[1].model_copy(update={"figures": None})
    full = _again(recorded[1], toy_runner)
    report = _again(record, toy_runner)
    assert report.ok, report.mismatches
    assert report.compared < full.compared


def test_a_narration_quoting_other_numbers_is_a_warning(recorded, toy_runner):
    told = Narration(paragraph="It holds 12.345 logits.", narrator_model="external/x")
    report = _again(recorded[1].model_copy(update={"narration": told}), toy_runner)
    assert report.ok, report.mismatches
    assert report.warnings == ["the narration quotes numbers that the record doesn't hold: 12.345"]


def test_a_narration_that_quotes_no_number_has_no_warning(recorded, toy_runner):
    told = Narration(paragraph="It copies the subject.", narrator_model="external/x")
    report = _again(recorded[1].model_copy(update={"narration": told}), toy_runner)
    assert report.ok, report.mismatches
    assert report.warnings == []


def test_a_unit_that_is_no_longer_attributed_fails(recorded, toy_runner):
    record = recorded[1]
    gone = record.units[0].model_copy(update={"unit": "L0/N999+"})
    report = _again(record.model_copy(update={"units": [gone, *record.units[1:]]}), toy_runner)
    assert "L0/N999+: recorded 'attributed', reproduced 'no attribution'" in report.mismatches


def test_a_circuit_is_compared_and_a_lost_definition_is_a_warning(recorded, toy_runner):
    from openinterpretability.core.explanation_records import ExplainedUnit
    from openinterpretability.explain_groups import GroupDef

    record = recorded[1]
    circuit = GroupDef("circuit", "C4", "C4", ((1, 2, 1.0),))
    inputs = record.inputs
    means = ex.unit_means(toy_runner, "toy", TEXTS, inputs.means_tokens)
    result = ex.explain(toy_runner, "toy", record.prompt, means, {}, token=inputs.token_id,
                        position=record.position, top=inputs.top, edges=inputs.edges,
                        groups=[circuit])  # fmt: skip
    (found,) = result.groups

    def group(name, attribution):
        return ExplainedUnit(unit=name, kind="circuit", share=0.0, attribution=attribution,
                             status="unstudied")  # fmt: skip

    honest = record.model_copy(update={"units": [*record.units, group("C4", found.attribution),
                                                 group("C9", 1.0)]})  # fmt: skip
    report = reproduce_explanation(honest, toy_runner, TEXTS, Tolerance(), groups=[circuit])
    assert report.ok, report.mismatches
    assert report.warnings == ["C9: its definition is no longer in the repository"]
    edited = honest.model_copy(update={"units": [*record.units, group("C4", 99.0)]})
    report = reproduce_explanation(edited, toy_runner, TEXTS, Tolerance(), groups=[circuit])
    assert any(m.startswith("C4 attribution: recorded 99.0") for m in report.mismatches)


def test_a_record_without_its_coverage_detail_fails(recorded, toy_runner):
    report = _again(recorded[1].model_copy(update={"coverage_detail": None}), toy_runner)
    assert "coverage detail: recorded True, reproduced False" in report.mismatches


def test_a_reading_of_a_unit_the_figures_no_longer_hold_fails(recorded, toy_runner):
    record = recorded[1]
    reading = {**record.figures.reading, "L9/N9+": [1.0, 2.0]}
    figures = record.figures.model_copy(update={"reading": reading})
    report = _again(record.model_copy(update={"figures": figures}), toy_runner)
    assert report.mismatches == ["L9/N9+ reading: recorded 2, reproduced None"]


def test_figures_without_an_ablation_table_skip_its_top_token(recorded, toy_runner):
    record = recorded[1]
    assert record.figures.ablation  # the toy record has one
    figures = record.figures.model_copy(update={"ablation": []})
    report = _again(record.model_copy(update={"figures": figures}), toy_runner)
    assert report.ok, report.mismatches
    assert report.compared < _again(record, toy_runner).compared


def _reproducer(toy_runner, monkeypatch, **options):
    from openinterpretability.cli_reproduce import _Reproducer

    monkeypatch.setattr("openinterpretability.cli._load_runner", lambda *a, **k: toy_runner)
    defaults = {"signers": "", "landed": {}, "tolerance": Tolerance(), "scan_corpora": {},
                "allowed_models": (), "device": "cpu"}  # fmt: skip
    return _Reproducer(**(defaults | options))


def test_an_invalid_explanation_record_fails(recorded, toy_runner, monkeypatch):
    from openinterpretability.reproduce_explanation import check_path

    path = recorded[0]
    path.write_text("{}")
    assert check_path(_reproducer(toy_runner, monkeypatch), path) == (
        "FAILED", ["invalid explanation record"], [])  # fmt: skip


def test_a_new_explanation_record_must_be_reproducible(recorded, toy_runner, monkeypatch):
    from openinterpretability.reproduce_explanation import check_path

    path, record, _ = recorded
    path.write_text(record.model_copy(update={"inputs": None}).model_dump_json())
    reproducer = _reproducer(toy_runner, monkeypatch, new={path.resolve()})
    status, (message,), _ = check_path(reproducer, path)
    assert status == "FAILED"
    assert message.startswith("a new explanation record must be reproducible: it was written")


def test_a_new_explanation_record_that_matches_the_registry_reproduces(recorded, toy_runner,
                                                                       monkeypatch):  # fmt: skip
    from openinterpretability.reproduce_explanation import check_path

    path, record, _ = recorded
    corpora = {record.inputs.means_corpus_sha256: TEXTS}
    reproducer = _reproducer(toy_runner, monkeypatch, new={path.resolve()}, scan_corpora=corpora)
    status, mismatches, _ = check_path(reproducer, path)
    assert (status, mismatches) == ("REPRODUCED", [])


def test_an_explanation_of_a_model_ci_may_not_run_fails(recorded, toy_runner, monkeypatch):
    from openinterpretability.reproduce_explanation import check_path

    reproducer = _reproducer(toy_runner, monkeypatch, allowed_models=("other/model",))
    status, (message,), _ = check_path(reproducer, recorded[0])
    assert status == "FAILED" and message == "model toy is not on the allow-list (other/model)"


def test_an_explanation_whose_means_corpus_cannot_be_had_fails(recorded, toy_runner,
                                                               monkeypatch):  # fmt: skip
    from openinterpretability.reproduce_explanation import check_path

    status, (message,), _ = check_path(_reproducer(toy_runner, monkeypatch), recorded[0])
    assert status == "FAILED"
    assert message.startswith("its means corpus can't be had:") and "no source" in message
