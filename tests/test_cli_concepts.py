from click.testing import CliRunner

from openinterpretability.cli import cli
from openinterpretability.core.probing import ConceptSource
from openinterpretability.harness.probing import read_labels
from tests.test_concepts import hub  # noqa: F401  (the fake Hub fixture)


def test_cli_needs_out_with_a_label(hub):  # noqa: F811
    result = CliRunner().invoke(cli, ["concepts", "--dataset", "ag_news", "--label", "Sports"])
    assert result.exit_code == 2
    assert "--out is required with --label." in result.output


def test_cli_writes_the_concept_set_and_its_licence(tmp_path, hub):  # noqa: F811
    out = tmp_path / "per.jsonl"
    args = ["concepts", "--dataset", "conll2003-ner", "--label", "PER", "--out", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert f"Wrote {out}: eriktks/conll2003@sha1" in result.output
    assert "Licence: other" in result.output
    assert "warning" not in result.output
    [(text, spans)] = read_labels(out)
    assert [text[s:e] for s, e in spans] == ["John", "Smith"]


def test_cli_reports_a_missing_label_as_a_click_error(tmp_path, hub):  # noqa: F811
    out = tmp_path / "loc.jsonl"
    args = ["concepts", "--dataset", "conll2003-ner", "--label", "LOC", "--out", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert "Error:" in result.output and "it has: PER, ORG" in result.output
    assert not out.exists()


def test_cli_warns_about_skipped_sentences_and_an_unstated_licence(tmp_path, monkeypatch):
    source = ConceptSource(
        preset="ud-upos",
        dataset="universal-dependencies/universal_dependencies",
        revision="abcdef0123",
        files="parquet/en_ewt/dev.parquet",
        column="upos",
        label="DET",
        level="token",
        documents=2,
        positive_documents=1,
        spans=1,
        skipped_sentences=3,
        seed=0,
        max_chars=3000,
        license=None,
        sha256="0" * 64,
    )
    monkeypatch.setattr("openinterpretability.concepts.build_concept", lambda *a, **k: source)
    out = tmp_path / "det.jsonl"
    args = ["concepts", "--dataset", "ud-upos", "--label", "DET", "--out", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "warning: 3 sentences' words not found" in result.output
    assert "Licence: not stated on the dataset card" in result.output
