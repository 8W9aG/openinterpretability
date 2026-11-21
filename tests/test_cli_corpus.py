"""``open-interpretability corpus``: rebuilding a record's corpus and building blends."""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from openinterpretability.cli import cli
from openinterpretability.core.corpora import CorpusSource
from openinterpretability.corpus import source_path


def _source(**kw) -> CorpusSource:
    fields = dict(dataset="org/a", revision="r1", files="*.parquet", documents=2, seed=7,
                  max_chars=9, docs_per_row_group=20, sha256="s" * 64)  # fmt: skip
    return CorpusSource(**(fields | kw))


def _record_scanning(monkeypatch, source: CorpusSource):
    """A record file whose provenance names ``source``, read without a real record."""
    provenance = SimpleNamespace(corpus="c.jsonl", corpus_sha256=source.sha256,
                                 corpus_source=source)  # fmt: skip
    monkeypatch.setattr(
        "openinterpretability.cli_review.load_submission",
        lambda path: SimpleNamespace(provenance=provenance),
    )


def _fake_hub(monkeypatch, built: CorpusSource):
    """Build from the Hub without the network: write the corpus, return ``built``."""

    def build(dataset, out, **kw):
        out.write_text('{"text": "x"}\n')
        source_path(out).write_text(built.model_dump_json())
        return built

    monkeypatch.setattr("openinterpretability.corpus_rebuild.build_from_hub", build)


def test_same_as_rebuilds_the_records_corpus_and_reports_it(monkeypatch, tmp_path):
    source = _source(lang_detector="fasttext-lid.176", code_detector="magika")
    _record_scanning(monkeypatch, source)
    _fake_hub(monkeypatch, source)
    record = tmp_path / "record.json"
    record.write_text("{}")
    out = tmp_path / "c.jsonl"
    result = CliRunner().invoke(cli, ["corpus", "--same-as", str(record), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert f"Wrote 2 documents to {out}" in result.output
    assert "Languages detected by fasttext-lid.176" in result.output
    assert "Programming languages detected by magika" in result.output
    assert "Licence: not stated on the dataset card" in result.output
    assert out.exists()


def test_same_as_names_the_record_when_the_rebuild_differs(monkeypatch, tmp_path):
    _record_scanning(monkeypatch, _source())
    _fake_hub(monkeypatch, _source(sha256="d" * 64))
    record = tmp_path / "record.json"
    record.write_text("{}")
    out = tmp_path / "c.jsonl"
    result = CliRunner().invoke(cli, ["corpus", "--same-as", str(record), "--out", str(out)])
    assert result.exit_code != 0
    assert f"{record}: the rebuilt corpus has SHA-256 {'d' * 64}" in result.output
    assert not out.exists()


def test_a_blend_lists_its_parts_and_warns_when_short(monkeypatch, tmp_path):
    parts = [_source(documents=1), _source(dataset="org/b", documents=1)]
    built = _source(documents=2, license="cc-by-4.0", components=parts)
    monkeypatch.setattr("openinterpretability.blends.build_blend", lambda name, out, **kw: built)
    args = ["corpus", "--docs", "5", "--out", str(tmp_path / "c.jsonl")]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert f"  1 from {parts[0]}" in result.output and f"  1 from {parts[1]}" in result.output
    assert "warning: only 2 of 5 documents found" in result.output
    assert "Licence: cc-by-4.0" in result.output
