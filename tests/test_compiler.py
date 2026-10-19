import json

import duckdb
import pyarrow.parquet as pq
import pytest

from openinterp.builder.compiler import compile_site, histogram_bars, load_records
from openinterp.builder.parquet_export import REGISTRY_SCHEMA
from openinterp.core.exceptions import RegistryError
from openinterp.core.schemas import AuditRecord


@pytest.fixture
def records(record, verdict):
    rejected = AuditRecord(
        submission=record.submission.model_copy(
            update={"identifier": record.submission.identifier.model_copy(update={"neuron_idx": 9})}
        ),
        verdict=verdict.model_copy(update={"status": "REJECTED", "confidence_score": 0.2}),
    )
    return [record, rejected]


def test_load_records(tmp_path, record):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.json").write_text(record.model_dump_json())
    assert load_records(tmp_path) == [record]

    (tmp_path / "bad.json").write_text(json.dumps({"submission": {}}))
    with pytest.raises(RegistryError, match="bad.json"):
        load_records(tmp_path)


def test_compile_site_publishes_only_verified(tmp_path, records):
    report = compile_site(records, tmp_path)
    assert (report.records_loaded, report.records_published) == (2, 1)

    page = tmp_path / "neurons" / "Qwen--Qwen2.5-0.5B" / "L3" / "N117-positive.html"
    assert page.exists()
    assert not (tmp_path / "neurons" / "Qwen--Qwen2.5-0.5B" / "L3" / "N9-positive.html").exists()

    html = page.read_text()
    assert "data-pagefind-body" in html
    assert 'href="../../../index.html"' in html  # relative root resolves from nested pages
    assert "Fires on the closing quote of reported speech." in html
    assert "&#34;Hello,&#34; she said." in html  # prompts are HTML-escaped
    assert "<rect" in html  # histogram rendered
    assert "VERIFIED" in html

    index = (tmp_path / "index.html").read_text()
    assert "neurons/Qwen--Qwen2.5-0.5B/L3/N117-positive.html" in index
    assert "1/24 layers" in index  # coverage uses model_num_layers
    assert "openai/gpt-test" in index  # agent leaderboard
    assert "duckdb-wasm@1.29.0" in index


def test_include_unverified(tmp_path, records):
    assert compile_site(records, tmp_path, include_unverified=True).records_published == 2


def test_parquet_export(tmp_path, records):
    compile_site(records, tmp_path)
    table = pq.read_table(tmp_path / "data" / "registry.parquet")
    assert table.schema.equals(REGISTRY_SCHEMA)
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["slug"] == "Qwen--Qwen2.5-0.5B/L3/N117-positive"
    assert row["counterfactuals"][0]["observed_delta"] == pytest.approx(0.71)
    assert row["tests_passed"] == 1
    assert AuditRecord.model_validate_json(row["record_json"]) == records[0]

    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    snapshot = tmp_path / manifest["snapshot"]
    assert snapshot.read_bytes() == (tmp_path / "data" / "registry.parquet").read_bytes()

    count = duckdb.sql(f"SELECT count(*) FROM '{snapshot}' WHERE is_lookahead = false").fetchone()
    assert count == (1,)


def test_empty_registry_builds(tmp_path):
    report = compile_site([], tmp_path)
    assert report.records_published == 0
    assert "No verified audits yet" in (tmp_path / "index.html").read_text()


def test_histogram_bars(submission):
    bars = histogram_bars(submission.activation_histogram)
    assert len(bars) == 3
    assert bars[0]["h"] == pytest.approx(140)  # tallest bin fills the chart
    assert bars[0]["h"] > bars[1]["h"] > bars[2]["h"]
    assert histogram_bars(None) == []


def test_cli_build(tmp_path, record):
    from click.testing import CliRunner

    from openinterp.cli import cli

    audits = tmp_path / "audits"
    audits.mkdir()
    (audits / "a.json").write_text(record.model_dump_json())
    result = CliRunner().invoke(
        cli, ["build", "--audits-dir", str(audits), "--output-dir", str(tmp_path / "dist")]
    )
    assert result.exit_code == 0, result.output
    assert "Published 1/1 records" in result.output
    assert (tmp_path / "dist" / "index.html").exists()


def _tree(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_build_is_deterministic(tmp_path, records):
    """Unchanged records must rebuild byte-for-byte, so hourly deploys upload nothing."""
    from datetime import UTC, datetime

    compile_site(records, tmp_path / "a")
    compile_site(list(reversed(records)), tmp_path / "b")  # input order must not matter
    a, b = _tree(tmp_path / "a"), _tree(tmp_path / "b")
    assert a == b

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    if today != "2026-10-01":  # the fixture's own review date
        assert not any(today.encode() in content for content in a.values())


def test_rebuild_removes_stale_pages(tmp_path, records, record):
    compile_site(records, tmp_path, include_unverified=True)
    stale = tmp_path / "neurons" / "Qwen--Qwen2.5-0.5B" / "L3" / "N9-positive.html"
    assert stale.exists()
    (tmp_path / "keep.txt").write_text("not ours")
    compile_site([record], tmp_path)
    assert not stale.exists()
    assert (tmp_path / "keep.txt").exists()
    assert len(list((tmp_path / "data" / "snapshots").iterdir())) == 1
