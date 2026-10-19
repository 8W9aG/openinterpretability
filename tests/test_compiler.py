import json
import re
from datetime import timedelta

import duckdb
import pyarrow.parquet as pq
import pytest

from openinterp.builder.compiler import compile_site, histogram_bars, load_records
from openinterp.builder.parquet_export import REGISTRY_SCHEMA
from openinterp.core.exceptions import RegistryError
from openinterp.core.schemas import AdversarialTestResult, AuditRecord

MODEL = "Qwen--Qwen2.5-0.5B"


def probes(neg_pass, neg_fail, pos_pass, pos_fail):
    serial = iter(range(10_000))

    def r(kind, passed, n):
        # Distinct word sets, so probes never cluster as near-duplicates.
        return [
            AdversarialTestResult(kind=kind, prompt=f"w{i}a w{i}b w{i}c", passed=passed)
            for i in (next(serial) for _ in range(n))
        ]

    return (
        r("hard_negative", True, neg_pass)
        + r("hard_negative", False, neg_fail)
        + r("positive_holdout", True, pos_pass)
        + r("positive_holdout", False, pos_fail)
    )


def experiment(record, *, idx, pole="positive", status, counts, hours=0, text=None):
    ident = record.submission.identifier.model_copy(update={"neuron_idx": idx, "pole": pole})
    sub = record.submission.model_copy(update={"identifier": ident})
    if text:
        sub = sub.model_copy(update={"natural_language_explanation": text})
    verdict = record.verdict.model_copy(
        update={
            "status": status,
            "adversarial_test_results": probes(*counts),
            "timestamp": record.verdict.timestamp + timedelta(hours=hours),
        }
    )
    return AuditRecord(submission=sub, verdict=verdict)


@pytest.fixture
def records(record):
    return [
        # N117+: an early, thinly-probed attempt, then a strong verification.
        experiment(
            record, idx=117, status="NEEDS_REVISION", counts=(2, 1, 1, 1), text="Early idea."
        ),
        experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0), hours=2),
        # N117−: a well-probed null result.
        experiment(
            record,
            idx=117,
            pole="negative",
            status="REJECTED",
            counts=(4, 6, 1, 4),
            hours=1,
            text="Fires on commas.",
        ),
        # N9+ in the same layer: rejected.
        experiment(record, idx=9, status="REJECTED", counts=(3, 3, 1, 3), text="Fires on dates."),
    ]


def test_load_records(tmp_path, record):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.json").write_text(record.model_dump_json())
    assert load_records(tmp_path) == [record]

    (tmp_path / "bad.json").write_text(json.dumps({"submission": {}}))
    with pytest.raises(RegistryError, match="bad.json"):
        load_records(tmp_path)


def test_every_experiment_published_in_hierarchy(tmp_path, records):
    report = compile_site(records, tmp_path)
    assert report.experiments_published == 4
    model = tmp_path / "models" / MODEL
    for page in ["index.html", "L3/index.html", "L3/N117.html", "L3/N9.html"]:
        assert (model / page).exists(), page

    neuron = (model / "L3" / "N117.html").read_text()
    assert "data-pagefind-body" in neuron
    assert 'href="../../../index.html"' in neuron  # relative root from nested pages
    assert "Fires on commas." in neuron and "Early idea." in neuron  # null + superseded shown
    assert "superseded" in neuron and "latest for this pole" in neuron
    assert "&#34;Hello,&#34; she said." in neuron  # prompts are HTML-escaped
    assert "<rect" in neuron  # histogram rendered

    layer = (model / "L3" / "index.html").read_text()
    assert 'href="../../../models/Qwen--Qwen2.5-0.5B/L3/N9.html"' in layer
    assert "1/24 layers" in (model / "index.html").read_text()


def test_experiments_ordered_by_conclusiveness(tmp_path, records):
    compile_site(records, tmp_path)
    neuron = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    order = re.findall(r'<section id="(\w+)-(\d+T\d+Z)"', neuron)
    # Strong verification, then the well-probed null, then the thin early attempt.
    assert [pole for pole, _ in order] == ["positive", "negative", "positive"]
    assert "strong evidence for" in neuron and "evidence against" in neuron


def test_index_highlights_conclusive_results_both_ways(tmp_path, records):
    compile_site(records, tmp_path)
    index = (tmp_path / "index.html").read_text()
    assert "Most conclusive results" in index
    assert "Fires on commas." in index  # a null result is featured
    assert "models/Qwen--Qwen2.5-0.5B/L3/N117.html#positive-" in index
    assert "openai/gpt-test" in index
    assert "duckdb-wasm@1.29.0" in index


def test_parquet_has_every_experiment(tmp_path, records):
    compile_site(records, tmp_path)
    table = pq.read_table(tmp_path / "data" / "registry.parquet")
    assert table.schema.equals(REGISTRY_SCHEMA)
    rows = table.to_pylist()
    assert len(rows) == 4
    latest = {(r["neuron_idx"], r["pole"]): r for r in rows if r["is_latest"]}
    assert latest[(117, "positive")]["status"] == "VERIFIED"
    assert sum(not r["is_latest"] for r in rows) == 1
    assert all(0 <= r["conclusiveness"] <= 1 for r in rows)

    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    snapshot = tmp_path / manifest["snapshot"]
    assert snapshot.read_bytes() == (tmp_path / "data" / "registry.parquet").read_bytes()
    nulls = duckdb.sql(f"SELECT count(*) FROM '{snapshot}' WHERE status = 'REJECTED'").fetchone()
    assert nulls == (2,)


def test_empty_registry_builds(tmp_path):
    report = compile_site([], tmp_path)
    assert report.experiments_published == 0
    assert "No experiments yet" in (tmp_path / "index.html").read_text()


def test_histogram_bars(submission):
    bars = histogram_bars(submission.activation_histogram)
    assert len(bars) == 3
    assert bars[0]["h"] == pytest.approx(140)  # tallest bin fills the chart
    assert bars[0]["h"] > bars[1]["h"] > bars[2]["h"]
    assert histogram_bars(None) == []


def test_cli_build_from_wiki_archive(tmp_path, records):
    from click.testing import CliRunner

    from openinterp.builder.wiki import write_experiment_page
    from openinterp.cli import cli

    for rec in records:
        write_experiment_page(rec, tmp_path / "wiki")
    result = CliRunner().invoke(
        cli, ["build", "--wiki-dir", str(tmp_path / "wiki"), "--output-dir", str(tmp_path / "d")]
    )
    assert result.exit_code == 0, result.output
    assert "Published 4 experiments" in result.output


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
    if today not in ("2026-10-01", "2026-10-02"):  # the fixtures' own review dates
        assert not any(today.encode() in content for content in a.values())


def test_rebuild_removes_stale_pages(tmp_path, records):
    compile_site(records, tmp_path)
    stale = tmp_path / "models" / MODEL / "L3" / "N9.html"
    assert stale.exists()
    (tmp_path / "keep.txt").write_text("not ours")
    compile_site(records[:3], tmp_path)
    assert not stale.exists()
    assert (tmp_path / "keep.txt").exists()
    assert len(list((tmp_path / "data" / "snapshots").iterdir())) == 1


def test_vocabulary_lens_rendered(tmp_path, record):
    from openinterp.core.schemas import VocabularyLens

    lens = VocabularyLens(
        logit_promoted=["oor"],
        logit_suppressed=["x"],
        jacobian_promoted=[" including"],
        jacobian_suppressed=[" excluding"],
        overlap=0.0,
        direction_cosine=0.31,
        logit_kurtosis=3.2,
        jacobian_kurtosis=9.5,
        prompts=100,
    )
    rec = record.model_copy(
        update={"submission": record.submission.model_copy(update={"vocabulary_lens": lens})}
    )
    compile_site([rec], tmp_path)
    page = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "Jacobian lens" in page and " including" in page and "cos(w, J·w) 0.31" in page
    row = pq.read_table(tmp_path / "data" / "registry.parquet").to_pylist()[0]
    assert row["jacobian_promoted"] == [" including"] and row["lens_overlap"] == 0.0
