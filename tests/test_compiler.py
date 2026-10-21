import json
import re
from datetime import timedelta

import duckdb
import pyarrow.parquet as pq
import pytest

from openinterp.builder.compiler import (
    causal_chart,
    compile_site,
    histogram_bars,
    load_records,
    probe_chart,
)
from openinterp.builder.parquet_export import REGISTRY_SCHEMA
from openinterp.core.exceptions import RegistryError
from openinterp.core.schemas import (
    AdversarialTestResult,
    AuditRecord,
    CausalPairEffect,
    PatchEffect,
    ReviewPolicy,
)

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
    model_page = (model / "index.html").read_text()
    # Coverage counts neurons, not layers: 24 layers × 4,864 neurons.
    assert "2/116,736 neurons studied (0.0017%)" in model_page
    assert "1/116,736 with a verified pole (0.00086%)" in model_page
    assert "1/24 layers" not in model_page


def test_coverage_uses_recorded_mlp_width(tmp_path, records):
    narrow = [
        r.model_copy(update={"submission": r.submission.model_copy(update={"model_mlp_width": 4})})
        for r in records
    ]
    compile_site(narrow, tmp_path)
    index = (tmp_path / "index.html").read_text()
    assert "2/96 neurons studied (2.1%)" in index
    assert "1/96 verified (1.0%)" in index


def test_pct_fine():
    from openinterp.builder.compiler import _pct_fine

    assert _pct_fine(None) == "—"
    assert _pct_fine(0) == "0%"
    assert _pct_fine(0.5) == "50%"
    assert _pct_fine(0.021) == "2.1%"
    assert _pct_fine(1 / 116_736) == "0.00086%"


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


def test_revisions_linked_both_ways(tmp_path, records):
    verified = records[1]  # revises records[0], the early attempt on N117+
    sub = verified.submission.model_copy(update={"supersedes": "20261001T120000Z"})
    records[1] = verified.model_copy(update={"submission": sub})
    compile_site(records, tmp_path)
    neuron = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    assert 'Revises <a class="font-mono underline" href="#positive-20261001T120000Z">' in neuron
    assert 'Revised by <a class="font-mono underline" href="#positive-20261001T140000Z">' in neuron
    rows = pq.read_table(tmp_path / "data" / "registry.parquet").to_pylist()
    assert sorted(r["supersedes"] or "" for r in rows) == ["", "", "", "20261001T120000Z"]


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


def _scored(verdict):
    def t(kind, activation, passed):
        return AdversarialTestResult(
            kind=kind, prompt=f"p {activation}", passed=passed, activation=activation
        )

    return verdict.model_copy(
        update={
            "policy": ReviewPolicy(activation_threshold=0.25),
            "adversarial_test_results": [
                t("hard_negative", 0.1, True),
                t("hard_negative", 0.1, True),  # same value: dodged, not hidden
                t("hard_negative", 3.0, False),  # beyond the scan max widens the axis
                t("positive_holdout", 2.0, True),
                t("positive_holdout", 0.2, False),
                AdversarialTestResult(kind="counterfactual", prompt="a -> b", passed=True),
            ],
        }
    )


def test_probe_chart(verdict):
    chart = probe_chart(_scored(verdict), 2.0)
    assert chart["threshold"] == pytest.approx(0.5)
    negatives, holdouts = chart["rows"]
    assert (negatives["n"], negatives["passed"], holdouts["n"], holdouts["passed"]) == (3, 2, 2, 1)
    assert chart["zero_x"] == chart["left"]
    assert negatives["dots"][-1]["cx"] == chart["right"]  # 3.0 is the axis end
    assert chart["zero_x"] < chart["threshold_x"] < chart["max_x"] < chart["right"]
    a, b = negatives["dots"][:2]
    assert a["cx"] == b["cx"] and a["cy"] != b["cy"]
    assert "failed · 0.200 (10% of max)" in holdouts["dots"][0]["label"]
    assert probe_chart(verdict, 2.0) is None  # no recorded policy, so no threshold to draw
    assert probe_chart(_scored(verdict), None) is None


def test_probe_chart_rendered(tmp_path, record):
    scored = record.model_copy(update={"verdict": _scored(record.verdict)})
    compile_site([scored], tmp_path)
    page = next((tmp_path / "models").rglob("N*.html")).read_text()
    assert "Probe strengths against the activation threshold" in page
    assert "2 of 3 hard negatives passed; 1 of 2 positive holdouts passed" in page
    assert "threshold 0.62 (25% of max)" in page


def _causal_pairs(causal):
    def patch(percentile, effect):
        return PatchEffect(effect=effect, percentile=percentile, null_median=0.01, null_max=0.05)

    return causal.model_copy(
        update={
            "target": "next_token_kl",
            "null_neurons": 32,
            "pairs_scored": 2,
            "sufficiency_effect": 0.1,
            "necessity_effect": 0.05,
            "pairs": [
                CausalPairEffect(
                    pair_index=0, sufficiency=patch(0.9, 0.2), necessity=patch(None, 0.0)
                ),
                CausalPairEffect(
                    pair_index=1, sufficiency=patch(0.3, 0.004), necessity=patch(1.0, 0.1)
                ),
            ],
        }
    )


def test_causal_chart(submission, verdict):
    assert causal_chart(verdict.causal_metrics, submission.counterfactuals) is None  # pre-0.5
    chart = causal_chart(_causal_pairs(verdict.causal_metrics), submission.counterfactuals)
    first, second = chart["rows"]
    assert (first["label"], second["label"]) == ("Pair 1", "Pair 2")
    full = chart["right"] - chart["left"]
    assert first["bars"][0]["w"] == pytest.approx(0.9 * full, abs=0.01)
    assert first["bars"][1]["w"] is None  # no matched units: nothing to rank against
    assert second["bars"][1]["w"] == pytest.approx(full)
    assert chart["baseline_x"] == pytest.approx(chart["left"] + full / 2)
    assert submission.counterfactuals[0].base_prompt in first["bars"][0]["label"]
    assert [b["value"] for b in chart["median"]["bars"]] == [0.6, 0.4]
    assert chart["median"]["y"] > second["y"]

    medians_only = causal_chart(
        verdict.causal_metrics.model_copy(update={"target": "next_token_kl"}), []
    )
    assert medians_only["rows"] == [] and medians_only["rule_y"] is None


def test_causal_chart_rendered(tmp_path, record):
    causal = _causal_pairs(record.verdict.causal_metrics)
    verdict = record.verdict.model_copy(update={"causal_metrics": causal})
    compile_site([record.model_copy(update={"verdict": verdict})], tmp_path)
    page = next((tmp_path / "models").rglob("N*.html")).read_text()
    assert "Causal effects per minimal pair" in page
    assert "Sufficiency and necessity percentiles for 2 minimal pairs" in page
    assert "Per-pair numbers" in page


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


def test_input_attribution_rendered(tmp_path, record):
    from openinterp.builder.wiki import write_experiment_page
    from openinterp.core.attribution import (
        AttributedToken,
        ContextExample,
        DistanceShare,
        InputAttribution,
    )
    from openinterp.harness.input_attribution import DISTANCE_BANDS

    shares = [0.5, 0.1, 0.1, 0.1, 0.2, 0.0]
    attribution = InputAttribution(
        baseline_token="<|endoftext|>",
        tolerance=0.1,
        examples=[
            ContextExample(
                example=0,
                peak_token='"',
                tokens=4,
                steps=16,
                converged=True,
                top=[
                    AttributedToken(offset=0, token='"', attribution=0.5, ablation_effect=0.6),
                    AttributedToken(offset=-3, token="Hi", attribution=0.3, ablation_effect=0.2),
                ],
                completeness_error=0.05,
            )
        ],
        distances=[
            DistanceShare(min_distance=lo, max_distance=hi, share=s)
            for (lo, hi), s in zip(DISTANCE_BANDS, shares, strict=True)
        ],
        mean_distance=7.25,
        ablation_agreement=0.8,
    )
    sub = record.submission.model_copy(update={"input_attribution": attribution})
    rec = record.model_copy(update={"submission": sub})
    compile_site([rec], tmp_path / "site")
    page = (tmp_path / "site" / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "Input attribution" in page and "65+ back" in page and "ρ = 0.80" in page
    row = pq.read_table(tmp_path / "site" / "data" / "registry.parquet").to_pylist()[0]
    assert row["current_token_share"] == 0.5 and row["context_mean_distance"] == 7.25

    md = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "### Input attribution" in md and "| Share | 50% | 10% |" in md
    assert '| 1 | `"` | `"` 0, 0.50 / 0.60; `Hi` -3, 0.30 / 0.20 |' in md


def test_token_heat_windows_long_snippets_around_the_peak():
    from openinterp.builder.compiler import token_heat
    from openinterp.core.schemas import ActivatingExample

    acts = [0.0] * 100
    acts[60] = 2.0
    acts[61] = 3.0
    ex = ActivatingExample(
        text="x" * 100,
        tokens=[f"t{i}" for i in range(100)],
        activations=acts,
        max_activation=3.0,
        max_token_index=61,
    )
    heat = token_heat(ex, 4.0, before=5, after=2)
    assert [t.text for t in heat.window] == [f"t{i}" for i in range(56, 64)]
    assert heat.clipped_start and heat.clipped_end and len(heat.tokens) == 100
    (peak,) = [t for t in heat.window if t.peak]
    assert (peak.text, peak.shade, peak.value) == ("t61", 0.75, 3.0)  # shade vs the neuron's max

    short = token_heat(ex.model_copy(update={"max_token_index": 1}), 4.0, before=5, after=200)
    assert not short.clipped_start and not short.clipped_end

    odd = ex.model_copy(
        update={"tokens": ["a\n", *ex.tokens[1:]], "activations": [-0.0, *acts[1:]]}
    )
    first = token_heat(odd, 4.0).tokens[0]
    assert first.text == "a↵" and str(first.value) == "0.0"


def test_heatmap_renders_exact_values_and_peak(tmp_path, record):
    compile_site([record], tmp_path)
    page = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    assert 'class="tok heat peak" style="--a: 1.0" data-v="2.500 · peak"' in page
    assert 'data-v="0.300"' in page  # exact stored value, not a rounded shade
    assert "--heat-rgb: 234 88 12" in page  # positive pole hue
    assert '<summary class="cursor-pointer">Values</summary>' in page


def test_home_lists_most_recent_experiments_newest_first(tmp_path, records):
    from openinterp.builder.compiler import registry_stats

    report = compile_site(records, tmp_path)
    recent = registry_stats(report.parquet)["recent"]
    assert [r["reviewed_iso"] for r in recent] == sorted(
        (r["reviewed_iso"] for r in recent), reverse=True
    )
    assert recent[0]["reviewed_iso"] == "2026-10-01T14:00:00Z"  # the +2 h verification
    assert recent[0]["reviewed_time"] == "14:00 UTC" and recent[0]["href"].endswith(
        recent[0]["experiment_id"]
    )
    home = (tmp_path / "index.html").read_text()
    section = home.split("Most recent experiments")[1].split("Most conclusive results")[0]
    assert section.index("Fires on commas.") < section.index("Fires on dates.")  # +1 h before +0 h
    assert '<time datetime="2026-10-01T14:00:00Z">2026-10-01</time>' in section


def test_explanation_scores_rendered_mistakes_first(tmp_path, record):
    from openinterp.core.explanation import ExplanationScores, ScoredWindow
    from openinterp.harness.explanation_scores import finalize_scores

    def window(text, acts, label, judged, marks=()):
        return ScoredWindow(text=text, tokens=list(text), activations=acts, label=label,
                            judged=judged, highlighted=list(marks),
                            quantile=0 if max(acts) >= 0.6 else None)  # fmt: skip

    scores = finalize_scores(
        ExplanationScores(
            slug=record.submission.identifier.slug, explanation="x", threshold=0.6,
            window_tokens=32, quantiles=1, quantile_edges=[], seed=3, corpus="c",
            candidate_windows=900, active_windows=40, judge_model="external/judge",
            detection_windows=[window("ok", [0.0, 0.0], False, False),
                               window("hm", [0.0, 2.0], True, False)],
            fuzzing_windows=[window("fz", [2.0, 0.0], True, True, marks=[0])],
        )
    )  # fmt: skip
    rec = record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"explanation_scores": scores})}
    )
    compile_site([rec], tmp_path)
    page = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "Explanation scores on held-out text" in page and "40 active of 900" in page
    assert page.index(">missed<") < page.index(">correct rejection<")  # mistakes first
    assert "peak marked" in page and "marks right · judged right" in page
    row = pq.read_table(tmp_path / "data" / "registry.parquet").to_pylist()[0]
    assert (row["detection_score"], row["fuzzing_score"]) == (0.5, 1.0)
