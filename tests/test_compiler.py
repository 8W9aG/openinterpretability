import json
import re
from datetime import timedelta

import duckdb
import pyarrow.parquet as pq
import pytest

from openinterp.builder.charts import causal_chart, histogram_bars, probe_chart, vocab_chart
from openinterp.builder.compiler import compile_site, load_records
from openinterp.builder.parquet_export import REGISTRY_SCHEMA
from openinterp.core.exceptions import RegistryError
from openinterp.core.schemas import (
    ActivatingExample,
    AdversarialTestResult,
    AuditRecord,
    CausalPairEffect,
    OppositePole,
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
    assert "earlier hypothesis" in neuron and "counts toward status" in neuron
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


def test_layer_map(tmp_path, records):
    from openinterp.builder.charts import layer_map
    from openinterp.builder.tree import ModelNode, build_tree

    (model,) = build_tree(records)
    chart = layer_map(model)
    assert chart["counts"] == {"VERIFIED": 1, "CONTESTED": 0, "NEEDS_REVISION": 0, "REJECTED": 1}
    assert chart["unexplored"] == 116_736 - 2
    assert len(chart["rows"]) == 24 and chart["label_every"] == 2
    # Worst status drawn first, so the verified neuron is on top; N117 shows its best pole.
    assert [(m["status"], m["label"]) for m in chart["marks"]] == [
        ("REJECTED", "L3 N9 · positive REJECTED"),
        ("VERIFIED", "L3 N117 · positive VERIFIED, negative REJECTED"),
    ]
    n9, n117 = chart["marks"]
    assert chart["left"] < n9["x"] < n117["x"] < chart["right"]
    assert n9["y"] == n117["y"] == chart["rows"][3][1]
    assert [label for _, label in chart["ticks"]] == ["0", "2,000", "4,000"]
    assert layer_map(ModelNode("m", "m", 2, "m/index.html")) is None  # no width, no map

    compile_site(records, tmp_path)
    page = (tmp_path / "models" / MODEL / "index.html").read_text()
    assert "Layer map: 2 studied neurons of 116,736" in page
    assert '<a href="../../models/Qwen--Qwen2.5-0.5B/L3/N117.html">' in page
    assert "Studied neurons per layer" not in page  # the map replaces the histogram


def test_pieces_bars(tmp_path, record, records):
    from openinterp.builder.charts import layer_pieces, model_pieces
    from openinterp.builder.tree import ModelNode, build_tree

    # N120 shares N117's 38-neuron block (114–151); the block shows the better status.
    records = [*records, experiment(record, idx=120, status="REJECTED", counts=(3, 3, 1, 3))]
    (model,) = build_tree(records)
    bars = model_pieces(model)
    assert bars["block"] == 38 and bars["cells"] == 24 * 128 and bars["lit"] == 2
    assert len(bars["rows"]) == 24 and not bars["rows"][0]["cells"]
    n9, n117 = bars["rows"][3]["cells"]
    assert (n9["x"], n9["status"]) == (0, "REJECTED")
    assert (n117["x"], n117["status"]) == (3 * 5, "VERIFIED")
    assert n117["label"].startswith("L3 N114–151: N117 positive VERIFIED")
    assert "N120 positive REJECTED" in n117["label"]

    grid = layer_pieces(model, model.layers[0])
    assert len(grid["rows"]) == 38 and grid["unexplored"] == 4864 - 3
    assert grid["counts"] == {"VERIFIED": 1, "CONTESTED": 0, "NEEDS_REVISION": 0, "REJECTED": 2}
    cell = next(c for c in grid["cells"] if c["href"].endswith("/N120.html"))
    assert (cell["x"], cell["y"]) == (grid["left"] + 120 * 5, 0)
    assert model_pieces(ModelNode("m", "m", 2, "m/index.html")) is None  # no width, no bars

    compile_site(records, tmp_path)
    index = (tmp_path / "index.html").read_text()
    assert "Studied neurons per layer" not in index  # the pieces bars replace the histogram
    assert "2 of 3,072 cells hold a studied neuron" in index
    layer = (tmp_path / "models" / MODEL / "L3" / "index.html").read_text()
    assert "Pieces map of layer 3: one cell per neuron, 128 to a row, 3 of 4,864 studied" in layer
    assert '<a href="../../../models/Qwen--Qwen2.5-0.5B/L3/N120.html"><title>' in layer


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
    status = {(r["neuron_idx"], r["pole"]): r["unit_status"] for r in rows}
    assert status == {
        (117, "positive"): "VERIFIED",
        (117, "negative"): "REJECTED",
        (9, "positive"): "REJECTED",
    }
    assert sum(r["counts_toward_status"] for r in rows) == 3  # not the early N117+ idea
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


def _lens(**scores):
    from openinterp.core.schemas import VocabularyLens

    return VocabularyLens(
        logit_promoted=["oor", "x"],
        logit_suppressed=["y", "z"],
        jacobian_promoted=[" including", "inc"],
        jacobian_suppressed=[" excluding", "\n"],
        overlap=0.0,
        direction_cosine=0.31,
        logit_kurtosis=3.2,
        jacobian_kurtosis=9.5,
        prompts=100,
        **scores,
    )


Z_SCORES = {
    "logit_promoted_z": [4.0, 3.0],
    "logit_suppressed_z": [-2.0, -1.5],
    "jacobian_promoted_z": [9.0, 6.0],
    "jacobian_suppressed_z": [-7.0, -5.0],
}


def test_vocab_chart():
    assert vocab_chart(None) is None and vocab_chart(_lens()) is None  # no z-scores recorded
    chart = vocab_chart(_lens(**Z_SCORES))
    jacobian, logit = chart["panels"]
    assert (jacobian["title"], logit["title"]) == ("Jacobian lens", "Logit lens")
    # Most promoted at the top, most suppressed at the bottom.
    assert [r["text"] for r in jacobian["rows"]] == ["␣including", "inc", "↵", "␣excluding"]
    assert [r["promoted"] for r in jacobian["rows"]] == [True, True, False, False]
    ys = [r["y"] for r in jacobian["rows"]]
    assert ys == sorted(ys) and ys[2] - ys[1] > ys[1] - ys[0]  # a gap between the sides
    # One shared axis: |z| = 9 reaches 10 on a 5-step scale, centred on zero.
    zero, half = chart["zero_x"], chart["width"] / 2 - 6
    assert [label for _, label in chart["ticks"]] == ["-10", "-5", "0", "+5", "+10"]
    tip = float(jacobian["rows"][0]["d"].split("Q")[1].split()[0])
    assert tip == pytest.approx(zero + 0.9 * half, abs=0.01)
    assert logit["rows"][0]["anchor"] == "end" and logit["rows"][-1]["anchor"] == "start"
    assert "z = -7.00 · #1 suppressed" in jacobian["rows"][-1]["label"]
    assert chart["table"][0]["jacobian_suppressed"] == (" excluding", -7.0)

    mismatched = _lens(**{**Z_SCORES, "logit_promoted_z": [4.0]})
    assert vocab_chart(mismatched) is None


def test_vocabulary_lens_rendered(tmp_path, record):
    from openinterp.builder.wiki import write_experiment_page

    def with_lens(lens):
        sub = record.submission.model_copy(update={"vocabulary_lens": lens})
        return record.model_copy(update={"submission": sub})

    compile_site([with_lens(_lens())], tmp_path / "old")
    page = (tmp_path / "old" / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "Vocabulary projection" in page and "predates vocabulary z-scores" in page
    assert " including" in page and "cos(w, J·w) 0.31" in page

    rec = with_lens(_lens(**Z_SCORES))
    compile_site([rec], tmp_path / "new")
    page = (tmp_path / "new" / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "predates vocabulary" not in page and "␣including · z = +9.00 · #1 promoted" in page
    row = pq.read_table(tmp_path / "new" / "data" / "registry.parquet").to_pylist()[0]
    assert row["jacobian_promoted"] == [" including", "inc"] and row["lens_overlap"] == 0.0
    assert row["jacobian_promoted_z"] == [9.0, 6.0] and row["logit_promoted_z"] == [4.0, 3.0]

    wiki = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "| Promotes | ` including` +9.0 `inc` +6.0 | `oor` +4.0 `x` +3.0 |" in wiki
    assert "| Suppresses | ` excluding` -7.0" in wiki


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


def _self_reviewed(rec):
    model = rec.submission.submitting_agent_model
    return rec.model_copy(
        update={"verdict": rec.verdict.model_copy(update={"reviewer_agent_model": model})}
    )


def test_self_reviews_badged_and_counted(tmp_path, records):
    from openinterp.builder.compiler import registry_stats

    records[1] = _self_reviewed(records[1])  # N117+'s latest, VERIFIED
    report = compile_site(records, tmp_path)
    totals = registry_stats(report.parquet)["totals"]
    assert (totals["self_reviewed"], totals["verified_self_reviewed"]) == (1, 1)
    assert pq.read_table(report.parquet).column("self_reviewed").to_pylist().count(True) == 1

    layer = tmp_path / "models" / MODEL / "L3"
    neuron = (layer / "N117.html").read_text()
    assert neuron.count(">self-reviewed</span>") == 2  # the positive pole's card and experiment
    assert "same model as the proposer" in neuron
    assert ">self-reviewed</span>" not in (layer / "N9.html").read_text()
    home = (tmp_path / "index.html").read_text()
    assert "1 of 4 experiments were" in home and "So were 1 of the 1 current verified" in home


def test_builtin_sample_scans_badged(tmp_path, records):
    from openinterp.core.schemas import BUILTIN_CORPUS, ExperimentProvenance

    rec = records[1]  # N117+'s latest
    provenance = ExperimentProvenance(
        dtype="float32",
        corpus=BUILTIN_CORPUS,
        n_snippets=1,
        tokens_scanned=552,
        selection="random",
        harness_version="t",
    )
    sub = rec.submission.model_copy(update={"provenance": provenance})
    records[1] = rec.model_copy(update={"submission": sub})
    compile_site(records, tmp_path)
    layer = tmp_path / "models" / MODEL / "L3"
    neuron = (layer / "N117.html").read_text()
    assert neuron.count(">built-in sample</span>") == 2  # the positive pole's card and experiment
    assert ">built-in sample</span>" not in (layer / "N9.html").read_text()


def _reviewed_by(rec, reviewer, *, status, hours, counts=(10, 0, 5, 0)):
    """A re-review of ``rec``'s submission: a new probe set from ``reviewer``."""
    verdict = rec.verdict.model_copy(
        update={
            "status": status,
            "reviewer_agent_model": reviewer,
            # Prompts of its own, so it isn't a re-measurement of an earlier probe set.
            "adversarial_test_results": [
                t.model_copy(update={"prompt": f"{t.prompt} {reviewer}{hours}"})
                for t in probes(*counts)
            ],
            "timestamp": rec.verdict.timestamp + timedelta(hours=hours),
        }
    )
    return rec.model_copy(update={"verdict": verdict})


def _standing(records, idx=117, pole="positive"):
    from openinterp.builder.tree import build_tree

    (model,) = build_tree(records)
    (unit,) = [n for layer in model.layers for n in layer.units if n.idx == idx]
    return unit.standings[pole], unit


def test_weaker_rerun_contests_instead_of_overturning(tmp_path, records):
    """An independent VERIFIED and a later independent NEEDS_REVISION of the same hypothesis:
    the pole is CONTESTED, not NEEDS_REVISION, and not counted as verified."""
    from openinterp.builder.compiler import registry_stats

    verified = records[1]
    records.append(
        _reviewed_by(
            verified, "deepseek/flash", status="NEEDS_REVISION", hours=3, counts=(5, 5, 1, 4)
        )
    )
    standing, unit = _standing(records)
    assert standing.status == "CONTESTED" and standing.contested
    assert [e.record.verdict.status for e in standing.readings] == ["NEEDS_REVISION", "VERIFIED"]
    assert not unit.verified

    report = compile_site(records, tmp_path)
    stats = registry_stats(report.parquet)
    assert (stats["totals"]["verified"], stats["totals"]["contested"]) == (0, 1)
    assert stats["verified"] == []
    neuron = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "CONTESTED" in neuron and "Reviews of this hypothesis disagree" in neuron
    home = (tmp_path / "index.html").read_text()
    assert 'id="contested"' in home and "1 unit is contested" in home
    assert "0 of the 1 hypotheses reviewed with more than one probe set" in home


def test_independent_review_outweighs_self_review(records):
    verified = records[1]  # reviewed by anthropic/claude-test, proposed by openai/gpt-test
    records.append(
        _self_reviewed(_reviewed_by(verified, "x", status="REJECTED", hours=3, counts=(2, 8, 0, 5)))
    )
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and not standing.self_reviewed
    roles = {e.record.verdict.status: e.role for e in unit.experiments if e.side == "positive"}
    assert roles == {
        "NEEDS_REVISION": "earlier_hypothesis",
        "VERIFIED": "current",
        "REJECTED": "outweighed",
    }


def test_self_reviews_alone_can_contest(records):
    verified = _self_reviewed(records[1])
    records[1] = verified
    records.append(
        _reviewed_by(
            verified,
            verified.verdict.reviewer_agent_model,
            status="REJECTED",
            hours=3,
            counts=(2, 8, 0, 5),
        )
    )
    standing, _ = _standing(records)
    assert standing.status == "CONTESTED" and standing.self_reviewed


def test_remeasurement_replaces_its_reading(records):
    """Re-evaluating a probe set under a newer protocol replaces that reading: the old
    reading can't contest the new one."""
    verified = records[1]
    remeasured = verified.model_copy(
        update={
            "verdict": verified.verdict.model_copy(
                update={
                    "status": "NEEDS_REVISION",
                    "timestamp": verified.verdict.timestamp + timedelta(hours=3),
                }
            )
        }
    )
    records.append(remeasured)
    standing, unit = _standing(records)
    assert standing.status == "NEEDS_REVISION" and len(standing.readings) == 1
    assert [e.role for e in unit.experiments if e.record is verified] == ["remeasured"]


def test_revision_replaces_contested_hypothesis(records):
    verified = records[1]
    records.append(
        _reviewed_by(
            verified, "deepseek/flash", status="NEEDS_REVISION", hours=3, counts=(5, 5, 1, 4)
        )
    )
    revised = experiment(
        records[0], idx=117, status="VERIFIED", counts=(10, 0, 5, 0), hours=6, text="Narrower."
    )
    records.append(revised)
    standing, _ = _standing(records)
    assert standing.status == "VERIFIED"
    assert [e.record.submission.natural_language_explanation for e in standing.readings] == [
        "Narrower."
    ]


def test_reviewer_agreement(records):
    from openinterp.builder.standing import reviewer_agreement
    from openinterp.builder.tree import build_tree

    verified = records[1]
    records.append(_reviewed_by(verified, "deepseek/flash", status="VERIFIED", hours=3))
    records.append(
        _reviewed_by(verified, "other/model", status="NEEDS_REVISION", hours=4, counts=(5, 5, 1, 4))
    )
    agreement = reviewer_agreement(build_tree(records))
    assert agreement["claims"] == 1 and agreement["unanimous"] == 0
    assert agreement["pairs"] == 3 and agreement["pair_agreement"] == round(1 / 3, 4)
    assert agreement["cross_model_pairs"] == 3


def _with_opposite(rec):
    example = ActivatingExample(
        text="x, y", tokens=["x", ",", " y"], activations=[0.0, 0.9, 0.1],
        max_activation=0.9, max_token_index=1,
    )  # fmt: skip
    opposite = OppositePole(pole="negative", top_examples=[example])
    return rec.model_copy(
        update={"submission": rec.submission.model_copy(update={"opposite_pole": opposite})}
    )


def test_neuron_page_shows_both_poles(tmp_path, records):
    # Without an opposite-pole scan, each pole's newest experiment supplies its examples.
    compile_site(records, tmp_path)
    page = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    section = page.split('id="both-poles"')[1].split("</section>")[0]
    assert "newest experiment" in section and section.count("From <a") == 2
    assert "--heat-rgb: 234 88 12" in section and "--heat-rgb: 2 132 199" in section

    # N9 has only a positive experiment; its negative column waits for a scan.
    page = (tmp_path / "models" / MODEL / "L3" / "N9.html").read_text()
    assert "Not scanned yet." in page.split('id="both-poles"')[1]

    # One scan that read both poles supplies both columns.
    compile_site([*records[:3], _with_opposite(records[3])], tmp_path)
    page = (tmp_path / "models" / MODEL / "L3" / "N9.html").read_text()
    section = page.split('id="both-poles"')[1].split("</section>")[0]
    assert "from one scan" in section and "Not scanned yet." not in section
    assert 'data-v="0.900 · peak"' in section and "max 0.90" in section


def test_wiring_rendered_with_links_to_studied_units(tmp_path, record):
    from openinterp.builder.wiki import write_experiment_page
    from openinterp.core.wiring import Wiring, WiringEdge

    def edge(layer, index, weight, kind="mlp_neuron", **channels):
        return WiringEdge(kind=kind, layer=layer, index=index, weight=weight, channels=channels)

    weights = Wiring(
        method="weights",
        upstream_neurons=[edge(1, 5, -0.42, gate=-0.42, up=0.1), edge(2, 9, 0.3, gate=0.1, up=0.3)],
        upstream_heads=[edge(3, 2, 0.8, kind="attn_head", gate=0.8, up=0.2)],
        downstream_neurons=[edge(4, 1, 0.25, gate=0.25, up=0.0)],
        downstream_heads=[edge(5, 0, 1.5, kind="attn_head", q=0.1, k=1.5, v=0.2)],
    )
    activations = Wiring(
        method="activations",
        upstream_neurons=[edge(1, 5, 0.07)],
        upstream_heads=None,
        downstream_neurons=[],
        prompts=4,
    )
    sub = record.submission.model_copy(update={"wiring": weights})
    verdict = record.verdict.model_copy(update={"wiring": activations})
    rec = record.model_copy(update={"submission": sub, "verdict": verdict})
    other_ident = sub.identifier.model_copy(update={"layer": 1, "neuron_idx": 5})
    other = record.model_copy(
        update={"submission": record.submission.model_copy(update={"identifier": other_ident})}
    )
    compile_site([rec, other], tmp_path / "site")
    page = (tmp_path / "site" / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "From the weights" in page and "on 4 of the review" in page
    assert 'href="../../../models/' + MODEL + '/L1/N5.html">L1/N5</a>' in page
    assert ">L2/N9<" in page and "L2/N9</a>" not in page  # no page for it: not linked
    assert "gate -0.42 · up +0.1" in page and ">k</td>" in page
    assert "Not measured." in page and "None: no later layer." in page

    wiki = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "### Wiring" in wiki and wiki.count("### Wiring") == 1
    assert "- **Upstream neurons:** L1/N5 -0.42, L2/N9 +0.3" in wiki
    assert "- **Downstream heads:** L5/H0 +1.5" in wiki
    assert "- **Upstream heads:** not measured." in wiki
    assert "on 4 of the review's prompts" in wiki


def scan_provenance(sample=None):
    from openinterp.core.schemas import ExperimentProvenance

    return ExperimentProvenance(
        dtype="float32", corpus="wiki.jsonl", corpus_sha256="ab" * 32, n_snippets=4,
        tokens_scanned=327, selection="random", harness_version="t", corpus_sample=sample,
    )  # fmt: skip


def test_label_confounds_rendered(tmp_path, record):
    from openinterp.builder.wiki import write_experiment_page
    from openinterp.core.schemas import ConfoundProfile, LabelConfound, LabelStat

    language = LabelConfound(
        r2=0.07, mutual_information_bits=0.313, labelled_tokens=327, unlabelled_tokens=0,
        levels=[LabelStat(label="fr", count=63, mean_strength=0.05, active=21)],
    )  # fmt: skip
    confounds = ConfoundProfile(
        token_r2=0.9, position_r2=0.2, frequency_correlation=0.1, min_count=3,
        trigger_tokens=[], position_means=[0.0], labels={"language": language},
    )  # fmt: skip
    update = {"confounds": confounds, "provenance": scan_provenance()}
    rec = record.model_copy(update={"submission": record.submission.model_copy(update=update)})
    compile_site([rec], tmp_path / "site")
    page = (tmp_path / "site" / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "Language R² · MI" in page and "0.07 · 0.313 bits" in page
    assert "fr 19% · other 81%" in page  # 63 of 327 labelled tokens
    index = (tmp_path / "site" / "index.html").read_text()
    assert 'id="corpora"' in index and "fr 19% · other 81%" in index
    wiki = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "frequency r 0.10, language R² 0.07 (0.313 bits)." in wiki


def test_agents_page_rates_exclude_self_reviews(tmp_path, records):
    from openinterp.builder.agents import agent_slug
    from openinterp.core.costs import ExperimentCosts, LLMCall
    from openinterp.core.schemas import ExperimentProvenance

    records[1] = _self_reviewed(records[1])  # N117+'s VERIFIED review, by the proposer's model
    costs = ExperimentCosts(
        llm_calls=[
            LLMCall(agent="proposer", model="openai/gpt-test", cost_usd=0.5),
            LLMCall(agent="reviewer", model="anthropic/claude-test", cost_usd=0.25),
        ]
    )
    provenance = ExperimentProvenance(
        dtype="float32",
        corpus="c",
        n_snippets=1,
        tokens_scanned=1,
        selection="random",
        harness_version="t",
        costs=costs,
    )
    sub = records[3].submission.model_copy(update={"provenance": provenance})
    records[3] = records[3].model_copy(update={"submission": sub})  # N9+, REJECTED
    compile_site(records, tmp_path)

    assert agent_slug("external/deepseek-flash[1m]") == "deepseek-flash"
    page = (tmp_path / "agents" / "index.html").read_text()
    proposer = (tmp_path / "agents" / "gpt-test.html").read_text()
    reviewer = (tmp_path / "agents" / "claude-test.html").read_text()
    # The proposer's model proposed 4 and reviewed 1 of them itself; that VERIFIED one
    # is left out of both its rates, so 0 of 3 independent proposals were verified.
    assert '0% <span class="text-slate-500">(0/3)</span>' in proposer
    assert '<td class="px-3 py-2">reviewer</td>' in proposer
    assert "$0.500" in page and "$0.250" in reviewer
    assert "Took part in 3 experiments" in reviewer
    # Agent names link to their pages from the home page and experiment pages.
    home = (tmp_path / "index.html").read_text()
    assert 'href="agents/gpt-test.html"' in home and "Agent costs" not in home
    neuron = (tmp_path / "models" / MODEL / "L3" / "N9.html").read_text()
    assert 'href="../../../agents/claude-test.html"' in neuron
    # The old per-spelling pages redirect to the merged ones.
    moved = (tmp_path / "agents" / "openai-gpt-test.html").read_text()
    assert 'url=gpt-test.html"' in moved


def test_agents_page_merges_spellings_of_one_model(tmp_path, records):
    from openinterp.core.costs import ExperimentCosts, LLMCall
    from openinterp.core.schemas import ExperimentProvenance

    def spelled(rec, proposer, reviewer):
        sub = rec.submission.model_copy(update={"submitting_agent_model": proposer})
        verdict = rec.verdict.model_copy(update={"reviewer_agent_model": reviewer})
        return rec.model_copy(update={"submission": sub, "verdict": verdict})

    records[0] = spelled(records[0], "external/deepseek-flash[1m]", "anthropic/claude-test")
    records[1] = spelled(records[1], "external/deepseek-flash", "external/deepseek-flash[1m]")
    provenance = ExperimentProvenance(
        dtype="float32",
        corpus="c",
        n_snippets=1,
        tokens_scanned=1,
        selection="random",
        harness_version="t",
        costs=ExperimentCosts(
            llm_calls=[LLMCall(agent="proposer", model="external/deepseek-flash", cost_usd=1.0)]
        ),
    )
    for i in (0, 1):
        sub = records[i].submission.model_copy(update={"provenance": provenance})
        records[i] = records[i].model_copy(update={"submission": sub})
    compile_site(records, tmp_path)

    page = (tmp_path / "agents" / "deepseek-flash.html").read_text()
    assert "Took part in 2 experiments" in page
    assert "external/deepseek-flash[1m]" in page  # listed under "Recorded as"
    assert "$2.00" in page  # both experiments' costs, under one name
    # Spelled two ways, but one model proposed and reviewed it: a self-review.
    assert records[1].self_reviewed
    for old in ("external-deepseek-flash-1m", "external-deepseek-flash"):
        assert 'url=deepseek-flash.html"' in (tmp_path / "agents" / f"{old}.html").read_text()
    home = (tmp_path / "index.html").read_text()
    assert ">deepseek-flash</a>" in home and ">external/deepseek-flash[1m]</a>" not in home


def test_corpora_view_takes_each_models_largest_whole_corpus_scan(record):
    from types import SimpleNamespace

    from openinterp.builder.corpora import corpora_view, language_mix
    from openinterp.core.schemas import ConfoundProfile, CorpusSample, LabelConfound, LabelStat

    def experiment(tokens, sample=None, labelled=True):
        levels = [LabelStat(label="en", count=tokens, mean_strength=0.0, active=0)]
        language = LabelConfound(
            r2=0.0, mutual_information_bits=0.0, labelled_tokens=tokens, unlabelled_tokens=0,
            levels=levels,
        )  # fmt: skip
        confounds = ConfoundProfile(
            token_r2=0.1, position_r2=0.1, frequency_correlation=0.0, min_count=3,
            trigger_tokens=[], position_means=[0.0],
            labels={"language": language} if labelled else None,
        )  # fmt: skip
        update = {"confounds": confounds, "provenance": scan_provenance(sample)}
        sub = record.submission.model_copy(update=update)
        return SimpleNamespace(record=record.model_copy(update={"submission": sub}))

    sample = CorpusSample(
        seed=0, max_tokens=10, documents=1, corpus_documents=2, tokens=9, sha256="x"
    )
    (corpus,) = corpora_view(
        [experiment(50), experiment(90, sample), experiment(70), experiment(5, labelled=False)]
    )
    assert corpus.experiments == 4 and len(corpus.mixes) == 1
    assert (corpus.mixes[0].tokens, corpus.mixes[0].sampled) == (70, False)
    assert language_mix(experiment(5, labelled=False).record.submission) is None


def test_agents_page_credits_contributors(tmp_path, records):
    from openinterp.builder.agents import contributors_view
    from openinterp.builder.tree import build_tree

    def credited(rec, proposer=None, reviewer=None):
        sub = rec.submission.model_copy(update={"contributor": proposer})
        verdict = rec.verdict.model_copy(update={"contributor": reviewer})
        return rec.model_copy(update={"submission": sub, "verdict": verdict})

    records[1] = credited(records[1], "alice", "bob")  # N117+ VERIFIED
    records[2] = credited(records[2], "alice", "bob")  # N117− well-probed null
    records[3] = credited(records[3], "bob")  # N9+ REJECTED, thinly probed
    compile_site(records, tmp_path)

    (model,) = build_tree(records)
    experiments = [e for layer in model.layers for n in layer.units for e in n.experiments]
    rows = {c.handle: c for c in contributors_view(experiments)}
    assert (rows["alice"].experiments, rows["alice"].verified) == (2, 1)
    assert (rows["alice"].conclusive_nulls, rows["alice"].reviews) == (1, 0)
    assert (rows["bob"].experiments, rows["bob"].conclusive_nulls, rows["bob"].reviews) == (1, 0, 2)
    assert [c.handle for c in contributors_view(experiments)] == ["bob", "alice"]
    page = (tmp_path / "agents" / "index.html").read_text()
    assert 'id="contributors"' in page and ">alice</td>" in page
    neuron = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "run by</span> alice" in neuron and "run by</span> bob" in neuron
    row = {
        r["neuron_idx"]: r
        for r in pq.read_table(tmp_path / "data" / "registry.parquet").to_pylist()
        if r["pole"] == "positive" and r["is_latest"]
    }
    assert (row[117]["submitting_contributor"], row[117]["reviewer_contributor"]) == (
        "alice",
        "bob",
    )
    assert (row[9]["submitting_contributor"], row[9]["reviewer_contributor"]) == ("bob", None)


def test_agents_page_without_contributors(tmp_path, records):
    compile_site(records, tmp_path)
    assert "No contributors recorded yet." in (tmp_path / "agents" / "index.html").read_text()
