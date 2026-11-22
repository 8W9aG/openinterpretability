import json
import re
from datetime import timedelta
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from openinterpretability.builder.charts import (
    causal_chart,
    histogram_bars,
    probe_chart,
    vocab_chart,
)
from openinterpretability.builder.compiler import compile_site, load_records
from openinterpretability.builder.model_facts import (
    ModelFacts,
    facts_from_config,
    format_parameters,
)
from openinterpretability.builder.parquet_export import REGISTRY_SCHEMA
from openinterpretability.core.exceptions import RegistryError
from openinterpretability.core.licenses import DATA_LICENSE, DATA_LICENSE_URL
from openinterpretability.core.project import site_url
from openinterpretability.core.schemas import (
    PROTOCOL_VERSION,
    ActivatingExample,
    AdversarialTestResult,
    AuditRecord,
    CausalPairEffect,
    OppositePole,
    PatchEffect,
    ReviewPolicy,
)
from tests.site_pages import experiment_pages, unit_pages

MODEL = "Qwen--Qwen2.5-0.5B"
PAGE = f"models/{MODEL}/L3/N117-positive/20261001T"  # an experiment page, but its time


def card_stat(name: str, value: str) -> str:
    """One statistic of a model card, as the page writes it (#5724)."""
    return f'{name}</dt><dd class="tabular-nums">{value}'


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

    neuron = unit_pages(model / "L3" / "N117.html")
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


def test_model_page_shows_the_model_license(tmp_path, records, monkeypatch):
    terms = 'license = "x-1"\nurl = "https://u"\ngated = true\npublishing_allowed = false\n'
    pyproject = (
        f'[tool.openinterpretability.licenses."Qwen/Qwen2.5-0.5B"]\n{terms}conditions = "Say so."\n'
    )
    (tmp_path / "pyproject.toml").write_text(pyproject)
    monkeypatch.chdir(tmp_path)
    compile_site(records, tmp_path / "site")
    page = (tmp_path / "site" / "models" / MODEL / "index.html").read_text()
    assert (
        '<a class="text-brand-700 underline dark:text-brand-400" href="https://u">x-1</a>' in page
    )
    assert "gated" in page and "not gated" not in page
    assert "generated text may not be published" in page
    assert "Say so." in page

    # A table with no licences. An empty file would mean the project's own settings (#3693).
    (tmp_path / "pyproject.toml").write_text("[tool.openinterpretability]\n")
    compile_site(records, tmp_path / "site")
    page = (tmp_path / "site" / "models" / MODEL / "index.html").read_text()
    assert "Licence not recorded" in page


def test_coverage_uses_recorded_mlp_width(tmp_path, records):
    narrow = [
        r.model_copy(update={"submission": r.submission.model_copy(update={"model_mlp_width": 4})})
        for r in records
    ]
    compile_site(narrow, tmp_path)
    index = (tmp_path / "index.html").read_text()
    assert card_stat("Neurons studied", "2 of 96 (2.1%)") in index
    assert card_stat("Verified", "1 of 96 (1.0%)") in index
    assert card_stat("Target", "80% verified") in index


def test_old_records_get_a_width_in_the_build_only(tmp_path, records):
    import pyarrow.parquet as pq

    with_width = records[0].model_copy(
        update={"submission": records[0].submission.model_copy(update={"model_mlp_width": 4})}
    )
    without = records[1]
    assert without.submission.model_mlp_width is None
    compile_site([with_width, without], tmp_path)
    rows = pq.read_table(tmp_path / "data" / "registry.parquet").to_pylist()
    assert {row["model_mlp_width"] for row in rows} == {4}
    signed = [row["record_json"] for row in rows]
    assert without.model_dump_json() in signed  # the record itself stays as it was signed


def test_a_model_status_grid(tmp_path, record, records):
    from openinterpretability.builder.pieces import GRID_PITCH, status_grid, studied_poles
    from openinterpretability.builder.statuses import STATUS_HEIGHT
    from openinterpretability.builder.stopped import StoppedNeuron
    from openinterpretability.builder.tree import build_tree

    # N120 shares N117's 38-neuron block (114–151); the block shows the better status.
    records = [*records, experiment(record, idx=120, status="REJECTED", counts=(3, 3, 1, 3))]
    (model,) = build_tree(records, [])
    grid = status_grid(model)
    assert grid["block"] == 38 and len(grid["rows"]) == 24  # 4,864 neurons in 128 cells
    assert dict((s, n) for s, _, n in grid["legend"]) == {
        "VERIFIED": 1, "CONTESTED": 0, "NEEDS_REVISION": 0, "REJECTED": 2, "STOPPED": 0,
    }  # fmt: skip
    assert grid["unexplored"] == 116_736 - 3
    n9, n117 = grid["cells"]
    assert (n9["status"], n117["status"]) == ("REJECTED", "VERIFIED")
    assert n117["x"] - n9["x"] == 3 * GRID_PITCH and n117["href"] == n9["href"]  # the layer
    assert n117["label"].startswith("L3 N114–151, 3 of 76 poles studied: N117 positive VERIFIED")
    # Height gives the status without colour, from the bottom of the row.
    assert (n117["height"], n9["height"]) == (STATUS_HEIGHT["VERIFIED"], STATUS_HEIGHT["REJECTED"])
    assert n9["y"] + n9["height"] == n117["y"] + n117["height"] == grid["rows"][3]["y"] + 10
    both = StoppedNeuron(1, 2, ["negative default-on", "positive dead"])
    assert studied_poles(both) == 2 and studied_poles(StoppedNeuron(1, 2, ["negative dead"])) == 1

    compile_site(records, tmp_path)
    page = (tmp_path / "models" / MODEL / "index.html").read_text()
    assert "Status grid of Qwen/Qwen2.5-0.5B: one row per layer" in page
    assert '<a href="../../models/Qwen--Qwen2.5-0.5B/L3/index.html">' in page
    assert "Studied neurons per layer" not in page  # the grid replaces the histogram


def test_a_layer_status_grid(record, records):
    from openinterpretability.builder.pieces import status_grid
    from openinterpretability.builder.tree import ModelNode, build_tree

    records = [*records, experiment(record, idx=120, status="REJECTED", counts=(3, 3, 1, 3))]
    (model,) = build_tree(records, [])
    grid = status_grid(model, model.layers[0])
    assert grid["block"] == 1 and len(grid["rows"]) == 38 and grid["unexplored"] == 4864 - 3
    assert [r["label"] for r in grid["rows"] if r["shown"]][:2] == ["N0", "N512"]
    cell = next(c for c in grid["cells"] if c["href"].endswith("/N120.html"))
    assert cell["label"] == "N120 positive REJECTED, 1 of 2 poles studied"
    assert status_grid(ModelNode("m", "m", 2, "m/index.html")) is None  # no width, no grid


def test_no_two_cells_of_a_status_grid_overlap(record, records):
    from openinterpretability.builder.pieces import GRID_PITCH, status_grid
    from openinterpretability.builder.tree import build_tree

    more = [experiment(record, idx=i, status="REJECTED", counts=(3, 3, 1, 3)) for i in range(40)]
    (model,) = build_tree([*records, *more], [])
    for grid in (status_grid(model), status_grid(model, model.layers[0])):
        boxes = {(c["x"], c["y"] + c["height"]) for c in grid["cells"]}
        assert len(boxes) == len(grid["cells"])  # one cell per place
        assert all(c["x"] + GRID_PITCH - 1 <= grid["width"] for c in grid["cells"])


def test_each_status_grid_legend_names_every_status_its_cells_use(tmp_path, records):
    import re

    compile_site(records, tmp_path)
    pages = [tmp_path / "index.html", tmp_path / "models" / MODEL / "index.html",
             *(tmp_path / "models" / MODEL).glob("L*/index.html")]  # fmt: skip
    seen = 0
    for page in pages:
        for figure in re.findall(
            r"<figure[^>]*data-status-grid>.*?</figure>", page.read_text(), re.S
        ):
            used = set(re.findall(r'data-status="([A-Z_]+)"', figure))
            assert used <= set(re.findall(r'data-swatch="([A-Z_]+)"', figure)), page
            seen += bool(used)
    assert seen >= 3  # the card, the model page and a layer page
    index = pages[0].read_text()
    assert "Studied neurons per layer" not in index  # the grid replaces the histogram
    assert "Neurons studied</dt>" in index and "Target</dt>" in index  # one layout per card
    layer = (tmp_path / "models" / MODEL / "L3" / "index.html").read_text()
    assert "layer 3: one row of 128 neurons, each cell one neuron. 2 studied neurons" in layer
    assert '<a href="../../../models/Qwen--Qwen2.5-0.5B/L3/N117.html"><rect' in layer


def test_pct_fine():
    from openinterpretability.builder.compiler import _pct_fine

    assert _pct_fine(None) == "—"
    assert _pct_fine(0) == "0%"
    assert _pct_fine(0.5) == "50%"
    assert _pct_fine(0.021) == "2.1%"
    assert _pct_fine(1 / 116_736) == "0.00086%"
    assert _pct_fine(-0.0009418028191878567) == "-0.094%"  # a random baseline below zero


def test_experiments_ordered_by_conclusiveness(tmp_path, records):
    compile_site(records, tmp_path)
    neuron = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    order = re.findall(r'<tr id="(\w+)-(\d+T\d+Z)"', neuron)
    # Strong verification, then the well-probed null, then the thin early attempt.
    assert [pole for pole, _ in order] == ["positive", "negative", "positive"]
    assert "strong evidence for" in neuron and "evidence against" in neuron


def test_each_experiment_has_its_own_page_and_the_unit_page_lists_them(tmp_path, records):
    report = compile_site(records, tmp_path)
    layer = tmp_path / "models" / MODEL / "L3"
    unit = layer / "N117.html"
    pages = [p.relative_to(layer).as_posix() for p in experiment_pages(unit)]
    # The record paths under wiki/experiments/, one directory per pole.
    assert [p.split("/")[0] for p in pages] == ["N117-negative", "N117-positive", "N117-positive"]
    assert all(layer / p in report.pages for p in pages)

    neuron = unit.read_text()
    assert "<section id=" not in neuron  # no experiment is rendered inline any more
    assert "Evidence and review" not in neuron
    links = re.findall(r'<tr id="(\w+-\d+T\d+Z)"', neuron)
    assert len(links) == 3
    # The old anchors move on to the experiments' pages.
    moved = json.loads(re.search(r"var moved = (\{.*?\});", neuron).group(1))
    assert sorted(moved) == sorted(links)
    for anchor, href in moved.items():
        side, id_ = anchor.split("-", 1)
        assert href == f"../../../models/{MODEL}/L3/N117-{side}/{id_}.html"
        assert f'href="{href}"' in neuron

    page = (layer / pages[0]).read_text()
    assert "Adversarial review" in page and "data-pagefind-body" in page
    assert 'href="../../../../index.html"' in page  # relative root, one level deeper
    assert f'<a class="underline" href="../../../../models/{MODEL}/L3/N117.html">' in page
    assert f"wiki/experiments/{MODEL}/L3/{pages[0].removesuffix('.html')}.json" in page


def test_index_has_no_most_conclusive_list(tmp_path, records):
    # #4616: the home page dropped its Most conclusive results list. The recent list
    # still features a null result, with a link to its experiment.
    compile_site(records, tmp_path)
    index = (tmp_path / "index.html").read_text()
    assert "Most conclusive results" not in index
    assert "Fires on commas." in index  # a null result is featured
    assert "models/Qwen--Qwen2.5-0.5B/L3/N117-positive/2026" in index
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
    nulls = duckdb.sql(f"SELECT count(*) FROM '{snapshot}' WHERE status = 'REJECTED'").fetchall()[0]
    assert nulls == (2,)


def test_revisions_linked_both_ways(tmp_path, records):
    verified = records[1]  # revises records[0], the early attempt on N117+
    sub = verified.submission.model_copy(update={"supersedes": "20261001T120000Z"})
    records[1] = verified.model_copy(update={"submission": sub})
    compile_site(records, tmp_path)
    neuron = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
    assert (
        'Revises <a class="font-mono underline" href="../../../../' + PAGE + '120000Z.html">'
        in neuron
    )
    assert (
        'Revised by <a class="font-mono underline" href="../../../../' + PAGE + '140000Z.html">'
        in neuron
    )
    rows = pq.read_table(tmp_path / "data" / "registry.parquet").to_pylist()
    assert sorted(r["supersedes"] or "" for r in rows) == ["", "", "", "20261001T120000Z"]


def test_empty_registry_builds(tmp_path):
    report = compile_site([], tmp_path)
    assert report.experiments_published == 0
    assert "No experiments yet" in (tmp_path / "index.html").read_text()


def test_footer_links_repo_and_contact(tmp_path):
    from openinterpretability.core.project import project_links

    compile_site([], tmp_path)
    index = (tmp_path / "index.html").read_text()
    assert 'href="https://github.com/8W9aG/openinterpretability"' in index
    email = project_links()["contact_email"]  # pyproject.toml's author email, the one source
    assert "@" in email and f'href="mailto:{email}"' in index


def test_every_page_links_the_x_account(tmp_path):
    from openinterpretability.core.project import project_links

    compile_site([], tmp_path)
    links = project_links()  # pyproject.toml's [project.urls], the one source
    handle = "@" + links["x_url"].rstrip("/").rsplit("/", 1)[-1]
    for page in ("index.html", "about.html"):
        html = (tmp_path / page).read_text()
        assert f'href="{links["x_url"]}">X</a>' in html
        assert f'<meta name="twitter:site" content="{handle}">' in html


def test_every_page_names_its_canonical_url(tmp_path, records):
    report = compile_site(records, tmp_path)
    site = site_url()
    assert site == "https://open-interpretability.com/"
    for page in report.pages:
        rel = page.relative_to(tmp_path).as_posix()
        if page.suffix == ".html" and "Moved" not in (text := page.read_text()):
            assert f'<link rel="canonical" href="{site}{rel}">' in text, rel


def test_about_page_quotes_protocol_and_licence(tmp_path):
    compile_site([], tmp_path)
    assert 'href="about.html"' in (tmp_path / "index.html").read_text()
    about = (tmp_path / "about.html").read_text()
    assert 'href="index.html"' in about  # root-relative links resolve from the site root
    assert f'<span class="font-mono">{PROTOCOL_VERSION}</span>' in about
    assert f"at least {ReviewPolicy().min_hard_negatives} hard negatives" in about
    assert "/blob/main/docs/PROTOCOL.md" in about
    assert "MIT licence" in about
    assert DATA_LICENSE in about and DATA_LICENSE_URL in about


def test_the_data_licence_is_stated_where_people_read_it():
    # One licence, from DATA_LICENSE: the README and wiki/LICENSE.md must name it.
    root = Path(__file__).resolve().parents[1]
    for path in ("README.md", "wiki/LICENSE.md"):
        assert DATA_LICENSE in (root / path).read_text(), path
    assert DATA_LICENSE_URL in (root / "wiki/LICENSE.md").read_text()


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
    page = unit_pages(next((tmp_path / "models").rglob("N*.html")))
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


def test_charts_draw_at_their_natural_size_with_one_text_size(tmp_path, record):
    # #5660: the class that sets every chart text's size reaches the page unescaped.
    # Otherwise Tailwind never generates it, and the text keeps the size it was drawn with.
    from openinterpretability.builder.charts import CHART_TEXT_PX

    compile_site([record], tmp_path)
    page = unit_pages(next((tmp_path / "models").rglob("N*.html")))
    assert f"[&_text]:text-[{CHART_TEXT_PX}px]" in page
    assert "&amp;_text" not in page
    assert 'tabindex="0" role="region" aria-label="' in page  # the scroll box


def test_causal_chart_rendered(tmp_path, record):
    causal = _causal_pairs(record.verdict.causal_metrics)
    verdict = record.verdict.model_copy(update={"causal_metrics": causal})
    compile_site([record.model_copy(update={"verdict": verdict})], tmp_path)
    page = unit_pages(next((tmp_path / "models").rglob("N*.html")))
    assert "Minimal pairs" in page
    assert "Sufficiency and necessity percentiles for 2 minimal pairs" in page
    # The per-pair numbers are merged into the pair table (#5566).
    assert '<th class="px-3 py-2 text-right">Sufficiency</th>' in page


def test_cli_build_from_wiki_archive(tmp_path, records, monkeypatch):
    from click.testing import CliRunner

    from openinterpretability.builder import model_facts
    from openinterpretability.builder.wiki import write_experiment_page
    from openinterpretability.cli import cli

    monkeypatch.setattr(model_facts, "model_config", lambda model, download=True: None)
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
        # Pages rendered from the repo's own documents show the dates written in them, not
        # the clock. On the day a document names (a release in CHANGELOG.md, a literature
        # page read today), that date is today (#4747).
        documents = ("changelog.html", "about.html")
        clocked = {path: content for path, content in a.items()
                   if path not in documents and not path.startswith("research/")}  # fmt: skip
        assert not any(today.encode() in content for content in clocked.values())


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
    from openinterpretability.core.schemas import VocabularyLens

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
    from openinterpretability.builder.wiki import write_experiment_page

    def with_lens(lens):
        sub = record.submission.model_copy(update={"vocabulary_lens": lens})
        return record.model_copy(update={"submission": sub})

    compile_site([with_lens(_lens())], tmp_path / "old")
    page = unit_pages(tmp_path / "old" / "models" / MODEL / "L3" / "N117.html")
    assert "Vocabulary projection" in page and "predates vocabulary z-scores" in page
    assert " including" in page and "cos(w, J·w) 0.31" in page

    rec = with_lens(_lens(**Z_SCORES))
    compile_site([rec], tmp_path / "new")
    page = unit_pages(tmp_path / "new" / "models" / MODEL / "L3" / "N117.html")
    assert "predates vocabulary" not in page and "␣including · z = +9.00 · #1 promoted" in page
    row = pq.read_table(tmp_path / "new" / "data" / "registry.parquet").to_pylist()[0]
    assert row["jacobian_promoted"] == [" including", "inc"] and row["lens_overlap"] == 0.0
    assert row["jacobian_promoted_z"] == [9.0, 6.0] and row["logit_promoted_z"] == [4.0, 3.0]

    wiki = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "| Promotes | ` including` +9.0 `inc` +6.0 | `oor` +4.0 `x` +3.0 |" in wiki
    assert "| Suppresses | ` excluding` -7.0" in wiki


def test_input_attribution_rendered(tmp_path, record):
    from openinterpretability.builder.wiki import write_experiment_page
    from openinterpretability.core.attribution import (
        AttributedToken,
        ContextExample,
        DistanceShare,
        InputAttribution,
    )
    from openinterpretability.harness.input_attribution import DISTANCE_BANDS

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
    page = unit_pages(tmp_path / "site" / "models" / MODEL / "L3" / "N117.html")
    assert "Input attribution" in page and "65+ back" in page and "ρ = 0.80" in page
    row = pq.read_table(tmp_path / "site" / "data" / "registry.parquet").to_pylist()[0]
    assert row["current_token_share"] == 0.5 and row["context_mean_distance"] == 7.25

    md = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "### Input attribution" in md and "| Share | 50% | 10% |" in md
    assert '| 1 | `"` | `"` 0, 0.50 / 0.60; `Hi` -3, 0.30 / 0.20 |' in md


def test_token_heat_windows_long_snippets_around_the_peak():
    from openinterpretability.builder.compiler import token_heat
    from openinterpretability.core.schemas import ActivatingExample

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
    page = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
    assert '<i class="p" style=--a:1 data-v="2.500 · peak">' in page
    assert 'data-v="0.300"' in page  # exact stored value, not a rounded shade
    assert "--heat-rgb: 234 88 12" in page  # positive pole hue
    assert '<summary class="cursor-pointer">Values</summary>' in page


def test_home_lists_most_recent_experiments_newest_first(tmp_path, records):
    from openinterpretability.builder.compiler import registry_stats

    report = compile_site(records, tmp_path)
    recent = registry_stats(report.parquet)["recent"]
    assert [r["reviewed_iso"] for r in recent] == sorted(
        (r["reviewed_iso"] for r in recent), reverse=True
    )
    assert recent[0]["reviewed_iso"] == "2026-10-01T14:00:00Z"  # the +2 h verification
    assert recent[0]["reviewed_time"] == "14:00 UTC" and recent[0]["href"].endswith(
        f"/N117-positive/{recent[0]['experiment_id']}.html"
    )
    home = (tmp_path / "index.html").read_text()
    section = home.split("Most recent experiments</h2>")[1].split("</section>")[0]
    assert section.index("Fires on commas.") < section.index("Fires on dates.")  # +1 h before +0 h
    assert '<time datetime="2026-10-01T14:00:00Z">2026-10-01</time>' in section


def test_explanation_scores_rendered_mistakes_first(tmp_path, record):
    from openinterpretability.core.explanation import ExplanationScores, ScoredWindow
    from openinterpretability.harness.explanation_scores import finalize_scores

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
    page = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
    assert "Explanation scores on held-out text" in page and "40 active of 900" in page
    assert page.index(">missed<") < page.index(">correct rejection<")  # mistakes first
    assert '<i class="p m"' in page and "marks right · judged right" in page
    row = pq.read_table(tmp_path / "data" / "registry.parquet").to_pylist()[0]
    assert (row["detection_score"], row["fuzzing_score"]) == (0.5, 1.0)


def _self_reviewed(rec):
    model = rec.submission.submitting_agent_model
    return rec.model_copy(
        update={"verdict": rec.verdict.model_copy(update={"reviewer_agent_model": model})}
    )


def test_self_reviews_badged_and_counted(tmp_path, records):
    from openinterpretability.builder.compiler import registry_stats

    records[1] = _self_reviewed(records[1])  # N117+'s latest, VERIFIED
    report = compile_site(records, tmp_path)
    totals = registry_stats(report.parquet)["totals"]
    assert (totals["self_reviewed"], totals["verified_self_reviewed"]) == (1, 1)
    assert pq.read_table(report.parquet).column("self_reviewed").to_pylist().count(True) == 1

    layer = tmp_path / "models" / MODEL / "L3"
    neuron = unit_pages(layer / "N117.html")
    # The positive pole's card, its row in the table, and its experiment's page.
    assert neuron.count(">self-reviewed</span>") == 3
    assert "same model as the proposer" in neuron
    assert ">self-reviewed</span>" not in unit_pages(layer / "N9.html")
    home = (tmp_path / "index.html").read_text()
    assert "1 of 4 experiments were" in home and "So were 1 of the 1 current verified" in home


def test_builtin_sample_scans_badged(tmp_path, records):
    from openinterpretability.core.schemas import BUILTIN_CORPUS, ExperimentProvenance

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
    neuron = unit_pages(layer / "N117.html")
    assert neuron.count(">built-in sample</span>") == 2  # the positive pole's card and experiment
    assert ">built-in sample</span>" not in unit_pages(layer / "N9.html")


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
    from openinterpretability.builder.tree import build_tree

    (model,) = build_tree(records, [])
    (unit,) = [n for layer in model.layers for n in layer.units if n.idx == idx]
    return unit.standings[pole], unit


def test_a_needs_revision_reading_does_not_contest_a_verified_one(tmp_path, records):
    """An independent VERIFIED, then an independent NEEDS_REVISION of the same hypothesis.

    NEEDS_REVISION never contests a VERIFIED claim (#4900): the pole stays VERIFIED.
    """
    from openinterpretability.builder.compiler import registry_stats

    verified = records[1]
    records.append(
        _reviewed_by(
            verified, "deepseek/flash", status="NEEDS_REVISION", hours=3, counts=(5, 5, 1, 4)
        )
    )
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and not standing.contested and unit.verified
    assert [e.record.verdict.status for e in standing.readings] == ["VERIFIED", "NEEDS_REVISION"]

    report = compile_site(records, tmp_path)
    stats = registry_stats(report.parquet)
    assert (stats["totals"]["verified"], stats["totals"]["contested"]) == (1, 0)
    neuron = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
    assert "Reviews of this hypothesis split evenly" not in neuron


@pytest.mark.parametrize(
    ("verified", "rejected", "status"),
    [(0.9, 0.7, "VERIFIED"), (0.6, 0.8, "REJECTED"), (0.80, 0.78, "CONTESTED")],
)
def test_scores_settle_verified_against_rejected(records, monkeypatch, verified, rejected,
                                                 status):  # fmt: skip
    """An even VERIFIED–REJECTED split, and a REJECTED rival, go to the score (#4900)."""
    from openinterpretability.builder import standing as st

    table = {"VERIFIED": verified, "REJECTED": rejected}
    monkeypatch.setattr(st, "score", lambda e: table.get(e.record.verdict.status))
    split = [*records, _reviewed_by(records[1], "deepseek/flash", status="REJECTED", hours=3,
                                    counts=(5, 5, 1, 4))]  # fmt: skip
    assert _standing(split)[0].status == status
    rival = [*records, _rival(records, status="REJECTED", hours=4, text="Fires on vowels.")]
    standing, _ = _standing(rival)
    assert standing.status == status
    if status == "VERIFIED":  # the earlier hypothesis keeps the standing, and its readings count
        assert standing.lead.record is records[1] and standing.lead.counts


def test_a_newer_needs_revision_hypothesis_leaves_a_verified_one_standing(records):
    """The L2/H3 case of #4780: a self-reviewed NEEDS_REVISION rival doesn't contest (#4900)."""
    records.append(_rival(records, status="NEEDS_REVISION", hours=4, text="Fires on nouns."))
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and unit.verified and standing.prior.record is records[1]
    roles = {e.record.submission.natural_language_explanation: e.role
             for e in unit.experiments if e.side == "positive"}  # fmt: skip
    assert roles["Fires on nouns."] == "failed_rival"


def test_a_majority_of_verified_readings_settles_a_contest(tmp_path, records):
    """Two independent VERIFIED readings outvote one NEEDS_REVISION (#3212)."""
    verified = records[1]
    records.append(
        _reviewed_by(
            verified, "deepseek/flash", status="NEEDS_REVISION", hours=3, counts=(5, 5, 1, 4)
        )
    )
    records.append(_reviewed_by(verified, "muse/spark", status="VERIFIED", hours=5))
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and not standing.contested and unit.verified
    assert standing.lead.record.verdict.status == "VERIFIED" and standing.dissent == 1
    assert [e.record.verdict.status for e in standing.readings] == [
        "VERIFIED", "VERIFIED", "NEEDS_REVISION",
    ]  # fmt: skip
    compile_site(records, tmp_path)
    neuron = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
    assert "Most reviews of this hypothesis agree (2 of 3)" in neuron


def test_needs_revision_does_not_join_rejected_against_verified(records):
    """VERIFIED, NEEDS_REVISION and REJECTED readings: one against one, so the scores decide.

    With no score on either side the pole is CONTESTED (#4900). Before, NEEDS_REVISION and
    REJECTED outvoted VERIFIED.
    """
    verified = records[1]
    records.append(
        _reviewed_by(
            verified, "deepseek/flash", status="NEEDS_REVISION", hours=3, counts=(5, 5, 1, 4)
        )
    )
    records.append(
        _reviewed_by(verified, "muse/spark", status="REJECTED", hours=5, counts=(2, 8, 0, 5))
    )
    standing, unit = _standing(records)
    assert standing.status == "CONTESTED" and not unit.verified


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
    """Re-evaluating a probe set under a newer protocol replaces that reading.

    The old reading can't contest the new one.
    """
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


def _rival(records, *, status, hours, text, counts=(2, 8, 0, 5)):
    """A new hypothesis on N117+, after its VERIFIED one."""
    return experiment(records[0], idx=117, status=status, counts=counts, hours=hours, text=text)


def test_failed_rival_contests_a_verified_hypothesis(tmp_path, records):
    """VERIFIED, then a different hypothesis that is REJECTED: CONTESTED, not REJECTED (#4654)."""
    from openinterpretability.builder.compiler import registry_stats

    records.append(_rival(records, status="REJECTED", hours=4, text="Fires on vowels."))
    standing, unit = _standing(records)
    assert standing.status == "CONTESTED" and standing.contested_by_hypothesis
    assert not unit.verified
    assert standing.lead.record.submission.natural_language_explanation == "Fires on vowels."
    assert standing.prior.record is records[1]
    roles = {
        e.record.submission.natural_language_explanation: e.role
        for e in unit.experiments
        if e.side == "positive"
    }
    assert roles == {
        "Early idea.": "earlier_hypothesis",
        records[1].submission.natural_language_explanation: "contested_hypothesis",
        "Fires on vowels.": "current",
    }

    report = compile_site(records, tmp_path)
    assert registry_stats(report.parquet)["totals"]["contested"] == 1
    neuron = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
    assert "a newer hypothesis came out" in neuron and ">rejected</a> after an earlier" in neuron
    assert "A new hypothesis that reaches VERIFIED settles it." in neuron
    assert "split evenly" not in neuron


def test_a_second_failed_rival_leaves_it_contested(records):
    records.append(_rival(records, status="REJECTED", hours=4, text="Fires on vowels."))
    records.append(_rival(records, status="NEEDS_REVISION", hours=5, text="Fires on nouns."))
    standing, _ = _standing(records)
    assert standing.status == "CONTESTED" and standing.prior.record is records[1]


def test_a_verified_rival_settles_the_contest(records):
    """VERIFIED, then REJECTED, then a third hypothesis that is VERIFIED: VERIFIED."""
    records.append(_rival(records, status="REJECTED", hours=4, text="Fires on vowels."))
    records.append(
        _rival(records, status="VERIFIED", hours=6, text="Narrower.", counts=(10, 0, 5, 0))
    )
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and standing.prior is None and unit.verified
    assert standing.lead.record.submission.natural_language_explanation == "Narrower."


def test_rejected_then_verified_is_verified(records):
    """The ``records`` fixture: N117+ was NEEDS_REVISION, then a new hypothesis VERIFIED."""
    standing, unit = _standing(records)
    assert standing.status == "VERIFIED" and standing.prior is None and unit.verified


def test_only_a_verified_earlier_hypothesis_contests(records):
    """N9+ was REJECTED; a new NEEDS_REVISION hypothesis is no contest."""
    records.append(
        experiment(
            records[0], idx=9, status="NEEDS_REVISION", counts=(5, 5, 1, 4), hours=3, text="Dates?"
        )
    )
    standing, _ = _standing(records, idx=9)
    assert standing.status == "NEEDS_REVISION" and standing.prior is None


def test_reviewer_agreement(records):
    from openinterpretability.builder.standing import reviewer_agreement
    from openinterpretability.builder.tree import build_tree

    verified = records[1]
    records.append(_reviewed_by(verified, "deepseek/flash", status="VERIFIED", hours=3))
    records.append(
        _reviewed_by(verified, "other/model", status="NEEDS_REVISION", hours=4, counts=(5, 5, 1, 4))
    )
    agreement = reviewer_agreement(build_tree(records, []))
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
    page = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
    section = page.split('id="both-poles"')[1].split("</section>")[0]
    assert "newest experiment" in section and section.count("From <a") == 2
    assert "--heat-rgb: 234 88 12" in section and "--heat-rgb: 2 132 199" in section

    # N9 has only a positive experiment; its negative column waits for a scan.
    page = unit_pages(tmp_path / "models" / MODEL / "L3" / "N9.html")
    assert "Not scanned yet." in page.split('id="both-poles"')[1]

    # One scan that read both poles supplies both columns.
    compile_site([*records[:3], _with_opposite(records[3])], tmp_path)
    page = unit_pages(tmp_path / "models" / MODEL / "L3" / "N9.html")
    section = page.split('id="both-poles"')[1].split("</section>")[0]
    assert "from one scan" in section and "Not scanned yet." not in section
    assert 'data-v="0.900 · peak"' in section and "max 0.90" in section


def test_wiring_rendered_with_links_to_studied_units(tmp_path, record):
    from openinterpretability.builder.wiki import write_experiment_page
    from openinterpretability.core.wiring import Wiring, WiringEdge

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
    page = unit_pages(tmp_path / "site" / "models" / MODEL / "L3" / "N117.html")
    assert "From the weights" in page and "on 4 of the review" in page
    assert 'href="../../../../models/' + MODEL + '/L1/N5.html">L1/N5</a>' in page
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
    from openinterpretability.core.schemas import ExperimentProvenance

    return ExperimentProvenance(
        dtype="float32", corpus="wiki.jsonl", corpus_sha256="ab" * 32, n_snippets=4,
        tokens_scanned=327, selection="random", harness_version="t", corpus_sample=sample,
    )  # fmt: skip


def test_label_confounds_rendered(tmp_path, record):
    from openinterpretability.builder.wiki import write_experiment_page
    from openinterpretability.core.schemas import ConfoundProfile, LabelConfound, LabelStat

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
    page = unit_pages(tmp_path / "site" / "models" / MODEL / "L3" / "N117.html")
    assert "Language R² · MI" in page and "0.07 · 0.313 bits" in page
    assert "fr 19% · other 81%" in page  # 63 of 327 labelled tokens
    index = (tmp_path / "site" / "index.html").read_text()
    assert 'id="corpora"' not in index  # the experiment shows its own corpus (#3298)
    assert "Corpus</dt>" in page and "wiki.jsonl" in page
    wiki = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "frequency r 0.10, language R² 0.07 (0.313 bits)." in wiki


def test_agents_page_rates_exclude_self_reviews(tmp_path, records):
    from openinterpretability.builder.agents import agent_slug
    from openinterpretability.core.costs import ExperimentCosts, LLMCall
    from openinterpretability.core.schemas import ExperimentProvenance

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
    # The proposer's model proposed 4 and reviewed 1 of them itself. That VERIFIED one is
    # left out of both its rates. So 0 of 3 independent proposals were verified.
    assert '0% <span class="text-slate-500">(0/3)</span>' in proposer
    assert '<td class="px-3 py-2">reviewer</td>' in proposer
    assert "$0.500" in page and "$0.250" in reviewer
    assert "Took part in 3 experiments" in reviewer
    # Agent names link to their pages from the home page and experiment pages.
    home = (tmp_path / "index.html").read_text()
    assert 'href="agents/gpt-test.html"' in home and "Agent costs" not in home
    neuron = unit_pages(tmp_path / "models" / MODEL / "L3" / "N9.html")
    assert 'href="../../../agents/claude-test.html"' in neuron
    # The old per-spelling pages redirect to the merged ones.
    moved = (tmp_path / "agents" / "openai-gpt-test.html").read_text()
    assert 'url=gpt-test.html"' in moved


def test_agents_page_merges_spellings_of_one_model(tmp_path, records):
    from openinterpretability.core.costs import ExperimentCosts, LLMCall
    from openinterpretability.core.schemas import ExperimentProvenance

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


def test_language_mix_of_one_scan(record):
    # Each experiment shows its own scan's mix; the home page's corpora table is gone (#3298).
    from types import SimpleNamespace

    from openinterpretability.builder.corpora import language_mix
    from openinterpretability.core.schemas import (
        ConfoundProfile,
        CorpusSample,
        LabelConfound,
        LabelStat,
    )

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
    whole = language_mix(experiment(70).record.submission)
    sampled = language_mix(experiment(90, sample).record.submission)
    assert (whole.tokens, whole.sampled, whole.shown[0].label) == (70, False, "en")
    assert (sampled.tokens, sampled.sampled) == (90, True)
    assert language_mix(experiment(5, labelled=False).record.submission) is None


def test_agents_page_credits_contributors(tmp_path, records):
    from openinterpretability.builder.agents import contributors_view
    from openinterpretability.builder.tree import build_tree

    def credited(rec, proposer=None, reviewer=None):
        sub = rec.submission.model_copy(update={"contributor": proposer})
        verdict = rec.verdict.model_copy(update={"contributor": reviewer})
        return rec.model_copy(update={"submission": sub, "verdict": verdict})

    records[1] = credited(records[1], "alice", "bob")  # N117+ VERIFIED
    records[2] = credited(records[2], "alice", "bob")  # N117− well-probed null
    records[3] = credited(records[3], "bob")  # N9+ REJECTED, thinly probed
    compile_site(records, tmp_path)

    (model,) = build_tree(records, [])
    experiments = [e for layer in model.layers for n in layer.units for e in n.experiments]
    rows = {c.handle: c for c in contributors_view(experiments)}
    assert (rows["alice"].experiments, rows["alice"].verified) == (2, 1)
    assert (rows["alice"].conclusive_nulls, rows["alice"].reviews) == (1, 0)
    assert (rows["bob"].experiments, rows["bob"].conclusive_nulls, rows["bob"].reviews) == (1, 0, 2)
    assert [c.handle for c in contributors_view(experiments)] == ["bob", "alice"]
    page = (tmp_path / "agents" / "index.html").read_text()
    assert 'id="contributors"' in page and 'href="../users/alice.html">alice</a></td>' in page
    neuron = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
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
    assert "No contributors yet." in (tmp_path / "agents" / "index.html").read_text()


def _write_paper(papers, slug, text):
    folder = papers / slug
    (folder / "media").mkdir(parents=True)
    (folder / "media" / "fig.svg").write_text("<svg/>")
    (folder / "README.md").write_text(text)


def test_research_section_lists_papers_in_order_and_renders_them(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "README.md").write_text("- [Two](two/README.md)\n- [One](one/README.md)\n")
    _write_paper(
        papers,
        "one",
        "# First paper\n\nIts **summary**.\n\n![A figure](media/fig.svg)\n\n"
        "See [two](../two/README.md) and [code](../../scripts/x.py).\n\n| a |\n| --- |\n| 1 |\n",
    )
    _write_paper(papers, "two", "# Second paper\n\nOther summary.\n")
    out = tmp_path / "dist"
    compile_site([], out, papers_dir=papers)

    index = (out / "research" / "index.html").read_text()
    assert index.index("Second paper") < index.index("First paper")
    assert "Its summary." in index
    page = (out / "research" / "one" / "index.html").read_text()
    assert 'href="../two/index.html"' in page
    assert "/blob/main/scripts/x.py" in page
    assert '<table tabindex="0">' in page  # a table scrolls, so a keyboard must reach it (#5649)
    assert (out / "research" / "one" / "media" / "fig.svg").exists()
    assert 'href="research/index.html"' in (out / "index.html").read_text()


def test_paper_pages_and_the_index_name_the_authors(tmp_path):
    # The authors come from the paper's CITATION.cff, in order (#5260).
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "README.md").write_text("- [One](one/README.md)\n")
    _write_paper(papers, "one", "# First paper\n\nIts summary.\n")
    (papers / "one" / "CITATION.cff").write_text(
        "authors:\n- given-names: Will\n  family-names: Sackfield\n- name: Open Interpretability\n"
    )
    out = tmp_path / "dist"
    compile_site([], out, papers_dir=papers)
    for page in ("research/index.html", "research/one/index.html"):
        assert "Will Sackfield, Open Interpretability" in (out / page).read_text()


def test_research_section_copies_an_interactive_page_as_it_is(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "README.md").write_text("- [Live](live/README.md)\n")
    folder = papers / "live"
    folder.mkdir()
    (folder / "README.md").write_text("# Live page\n\nTry it.\n")
    (folder / "index.html").write_text("<!doctype html><p>own page")
    (folder / "data.js").write_text("window.X = 1;\n")
    out = tmp_path / "dist"
    compile_site([], out, papers_dir=papers)
    assert (out / "research" / "live" / "index.html").read_text() == "<!doctype html><p>own page"
    assert (out / "research" / "live" / "data.js").exists()
    assert not (out / "research" / "live" / "README.md").exists()
    assert "Interactive" in (out / "research" / "index.html").read_text()


def test_research_section_is_empty_without_papers(tmp_path):
    compile_site([], tmp_path)
    assert "No papers yet" in (tmp_path / "research" / "index.html").read_text()


def test_repo_papers_are_all_listed_on_the_index():
    from pathlib import Path

    papers = Path(__file__).parents[1] / "papers"
    listed = (papers / "README.md").read_text()
    for folder in papers.iterdir():
        if folder.is_dir():
            assert f"({folder.name}/README.md)" in listed


def _stop(*, layer=3, idx=117, pole="negative", outcome="default_on", share=0.55):
    from datetime import UTC, datetime

    from openinterpretability.core.draws import ChosenStop
    from openinterpretability.core.identifiers import NeuronIdentifier
    from openinterpretability.core.schemas import ExperimentProvenance

    return ChosenStop(
        identifier=NeuronIdentifier(model_name=MODEL, layer=layer, neuron_idx=idx, pole=pole),
        outcome=outcome,
        active_share=share,
        provenance=ExperimentProvenance(
            dtype="float32",
            corpus="stop-corpus",
            corpus_sha256="ab" * 32,
            n_snippets=1,
            tokens_scanned=100,
            selection="manual",
            harness_version="test",
        ),
        timestamp=datetime(2026, 11, 5, 23, 8, tzinfo=UTC),
    )


def test_unit_page_lists_a_stop_with_its_experiments(tmp_path, record):
    # The positive pole has an experiment. The negative pole has only the stop (#3297).
    studied = experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0))
    compile_site([studied], tmp_path, stops=[_stop()])
    page = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
    experiments = page[page.index('id="experiments"') :]
    assert "Stopped: default-on" in experiments and "55%" in experiments
    assert "chosen" in experiments and "Scans that stopped come last" in experiments
    assert f"/blob/main/wiki/stops/{MODEL}/L3/N117-negative/20261105T230800Z.md" in page
    assert "No experiment: the scan stopped before a hypothesis" in page
    assert "Not studied yet" not in page  # the stopped pole isn't offered as a candidate
    assert 'id="scans-that-stopped"' not in page


def test_unit_page_without_a_stop_lists_none(tmp_path, record):
    studied = experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0))
    compile_site([studied], tmp_path, stops=[_stop(idx=9)])
    page = unit_pages(tmp_path / "models" / MODEL / "L3" / "N117.html")
    assert "Stopped:" not in page and "Not studied yet" in page


def test_a_neuron_whose_scans_all_stopped_has_a_page(tmp_path, record):
    from openinterpretability.core.draws import UnauditableDraw

    studied = experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0))
    chosen = _stop_on_model(studied, idx=9, pole="positive", outcome="dead", share=None)
    fields = _stop_on_model(studied, idx=9).model_dump(exclude={"latest_record"})
    draw = UnauditableDraw.model_validate(fields)  # a random draw that stopped
    compile_site([studied], tmp_path, stops=[chosen], draws=[draw])
    page = (tmp_path / "models" / MODEL / "L3" / "N9.html").read_text()
    assert "Stopped: dead" in page and "Stopped: default-on" in page
    assert "random draw" in page and "chosen" in page
    assert f"/blob/main/wiki/draws/{MODEL}/L3/N9-negative/20261105T230800Z.json" in page
    assert "Not studied yet" not in page
    layer = (tmp_path / "models" / MODEL / "L3" / "index.html").read_text()
    assert 'href="N9.html"' in layer or "L3/N9.html" in layer
    assert "Scans that stopped on units with no page" not in (tmp_path / "index.html").read_text()


@pytest.mark.parametrize(("proven", "verified"), [(True, 2), (False, 1)])
def test_home_page_counts_a_proven_dead_stop_as_verified(tmp_path, record, proven, verified):
    studied = experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0))
    dead = _stop(idx=9, pole="positive", outcome="dead", share=None)
    # MODEL is the slug in paths; the coverage count matches the repo name in identifiers.
    model = studied.submission.identifier.model_name
    ident = dead.identifier.model_copy(update={"model_name": model})
    dead = dead.model_copy(update={"identifier": ident, "proven_dead": proven})
    compile_site([studied], tmp_path, stops=[dead])
    home = " ".join((tmp_path / "index.html").read_text().split())
    assert card_stat("Verified", f"{verified} of 116,736") in home
    assert ("1 of them proven dead" in home) is proven


def _stop_on_model(studied, **kwargs):
    """A stop on the model of ``studied``, named as its identifiers name it."""
    stop = _stop(**kwargs)
    model = studied.submission.identifier.model_name
    ident = stop.identifier.model_copy(update={"model_name": model})
    return stop.model_copy(update={"identifier": ident})


def test_a_neuron_with_only_stopped_scans_counts_as_studied(tmp_path, record):
    studied = experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0))
    dead = _stop_on_model(studied, idx=9, pole="positive", outcome="dead", share=None)
    default_on = _stop_on_model(studied, idx=9, pole="negative")
    compile_site([studied], tmp_path, stops=[dead, default_on])
    home = " ".join((tmp_path / "index.html").read_text().split())
    tile_value = '<dd class="mt-auto pt-1 font-mono text-2xl tabular-nums">2</dd>'
    assert f"Neurons studied</dt> {tile_value}" in home
    # Each tile is a flex column with its value at the bottom, so values share a baseline (#5832).
    assert '<div class="flex flex-col bg-white p-4 dark:bg-slate-950"> <dt' in home
    assert card_stat("Neurons studied", "2 of 116,736") in home
    assert card_stat("Verified", "1 of 116,736") in home
    model_page = " ".join((tmp_path / "models" / MODEL / "index.html").read_text().split())
    assert "2/116,736 neurons studied" in model_page and "STOPPED 1" in model_page
    layer = " ".join((tmp_path / "models" / MODEL / "L3" / "index.html").read_text().split())
    assert "2 neurons studied (1 with only stopped scans)" in layer
    assert "N9 · scan stopped: negative default-on, positive dead" in layer
    assert (tmp_path / "models" / MODEL / "L3" / "N9.html").exists()  # it has a page (#3297)


def test_a_stop_on_a_neuron_with_an_experiment_is_not_counted_twice(tmp_path, record):
    studied = experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0))
    compile_site([studied], tmp_path, stops=[_stop_on_model(studied, idx=117)])
    home = " ".join((tmp_path / "index.html").read_text().split())
    assert card_stat("Neurons studied", "1 of 116,736") in home
    layer = " ".join((tmp_path / "models" / MODEL / "L3" / "index.html").read_text().split())
    assert "1 neurons studied ·" in layer and "only stopped scans" not in layer


def test_a_random_draw_on_a_dead_neuron_counts_as_studied(tmp_path, record):
    from openinterpretability.core.draws import UnauditableDraw

    studied = experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0))
    stop = _stop_on_model(studied, idx=40, layer=5, pole="positive", outcome="dead", share=None)
    draw = UnauditableDraw(**stop.model_dump(exclude={"protocol_version", "latest_record"}))
    compile_site([studied], tmp_path, draws=[draw])
    home = " ".join((tmp_path / "index.html").read_text().split())
    assert card_stat("Neurons studied", "2 of 116,736") in home
    assert (tmp_path / "models" / MODEL / "L5" / "index.html").exists()


MODELS_PYPROJECT = """[tool.openinterpretability]
models = ["Qwen/Qwen2.5-0.5B", "org/unstudied", "org/unreadable"]

[tool.openinterpretability.licenses."Qwen/Qwen2.5-0.5B"]
license = "apache-2.0"
url = "https://u"
gated = false
publishing_allowed = true
"""
MODEL_FACTS = {
    "Qwen/Qwen2.5-0.5B": ModelFacts(494_032_768, 24, 4864, "silu"),
    "org/unstudied": ModelFacts(124_439_808, 12, 3072, "gelu_new"),
}


def _rows(page: str) -> list[list[str]]:
    body = page[page.index("<tbody") : page.index("</tbody>")]
    rows = re.findall(r"<tr>(.*?)</tr>", body, re.S)
    return [re.sub(r"<[^>]+>", " ", r).split() for r in rows]


def test_models_page_lists_every_configured_model(tmp_path, records, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(MODELS_PYPROJECT)
    monkeypatch.chdir(tmp_path)
    compile_site(records, tmp_path / "site", model_facts=MODEL_FACTS)
    page = (tmp_path / "site" / "models" / "index.html").read_text()

    rows = _rows(page)
    assert [r[0] for r in rows] == ["Qwen/Qwen2.5-0.5B", "org/unstudied", "org/unreadable"]
    qwen, unstudied, unreadable = rows
    assert qwen[1:6] == ["494M", "24", "4,864", "silu", "apache-2.0"]
    assert unstudied[1:7] == ["124M", "12", "3,072", "gelu_new", "not", "recorded"]
    assert unstudied[-3:] == ["0", "0", "—"]  # no coverage run
    assert unreadable[1:] == ["—", "—", "—", "—", "not", "recorded", "0", "0", "—"]

    # A model with no record still gets a page when its config gives its layers.
    assert 'href="../models/Qwen--Qwen2.5-0.5B/index.html"' in page
    assert 'href="../models/org--unstudied/index.html"' in page
    assert (tmp_path / "site" / "models" / "org--unstudied" / "index.html").exists()
    assert "org--unreadable" not in page
    # The header links the page from every page.
    assert '<a href="models/index.html"' in (tmp_path / "site" / "index.html").read_text()


def test_models_page_counts_experiments_and_verified(tmp_path, records, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(MODELS_PYPROJECT)
    monkeypatch.chdir(tmp_path)
    compile_site(records, tmp_path / "site", model_facts=MODEL_FACTS)
    qwen = _rows((tmp_path / "site" / "models" / "index.html").read_text())[0]
    assert qwen[-3:-1] == ["4", "1"]  # the fixture: 4 experiments, 1 neuron with a VERIFIED pole


def test_models_page_shows_each_models_pieces_bars_as_the_home_page_does(tmp_path, records):
    import re

    compile_site(records, tmp_path / "site")
    models = (tmp_path / "site" / "models" / "index.html").read_text()
    home = (tmp_path / "site" / "index.html").read_text()

    def card(page):  # the card's statistics and its bars, without the page-relative link
        found = re.search(r'aria-label="Status grid of [^"]+"', page)
        stats = re.search(r"\d+ experiments ·[^<]+", page)
        return found and found.group(0), stats and " ".join(stats.group(0).split())

    assert card(models) == card(home) and card(models)[0]  # the same card on both pages
    assert 'href="../models/' in models  # its link is relative to the Models page


def test_facts_from_config_counts_parameters_not_buffers():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=2, n_embd=8, n_head=2, vocab_size=10, n_positions=4)
    facts = facts_from_config(config.to_dict())
    real = GPT2LMHeadModel(config)
    assert facts.parameters == sum(p.numel() for p in real.parameters())  # tied once
    assert facts.parameters < sum(t.numel() for t in real.state_dict().values())
    assert (facts.layers, facts.mlp_width, facts.activation) == (2, 32, "gelu_new")


def test_facts_from_unknown_architecture():
    facts = facts_from_config({"model_type": "no-such-model", "num_hidden_layers": 3})
    assert facts == ModelFacts(None, None, None, None)


def test_format_parameters():
    assert format_parameters(124_439_808) == "124M"
    assert format_parameters(1_543_714_304) == "1.54B"
    assert format_parameters(None) == "—"
    assert format_parameters(512) == "512"


def test_the_page_says_where_each_step_ran_and_whether_it_streamed(tmp_path, record):
    from openinterpretability.core.schemas import RunEnvironment

    def env(device, streamed=None):
        versions = {"openinterp_version": "0", "torch_version": "0"}
        return RunEnvironment(device=device, dtype="float32", streamed=streamed, **versions)

    from openinterpretability.core.schemas import ExperimentProvenance

    sub = record.submission
    provenance = ExperimentProvenance(
        dtype="float32",
        corpus="toy",
        n_snippets=1,
        tokens_scanned=10,
        selection="manual",
        harness_version="0",
        environment=env("cpu", True),
    )
    streamed = record.model_copy(
        update={
            "submission": sub.model_copy(update={"provenance": provenance}),
            "verdict": record.verdict.model_copy(update={"environment": env("mps")}),
        }
    )
    compile_site([streamed], tmp_path)
    page = unit_pages(next((tmp_path / "models").rglob("N*.html")))
    text = re.sub(r"<[^>]+>", "", page)
    assert "scan cpu (streamed from disk) · review mps" in text


# Earlier tests leave native threads in this process (the CLI build has none), so the fork
# warns. The workers only render templates, so they need no lock that a thread could hold.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_parallel_render_writes_the_same_site_as_one_process(tmp_path, records):
    one = compile_site(records, tmp_path / "one", jobs=1)
    two = compile_site(records, tmp_path / "two", jobs=2)
    rel = [p.relative_to(tmp_path / "one") for p in one.pages]
    assert rel == [p.relative_to(tmp_path / "two") for p in two.pages]  # same order too
    for path in (tmp_path / "one").rglob("*"):
        if path.is_file():
            assert (
                path.read_bytes()
                == (tmp_path / "two" / path.relative_to(tmp_path / "one")).read_bytes()
            )


def test_heat_html_is_short_and_escaped():
    from openinterpretability.builder.heat import HeatToken, heat_html

    tokens = [
        HeatToken('<a & "b">', 0.5, 1.23456, peak=False),
        HeatToken("x", 1.0, 2.0, peak=True),
        HeatToken("↵", 0.0, 0.0, peak=False, marked=True),
        HeatToken(" y", 0.0, 0.0, peak=False),
        HeatToken(" w", 0.0, -0.0004, peak=False),
        HeatToken(" z", 0.007, 0.0004, peak=False),
    ]
    assert str(heat_html(tokens)) == (
        '<span class="hm">'
        '<i style=--a:.5 data-v="1.235">&lt;a &amp; &#34;b&#34;&gt;</i>'
        '<i class="p" style=--a:1 data-v="2.000 · peak">x</i>'
        '<i class="m" data-v="0.000 · marked">↵</i>'
        "<i> y</i>"  # no shade and a value of 0.000: base.html supplies both
        '<i data-v="-0.000"> w</i>'
        "<i style=--a:.007> z</i>"
        "</span>"
    )


def test_values_rows_are_short_and_mark_the_peak():
    from openinterpretability.builder.heat import PEAK_ROW, HeatToken, values_rows

    tokens = [HeatToken("<x>", 0.2, 0.25, peak=False), HeatToken("y", 1.0, 1.0, peak=True)]
    assert str(values_rows(tokens)) == (
        '<tr><td><code class="tok">&lt;x&gt;</code><td>0.250'
        f'<tr class="{PEAK_ROW}"><td><code class="tok">y</code><td>1.000'
    )


def test_home_page_shows_dashboard(tmp_path, records):
    compile_site(records, tmp_path)
    index = (tmp_path / "index.html").read_text()
    assert 'id="dashboard"' in index
    assert "Verdicts per model" in index and "Verdicts per layer" in index


def test_protocol_mix_of_verified_units_is_reported(tmp_path, records):
    from openinterpretability.builder.compiler import registry_stats
    from openinterpretability.core.schemas import PROTOCOL_VERSION, ReviewPolicy

    old = ReviewPolicy(protocol_version="0.7")
    records[1] = records[1].model_copy(
        update={"verdict": records[1].verdict.model_copy(update={"policy": old})}
    )
    report = compile_site(records, tmp_path)
    stats = registry_stats(report.parquet)
    mix = {row["protocol_version"]: row["units"] for row in stats["protocol_mix"]}
    assert mix.get("0.7") == 1 and stats["current_protocol"] == PROTOCOL_VERSION
    assert sum(mix.values()) == stats["totals"]["verified"]
    home = (tmp_path / "index.html").read_text()
    assert "verified claims were judged under the current protocol" in home
    assert "1 under 0.7" in home


def test_the_registry_table_is_the_same_in_batches(records, monkeypatch):
    """Batches only bound memory: the table is the same with one row per batch (#4652)."""
    from openinterpretability.builder import parquet_export as px
    from openinterpretability.builder.tree import build_tree

    experiments = [e for m in build_tree(records, []) for la in m.layers for n in la.units
                   for e in n.experiments]  # fmt: skip
    whole = px.records_to_table(experiments)
    monkeypatch.setattr(px, "BATCH_ROWS", 1)
    batched = px.records_to_table(experiments)
    assert batched.equals(whole)
    assert whole.column("record_json")[0].as_py().startswith("{")
    slim = px.records_to_table(experiments, with_json=False)
    assert "record_json" not in slim.column_names
    assert slim.equals(whole.drop_columns(["record_json"]))


def test_the_registry_file_is_written_a_batch_at_a_time(records, tmp_path, monkeypatch):
    """Each batch is its own row group, and the file reads back as the whole table (#4652)."""
    import pyarrow.parquet as pq

    from openinterpretability.builder import parquet_export as px
    from openinterpretability.builder.tree import build_tree

    experiments = [e for m in build_tree(records, []) for la in m.layers for n in la.units
                   for e in n.experiments]  # fmt: skip
    monkeypatch.setattr(px, "BATCH_ROWS", 2)
    path = px.export_registry(experiments, tmp_path)
    assert pq.ParquetFile(path).metadata.num_row_groups == 2  # 4 rows
    assert pq.read_table(path).equals(px.records_to_table(experiments).combine_chunks())


def test_a_claim_is_a_hash_shared_by_re_reviews(records):
    from openinterpretability.builder.tree import claim_of

    verified = records[1]
    again = _reviewed_by(verified, "deepseek/flash", status="VERIFIED", hours=3)
    assert claim_of(again) == claim_of(verified) and len(claim_of(verified)) == 64
    assert claim_of(records[0]) != claim_of(verified)  # another hypothesis


def test_a_build_from_slim_records_writes_the_same_site(tmp_path, records):
    """Records held slim, and loaded in full per unit and per batch, change no file (#4652)."""
    from openinterpretability.builder.slim import SlimArchive
    from openinterpretability.builder.wiki import load_archive_paths

    wiki = tmp_path / "wiki"
    for i, rec in enumerate(records):
        path = wiki / "experiments" / f"r{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rec.model_dump_json())
    full = [r for _, r in load_archive_paths(wiki)]
    archive = SlimArchive([])
    slim = [r for _, r in load_archive_paths(wiki, keep=archive)]
    assert len(archive.paths) == len(records)
    assert all(not r.submission.top_examples for r in slim)
    compile_site(full, tmp_path / "a")
    compile_site(slim, tmp_path / "b", archive=archive)
    assert not slim  # handed over
    a = {p.relative_to(tmp_path / "a"): p.read_bytes() for p in (tmp_path / "a").rglob("*")
         if p.is_file()}  # fmt: skip
    b = {p.relative_to(tmp_path / "b"): p.read_bytes() for p in (tmp_path / "b").rglob("*")
         if p.is_file()}  # fmt: skip
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)


@pytest.mark.parametrize("copies", [1, 3])
def test_a_build_holds_only_a_batch_or_a_unit_in_full(tmp_path, records, monkeypatch, copies):
    """The records held in full at once don't grow with the archive (#4652).

    A build holds every record slim. It loads records in full only per unit page and per
    Parquet batch. With N or 3N records, at most a batch or the widest unit is full at once.
    """
    from contextlib import contextmanager

    from openinterpretability.builder import compiler, parquet_export, slim
    from openinterpretability.builder.wiki import load_archive_paths

    wiki = tmp_path / "wiki"
    many = [
        experiment(r, idx=r.submission.identifier.neuron_idx + 1000 * k,
                   pole=r.submission.identifier.pole, status=r.verdict.status,
                   counts=(10, 0, 5, 0), hours=k)
        for k in range(copies) for r in records
    ]  # fmt: skip
    for i, rec in enumerate(many):
        path = wiki / "experiments" / f"r{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rec.model_dump_json())
    widest = {"n": 0, "loads": 0}
    real = slim.full_records

    @contextmanager
    def counted(experiments):
        held = [e for e in experiments if e.source is not None]
        widest["n"] = max(widest["n"], len(held))
        widest["loads"] += len(held)
        with real(experiments):
            yield

    monkeypatch.setattr(compiler, "full_records", counted)
    monkeypatch.setattr(parquet_export, "full_records", counted)
    monkeypatch.setattr(parquet_export, "BATCH_ROWS", 2)
    archive = slim.SlimArchive([])
    held = [r for _, r in load_archive_paths(wiki, keep=archive)]
    compile_site(held, tmp_path / "out", archive=archive)
    assert widest["loads"] >= len(many)  # each record was loaded in full somewhere
    # A batch of 2, or the widest unit's 3 experiments (N117's two poles): the same for 3N.
    assert widest["n"] == 3


def test_explain_reads_the_standing_not_the_newest_record(tmp_path, records):
    """The labels explain reads: a newer NEEDS_REVISION rival leaves the unit VERIFIED (#4900)."""
    from openinterpretability.explain import registry_labels
    from openinterpretability.planner_states import states_of

    records.append(_rival(records, status="NEEDS_REVISION", hours=4, text="Fires on nouns."))
    pairs = [(tmp_path / f"{i}.json", r) for i, r in enumerate(records)]
    states = states_of(pairs, [])
    slug = records[1].submission.identifier.slug
    assert states[slug].record.verdict.status == "NEEDS_REVISION"  # the newest record
    status, explanation, _ = registry_labels(
        tmp_path, records[1].submission.identifier.model_name, states
    )[slug]
    assert status == "VERIFIED"
    assert explanation == records[1].submission.natural_language_explanation
