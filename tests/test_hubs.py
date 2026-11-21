"""Hubs as entities: identity, derived standing and artefact facts (hubs.py, #4285)."""

import pytest
from pydantic import ValidationError

from openinterpretability import hubs
from openinterpretability.builder.circuits import circuits_and_hubs
from openinterpretability.builder.tree import build_tree
from openinterpretability.core.identifiers import HubIdentifier
from openinterpretability.core.schemas import ReviewPolicy, scan_active_share
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.test_circuits import at, wiring


def test_a_hub_is_named_after_its_center():
    hub = HubIdentifier(model_name="Qwen/Qwen2.5-0.5B", layer=12, center="neuron", index=1)
    assert hub.slug == "Qwen--Qwen2.5-0.5B/L12/N1/HUB"
    head = HubIdentifier(model_name="m/x", layer=3, center="head", index=2)
    assert (head.center_slug, head.kind) == ("m--x/L3/H2", "hub")
    with pytest.raises(ValidationError):
        HubIdentifier(model_name="m/x", layer=3, center="feature", index=2)


def _one_hub(record):
    spoke = wiring("activations", down=[("mlp_neuron", 12, 1, 0.1)])
    records = [*(at(record, 2, i, activations=spoke) for i in range(10)), at(record, 12, 1)]
    (model,) = build_tree(records, [])
    return circuits_and_hubs(model, {})[1]


def test_a_hub_stands_as_its_center_and_reports_its_facts(record):
    found = _one_hub(record)
    norm = hubs.NormRow(hub="x", date="2026-11-17", norm=2.0, layer_median=1.0, percentile=0.99)
    ident = hubs.identifier(found[0].unit)
    (entity,) = hubs.hub_entities(found, {ident.slug: norm})
    assert entity.identifier == ident and entity.neighbours == 10
    assert entity.standings == found[0].unit.standings and entity.norm is norm
    (side,) = entity.standings
    lead = entity.standings[side].lead.record
    expected = lead.verdict.metrics.active_share if lead.verdict.metrics else None
    if expected is None:
        expected = scan_active_share(lead.submission, ReviewPolicy())
    assert entity.active_share[side] == expected
    assert entity.default_on[side] is (expected is not None and expected >= 0.5)
    assert hubs.hub_entities(found)[0].norm is None  # no row recorded yet


def test_norm_rows_come_from_the_weights_and_the_newest_wins(tiny_gpt2, tmp_path):
    runner = NeuronHookRunner(tiny_gpt2)
    neuron = HubIdentifier(model_name="tiny", layer=1, center="neuron", index=5)
    row = hubs.norm_row(runner, neuron, "2026-11-17")
    columns = tiny_gpt2.transformer.h[1].mlp.c_proj.weight.detach().norm(dim=1)  # Conv1D: [in, out]
    assert row.norm == pytest.approx(float(columns[5]), abs=1e-6)  # stored to 6 places
    assert row.layer_median == pytest.approx(float(columns.median()), abs=1e-6)
    assert row.percentile == pytest.approx(float((columns < columns[5]).float().mean()), abs=1e-6)
    head = HubIdentifier(model_name="tiny", layer=0, center="head", index=1)
    own = float(runner.head_output_weights(0, 1).norm())
    assert hubs.norm_row(runner, head, "2026-11-17").norm == pytest.approx(own, abs=1e-6)

    path = hubs.path_for("tiny", tmp_path)
    old = row.model_copy(update={"date": "2026-11-01", "norm": 9.0})
    path.write_text(old.model_dump_json() + "\n" + row.model_dump_json() + "\n")
    assert hubs.load_norms(path) == {neuron.slug: row}
    assert hubs.load_norms(tmp_path / "missing.jsonl") == {}


def test_record_facts_take_the_reviewed_active_share_first(record):
    from openinterpretability.core.schemas import ReviewMetrics

    metrics = ReviewMetrics(specificity=1.0, sensitivity=1.0, balanced_accuracy=1.0,
                            counterfactual_reproduction=1.0, lookahead_consistent=True,
                            independent_review=True, gates={"not_default_on": False},
                            active_share=0.75)  # fmt: skip
    verdict = record.verdict.model_copy(update={"metrics": metrics})
    share, default_on = hubs.record_facts(record.model_copy(update={"verdict": verdict}))
    assert share == 0.75 and default_on is True


def test_load_norms_skips_blank_lines_and_keeps_the_newest_in_any_order(tmp_path):
    new = hubs.NormRow(hub="h", date="2026-11-17", norm=2.0, layer_median=1.0, percentile=0.5)
    old = new.model_copy(update={"date": "2026-11-01", "norm": 9.0})
    path = tmp_path / "hubs.jsonl"
    path.write_text(new.model_dump_json() + "\n\n" + old.model_dump_json() + "\n")
    assert hubs.load_norms(path) == {"h": new}


def test_a_hub_line_without_facts_says_none_were_recorded():
    line = hubs.hub_line({"hub": "m/L1/H2/HUB", "href": "hubs/m/L1-H2.html"})
    assert "no facts recorded. A hub can be an artefact." in line
    assert line.startswith("hub m/L1/H2/HUB (") and "hubs/m/L1-H2.html" in line
