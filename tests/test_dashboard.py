from openinterp.builder.dashboard import DAILY_BINS, VERDICTS, dashboard, layer_columns, timeline
from openinterp.builder.tree import ModelNode

MODELS = [ModelNode("a", "a", 4, "a/index.html"), ModelNode("b", "b", 2, "b/index.html")]


def row(day, model, layer, status, n):
    return {"day": day, "model_name": model, "layer": layer, "status": status, "n": n}


ROWS = [
    row("2026-10-01", "a", 0, "VERIFIED", 2),
    row("2026-10-01", "a", 1, "REJECTED", 1),
    row("2026-10-03", "b", 1, "NEEDS_REVISION", 1),
]


def test_verdicts_in_status_order():
    assert VERDICTS == ("VERIFIED", "NEEDS_REVISION", "REJECTED")


def test_empty_registry_has_no_dashboard():
    assert dashboard([], MODELS) is None


def test_timeline_keeps_empty_days_and_cumulative_share():
    tl = timeline(ROWS)
    assert [c["date"] for c in tl["columns"]] == ["2026-10-01", "2026-10-02", "2026-10-03"]
    assert [c["n"] for c in tl["columns"]] == [3, 0, 1]
    assert [c["to_date"] for c in tl["columns"]] == [2 / 3, 2 / 3, 0.5]
    assert tl["unit"] == "day" and tl["share"] == 0.5
    first = tl["columns"][0]["segments"]
    assert first[0]["status"] == "VERIFIED"  # VERIFIED sits on the baseline
    assert first[0]["y"] + first[0]["h"] == tl["count_base"]


def test_long_timeline_bins_by_week():
    rows = [row("2026-01-01", "a", 0, "VERIFIED", 1), row("2026-12-31", "a", 0, "REJECTED", 1)]
    tl = timeline(rows)
    assert tl["unit"] == "week" and len(tl["columns"]) < DAILY_BINS
    assert sum(c["n"] for c in tl["columns"]) == 2


def test_layer_columns_give_every_layer_a_slot():
    charts = layer_columns(ROWS, MODELS)
    assert [c["name"] for c in charts] == ["a", "b"]
    assert [c["n"] for c in charts[0]["columns"]] == [2, 1, 0, 0]
    assert charts[1]["columns"][1]["label"] == "b L1 · 1 experiment: 1 NEEDS REVISION (0% VERIFIED)"
