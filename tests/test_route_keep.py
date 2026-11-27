"""The helpers of scripts/route_keep.py (#5164)."""

import sys
from pathlib import Path
from types import SimpleNamespace as N

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import route_keep as rk  # noqa: E402


def _record(all_cut=10.0, patching=0.4):
    m = N(clean=-0.12345, all_cut=all_cut, route_cut=3.5, sufficiency=0.61, patching=patching,
          null_cuts=[0.123456], null_patching=[None, 0.012345], corrupted_gap=2.0,
          reviewer_patching={"a": None, "b": 0.75}, reviewer_null_patching={"b": [0.0, None]},
          controls={"c1": True}, probes={"p1": False, "p2": True},
          top_tokens={"c1": " ball", "p1": " hat"})  # fmt: skip
    v = N(status="REJECTED", gates={"necessity": False, "sufficiency": True}, measures=m)
    return N(verdict=v)


def test_gate_row_rounds_the_measures_and_keeps_missing_values():
    row = rk.gate_row(_record(), "M", 7)
    assert (row["table"], row["model"], row["seed"], row["status"]) == ("gate", "M", 7, "REJECTED")
    assert row["necessity"] == 0.35 and row["clean"] == -0.1235
    assert row["null_patching"] == [None, 0.0123]
    assert row["reviewer_patching"] == {"a": None, "b": 0.75}
    assert row["reviewer_null_patching"] == {"b": [0.0, None]}
    assert row["gates"] == {"necessity": False, "sufficiency": True}


def test_gate_row_has_no_necessity_without_a_cut_and_no_patching_without_a_gap():
    row = rk.gate_row(_record(all_cut=0.0, patching=None), "M", 0)
    assert row["necessity"] is None and row["patching"] is None


def test_prompt_rows_list_every_control_and_probe_with_its_top_token():
    rows = rk.prompt_rows(_record(), "M")
    assert [(r["kind"], r["prompt"], r["as_claimed"], r["top"]) for r in rows] == [
        ("control", "c1", True, " ball"),
        ("probe", "p1", False, " hat"),
        ("probe", "p2", True, None),
    ]
