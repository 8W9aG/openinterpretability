"""``scripts/neuron_development.py``: VERIFIED poles through Pythia's checkpoints (#3376)."""

import importlib.util
import json
import math
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "neuron_development", Path(__file__).parents[1] / "scripts" / "neuron_development.py"
)
nd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nd)


def test_pass_rate_counts_only_the_measured_probes_of_one_kind():
    strengths = [1.0, 0.1, None, 0.9, 0.0]
    expected = [True, True, True, False, False]
    assert nd.pass_rate(strengths, expected, 0.5, want=True) == 0.5  # 1.0 passes, 0.1 fails
    assert nd.pass_rate(strengths, expected, 0.5, want=False) == 0.5  # 0.0 passes, 0.9 fails
    assert math.isnan(nd.pass_rate([None], [True], 0.5, want=True))


def test_pearson_is_nan_for_a_constant_list():
    assert nd.pearson([1, 2, 3], [2, 4, 6]) == 1.0
    assert math.isnan(nd.pearson([1, 1, 1], [1, 2, 3]))


def test_a_pole_is_born_when_it_holds_and_correlates_from_then_on():
    rows = [
        {"step": 0, "r_final": 0.1, "holds": False},
        {"step": 512, "r_final": 0.9, "holds": True},
        {"step": 1000, "r_final": 0.7, "holds": True},  # dips: not yet formed
        {"step": 2000, "r_final": 0.85, "holds": True},
        {"step": 4000, "r_final": 1.0, "holds": True},
    ]
    assert nd.born_at(rows) == 2000
    assert nd.born_at([{"step": 0, "r_final": 1.0, "holds": False}]) is None


def test_a_pole_is_taken_only_when_its_newest_record_is_verified(tmp_path):
    # #4881: a later REJECTED record leaves the pole out, though an older one is VERIFIED.
    for pole, statuses in (("N5-positive", ("VERIFIED", "VERIFIED", "REJECTED")),
                           ("N7-positive", ("REJECTED", "VERIFIED"))):  # fmt: skip
        folder = tmp_path / "experiments" / nd.SLUG / "L1" / pole
        folder.mkdir(parents=True)
        for month, status in enumerate(statuses, start=1):
            name = f"20260{month}01T000000Z"
            record = {"verdict": {"status": status}, "submission": {"n": name}}
            (folder / f"{name}.json").write_text(json.dumps(record))
    (record,) = nd.verified_records(tmp_path)
    assert record["submission"]["n"] == "20260201T000000Z"
    assert record["_slug"] == f"{nd.SLUG}/L1/N7-positive"
    assert nd.left_out(tmp_path) == ["L1/N5-positive"]


def test_each_row_correlates_with_the_checkpoint_before():
    record = {
        "_slug": "s",
        "verdict": {
            "adversarial_test_results": [],
            "policy": {"min_specificity": 0.8, "min_sensitivity": 0.6},
        },
        "submission": {"max_activation": 1.0},
    }
    by_step = {0: {"s": [1.0, 2.0, 3.0]}, 1: {"s": [1.0, 2.0, 4.0]}, 2: {"s": [3.0, 2.0, 1.0]}}
    rows = nd.measure_pole(record, by_step, (0, 1, 2), 3)
    assert [r["r_prev"] for r in rows][0] is None
    assert rows[1]["r_prev"] > 0.9 and rows[2]["r_prev"] < -0.9
