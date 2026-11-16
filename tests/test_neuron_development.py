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


def test_the_newest_verified_record_of_each_pole_is_taken(tmp_path):
    pole = tmp_path / "experiments" / nd.SLUG / "L1" / "N5-positive"
    pole.mkdir(parents=True)
    for name, status in (("20260101T000000Z", "VERIFIED"), ("20260201T000000Z", "VERIFIED"),
                         ("20260301T000000Z", "REJECTED")):  # fmt: skip
        record = {"verdict": {"status": status}, "submission": {"n": name}}
        (pole / f"{name}.json").write_text(json.dumps(record))
    (record,) = nd.verified_records(tmp_path)
    assert record["submission"]["n"] == "20260201T000000Z"
    assert record["_slug"] == f"{nd.SLUG}/L1/N5-positive"
