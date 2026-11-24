"""``scripts/conformal_line.py``: a conformal line for each VERIFIED pole (#6417)."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

_SPEC = importlib.util.spec_from_file_location(
    "conformal_line", Path(__file__).parents[1] / "scripts" / "conformal_line.py"
)
cl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cl)

POLICY = {
    "activation_threshold": 0.25,
    "min_specificity": 0.8,
    "min_sensitivity": 0.6,
    "probe_cluster_threshold": 0.5,
}


def _probe(kind, activation, expected=None, prompt=None):
    expected = kind == "positive_holdout" if expected is None else expected
    prompt = prompt or f"{kind} probe number {activation} about something unrelated"
    return {"kind": kind, "activation": activation, "expected_active": expected,
            "prompt": prompt}  # fmt: skip


def _record(probes, status="VERIFIED", kind="mlp_neuron_pole", max_activation=4.0, floor=None):
    ident = {"model_name": "m/x", "layer": 1, "kind": kind, "neuron_idx": 2, "pole": "positive"}
    return {"submission": {"identifier": ident, "max_activation": max_activation,
                           "first_token_floor": floor},
            "verdict": {"status": status, "policy": POLICY,
                        "adversarial_test_results": probes}}  # fmt: skip


def test_the_line_is_just_above_the_conformal_quantile():
    peaks = [float(i) for i in range(1, 20)]  # n = 19: the 18th smallest at α = 0.1
    assert cl.conformal_line(peaks, 0.1) == math.nextafter(18.0, math.inf)
    assert cl.conformal_line(list(reversed(peaks)), 0.1) == math.nextafter(18.0, math.inf)
    assert cl.conformal_line(peaks, 0.05) == math.nextafter(19.0, math.inf)  # rank 19 of 19
    assert cl.conformal_line(peaks[:18], 0.05) == math.inf  # rank 19 of 18: too few windows
    tied = [1.0] * 10
    assert cl.conformal_line(tied, 0.5) > 1.0  # a tie at the quantile doesn't fire


def test_probes_are_read_again_with_a_new_line():
    words = ["apple", "river", "candle", "mountain", "violin", "harbour", "pepper", "glacier"]
    probes = [_probe("hard_negative", a, prompt=f"{w} {w} {w}")
              for a, w in zip((0.1, 0.2, 0.3, 0.4, 2.0), words, strict=False)]  # fmt: skip
    probes += [_probe("positive_holdout", a, prompt=f"{w} {w} {w}")
               for a, w in zip((1.0, 3.0, 0.5), words[5:], strict=True)]  # fmt: skip
    probes += [_probe("hard_negative", None), _probe("counterfactual", 9.0, expected=False)]
    rates = cl.probe_rates(_record(probes), 1.0)
    assert math.isclose(rates["specificity"], 0.8) and math.isclose(rates["sensitivity"], 2 / 3)
    assert rates["holds"] is True
    assert (rates["quiet"], rates["quiet_fired"]) == (5, 1)  # 2.0 fires; the None isn't read
    assert cl.probe_rates(_record(probes), 0.35)["holds"] is False  # 0.4 fires: specificity 0.6
    assert cl.probe_rates(_record(probes), 1.01)["holds"] is False  # 1.0 misses: sensitivity 1/3


def test_near_duplicate_probes_count_as_one_as_in_the_review():
    same = "the cat sat on the mat today"
    probes = [
        _probe("hard_negative", 2.0, prompt=same),
        _probe("hard_negative", 2.0, prompt=same),
        _probe("hard_negative", 0.1, prompt="violins echo through the old harbour"),
    ]
    rates = cl.probe_rates(_record(probes), 1.0)
    assert math.isclose(rates["specificity"], 0.5)  # two clusters: one fails, one passes
    assert (rates["quiet"], rates["quiet_fired"]) == (3, 2)  # the raw count takes every probe


def test_an_inverse_claims_probes_group_by_kind():
    # An inverse claim predicts its hard negatives active (agents/review_measurements.py).
    probes = [
        _probe("hard_negative", 2.0, expected=True, prompt="a quiet pond at dawn"),
        _probe("positive_holdout", 0.1, expected=False, prompt="loud drums in the hall"),
    ]
    rates = cl.probe_rates(_record(probes), 1.0)
    assert rates["specificity"] == 1.0 and rates["sensitivity"] == 1.0
    assert (rates["quiet"], rates["quiet_fired"]) == (1, 0)  # its quiet probe is the holdout


def test_a_record_with_no_probes_of_a_kind_has_no_rate():
    rates = cl.probe_rates(_record([_probe("hard_negative", 0.1)]), 1.0)
    assert rates["specificity"] == 1.0 and math.isnan(rates["sensitivity"])
    assert rates["holds"] is False
    assert math.isnan(cl.probe_rates(_record([]), 1.0)["specificity"])


def test_the_records_own_line_is_its_share_of_its_maximum():
    assert cl.record_line(_record([], max_activation=4.0)) == 1.0


def test_the_split_is_seeded_and_halves_the_windows():
    windows = [str(i) for i in range(10)]
    cal, held = cl.split(windows)
    assert len(cal) == len(held) == 5 and sorted(cal + held) == sorted(windows)
    assert (cal, held) == cl.split(windows, seed=0)
    assert cl.split(windows, seed=1) != (cal, held)
    assert windows == [str(i) for i in range(10)]  # the input isn't shuffled in place


def test_only_the_newest_verified_neuron_pole_of_each_unit_is_taken(tmp_path):
    def write(unit, name, record):
        folder = tmp_path / "experiments" / unit
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.json").write_text(json.dumps(record))

    write("m--x/L1/N2-positive", "20260101T000000Z", _record([], status="REJECTED"))
    write("m--x/L1/N2-positive", "20260201T000000Z", _record([]))
    write("m--x/L1/N3-positive", "20260201T000000Z", _record([]))
    write("m--x/L1/N3-positive", "20260301T000000Z", _record([], status="REJECTED"))
    write("m--x/L2/N4-positive", "20260101T000000Z", _record([], kind="attention_head"))
    write("m--y/L1/N5-negative", "20260101T000000Z", _record([]))
    found = cl.newest_verified(tmp_path)
    assert sorted(found) == ["m--x", "m--y"]
    assert [r["_slug"] for r in found["m--x"]] == ["m--x/L1/N2-positive"]
    assert found["m--y"][0]["_slug"] == "m--y/L1/N5-negative"


def test_a_pole_row_gives_each_line_with_its_coverage_and_rates():
    probes = [_probe("hard_negative", 0.5), _probe("positive_holdout", 3.0)]
    record = _record(probes) | {"_slug": "m--x/L1/N2-positive"}
    cal = [float(i) for i in range(1, 100)]  # the line at α = 0.05: just above 95
    held = [0.5, 96.0, 50.0, 99.0]
    row = cl.pole_row(record, cal, held)
    assert (row["table"], row["slug"], row["layer"], row["pole"]) == (
        "pole", "m--x/L1/N2-positive", 1, "positive")  # fmt: skip
    assert row["own_line"] == 1.0 and row["own_cover"] == 0.25  # only 0.5 is under 1.0
    assert row["own"]["holds"] is True
    assert row["a0.05"]["line"] == math.nextafter(95.0, math.inf)
    assert row["a0.05"]["cover"] == 0.5 and row["a0.05"]["holds"] is False  # 3.0 misses
    assert row["a0.01"]["line"] == math.nextafter(99.0, math.inf)  # rank 99 of 99
    assert row["a0.01"]["cover"] == 1.0
    assert cl.pole_row(record, cal[:50], held)["a0.01"]["line"] == math.inf  # too few


def test_window_peaks_read_each_poles_layer_and_skip_a_floored_first_token():
    acts = {1: torch.tensor([[[9.0, 0.0, -1.0], [0.0, 2.0, -3.0], [1.0, 1.0, 0.5]]]),
            3: torch.tensor([[[0.0, 0.0, -5.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]]])}  # fmt: skip
    seen = []

    def layer_activations(texts, layers):
        seen.append((texts, layers))
        return acts

    runner = SimpleNamespace(layer_activations=layer_activations)
    records = [_record([]), _record([], floor=0.1), _record([])]
    records[2]["submission"]["identifier"] |= {"layer": 3, "neuron_idx": 2, "pole": "negative"}
    records[1]["submission"]["identifier"]["neuron_idx"] = 0
    peaks = cl.window_peaks(runner, ["w1", "w2"], records)
    assert seen == [(["w1"], [1, 3]), (["w2"], [1, 3])]  # one pass per window, both layers
    assert peaks[0] == [0.5, 0.5]  # Layer 1, neuron 2 reads −1, −3 and 0.5.
    assert peaks[1] == [1.0, 1.0]  # Neuron 0 drops its first token, 9.0.
    assert peaks[2] == [5.0, 5.0]  # A negative pole takes the largest −act.
