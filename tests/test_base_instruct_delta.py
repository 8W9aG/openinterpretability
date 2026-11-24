"""The base-to-instruct weight change at each pole (scripts/base_instruct_delta.py, #5133)."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import base_instruct_delta as bid  # noqa: E402


def _layer(width=8, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"gate_proj": torch.randn(width, d, generator=g, dtype=torch.float64),
            "up_proj": torch.randn(width, d, generator=g, dtype=torch.float64),
            "down_proj": torch.randn(d, width, generator=g, dtype=torch.float64)}  # fmt: skip


def test_a_change_on_one_neuron_shows_on_that_neuron_only():
    base = _layer()
    delta = {m: torch.zeros_like(w) for m, w in base.items()}
    delta["gate_proj"][3] = base["gate_proj"][3]  # neuron 3's gate row doubles
    rank, measures = bid.neuron_measures(base, {m: d + 0 * base[m] for m, d in delta.items()})
    assert rank["gate_proj"] == 1
    relative = measures["relative_gate_proj"]
    assert torch.isclose(relative[3], torch.tensor(1.0, dtype=relative.dtype))
    assert (relative[torch.arange(8) != 3] == 0).all()
    assert measures["relative_all"][3] > 0 and set(measures) >= {"top_r_share", "relative_all"}


def test_a_change_on_few_neurons_is_more_concentrated_than_its_rotation():
    delta = {m: torch.zeros_like(w) for m, w in _layer(width=40).items()}
    delta["up_proj"][0] = 5.0
    row = bid.rotation_row(delta, layer=1)
    assert row["top_share"] == 1.0 and row["top_share_rotated"] < 0.5


def test_pole_rows_keep_only_the_poles(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"table": "summary"}) + "\n" + json.dumps({"table": "pole", "slug": "a"}) + "\n"
    )
    assert bid.pole_rows(path) == [{"table": "pole", "slug": "a"}]


def test_a_measure_that_only_tracks_depth_has_no_within_layer_signal():
    # Failures sit in layer 1, where every score is high: a high raw AUROC, a shuffle p of 1.
    layers = [0] * 10 + [1] * 10
    scores = [0.1] * 10 + [0.9] * 10
    fails = [False] * 10 + [True, False] * 5
    assert round(bid.auroc(scores, fails), 4) == 0.8333  # (5·10 + 5·5·½) / (5·15)
    assert bid.shuffled_p(scores, fails, layers, shuffles=99) == 1.0
