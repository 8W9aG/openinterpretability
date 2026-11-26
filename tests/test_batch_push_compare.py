"""scripts/batch_push_compare.py: the push in training of the 32× batch, by range (#6806)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import batch_push_compare as bpc  # noqa: E402


def _layer(push, ratio, moved):
    return {"median_push": push, "median_push_ratio": ratio, "median_moved": moved}


def _window(step, push):
    # Layer 0 is not counted, so its values must not change the summary.
    layers = [_layer(99.0, 99.0, 99.0)] + [_layer(push * k, -0.1 * k, -1.0 * k) for k in (1, 2, 3)]
    return {"part": "push_batch_32x", "step": step, "layers": layers}


def test_a_range_takes_the_median_of_l1_to_l3_and_the_mean_over_windows():
    rows = [_window(100, -1.0), _window(200, 2.0), _window(300, -3.0),
            {"part": "push_batch_base", "step": 200, "layers": []}]  # fmt: skip
    got = bpc.summarise(rows, "push_batch_32x", 0, 200)
    assert got["windows"] == 2 and got["windows_down"] == 1  # medians -2 and +4
    assert got["median_push"] == pytest.approx(1.0)
    assert got["push_ratio"] == pytest.approx(-0.2) and got["moved"] == pytest.approx(-2.0)


def test_losses_read_only_the_runs_evaluations():
    rows = [
        {"part": "training_batch_push", "condition": "dropout_0", "step": 0, "loss": 10.8321},
        {"part": "training_batch_push", "condition": "dropout_0", "step": 2000, "loss": 5.791},
        {"part": "other", "condition": "dropout_0", "step": 4000, "loss": 1.0},
        {"part": "training_batch_push", "condition": "dropout_0", "step": 100, "layers": []},
    ]
    assert bpc.losses(rows) == {"dropout_0": {0: 10.832, 2000: 5.791}}
