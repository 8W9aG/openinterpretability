"""The interference report's helpers: studied units and directly edited neurons (#5230)."""

from __future__ import annotations

from openinterpretability.core.identifiers import model_slug
from openinterpretability.core.paths import experiments_dir
from openinterpretability.interference import report
from openinterpretability.patches.schema import Op

MODEL = "Qwen/Qwen2.5-0.5B"


def test_studied_reads_each_layer_and_neuron_with_an_experiment(tmp_path):
    root = experiments_dir(tmp_path) / model_slug(MODEL)
    for name in ["L3/N117-positive", "L3/N117-negative", "L10/N4-negative",
                 "Lx/N5-positive", "L2/notes"]:  # fmt: skip
        (root / name).mkdir(parents=True)
    assert report.studied(tmp_path, MODEL) == {(3, 117), (10, 4)}


def test_studied_is_empty_when_the_model_has_no_experiments(tmp_path):
    assert report.studied(tmp_path, MODEL) == set()


def test_edited_skips_refs_and_non_mlp_sites_and_reads_each_op_kind():
    def op(op_id, kind, targets, **params):
        return ("p:" + op_id, Op(op_id=op_id, type=kind, targets=targets, params=params))

    ops = [
        op("r", "sparse_add", ["p:g"], axis="col", index=[1], values=[0.0]),  # a ref
        op("e", "replace", ["embed.W_E"], value=0.0),  # no layer
        op("a", "replace", ["blocks.0.attn.W_O"], value=0.0),  # not the MLP
        op("p", "prune", ["blocks.2.mlp.W_out"], unit="neuron", index=7),
        op("c", "clamp", ["blocks.3.mlp.hook_post"], index=4, value=1.0),
        op("s", "scale", ["blocks.4.mlp.W_in"], factor=2.0),
        op("v", "steer", ["blocks.5.mlp.hook_post"], vector=[0.0]),  # no neuron to name
        op("x", "sparse_add", ["blocks.6.mlp.W_out"], axis="col", index=3, values=[0.0]),
    ]
    assert report.edited(ops) == {(2, 7), (3, 4), (4, None)}
