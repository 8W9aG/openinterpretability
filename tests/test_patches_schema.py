"""What the patch record and its operations refuse (patches/schema.py)."""

import pytest
from pydantic import ValidationError

from openinterpretability.patches.schema import Op, Patch, PatchError, read_patch

BASE = "toy/model@abc123"


@pytest.mark.parametrize(
    "op, message",
    [
        ({"op_id": "a", "type": "scale"}, "scale op 'a' lacks factor"),
        (
            {"op_id": "a", "type": "lowrank_add", "params": {"A": [[1.0]]}},
            "lowrank_add op 'a' lacks B",
        ),
        (
            {"op_id": "g", "type": "gate", "params": {"function": {"kind": "magic"}}},
            "unknown gate function 'magic'",
        ),
        (
            {"op_id": "m", "type": "add_module", "params": {"module": {"kind": "conv"}}},
            "unknown module kind 'conv'",
        ),
        (
            {"op_id": "p:1", "type": "scale", "params": {"factor": 2.0}},
            "op_id 'p:1' may not hold ':', which separates a ref",
        ),
    ],
)
def test_an_op_that_breaks_a_rule_is_refused(op, message):
    with pytest.raises(ValidationError, match=message):
        Op(**op)


def test_known_gate_functions_and_modules_are_accepted():
    gate = Op(op_id="g", type="gate", params={"function": {"kind": "threshold"}})
    module = Op(op_id="m", type="add_module", params={"module": {"kind": "mlp"}})
    assert (gate.params["function"]["kind"], module.params["module"]["kind"]) == (
        "threshold",
        "mlp",
    )


def _scale(op_id):
    return {"op_id": op_id, "type": "scale", "params": {"factor": 2.0}}


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"base": BASE, "ops": [_scale("a"), _scale("a")]}, "op_id values must be unique"),
        ({"base": "toy/model"}, "base 'toy/model' must be '<model>@<revision>'"),
        (
            {"base": BASE, "evidence": {"verdict": "maybe"}},
            r"verdict must be one of \('fixed', 'not fixed'\), not 'maybe'",
        ),
    ],
)
def test_a_patch_that_breaks_a_rule_is_refused(fields, message):
    with pytest.raises(ValidationError, match=message):
        Patch(**fields)


def test_reading_a_patch_with_a_bad_base_raises_a_patch_error():
    text = Patch(base=BASE).sealed().model_dump_json().replace(BASE, "no-revision")
    with pytest.raises(PatchError, match="must be '<model>@<revision>'"):
        read_patch(text)


def test_a_patch_with_a_verdict_reads_it_back():
    patch = Patch(base=BASE, ops=[_scale("a")], evidence={"verdict": "fixed"}).sealed()
    assert read_patch(patch.model_dump_json()).verdict == "fixed"
