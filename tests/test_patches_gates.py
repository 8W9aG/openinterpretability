"""Gate functions, positions and composites (openinterpretability/patches/gates.py)."""

import pytest
import torch

from openinterpretability.patches.gates import Gates, inputs_of
from openinterpretability.patches.schema import PatchError

POINT = "blocks.0.hook_resid_post"
# One sequence of three positions, two features each.
X = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]])


def load(value):
    return torch.as_tensor(value, dtype=torch.float32)


def _gates(gates, values=None):
    g = Gates(gates, load)
    g.values.update(values or {POINT: X})
    return g


def test_an_unknown_gate_function_is_refused():
    with pytest.raises(PatchError, match="unknown gate function 'magic'"):
        inputs_of({"kind": "magic"})


def test_a_threshold_gate_can_read_a_direction():
    g = _gates({"g": {"function": {"kind": "threshold", "input": POINT, "direction": [1, -1]},
                      "threshold": 0.5}})  # fmt: skip
    # Scores x·[1, -1] are 1, -2 and 2.
    assert g.output("g").tolist() == [[1.0, 0.0, 1.0]]


def test_an_mlp_gate_is_a_small_classifier():
    function = {"kind": "mlp", "input": POINT, "W1": [[1.0, 0.0], [0.0, 1.0]], "b1": [0.0, -1.0],
                "W2": [[1.0, 1.0]], "b2": [-2.5]}  # fmt: skip
    g = _gates({"g": {"function": function}})
    # relu(x + b1) summed, less 2.5: -1.5, -1.5 and 0.5.
    assert g.output("g").tolist() == [[0.0, 0.0, 1.0]]


def test_attention_gates_need_the_attention_weights():
    function = {"kind": "attention_score", "heads": [[0, 0]], "key_tokens": [5]}
    values = {"hook_tokens": torch.tensor([[5, 6, 7]]), "blocks.0.attn.hook_pattern": None}
    g = _gates({"g": {"function": function}}, values)
    with pytest.raises(PatchError, match="the model must run eager attention"):
        g.output("g")


def test_a_gate_opens_only_at_its_listed_positions():
    function = {"kind": "threshold", "input": POINT, "index": 0}
    g = _gates({"g": {"function": function, "threshold": -1.0, "positions": [0, 2]}})
    assert g.output("g").tolist() == [[1.0, 0.0, 1.0]]


def test_a_missing_gate_is_named():
    with pytest.raises(PatchError, match="no gate nowhere"):
        _gates({}).output("nowhere")


def _leaves():
    # "a" opens at positions 0 and 2, "b" at positions 1 and 2.
    return {
        "a": {"function": {"kind": "threshold", "input": POINT, "index": 0}, "threshold": 0.5},
        "b": {"function": {"kind": "threshold", "input": POINT, "index": 1}, "threshold": 0.5},
    }


@pytest.mark.parametrize(
    ("kind", "children", "expected"),
    [
        ("not", ["a"], [0.0, 1.0, 0.0]),
        ("or", ["a", "b"], [1.0, 1.0, 1.0]),
        ("product", ["a", "b"], [0.0, 0.0, 1.0]),
        ("min", ["a", "b"], [0.0, 0.0, 1.0]),
    ],
)
def test_composite_gates_combine_their_gates(kind, children, expected):
    gates = _leaves() | {"c": {"function": {"kind": kind, "gates": children}}}
    assert _gates(gates).output("c").tolist() == [expected]


def test_a_not_gate_takes_exactly_one_gate():
    gates = _leaves() | {"c": {"function": {"kind": "not", "gates": ["a", "b"]}}}
    with pytest.raises(PatchError, match="a 'not' gate takes exactly one gate"):
        _gates(gates).output("c")
