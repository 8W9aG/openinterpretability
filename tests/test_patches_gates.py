"""Gate functions, positions and composites (openinterpretability/patches/gates.py)."""

import pytest
import torch

from openinterpretability.patches.gates import Gates, _mask, inputs_of
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
    with pytest.raises(
        PatchError, match="^attention weights are missing: the model must run eager attention$"
    ):
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
    with pytest.raises(PatchError, match="^a 'not' gate takes exactly one gate$"):
        _gates(gates).output("c")


def test_each_kind_reads_its_inputs():
    assert inputs_of({"kind": "mlp", "input": POINT}) == [POINT]
    assert inputs_of({"kind": "threshold", "input": POINT}) == [POINT]
    heads = {"kind": "attention_score", "heads": [[2, 0], [0, 1], [2, 3]], "key_tokens": [5]}
    assert inputs_of(heads) == ["hook_tokens", "blocks.0.attn.hook_pattern",
                                "blocks.2.attn.hook_pattern"]  # fmt: skip
    assert inputs_of({"kind": "linear_probe", "input": POINT}) == [POINT]
    assert inputs_of({"kind": "token_match", "tokens": [1]}) == ["hook_tokens"]


def test_a_linear_probe_adds_its_bias_and_defaults_it_to_zero():
    with_b = _gates({"g": {"function": {"kind": "linear_probe", "input": POINT, "w": [1.0, 0.0],
                                        "b": 1.5}, "output": "soft"}})  # fmt: skip
    without = _gates({"g": {"function": {"kind": "linear_probe", "input": POINT, "w": [1.0, 0.0]},
                            "output": "soft"}})  # fmt: skip
    # Scores x·[1, 0] + b: 2.5, 1.5 and 4.5 with the bias, 1, 0 and 3 without.
    assert torch.allclose(with_b.output("g"), torch.sigmoid(torch.tensor([[2.5, 1.5, 4.5]])))
    assert torch.allclose(without.output("g"), torch.sigmoid(torch.tensor([[1.0, 0.0, 3.0]])))


def test_a_soft_gate_is_a_sigmoid_of_the_score_less_its_threshold_over_its_temperature():
    function = {"kind": "threshold", "input": POINT, "index": 0}
    soft = _gates({"g": {"function": function, "output": "soft", "threshold": 1.0,
                         "temperature": 2.0}})  # fmt: skip
    # Scores 1, 0 and 3, so (score - 1) / 2 is 0, -0.5 and 1.
    assert torch.allclose(soft.output("g"), torch.sigmoid(torch.tensor([[0.0, -0.5, 1.0]])))
    default = _gates({"g": {"function": function, "output": "soft"}})
    assert torch.allclose(default.output("g"), torch.sigmoid(torch.tensor([[1.0, 0.0, 3.0]])))


def test_a_forced_gate_is_fully_open_or_shut():
    assert _gates({"g": {"force": "open"}}).output("g").item() == 1.0
    assert _gates({"g": {"force": "shut"}}).output("g").item() == 0.0


def test_the_mask_opens_all_positions_or_the_last():
    assert torch.equal(_mask({}, torch.Size([1, 3]), "cpu"), torch.ones(1, 3))
    assert _mask({"positions": "last"}, torch.Size([1, 3]), "cpu").tolist() == [[0.0, 0.0, 1.0]]


def test_and_and_min_take_the_smallest_output_of_their_gates():
    soft = {"a": {"function": {"kind": "threshold", "input": POINT, "index": 0},
                  "output": "soft"},
            "b": {"function": {"kind": "threshold", "input": POINT, "index": 1},
                  "output": "soft"}}  # fmt: skip
    a, b = _gates(soft).output("a"), _gates(soft).output("b")
    for kind in ("and", "min"):
        gates = soft | {"c": {"function": {"kind": kind, "gates": ["a", "b"]}}}
        # A product would give a smaller value at every position.
        assert torch.allclose(_gates(gates).output("c"), torch.minimum(a, b))


def test_an_attention_gate_sums_each_query_over_its_matching_keys():
    function = {"kind": "attention_score", "heads": [[0, 0]], "key_tokens": [5]}
    # Query 0 attends 0.9 to the key token at position 0, query 1 0.2, query 2 0.6.
    pattern = torch.tensor([[[[0.9, 0.1, 0.0], [0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]]])
    values = {"hook_tokens": torch.tensor([[5, 6, 7]]), "blocks.0.attn.hook_pattern": pattern}
    g = _gates({"g": {"function": function, "output": "soft", "threshold": 0.5,
                      "temperature": 0.1}}, values)  # fmt: skip
    assert torch.allclose(
        g.output("g"), torch.sigmoid((torch.tensor([[0.9, 0.2, 0.6]]) - 0.5) / 0.1)
    )


def test_a_gate_that_reads_a_point_not_yet_reached_says_so():
    g = _gates({"g": {"function": {"kind": "threshold", "input": "blocks.3.mlp.hook_out",
                                   "index": 0}}}, {})  # fmt: skip
    with pytest.raises(PatchError, match="^a gate reads blocks.3.mlp.hook_out, which the forward "
                                         "pass hasn't reached yet$"):  # fmt: skip
        g.output("g")


@pytest.mark.parametrize(
    "function",
    [
        {"kind": "token_match", "tokens": [5]},
        {"kind": "linear_probe", "input": POINT, "w": [1.0, 0.0]},
        {"kind": "threshold", "input": POINT, "direction": [1.0, 0.0]},
        {
            "kind": "mlp",
            "input": POINT,
            "W1": [[1.0, 0.0]],
            "b1": [0.0],
            "W2": [[1.0]],
            "b2": [0.0],
        },
        {"kind": "attention_score", "heads": [[0, 0]], "key_tokens": [5]},
    ],
)
def test_every_gate_moves_its_parameters_to_the_input_s_device(function):
    # A tensor on the meta device can't be combined with one on the CPU, so a parameter left
    # behind on the CPU raises. The meta device stands in for a GPU here.
    meta = {"hook_tokens": torch.tensor([[5, 6, 7]], device="meta"),
            POINT: X.to("meta"),
            "blocks.0.attn.hook_pattern": torch.zeros(1, 1, 3, 3, device="meta")}  # fmt: skip
    out = _gates({"g": {"function": function, "positions": "last"}}, meta).output("g")
    assert out.device.type == "meta" and out.shape == (1, 3)


def test_an_attention_gate_averages_over_its_heads():
    function = {"kind": "attention_score", "heads": [[0, 0], [0, 1]], "key_tokens": [5]}
    pattern = torch.tensor([[[[1.0, 0.0, 0.0]] * 3, [[0.0, 1.0, 0.0]] * 3]])  # heads 0 and 1
    values = {"hook_tokens": torch.tensor([[5, 6, 7]]), "blocks.0.attn.hook_pattern": pattern}
    g = _gates({"g": {"function": function, "output": "soft", "temperature": 0.1}}, values)
    # Head 0 puts all its weight on the key, head 1 none, so the mean is 0.5 at every query.
    assert g.output("g").shape == (1, 3)
    assert torch.allclose(g.output("g"), torch.sigmoid(torch.full((1, 3), 0.5) / 0.1))


def test_a_hard_gate_stays_shut_at_its_threshold():
    function = {"kind": "threshold", "input": POINT, "index": 0}
    # Scores 1, 0 and 3: the gate opens only where the score is above 1.
    assert _gates({"g": {"function": function, "threshold": 1.0}}).output("g").tolist() == [
        [0.0, 0.0, 1.0]
    ]
