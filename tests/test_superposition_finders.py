"""The superposition finders and the ``superpositions`` commands (issue #3686)."""

from __future__ import annotations

import json

import pytest
import torch
from click.testing import CliRunner

from openinterp.audit_options import read_superposition
from openinterp.cli import cli
from openinterp.superposition_finders import (
    NUMBER_SOURCE,
    agreement_examples,
    compare,
    from_circuit,
    grammatical_number,
    sparse_probe,
    split_by_group,
)
from tests.test_wiring import qwen_runner  # noqa: F401  (fixture)


def row(rank: int, layer: int, idx: int, attribution: float) -> dict:
    return {"table": "neuron", "rank": rank, "layer": layer, "neuron_idx": idx,
            "attribution": attribution}  # fmt: skip


ROWS = [
    {"table": "curve", "basis": "attribution", "size": 10},
    row(1, 3, 40, 0.9),
    row(2, 0, 7, -0.8),
    row(3, 3, 5, 0.7),
    row(4, 1, 2, 0.6),
    row(5, 0, 1, 0.5),
    row(6, 3, 9, 0.0),
    row(7, 1, 8, 0.4),
]


def test_from_circuit_groups_by_layer_and_sorts_members():
    found = from_circuit(ROWS, top=6, min_members=2, source="test")
    assert [layer for layer, _ in found] == [0, 3]  # layer 1 has 1 neuron in the top 6
    by_layer = dict(found)
    assert by_layer[0].neurons == [1, 7] and by_layer[3].neurons == [5, 40]  # the zero is dropped
    assert by_layer[0].members[1].weight == -0.8 and by_layer[0].method == "circuit"


def test_from_circuit_keeps_layers_with_min_members():
    assert [layer for layer, _ in from_circuit(ROWS, top=7, min_members=2, source="t")] == [0, 1, 3]
    assert from_circuit(ROWS, top=7, min_members=3, source="t") == []
    with pytest.raises(ValueError):
        from_circuit(ROWS, min_members=1, source="t")


def test_sparse_probe_finds_the_planted_columns():
    gen = torch.Generator().manual_seed(0)
    acts = torch.randn(1000, 50, generator=gen) * torch.linspace(0.5, 3.0, 50) + 2.0
    planted = [4, 17, 33]
    labels = (acts[:, 4] - 2) / 0.6 - (acts[:, 17] - 2) / 1.3 + (acts[:, 33] - 2) / 2.2 > 0
    neurons, weights, accuracy = sparse_probe(acts, labels.long(), k=3, seed=0)
    assert neurons == planted and accuracy > 0.9
    assert weights[0] > 0 > weights[1] and weights[2] > 0


def test_split_by_group_keeps_each_group_on_one_side():
    groups = torch.arange(200) // 2
    train, test = split_by_group(groups, seed=0)
    assert len(train) + len(test) == 200 and len(test) == 40
    assert not set(groups[train].tolist()) & set(groups[test].tolist())


def test_from_circuit_command_writes_definitions_that_audit_reads(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    out = tmp_path / "defs.jsonl"
    args = ["superpositions", "from-circuit", "--rows", str(rows), "--model", "toy"]
    result = CliRunner().invoke(cli, [*args, "--top", "7", "--out", str(out)])
    assert result.exit_code == 0, result.output
    lines = out.read_text().splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        one = tmp_path / f"{i}.json"
        one.write_text(line)
        unit, model, layer, profile = read_superposition(one, "neuron")
        assert (unit, model, profile.method) == ("superposition", "toy", "circuit")
        assert layer == json.loads(line)["layer"]


def test_agreement_examples_pair_a_singular_with_a_plural():
    examples = agreement_examples(6, seed=0)
    assert len(examples) == 12
    assert all(a[1] + b[1] == 1 for a, b in zip(examples[::2], examples[1::2], strict=True))


def test_grammatical_number_on_the_tiny_qwen(qwen_runner):  # noqa: F811
    found = grammatical_number(qwen_runner, [0, 2], k=4, n=16, seed=0)
    assert [layer for layer, _, _ in found] == [0, 2]
    for _, profile, accuracy in found:
        assert len(profile.members) == 4 and profile.method == "sparse-probe"
        assert profile.source == NUMBER_SOURCE and 0.0 <= accuracy <= 1.0


def test_compare_gives_shared_neurons_and_cosine():
    (_, a), (_, b) = from_circuit(ROWS, top=7, min_members=2, source="t")[:2]
    assert compare(a, a) == ([1, 7], pytest.approx(1.0))
    assert compare(a, b) == ([], 0.0)
