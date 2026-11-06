"""The helpers behind docs/analyses/default-on-poles.md."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from default_on_clamps import (  # noqa: E402
    added_at_final_norm,
    clamp,
    keep_first,
    shuffle_within_neurons,
    shuffled_clamp,
    zero,
)
from default_on_decode import ridge  # noqa: E402
from default_on_poles import gelu_dip, gelu_slope, split_half, split_windows  # noqa: E402
from default_on_targets import (  # noqa: E402
    TokenRows,
    ambiguous_rows,
    auroc,
    bootstrap_interval,
    label_tokens,
    resample_groups,
    token_lookup,
)

from openinterp.harness.annotations import Span  # noqa: E402
from openinterp.harness.hooks import NeuronHookRunner  # noqa: E402
from tests.conftest import VOCAB, FakeTokenizer  # noqa: E402


def test_gelu_dip_is_gelus_minimum():
    grid = torch.linspace(-3, 0, 30001, dtype=torch.float64)
    assert gelu_dip() == pytest.approx(
        grid[torch.nn.functional.gelu(grid).argmin()].item(), abs=1e-3
    )
    assert gelu_slope(torch.tensor(gelu_dip())).abs() < 1e-5


def test_split_windows_keeps_documents_apart():
    docs = torch.arange(40).repeat_interleave(5)
    fit, test = split_windows(docs, 30, 40, seed=0)
    assert len(fit) == 30 and len(test) == 40
    assert not set(docs[fit].tolist()) & set(docs[test].tolist())


def test_clamp_and_zero_change_one_side_only():
    acts = torch.tensor([[[-2.0, 1.0], [3.0, -4.0]]])
    assert clamp(torch.tensor([-1.0, -1.0]), "negative")(acts).tolist() == [[[-1, 1], [3, -1]]]
    assert zero("positive")(acts).tolist() == [[[-2, 0], [0, -4]]]


def test_shuffle_keeps_each_neurons_values():
    change = torch.arange(24.0).reshape(1, 8, 3)
    out = shuffle_within_neurons(change, torch.Generator().manual_seed(0))
    assert not out.equal(change)
    assert out.sort(1).values.equal(change.sort(1).values)


def test_split_half_separates_token_driven_hits_from_chance():
    tokens = torch.arange(20).repeat(100)
    even = torch.arange(2000) % 40 < 20
    driven = (tokens < 10).unsqueeze(1)
    chance = torch.rand(2000, 1, generator=torch.Generator().manual_seed(0)) < 0.5
    assert split_half(driven, tokens, even).item() == pytest.approx(1.0)
    assert abs(split_half(chance, tokens, even).item()) < 0.8


def test_ridge_recovers_a_linear_map():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(500, 5, generator=generator)
    y = x @ torch.randn(5, 2, generator=generator) + 3
    assert ridge(x[:400], y[:400], x[400:]).sub(y[400:]).abs().max() < 0.5


def test_change_added_at_final_norm_matches_a_last_layer_clamp(tiny_gpt2):
    # The last layer's write reaches only the final norm, so both routes give one result.
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    inputs = {"input_ids": torch.randint(VOCAB, (2, 6))}
    apply = zero("negative")
    with torch.no_grad(), added_at_final_norm(runner, 1, apply):
        direct = runner.forward(inputs)
    total = runner.run_with_interventions(inputs, {1: apply})
    assert direct.sub(total).abs().max() < 1e-4


def test_keep_first_and_null_leave_the_first_token_alone():
    acts = torch.tensor([[[-5.0, 9.0], [-1.0, 2.0], [-2.0, 3.0], [-3.0, 4.0]]])
    value = torch.tensor([-2.0, -2.0])
    assert keep_first(clamp(value, "negative"))(acts)[0, 0].tolist() == [-5, 9]
    null = shuffled_clamp(value, "negative", torch.Generator().manual_seed(0))(acts)
    assert null[0, 0].tolist() == [-5, 9]
    assert (null - acts)[0, 1:].sum().item() == pytest.approx(0.0)


def test_auroc_ranks_ties_at_their_mean():
    labels = torch.tensor([0, 0, 1, 1])
    assert auroc(torch.tensor([0.1, 0.2, 0.3, 0.4]), labels) == pytest.approx(1.0)
    assert auroc(torch.tensor([0.4, 0.3, 0.2, 0.1]), labels) == pytest.approx(0.0)
    assert auroc(torch.ones(4), labels) == pytest.approx(0.5)
    assert auroc(torch.tensor([0.1, 0.3, 0.3, 0.4]), labels) == pytest.approx(0.875)


def test_token_lookup_shrinks_to_the_mean_and_covers_unseen_tokens():
    tokens = torch.tensor([1, 1, 2, 2])
    y = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    out = token_lookup(tokens, y, torch.tensor([1, 2, 3]))
    assert out.squeeze(1).tolist() == pytest.approx([2.5 / 3, 0.5 / 3, 0.5])


def test_label_tokens_marks_overlaps_only_for_listed_labels():
    offsets = [(0, 3), (3, 7), (7, 10)]
    spans = [Span(4, 7, "A"), Span(0, 1, "B"), Span(8, 9, "IGNORED")]
    y = label_tokens(offsets, spans, ["A", "B"])
    assert y.tolist() == [[0, 1], [1, 0], [0, 0]]


def test_ambiguous_rows_need_both_outcomes_in_the_fit_rows():
    fit = TokenRows(
        None, None, torch.tensor([[1.0], [0.0], [1.0], [0.0]]), torch.tensor([5, 5, 6, 7]), None
    )
    test = TokenRows(None, None, None, torch.tensor([5, 6, 7, 8]), None)
    assert ambiguous_rows(fit, test, 0).tolist() == [True, False, False, False]


def test_resample_groups_keeps_groups_whole():
    groups = torch.tensor([0, 0, 1, 1, 1, 2])
    rows = resample_groups(groups, torch.Generator().manual_seed(0))
    picked = groups[rows].tolist()
    for name in set(picked):
        assert picked.count(name) % groups.tolist().count(name) == 0
    assert bootstrap_interval(lambda r: float(len(r) > 0), groups) == [1.0, 1.0]
