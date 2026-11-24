"""``scripts/base_instruct_patch.py``: the instruct difference in each layer's poles (#5140)."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

_SPEC = importlib.util.spec_from_file_location(
    "base_instruct_patch", Path(__file__).parents[1] / "scripts" / "base_instruct_patch.py"
)
bip = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bip)


def test_kl_is_zero_for_the_same_logits_and_positive_otherwise():
    p = torch.tensor([[2.0, 0.0, 0.0]])
    assert bip.kl_per_position(p, p).tolist() == [0.0]
    q = torch.tensor([[0.0, 0.0, 0.0]])
    expected = sum(a * math.log(a / (1 / 3)) for a in torch.softmax(p[0], 0).tolist())
    assert math.isclose(float(bip.kl_per_position(p, q)[0]), expected, rel_tol=1e-5)


def test_the_means_read_the_responses_predictions_all_and_early():
    kl = torch.arange(20.0)
    # Response tokens at 5..16: their predictions sit at 4..15. Early is the 2nd to 10th.
    assert bip.response_means(kl, 5, 17) == (9.5, 9.0)
    assert math.isnan(bip.response_means(kl, 5, 6)[1])  # a one-token response has no early part


def test_random_poles_are_seeded_distinct_and_avoid_the_excluded():
    exclude = {(0, "positive"), (1, "negative")}
    poles = bip.random_poles(4, exclude, 5, seed=3)
    assert len(poles) == len(set(poles)) == 5 and not exclude & set(poles)
    assert poles == bip.random_poles(4, exclude, 5, seed=3)
    assert poles != bip.random_poles(4, exclude, 5, seed=4)
    assert len(bip.random_poles(1, set(), 9, seed=0)) == 2  # no more than the layer has


def test_a_patch_replaces_only_the_named_sign_and_positions_the_source_reaches():
    base = torch.tensor([[[1.0, -1.0, 5.0], [2.0, -2.0, 5.0], [3.0, -3.0, 5.0]]])
    src = torch.tensor([[-3.0, 3.0, 9.0], [4.0, -4.0, 9.0]])  # two positions, not three
    out = bip.patch_poles([(0, "positive"), (1, "negative")], src)(base)
    assert out.tolist() == [[[0.0, 0.0, 5.0], [4.0, -4.0, 5.0], [3.0, -3.0, 5.0]]]
    assert base[0, 0].tolist() == [1.0, -1.0, 5.0]  # the input is left as it was
    assert torch.equal(bip.patch_poles([], src)(base), base)


def test_each_condition_has_its_poles():
    named = bip.conditions([(1, "positive")], width=3, layer=2)
    assert named["verified"] == [(1, "positive")]
    assert len(named["all"]) == 6
    draws = [k for k in named if k.startswith("random")]
    assert len(draws) == bip.DRAWS and all(named[k] != [(1, "positive")] for k in draws)
    assert named["random0"] != named["random1"] or bip.DRAWS == 1


def test_the_partner_is_the_next_conversation_at_least_as_long():
    assert bip.partner([5, 3, 9, 4], 0) == 2
    assert bip.partner([5, 3, 9, 4], 1) == 2
    assert bip.partner([5, 3, 9, 4], 2) == 0  # none as long: the longest other one
    assert bip.partner([5, 5], 0) == 1


def test_a_batched_pass_gives_each_row_its_own_patch():
    seen = {}

    def run(ids, interventions):
        ((layer, intervene),) = interventions.items()
        act = torch.zeros(ids.shape[0], ids.shape[1], 2)
        seen["layer"], seen["act"] = layer, intervene(act)
        return seen["act"][..., :1].repeat(1, 1, 3)  # logits that follow the patched value

    conv = {"ids": torch.zeros(1, 4, dtype=torch.long), "start": 2, "layer": 7,
            "target": torch.zeros(4, 3)}  # fmt: skip
    plus = lambda act: act + 1  # noqa: E731
    found = bip.batched_kl(SimpleNamespace(run_with_interventions=run), conv, [None, plus])
    assert seen["layer"] == 7 and seen["act"][:, 0, 0].tolist() == [0.0, 1.0]
    assert found[0] == (0.0, 0.0) and found[1][0] == 0.0  # uniform logits either way


def test_the_verified_poles_come_from_the_base_models_newest_records(tmp_path):
    def write(unit, status, layer, neuron, pole):
        folder = tmp_path / "experiments" / unit
        folder.mkdir(parents=True)
        ident = {"kind": "mlp_neuron_pole", "layer": layer, "neuron_idx": neuron, "pole": pole}
        (folder / "20260101T000000Z.json").write_text(
            json.dumps({"submission": {"identifier": ident}, "verdict": {"status": status}})
        )

    write(f"{bip.SLUG}/L2/N5-positive", "VERIFIED", 2, 5, "positive")
    write(f"{bip.SLUG}/L2/N1-negative", "VERIFIED", 2, 1, "negative")
    write(f"{bip.SLUG}/L3/N4-positive", "REJECTED", 3, 4, "positive")
    write("other--model/L2/N9-positive", "VERIFIED", 2, 9, "positive")
    assert bip.verified_by_layer(tmp_path) == {2: [(1, "negative"), (5, "positive")]}
