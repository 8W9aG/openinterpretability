"""The helpers of scripts/weight_average_keep.py (#5404)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import weight_average_keep as wa  # noqa: E402
import weight_average_keep_stats as st  # noqa: E402


def test_scaled_multiplies_every_weight_by_alpha():
    delta = {"a": torch.tensor([1.0, -2.0]), "b": torch.tensor([[4.0]])}
    got = wa.scaled(delta, 0.25)
    assert got["a"].tolist() == [0.25, -0.5] and got["b"].tolist() == [[1.0]]
    assert wa.scaled(delta, 0.0)["a"].tolist() == [0.0, 0.0]


def test_random_like_matches_each_weights_scaled_norm_and_is_seeded():
    delta = {"a": torch.randn(5, 7), "b": torch.randn(3)}
    one, two = wa.random_like(delta, 0.5, 3), wa.random_like(delta, 0.5, 3)
    other = wa.random_like(delta, 0.5, 4)
    for name in delta:
        assert float(one[name].norm()) == pytest.approx(0.5 * float(delta[name].norm()), rel=1e-5)
        assert torch.equal(one[name], two[name])
        assert not torch.equal(one[name], other[name])
    negative = wa.random_like(delta, -1.5, 3)["a"]  # a negative step has the same size
    assert float(negative.norm()) == pytest.approx(1.5 * float(delta["a"].norm()), rel=1e-5)


def test_token_kl_is_zero_for_the_same_model_and_positive_otherwise():
    base = torch.log_softmax(torch.tensor([[[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]]]), -1)
    other = torch.log_softmax(torch.tensor([[[0.0, 2.0, -1.0], [0.0, 0.0, 0.0]]]), -1)
    assert wa.token_kl(base, base) == 0.0
    p = base[0, 0].exp()
    want = float((p * (base[0, 0] - other[0, 0])).sum()) / 2  # the second token has KL 0
    assert wa.token_kl(base, other) == pytest.approx(want, abs=1e-6)


class _Model(torch.nn.Module):
    """A language model whose logits at each position are its token id, one-hot, times 3."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(3.0))

    def forward(self, ids):
        return SimpleNamespace(logits=torch.nn.functional.one_hot(ids, 4).float() * self.scale)


def test_log_probs_are_normalised_rows_on_the_cpu_without_a_gradient():
    got = wa.log_probs(_Model(), torch.tensor([[0, 1, 2], [3, 3, 0]]))
    assert got.shape == (2, 3, 4) and got.device.type == "cpu" and not got.requires_grad
    assert torch.allclose(got.exp().sum(-1), torch.ones(2, 3))
    assert int(got[0, 1].argmax()) == 1 and int(got[1, 0].argmax()) == 3


def _rows():
    """Three poles along four steps and one random change: a flips at 0.75, b at 0.5 only."""
    rows = []
    fails = {("a", 0.75), ("a", 1.0), ("b", 0.5), ("a", -0.5)}
    for alpha in (-0.5, 0.5, 0.75, 1.0):
        flips = [s for s in "abc" if (s, alpha) in fails]
        rows.append({"table": "summary", "condition": "interpolate", "alpha": alpha,
                     "seed": None, "keeps": 3 - len(flips), "reproduced_on_base": 3,
                     "keep_share": (3 - len(flips)) / 3, "keep_ci95": [0, 1],
                     "change_norm": abs(alpha), "token_kl": abs(alpha) / 10})  # fmt: skip
        rows += [{"table": "pole", "condition": "interpolate", "alpha": alpha,
                  "seed": None, "slug": s, "base_passes": True, "keeps": s not in flips}
                 for s in "abc"]  # fmt: skip
    rows.append({"table": "summary", "condition": "random", "alpha": 1.0, "seed": 0, "keeps": 3,
                 "reproduced_on_base": 3, "keep_share": 1.0, "keep_ci95": [0, 1],
                 "change_norm": 1.0})  # fmt: skip
    rows += [{"table": "pole", "condition": "random", "alpha": 1.0, "seed": 0, "slug": s,
              "base_passes": True, "keeps": True} for s in "abc"]  # fmt: skip
    return rows


def test_steps_count_keeps_and_the_flips_shared_with_the_instruct_model():
    steps = {(r["condition"], r["alpha"]): r for r in st.steps(_rows())}
    assert steps[("interpolate", 1.0)]["flipped"] == ["a"]
    assert steps[("interpolate", -0.5)]["shared_with_instruct"] == 1
    assert steps[("interpolate", 0.5)]["shared_with_instruct"] == 0
    assert steps[("random", 1.0)]["flips"] == 0 and steps[("random", 1.0)]["token_kl"] is None


def test_paired_sets_each_step_against_its_random_change_and_against_the_instruct_model():
    rows = {(tuple(r["first"]), tuple(r["second"])): r for r in st.paired(_rows())}
    instruct_vs_random = rows[(("interpolate", 1.0, None), ("random", 1.0, 0))]
    assert (instruct_vs_random["only_first"], instruct_vs_random["only_second"]) == (1, 0)
    half = rows[(("interpolate", 0.5, None), ("interpolate", 1.0, None))]
    assert (half["only_first"], half["only_second"], half["both"]) == (1, 1, 0)
    assert len(rows) == 3  # no random change at 0.5 or 0.75 in these rows


def test_paths_follow_each_pole_that_flips_at_a_positive_step():
    paths = {r["slug"]: r for r in st.paths(_rows())}
    assert set(paths) == {"a", "b"}  # c never flips, and a's flip at -0.5 is not on the path
    assert paths["a"]["keeps"] == [True, False, False] and paths["a"]["one_run"]
    assert paths["b"]["first_fail"] == 0.5 and not paths["b"]["one_run"]
