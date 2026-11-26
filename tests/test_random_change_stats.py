"""The paired tests of scripts/random_change_stats.py, and the figure helpers they share (#5154)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import figures_random_change_keep as figs  # noqa: E402
import random_change_stats as stats  # noqa: E402


def test_mcnemar_is_the_two_sided_exact_binomial():
    assert stats.mcnemar(9, 1) == pytest.approx(22 / 1024)
    assert stats.mcnemar(1, 9) == stats.mcnemar(9, 1)
    assert stats.mcnemar(5, 5) == 1.0
    assert stats.mcnemar(0, 0) == 1.0


def test_overlap_p_is_the_hypergeometric_tail():
    # 2 flips each among 4 poles: an overlap of 2 has 1 chance in 6.
    assert stats.overlap_p(2, 4, 2, 2) == pytest.approx(1 / 6)
    assert stats.overlap_p(0, 4, 2, 2) == pytest.approx(1.0)


def _pole(slug, condition, seed, keeps, base_passes=True, pearson=0.9, ratio=1.0):
    return {"table": "pole", "model": "m", "slug": slug, "condition": condition,
            "seed": seed, "keeps": keeps, "base_passes": base_passes, "pearson": pearson,
            "ratio": ratio}  # fmt: skip


def _base(slug, specificity, sensitivity, passes=True):
    return {"table": "base", "model": "m", "slug": slug,
            "readings": {"specificity": specificity, "sensitivity": sensitivity,
                         "passes": passes}}  # fmt: skip


ROWS = [
    _pole("a", "instruct", None, False), _pole("b", "instruct", None, False),
    _pole("c", "instruct", None, True), _pole("d", "instruct", None, True),
    _pole("e", "instruct", None, False, base_passes=False),
    _pole("a", "random_all", 0, False, pearson=0.5), _pole("b", "random_all", 0, True),
    _pole("c", "random_all", 0, False, pearson=None), _pole("d", "random_all", 0, True, ratio=None),
    _pole("e", "random_all", 0, False, base_passes=False),
    _base("a", 0.8, 0.9), _base("b", 0.95, 0.65), _base("c", 0.9, 0.9), _base("d", 1.0, 1.0),
    _base("e", 0.5, 0.5, passes=False),
]  # fmt: skip


def test_comparisons_count_the_flips_of_the_poles_that_reproduce():
    (row,) = stats.comparisons(ROWS, "m")
    assert row["reproduced"] == 4 and row["instruct_flips"] == 2 and row["random_flips"] == 2
    assert row["only_instruct"] == 1 and row["only_random"] == 1 and row["both"] == 1
    assert row["expected_both"] == 1.0 and row["mcnemar_p"] == 1.0
    assert row["overlap_p"] == pytest.approx(5 / 6, abs=1e-3)


def test_activity_gives_the_medians_over_measured_poles():
    rows = {r["condition"]: r for r in stats.activity(ROWS, "m")}
    assert rows["random_all"]["median_pearson"] == 0.9  # of 0.5, 0.9 and 0.9
    assert rows["random_all"]["median_ratio"] == 1.0


def test_by_margin_orders_the_poles_nearest_a_gate_first():
    rows = {r["condition"]: r for r in stats.by_margin(ROWS, "m")}
    # Margins: a 0.0, b 0.05, c 0.1, d 0.2, so each quarter holds one pole.
    assert rows["instruct"]["quarter_sizes"] == [1, 1, 1, 1]
    assert rows["instruct"]["flips_by_quarter"] == [1, 1, 0, 0]
    assert rows["random_all"]["flips_by_quarter"] == [1, 0, 1, 0]


def test_the_figure_helpers_read_the_same_rows():
    assert figs.pole_order(ROWS, "m") == ["a", "b", "c", "d"]
    assert figs.flipped(ROWS, "m", "random_all", 0) == {"a", "c"}
    assert figs.kind("random_mlp") == "random" and figs.kind("instruct_mlp") == "instruct"
    summary = [{"table": "summary", "model": "m", "condition": "random_mlp", "seed": s}
               for s in (2, 0, 1)]  # fmt: skip
    assert [r["seed"] for r in figs.summaries(summary, "m", "random_mlp")] == [0, 1, 2]
