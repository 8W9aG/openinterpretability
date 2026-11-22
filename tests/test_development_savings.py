"""Explanations that fail after step 16,000, and whether they come back (#5119)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import figures_development_savings as savings  # noqa: E402


def _rows(pole, rates):
    """``pole`` rows at steps 16k on, each with (specificity, sensitivity) from ``rates``."""
    steps = (8000, 16000, 32000, 64000, 128000, 143000)
    return [{"table": "pole", "slug": f"M/L1/N{pole}-positive", "step": st,
             "specificity": sp, "sensitivity": se}
            for st, (sp, se) in zip(steps, rates, strict=True)]  # fmt: skip


HOLD, NEAR, DEEP = (0.9, 0.9), (0.75, 0.9), (0.05, 1.0)


def test_shortfall_reads_both_gates_in_probes():
    assert savings.shortfall(0.8, 0.6) == 0
    assert savings.shortfall(0.75, 0.6) == 1
    assert savings.shortfall(0.8, 0.5) == 1
    assert savings.shortfall(0.6, 0.3) == 4  # the larger of 4 and 3


def test_savings_keeps_poles_that_hold_at_16k_and_fail_later():
    rows = [
        *_rows(1, [NEAR, HOLD, NEAR, HOLD, HOLD, HOLD]),  # back at 64k
        *_rows(2, [HOLD, HOLD, DEEP, DEEP, NEAR, HOLD]),  # back only at the final step
        *_rows(3, [HOLD, HOLD, HOLD, HOLD, HOLD, HOLD]),  # never fails
        *_rows(4, [HOLD, NEAR, HOLD, HOLD, HOLD, HOLD]),  # fails at 16k itself
        {"table": "loss", "step": 0, "loss": 1.0},
    ]
    found = savings.savings(rows)
    assert [(p["slug"], p["back"]) for p in found] == [("M/L1/N1-positive", 64000),
                                                       ("M/L1/N2-positive", None)]  # fmt: skip
    assert found[1]["steps"] == [(16000, 0), (32000, 15), (64000, 15), (128000, 1), (143000, 0)]


def test_the_figure_names_each_pole_and_counts_the_near_failures():
    rows = [*_rows(1, [HOLD, HOLD, NEAR, HOLD, HOLD, HOLD]),
            *_rows(2, [HOLD, HOLD, DEEP, NEAR, HOLD, HOLD])]  # fmt: skip
    svg = savings.figure(rows).render()
    assert "L1/N1+" in svg and "L1/N2+" in svg
    assert "2 of the 3 failures are within two probes" in svg
    assert savings._label("M/L3/N7-negative") == "L3/N7−"
