"""Copies that have to exist agree with their source.

Each concept is defined once in code. Where a second copy can't import it (docs quote
numbers, a template or workflow retypes a list), a test here pins the copy to the
source. See .claude/skills/single-source-of-truth/.
"""

import re
from pathlib import Path
from typing import get_args

import pytest

from openinterp.builder.statuses import MAP_STATUSES, STATUS_BADGE, STATUS_FILL
from openinterp.core.schemas import PROTOCOL_VERSION, ReviewPolicy, VerdictStatus
from openinterp.reproduce import Tolerance

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (ROOT / "docs/PROTOCOL.md").read_text()
POLICY = ReviewPolicy()


def test_site_statuses_are_the_verdicts_plus_contested():
    assert set(MAP_STATUSES) == set(get_args(VerdictStatus)) | {"CONTESTED"}
    assert tuple(STATUS_FILL) == tuple(STATUS_BADGE) == MAP_STATUSES


def test_protocol_title_names_the_current_version():
    assert PROTOCOL.splitlines()[0] == f"# Audit protocol (v{PROTOCOL_VERSION})"


def _gate_default(gate: str) -> str:
    row = re.search(rf"^\| `{gate}` \| ([^|]+) \|", PROTOCOL, re.M)
    assert row, f"no row for {gate} in the PROTOCOL.md §4 gate table"
    return row.group(1).strip()


@pytest.mark.parametrize(
    ("gate", "quoted"),
    [
        ("enough_hard_negatives", f"≥ {POLICY.min_hard_negatives} clusters"),
        ("enough_holdouts", f"≥ {POLICY.min_holdouts} clusters"),
        ("specificity", f"≥ {POLICY.min_specificity:.2f}"),
        ("sensitivity", f"≥ {POLICY.min_sensitivity:.2f}"),
        ("counterfactual_reproduction", f"≥ {POLICY.min_counterfactual_reproduction:.2f} of pairs"),
        ("not_default_on", f"active share < {POLICY.max_active_share:.2f}"),
    ],
)
def test_protocol_gate_table_quotes_the_policy_defaults(gate, quoted):
    assert _gate_default(gate) == quoted


def test_protocol_prose_quotes_the_policy_defaults():
    quotes = [
        f"`activation_threshold × scale` (default {POLICY.activation_threshold:g})",
        f"max({POLICY.delta_abs_tolerance:.2f}, {POLICY.delta_rel_tolerance:.2f} × |predicted|)",
        f"balanced accuracy is below **{POLICY.reject_below_balanced_accuracy:.2f}**",
        f"up to {POLICY.joint_ablation_partners} partners (lift lower bound "
        f"≥ {POLICY.min_partner_lift:g} on ≥ {POLICY.min_partner_coactive} co-active tokens, "
        f"or |ρ| ≥ {POLICY.min_partner_correlation:g})",
        f"(defaults {Tolerance.atol:g} and {Tolerance.rtol:g})",
    ]
    for quote in quotes:
        assert quote in PROTOCOL, f"PROTOCOL.md no longer quotes: {quote}"


def test_registry_paths_come_from_core_paths():
    """The wiki directory's default and layout are written in core/paths.py only."""
    retyped = re.compile(r'Path\("wiki"\)|default="wiki"|/ "experiments"|/ "draws"')
    hits = [
        f"{path.relative_to(ROOT)}:{n}"
        for path in sorted((ROOT / "openinterp").rglob("*.py"))
        if path.name != "paths.py"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if retyped.search(line)
    ]
    assert not hits, f"use openinterp.core.paths instead: {hits}"
