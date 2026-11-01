"""Copies that have to exist agree with their source.

Each concept is defined once in code. Where a second copy can't import it (docs quote
numbers, a template or workflow retypes a list), a test here pins the copy to the
source. See .claude/skills/single-source-of-truth/.
"""

import importlib.util
import re
from pathlib import Path
from typing import get_args

import pytest

from openinterp.agents.reviewer import ProbeSet
from openinterp.builder.statuses import MAP_STATUSES, STATUS_BADGE, STATUS_FILL
from openinterp.core.schemas import (
    ACTIVATION_THRESHOLD,
    PROTOCOL_VERSION,
    ReviewPolicy,
    VerdictStatus,
)
from openinterp.corpus import DEFAULT_SAMPLE_TOKENS
from openinterp.harness.layer_cache import DEFAULT_GB
from openinterp.planner import BASE_WEIGHTS, DEFAULT_EPSILON, configured_windows
from openinterp.reproduce import Tolerance

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (ROOT / "docs/PROTOCOL.md").read_text()
POLICY = ReviewPolicy()

_SPEC = importlib.util.spec_from_file_location(
    "check_line_limit", ROOT / "scripts" / "check_line_limit.py"
)
check_line_limit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_line_limit)


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


def _probe_bounds() -> tuple[int, int]:
    limits = {type(m).__name__: m for m in ProbeSet.model_fields["probes"].metadata}
    return limits["MinLen"].min_length, limits["MaxLen"].max_length


def _quotes() -> list[tuple[str, str]]:
    """(file, text) pairs: numbers the README, docs, skills and agents quote from code."""
    lo, hi = _probe_bounds()
    gpt2 = configured_windows(ROOT / "pyproject.toml")["openai-community/gpt2"]
    skills = ".claude/skills"
    return [
        ("README.md", f"capped at {DEFAULT_GB:g} GB"),
        ("README.md", f"up to {DEFAULT_SAMPLE_TOKENS:,} tokens"),
        ("README.md", f"the {check_line_limit.MAX_LINES}-line limit"),
        ("README.md", f"a {check_line_limit.MAX_LINES}-line limit"),
        ("README.md", f"writes {lo}–{hi} adversarial probes"),
        ("README.md", f"fit only the first {gpt2} positions"),
        ("README.md", f"{gpt2}-token windows (`[tool.openinterp.window_tokens]`)"),
        ("docs/PROTOCOL.md", f"writes {lo}–{hi} probes"),
        (".claude/agents/neuron-reviewer.md", f"writes {lo}–{hi} probes"),
        (".claude/agents/neuron-reviewer.md", f"the prompt asks for ({lo}–{hi})"),
        (f"{skills}/choose-next-experiment/SKILL.md", f"ε (default {DEFAULT_EPSILON:g})"),
        (
            f"{skills}/explanation-simulation-scoring/SKILL.md",
            f"`{ACTIVATION_THRESHOLD:g} × max_activation`",
        ),
        (f"{skills}/native-neuron-audit/SKILL.md", f"scanned in {gpt2}-token windows"),
    ]


@pytest.mark.parametrize(("path", "quote"), _quotes())
def test_docs_quote_the_code(path, quote):
    assert quote in (ROOT / path).read_text(), f"{path} no longer quotes: {quote}"


def test_next_experiment_skill_quotes_the_base_weights():
    """choose-next-experiment lists BASE_WEIGHTS as `kind` w > `kind` = `kind` w > …"""
    skill = (ROOT / ".claude/skills/choose-next-experiment/SKILL.md").read_text()
    line = next(x for x in skill.splitlines() if x.startswith("- **Base weights**"))
    quoted = {}
    for group in line.split(":", 1)[1].split(" > "):
        weight = float(re.findall(r"\d+\.\d+", group)[-1])
        quoted |= dict.fromkeys(re.findall(r"`(\w+)`", group), weight)
    assert quoted == BASE_WEIGHTS
