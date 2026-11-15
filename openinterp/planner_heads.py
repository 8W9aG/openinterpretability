"""The unit of an explore draw: a random neuron pole or a random attention head (#3248).

Before this, explore draws always sampled an MLP neuron pole. So heads were audited only
by hand, and the registry had no head records for follow-ups to start from. Now, after the
model draw, an explore draw becomes a random head audit with probability ``head_share``.

The share is configured (``[tool.openinterp] head_share``), not weighted by unit count.
Qwen2.5-0.5B has 336 heads and about 233,000 poles. By count, heads would be under 0.2%
of draws, so they would almost never come up.

The kind is drawn from a generator seeded by the audit seed, not from the planner's own.
So the planner's sequence of draws doesn't shift. A seed that drew a neuron still draws
that neuron, unless it falls in the head share.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openinterp.planner import Candidate

DEFAULT_HEAD_SHARE = 0.1  # of explore draws; [tool.openinterp] head_share overrides it
_KIND_SALT = 0x4EAD  # mixes the audit seed into the kind draw's own generator


def head_share(pyproject: Path = Path("pyproject.toml")) -> float:
    """Return ``[tool.openinterp] head_share``, or the default."""
    from openinterp.planner import _config

    return float(_config(pyproject).get("head_share", DEFAULT_HEAD_SHARE))


def explore_audit(model: str, seed: int, window: int | None, unit: str | None = None) -> str:
    """The ``open-interpretability audit`` command of a random explore draw."""
    audit = f"open-interpretability audit --model {model} --random --seed {seed}"
    if unit:
        audit += f" --unit {unit}"
    if window:
        audit += f" --window-tokens {window}"
    return audit


def is_head_draw(audit_seed: int, share: float) -> bool:
    """Whether the explore draw with ``audit_seed`` studies a head, at ``share``."""
    return random.Random(audit_seed ^ _KIND_SALT).random() < share


def head_draw(model: str, audit_seed: int, pyproject: Path, window: int | None) -> Candidate | None:
    """A random head audit of ``model``, or None when the draw stays a neuron pole."""
    from openinterp.planner import Candidate, _audit_options

    share = head_share(pyproject)
    if not is_head_draw(audit_seed, share):
        return None
    audit = explore_audit(model, audit_seed, window, unit="head")
    return Candidate(
        kind="explore",
        model=model,
        weight=0.0,
        rationale=f"uniformly random attention head (unbiased among heads); heads get "
        f"{share:.0%} of explore draws ([tool.openinterp] head_share)",
        commands=[audit],
        skills=["attention-head-audit"],  # the explore skills are for neuron poles
        audit=_audit_options([audit]),
    )


def neuron_draw(
    model: str, audit_seed: int, window: int | None, chance: float | None, why: str
) -> Candidate:
    """A random neuron-pole audit of ``model``: the default explore draw.

    ``chance`` is the model's probability in the model draw, or None with one model.
    """
    from openinterp.planner import SKILLS, Candidate, _audit_options

    rationale = "uniformly random new neuron (unbiased; counts towards registry statistics)"
    if chance is not None:
        rationale += f"; {model} drawn with probability {chance:.0%} by {why}"
    audit = explore_audit(model, audit_seed, window)
    return Candidate(
        kind="explore",
        model=model,
        weight=0.0,
        rationale=rationale,
        commands=[audit],
        skills=SKILLS["explore"],
        audit=_audit_options([audit]),
    )
