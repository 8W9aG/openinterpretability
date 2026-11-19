"""Until-verified draws: keep experimenting on one unit until it is VERIFIED (issue #4485).

A draw runs one experiment and stops. A unit that comes close can then wait a long time
for its next attempt, and the context of the first one is lost. So one draw in
:data:`UNTIL_VERIFIED_SHARE` runs in until-verified mode. Turn 1 is the normal draw. After
each turn, a sequence ends on a verdict in :data:`TERMINAL`, on a unit with no superseding
follow-up, or after its last turn. Otherwise the next turn runs the unit's superseding
follow-up (:func:`next_turn`): ``revise`` after NEEDS_REVISION, ``new_hypothesis`` after
REJECTED. Every turn goes on the draw's one branch and PR.

The mode is drawn from the plan's seed (:func:`decide`), so ``next --seed`` repeats it. A
multi-turn draw counts as one draw: one ledger entry, and later turns don't call the planner.
Each turn's proposer sees the earlier turns through the ``supersedes`` chain
(:func:`openinterpretability.revision.previous_attempt`).
"""

from __future__ import annotations

import random
from pathlib import Path

UNTIL_VERIFIED_SHARE = 0.25  # the chance that a draw runs until verified
# Turns of one draw, at most. #3545's survival analysis of failed attempts can set this
# from data; until then, 3 caps a draw at three times a normal one.
MAX_TURNS = 3
# Verdicts that end a sequence: success, or a scan that found the unit dead or default-on.
TERMINAL = ("VERIFIED", "STOPPED")
# Draws that study a unit with an audit; other kinds (explanation, circuit) take one turn.
UNIT_KINDS = ("explore", "opposite_pole", "revise", "new_hypothesis", "rescan", "sister",
              "superposition", "independent_review", "replicate", "resolve", "challenge",
              "reevaluate")  # fmt: skip
SUPERSEDING = ("revise", "new_hypothesis", "rescan")  # a follow-up that writes a new hypothesis
SALT = 4485  # mixes the seed, so the mode doesn't follow the planner's own first draw


def decide(seed: int, forced: bool | None = None) -> bool:
    """Whether the draw of ``seed`` runs until verified: ``forced``, else a seeded draw."""
    if forced is not None:
        return forced
    return random.Random(seed * 1_000_003 + SALT).random() < UNTIL_VERIFIED_SHARE


def keep_going(kind: str, verdict: str, turn: int, max_turns: int) -> bool:
    """Whether a sequence takes another turn after turn ``turn`` ended in ``verdict``."""
    return kind in UNIT_KINDS and verdict not in TERMINAL and turn < max_turns


def next_turn(slug: str, wiki_dir: Path):
    """The unit's superseding follow-up as a :class:`~openinterpretability.planner.Plan`, or None.

    It reads the unit's state as it stands after the last turn, so the follow-up builds on
    that turn's record.
    """
    from datetime import UTC, datetime

    from openinterpretability.planner import Plan, candidates_for, load_states

    states = load_states(wiki_dir)
    state = states.get(slug)
    if state is None:
        return None
    found = [c for c in candidates_for(state, set(states), (), datetime.now(UTC))
             if c.kind in SUPERSEDING]  # fmt: skip
    if not found:
        return None
    choice = max(found, key=lambda c: c.weight)
    return Plan(seed=0, choice=choice, explored=False, candidates=found)


RULE = (
    "Until verified: after the post step, read the verdict. If it is VERIFIED or STOPPED, "
    "or this was turn {max_turns}, land the PR. Otherwise run "
    "`uv run open-interpretability next --continue {slug}` and follow its steps on this "
    "same branch and PR, then read the verdict again."
)


def rule(slug: str | None, max_turns: int) -> str:
    """The rule an agent follows for the next turn of an until-verified draw."""
    return RULE.format(slug=slug or "<SLUG>", max_turns=max_turns)
