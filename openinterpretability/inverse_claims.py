"""Inverse claims: when a default-on neuron pole goes quiet (issue #4951).

A scan stops on a default-on unit, one active on at least half its scan's tokens
(``evidence.check_default_on``, protocol 0.8). It can't be a detector claim: no hard
negative of a few words stays inactive. The maintainer decided on 2026-11-18 to study such
poles with an *inverse* claim. It says when the pole goes quiet, or drops below its usual
level. It is reviewed with the detector's gates, with the sides swapped:

- **Holdouts** are texts that the claim says silence the pole (``expected_active`` false).
  Sensitivity is the share of them that stay below the activity line.
- **Hard negatives** are near-misses that the claim says keep it on (``expected_active``
  true). Specificity is the share of them that stay at or above the line.
- **The default-on gate is swapped:** ``is_default_on`` replaces ``not_default_on``. The
  claim applies only to a default-on pole.

Every other gate is unchanged. Inverse claims have their own rule version,
:data:`INVERSE_RULE_VERSION`, so ``PROTOCOL_VERSION`` is untouched. They are for neuron
poles only. A VERIFIED inverse record explains its unit for coverage, as any VERIFIED
record does.
"""

from __future__ import annotations

INVERSE_RULE_VERSION = "1.0"
INVERSE_CAP = 5  # stopped units studied per explained prompt, by coverage gain (maintainer)


def probe_kind(expected_active: bool, inverse: bool) -> str:
    """A probe's kind: what the claim predicts for it, read on the claim's own side."""
    holdout = expected_active != inverse  # an inverse claim's positive case is "quiet"
    return "positive_holdout" if holdout else "hard_negative"


def gates(policy, negatives, holdouts, reproduction, lookahead, independent, beats,
          active_share, inverse: bool = False) -> dict[str, bool]:  # fmt: skip
    """The decision rule's gates (docs/PROTOCOL.md §4).

    An inverse claim swaps the default-on gate: ``is_default_on`` for ``not_default_on``.
    """
    out = {
        "enough_hard_negatives": negatives.clusters >= policy.min_hard_negatives,
        "enough_holdouts": holdouts.clusters >= policy.min_holdouts,
        "specificity": (negatives.rate or 0.0) >= policy.min_specificity,
        "sensitivity": (holdouts.rate or 0.0) >= policy.min_sensitivity,
        "counterfactual_reproduction": (reproduction or 0.0)
        >= policy.min_counterfactual_reproduction,
        "lookahead_consistent": lookahead,
        "independent_review": independent or not policy.require_independent_review,
        "beats_confound_baseline": beats,
    }
    if inverse:
        out["is_default_on"] = active_share is not None and active_share >= policy.max_active_share
    else:
        out["not_default_on"] = active_share is None or active_share < policy.max_active_share
    return out


def check_scan(ident, values, scale: float, provenance, inverse: bool) -> None:
    """Stop a detector scan on a default-on unit, and an inverse scan on any other unit."""
    import click

    from openinterpretability.evidence import Unauditable, check_default_on

    if not inverse:
        check_default_on(ident, values, scale, provenance)
        return
    if ident.kind != "mlp_neuron_pole":
        raise click.UsageError("--inverse is for neuron poles only")
    try:
        check_default_on(ident, values, scale, provenance)
    except Unauditable:
        return  # default-on: an inverse claim applies
    raise click.UsageError(
        f"{ident} is not default-on, so audit it as a detector, without --inverse"
    )


INVERSE_PROPOSAL = """\
THIS IS AN INVERSE CLAIM (docs/PROTOCOL.md, "Inverse claims"). The pole is default-on: it is \
active on most tokens, so it can't be a detector. Explain when it goes QUIET, below 0.25 x its \
maximum. The top-activating snippets below show where it is on; measure where it is off. In \
your answer, `explanation` says what silences it. `activating_tokens` lists the tokens where it \
goes quiet. Counterfactual pairs change a text so that the pole goes quiet or comes back on, \
with the measured delta.

"""

INVERSE_REVIEW = """\
THIS IS AN INVERSE CLAIM: the hypothesis says when a default-on pole goes QUIET. The sides are \
swapped. A positive holdout is a text that the hypothesis says SILENCES the pole \
(expected_active false). A hard negative is a near-miss text that the hypothesis says KEEPS IT \
ON (expected_active true), for example one that looks like a silencing text but isn't.

"""


def inverse_preamble(evidence) -> str:
    """The proposer prompt's opening for an inverse claim, else nothing."""
    return INVERSE_PROPOSAL if getattr(evidence, "claim", "detector") == "inverse" else ""


def count_line(submission, n: int, negatives: int, holdouts: int) -> str:
    """The reviewer's count line: probes by kind, read on the claim's own side."""
    if submission.claim != "inverse":
        return f"Write {n} probes: {negatives} hard negatives and {holdouts} positive holdouts."
    return INVERSE_REVIEW + review_counts(n, negatives, holdouts)


def review_counts(n: int, negatives: int, holdouts: int) -> str:
    """The reviewer's count line for an inverse claim."""
    return (f"Write {n} probes: {negatives} hard negatives that keep the pole on "
            f"(expected_active true) and {holdouts} positive holdouts that silence it "
            "(expected_active false).")  # fmt: skip
