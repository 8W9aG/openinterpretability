"""Reproducing records made in a 16-bit dtype (bfloat16 or float16), issue #2995.

16-bit rounding moves a number by a few percent from one device to another. So a 16-bit
record is reproduced in float32, and compared within ``reproduce.HALF_TOLERANCE``
(``docs/analyses/half-precision-reproduce.md``). A pass/fail can then flip where rounding
moves a number across its threshold. These helpers check that the record's own results
follow from its own numbers, and its decision from its results.
"""

from __future__ import annotations

HALF = frozenset({"bfloat16", "float16"})
ROUNDING = 1e-6  # records round activations and deltas to 6 decimals


def reproduction_dtype(dtype: str) -> str:
    """The dtype to reproduce a record in: float32 for a 16-bit record, else its own."""
    return "float32" if dtype in HALF else dtype


def recorded_pass(result, policy, scale: float) -> bool | None:
    """What the decision rule gives for ``result``'s own recorded number, or None.

    A probe passes when its activation is at or above the threshold exactly when it is
    expected to be active. A counterfactual passes when its Δ agrees with the expected Δ.
    Other kinds (lookahead) have no single number, so they get None. Records round
    numbers to 6 decimals, so a number within that rounding of its boundary gets None too.
    """
    from openinterp.agents.review_measurements import _delta_agrees

    if not result.measured or scale <= 0:
        return None
    if result.kind in {"hard_negative", "positive_holdout"}:
        if result.activation is None:
            return False

        def rule(x):
            return (x >= policy.activation_threshold * scale) == result.expected_active

        number = result.activation
    elif result.kind == "counterfactual":
        if result.observed_delta is None or result.expected_delta is None:
            return False

        def rule(x):
            return _delta_agrees(result.expected_delta, x, policy)

        number = result.observed_delta
    else:
        return None
    low, high = rule(number - ROUNDING), rule(number + ROUNDING)
    return low if low == high else None


def rederived(verdict):
    """``verdict`` with its metrics and status computed again from its own recorded results."""
    from openinterp.agents.reviewer import decide, review_metrics

    m = verdict.metrics
    baseline = (
        (m.confound_baseline, m.confound_baseline_accuracy)
        if m.confound_baseline is not None
        else None
    )
    metrics = review_metrics(
        verdict.adversarial_test_results,
        verdict.policy,
        independent=m.independent_review,
        baseline=baseline,
        active_share=m.active_share,
        first_token_floor=m.first_token_floor,
    )
    return verdict.model_copy(
        update={
            "metrics": metrics,
            "status": decide(metrics, verdict.policy),
            "confidence_score": metrics.balanced_accuracy,
        }
    )
