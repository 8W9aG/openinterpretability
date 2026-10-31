"""What the reviewer measures on the model: probe and counterfactual activity, the
lookahead check, and the causal and wiring readouts. :mod:`openinterp.agents.reviewer`
turns these into metrics and a verdict."""

from __future__ import annotations

import math
from functools import partial

from openinterp.core.exceptions import ArchitectureNotSupportedError
from openinterp.core.schemas import AdversarialTestResult, ReviewPolicy
from openinterp.harness.causal import causal_metrics
from openinterp.harness.covariance import joint_ablation
from openinterp.harness.lookahead import ablation_profile
from openinterp.harness.scan import has_first_token_floor
from openinterp.harness.units import peak_strength, submission_unit
from openinterp.harness.wiring_activations import activation_wiring


def settle_first_token_floor(submission, runner, policy: ReviewPolicy):
    """(submission, unit, scale), with ``first_token_floor`` judged here if the scan
    predates protocol 0.8, which judges it at the scan."""
    unit = submission_unit(submission)
    scale = submission.max_activation or max(
        peak_strength(runner, unit, p.base_prompt) or 0.0 for p in submission.counterfactuals
    )
    if submission.first_token_floor is None:
        read = partial(unit.strengths, runner)
        floor = has_first_token_floor(runner, read, scale, policy)
        submission = submission.model_copy(update={"first_token_floor": floor})
        unit = submission_unit(submission)
    return submission, unit, scale


def measure_probes(runner, unit, probes, scale: float, threshold: float):
    """Each probe's result, plus the measured probes and whether each was active.

    A one-token prompt to a unit with a first-token floor has nothing measured: it is
    recorded as unmeasured and counts in no metric."""
    results: list[AdversarialTestResult] = []
    measured_probes, actual = [], []
    for probe in probes:
        strength = peak_strength(runner, unit, probe.text)
        active = strength is not None and scale > 0 and strength >= threshold
        if strength is not None:
            measured_probes.append(probe)
            actual.append(active)
        results.append(
            AdversarialTestResult(
                kind="positive_holdout" if probe.expected_active else "hard_negative",
                prompt=probe.text,
                passed=strength is not None and active == probe.expected_active,
                expected_active=probe.expected_active,
                activation=None if strength is None else round(strength, 6),
                detail=probe.rationale,
                measured=strength is not None,
            )
        )
    return results, measured_probes, actual


def measure_counterfactuals(runner, unit, submission, scale: float, policy: ReviewPolicy):
    results = []
    for pair in submission.counterfactuals:
        base = peak_strength(runner, unit, pair.base_prompt)
        cf = peak_strength(runner, unit, pair.counterfactual_prompt)
        both = base is not None and cf is not None
        observed = (base - cf) / scale if both and scale > 0 else 0.0
        results.append(
            AdversarialTestResult(
                kind="counterfactual",
                prompt=f"{pair.base_prompt} -> {pair.counterfactual_prompt}",
                passed=both and scale > 0 and _delta_agrees(pair.expected_delta, observed, policy),
                expected_delta=pair.expected_delta,
                observed_delta=round(observed, 6) if both else None,
                measured=both,
            )
        )
    return results


def lookahead_result(runner, unit, submission) -> AdversarialTestResult:
    """Whether ablating the unit on the base prompts agrees with the claimed mechanism."""
    prompts = [p.base_prompt for p in submission.counterfactuals]
    measured = ablation_profile(runner, prompts, unit.ablation(runner))
    claimed = submission.lookahead.is_lookahead
    return AdversarialTestResult(
        kind="lookahead_consistency",
        prompt="; ".join(p.base_prompt for p in submission.counterfactuals),
        passed=measured.is_lookahead == claimed,
        detail=(
            f"claimed {'lookahead' if claimed else 'retrospective'}; measured "
            f"{'lookahead' if measured.is_lookahead else 'retrospective'} "
            f"(KL={measured.kl_divergence:.4g}, tokens={measured.target_upcoming_tokens[:5]})"
        ),
    )


def causal_readouts(submission, runner, probes, policy: ReviewPolicy, scale: float):
    """(causal metrics, joint ablation with the unit's correlated partners)."""
    causal = causal_metrics(
        submission,
        runner,
        freeze_downstream=policy.freeze_downstream,
        null_neurons=policy.causal_null_neurons,
        min_change=policy.delta_abs_tolerance * scale,
    )
    return causal, _joint_ablation(submission, runner, probes, policy)


def _joint_ablation(submission, runner, probes, policy):
    # Ranked by co-activation lift (the scan's order); Spearman is the fallback signal.
    partners: list[int] = []
    for c in submission.correlated_neurons:
        lifted = c.coactivation_lift_lower >= policy.min_partner_lift and (
            c.coactive_tokens >= policy.min_partner_coactive
        )
        if (lifted or abs(c.spearman) >= policy.min_partner_correlation) and (
            c.neuron_idx not in partners
        ):
            partners.append(c.neuron_idx)
    partners = partners[: policy.joint_ablation_partners]
    if not partners or submission.identifier.kind != "mlp_neuron_pole":
        return None
    ident = submission.identifier
    prompts = [p.base_prompt for p in submission.counterfactuals]
    prompts += [p.text for p in probes if p.expected_active]
    return joint_ablation(runner, prompts, ident.layer, ident.neuron_idx, ident.pole, partners)


def _activation_wiring(submission, runner, probes, threshold):
    """Wiring by attribution at the unit's peak on the prompts it should fire on (the
    counterfactuals' base prompts and the holdouts), counting those where it did."""
    if threshold <= 0:
        return None
    prompts = [p.base_prompt for p in submission.counterfactuals]
    prompts += [p.text for p in probes if p.expected_active]
    try:
        return activation_wiring(runner, submission_unit(submission), prompts, threshold)
    except ArchitectureNotSupportedError:
        return None


def _delta_agrees(expected: float, observed: float, policy: ReviewPolicy) -> bool:
    """Observed Δ must land within max(abs_tol, rel_tol·|expected|) of the prediction.

    Non-control pairs (|expected| > abs_tol) must also have the predicted sign.
    """
    tolerance = max(policy.delta_abs_tolerance, policy.delta_rel_tolerance * abs(expected))
    if abs(observed - expected) > tolerance:
        return False
    if abs(expected) <= policy.delta_abs_tolerance:
        return True
    return math.copysign(1, observed) == math.copysign(1, expected)
