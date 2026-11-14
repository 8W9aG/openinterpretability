"""Neuronpedia's explanations through the adversarial review.

The command is `open-interpretability neuronpedia-review`.
"""

from openinterp.cli_neuronpedia import baseline_submission
from openinterp.crossref import Explanation
from openinterp.neuronpedia_review import NOT_PROBE_GATES, ReviewRow, probe_gates, report

THEIRS = Explanation(description="references to the number 39", explainer="gpt-4o-mini")


def test_the_copy_claims_only_neuronpedias_explanation(record):
    lookahead = record.submission.lookahead.model_copy(
        update={"is_lookahead": True, "target_upcoming_tokens": [" the"]}
    )
    sub = record.submission.model_copy(update={"lookahead": lookahead, "prior_art": []})
    copy = baseline_submission(record.model_copy(update={"submission": sub}), THEIRS)
    assert copy.natural_language_explanation == THEIRS.description
    assert copy.activating_tokens == []  # the agent's claimed tokens would leak its hypothesis
    assert copy.lookahead.is_lookahead is False and copy.lookahead.target_upcoming_tokens == []
    assert copy.lookahead.kl_divergence == lookahead.kl_divergence  # measured, so kept
    assert copy.submitting_agent_model == "neuronpedia/gpt-4o-mini"
    assert copy.prior_art is None
    assert copy.top_examples == record.submission.top_examples  # the same scan evidence


def test_only_the_gates_that_probes_decide_count():
    gates = {"specificity": True, "sensitivity": False, **dict.fromkeys(NOT_PROBE_GATES, False)}
    assert probe_gates(gates) == {"specificity": True, "sensitivity": False}


def _side(acc, ok):
    gates = {"specificity": ok, "sensitivity": True}
    return {"balanced_accuracy": acc, "specificity": acc, "sensitivity": 1.0, "gates": gates,
            "probe_verified": ok}  # fmt: skip


def test_report_counts_latents_that_pass_every_probe_gate():
    row = dict(experiment_id="e", url="https://np/1", agent_status="VERIFIED",
               agent_explanation="a", neuronpedia_explanation="n | x", explainer="m")  # fmt: skip
    rows = [
        ReviewRow(slug="u/F1", agent=_side(0.9, True), neuronpedia=_side(0.6, False), **row),
        ReviewRow(slug="u/F2", agent=_side(0.8, True), neuronpedia=_side(0.85, True), **row),
        ReviewRow(slug="u/F3", agent=_side(0.8, True), **row),  # not reviewed yet
    ]
    md = report(rows)
    assert "Reviewed: 2 of 3 latents." in md and "Two things favour the agent" in md
    assert "| Passes every probe gate | 2 of 2 | 1 of 2 |" in md
    assert "| Balanced accuracy (mean, n = 2) | 0.85 | 0.72 (sign test p = 1) |" in md
    assert "failed: specificity ×1." in md
    assert "n \\| x" in md and "not reviewed yet" in md
