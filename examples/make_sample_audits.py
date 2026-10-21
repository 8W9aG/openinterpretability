"""Write synthetic audit records to examples/audits/ for `make sample-run`.

These are illustrative fixtures for exercising the site builder, NOT real audit
results: the numbers are invented and no model was run.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

from openinterp.core.explanation import ExplanationScores, ScoredWindow
from openinterp.core.schemas import (
    ActivatingExample,
    AdversarialTestResult,
    AuditRecord,
    CausalMetrics,
    CounterfactualPair,
    Histogram,
    LookaheadProfile,
    NeuronAuditSubmission,
    NeuronIdentifier,
    VerificationVerdict,
)
from openinterp.harness.explanation_scores import finalize_scores

MODEL = "Qwen/Qwen2.5-0.5B"
FIXTURE = "example-fixture"
SPECS = [
    # layer, neuron, pole, explanation, tokens, lookahead targets, status, confidence
    (
        4,
        1203,
        "positive",
        "Closing quotation mark that ends a span of reported speech.",
        ['"', "”"],
        [],
        "VERIFIED",
        0.91,
    ),
    (
        11,
        377,
        "negative",
        "Prepares a year token after 'in' when the context is historical.",
        [" in"],
        [" 19", " 18"],
        "VERIFIED",
        0.83,
    ),
    (
        17,
        4410,
        "positive",
        "Python 'return' keyword inside a function body.",
        [" return"],
        [],
        "VERIFIED",
        0.78,
    ),
    (20, 88, "positive", "Generic sentence-final punctuation.", ["."], [], "REJECTED", 0.31),
]
EXAMPLE_TEXT = [
    ('"Come in," she said.', 4),
    ("In 1969, Apollo 11 landed.", 1),
    ("def f(x):\n    return x", 5),
]


def example(rng: random.Random, text: str, peak: int) -> ActivatingExample:
    tokens = text.replace("\n", " \n").split(" ")
    tokens = [t if i == 0 else " " + t for i, t in enumerate(tokens)]
    peak = min(peak, len(tokens) - 1)
    acts = [round(rng.uniform(0, 0.4), 3) for _ in tokens]
    acts[peak] = round(rng.uniform(2.0, 3.0), 3)
    return ActivatingExample(
        text=text, tokens=tokens, activations=acts, max_activation=acts[peak], max_token_index=peak
    )


def sample_scores(rng: random.Random, sub: NeuronAuditSubmission) -> ExplanationScores:
    """Invented held-out windows and judge answers, so pages show the scores section."""
    scale = sub.max_activation or 1.0
    windows = []
    for n in range(20):
        ex = sub.top_examples[n % len(sub.top_examples)]
        active = n % 2 == 0
        acts = [a if active else round(a * 0.1, 3) for a in ex.activations]
        windows.append(
            ScoredWindow(
                text=ex.text,
                tokens=ex.tokens,
                activations=acts,
                quantile=n % 5 if active else None,
                label=active,
                judged=active if rng.random() < 0.75 else not active,
                similarity=round(rng.uniform(0.0, 0.4) + 0.2 * active, 3),
            )
        )
    return finalize_scores(
        ExplanationScores(
            slug=sub.identifier.slug,
            explanation=sub.natural_language_explanation,
            threshold=0.25 * scale,
            window_tokens=32,
            quantiles=5,
            quantile_edges=[round(scale * q, 3) for q in (0.3, 0.4, 0.5, 0.7)],
            seed=0,
            corpus="synthetic",
            candidate_windows=4000,
            active_windows=rng.randint(40, 200),
            judge_model=FIXTURE,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            detection_windows=windows,
        )
    )


def main() -> None:
    rng = random.Random(0)
    out = Path(__file__).parent / "audits"
    out.mkdir(exist_ok=True)
    for n, (layer, idx, pole, expl, toks, targets, status, conf) in enumerate(SPECS):
        ident = NeuronIdentifier(model_name=MODEL, layer=layer, neuron_idx=idx, pole=pole)
        text, peak = EXAMPLE_TEXT[n % len(EXAMPLE_TEXT)]
        examples = [example(rng, text, peak) for _ in range(3)]
        counts = sorted((int(rng.expovariate(1 / 400)) for _ in range(24)), reverse=True)
        sub = NeuronAuditSubmission(
            identifier=ident,
            natural_language_explanation=expl,
            activating_tokens=toks,
            counterfactuals=[
                CounterfactualPair(
                    base_prompt=text,
                    counterfactual_prompt=text.upper(),
                    expected_delta=0.7,
                    observed_delta=round(rng.uniform(0.5, 0.9), 3),
                ),
                CounterfactualPair(
                    base_prompt=text,
                    counterfactual_prompt=text + " Indeed.",
                    expected_delta=0.0,
                    observed_delta=round(rng.uniform(-0.1, 0.1), 3),
                ),
            ],
            lookahead=LookaheadProfile(
                is_lookahead=bool(targets),
                target_upcoming_tokens=targets,
                kl_divergence=round(rng.uniform(0.001, 0.05), 4),
            ),
            submitting_agent_model=FIXTURE,
            max_activation=max(e.max_activation for e in examples),
            top_examples=examples,
            activation_histogram=Histogram(
                bin_edges=[round(i * 0.125, 3) for i in range(25)], counts=counts
            ),
            model_num_layers=24,
        )
        tests = [
            AdversarialTestResult(
                kind="hard_negative",
                prompt=f"Synthetic {chr(97 + i)}{i} probe{i}",
                passed=rng.random() < conf,
                expected_active=False,
                activation=round(rng.uniform(0, 0.5), 3),
            )
            for i in range(12)
        ]
        verdict = VerificationVerdict(
            status=status,
            confidence_score=conf,
            adversarial_test_results=tests,
            reviewer_agent_model=FIXTURE,
            timestamp=datetime(2026, 10, 1, 12, n, tzinfo=UTC),
            causal_metrics=CausalMetrics(
                sufficiency_score=round(rng.uniform(0.3, 0.9), 3),
                necessity_score=round(rng.uniform(0.2, 0.8), 3),
                hydra_mitigated=True,
            ),
            explanation_scores=sample_scores(rng, sub),
            notes="Synthetic example record (not a real audit).",
        )
        path = out / f"{ident.slug.replace('/', '_')}.json"
        path.write_text(
            AuditRecord(submission=sub, verdict=verdict).model_dump_json(indent=2) + "\n"
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
