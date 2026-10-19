"""Write synthetic audit records to examples/audits/ for `make sample-run`.

These are illustrative fixtures for exercising the site builder, NOT real audit
results: the numbers are invented and no model was run.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

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
            notes="Synthetic example record (not a real audit).",
        )
        path = out / f"{ident.slug.replace('/', '_')}.json"
        path.write_text(
            AuditRecord(submission=sub, verdict=verdict).model_dump_json(indent=2) + "\n"
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
