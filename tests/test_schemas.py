from datetime import datetime

import pytest
from pydantic import ValidationError

from openinterp.core.schemas import (
    ActivatingExample,
    AuditRecord,
    CausalMetrics,
    CounterfactualPair,
    Histogram,
    NeuronIdentifier,
    VerificationVerdict,
)


def test_identifier_slug_is_url_safe(identifier):
    assert identifier.slug == "Qwen--Qwen2.5-0.5B/L3/N117-positive"
    assert str(identifier) == "Qwen/Qwen2.5-0.5B L3/N117+"


@pytest.mark.parametrize(
    "overrides",
    [
        {"pole": "both"},
        {"layer": -1},
        {"neuron_idx": "5"},  # strict: no str -> int coercion
        {"layer": 2.0},  # strict: no float -> int coercion
        {"model_name": ""},
    ],
)
def test_identifier_rejects_invalid(overrides):
    data = {"model_name": "m", "layer": 0, "neuron_idx": 0, "pole": "positive"} | overrides
    with pytest.raises(ValidationError):
        NeuronIdentifier(**data)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        NeuronIdentifier(model_name="m", layer=0, neuron_idx=0, pole="positive", typo=1)


def test_counterfactual_prompts_must_differ():
    with pytest.raises(ValidationError, match="must differ"):
        CounterfactualPair(base_prompt="a", counterfactual_prompt="a", expected_delta=0.5)


def test_scores_bounded():
    with pytest.raises(ValidationError):
        CausalMetrics(sufficiency_score=1.2, necessity_score=0.1, hydra_mitigated=False)


def test_verdict_requires_aware_timestamp(verdict):
    data = verdict.model_dump() | {"timestamp": datetime(2026, 1, 1)}
    with pytest.raises(ValidationError):
        VerificationVerdict(**data)


def test_verdict_status_literal(verdict):
    with pytest.raises(ValidationError):
        VerificationVerdict(**(verdict.model_dump() | {"status": "MAYBE"}))


def test_histogram_shape():
    with pytest.raises(ValidationError):
        Histogram(bin_edges=[0.0, 1.0], counts=[1, 2])


def test_example_alignment():
    with pytest.raises(ValidationError):
        ActivatingExample(
            text="a b", tokens=["a", "b"], activations=[1.0], max_activation=1.0, max_token_index=0
        )


def test_record_json_round_trip(record):
    restored = AuditRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.is_verified


def test_unreviewed_record_is_not_verified(submission):
    assert not AuditRecord(submission=submission).is_verified


def test_models_are_frozen(identifier):
    with pytest.raises(ValidationError):
        identifier.layer = 4
