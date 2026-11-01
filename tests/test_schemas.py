import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from openinterp.core.schemas import (
    ActivatingExample,
    AuditRecord,
    CausalMetrics,
    CounterfactualPair,
    HeadIdentifier,
    Histogram,
    NeuronAuditSubmission,
    NeuronIdentifier,
    OppositePole,
    ScanEvidence,
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


def _opposite(pole):
    example = ActivatingExample(
        text="a", tokens=["a"], activations=[1.0], max_activation=1.0, max_token_index=0
    )
    return OppositePole(pole=pole, top_examples=[example])


def test_opposite_pole_is_the_other_pole_of_a_neuron(submission):
    other = submission.model_copy(update={"opposite_pole": _opposite("negative")})
    assert NeuronAuditSubmission.model_validate(other.model_dump()).opposite_pole.pole == "negative"
    with pytest.raises(ValidationError, match="does not name"):
        submission.model_validate(
            submission.model_dump() | {"opposite_pole": _opposite("positive")}
        )
    head = HeadIdentifier(model_name="gpt2", layer=5, head_idx=1)
    with pytest.raises(ValidationError, match="only for neuron poles"):
        ScanEvidence(identifier=head, top_examples=[], opposite_pole=_opposite("negative"))


def test_histogram_share_counts_bins_wholly_above_the_line():
    from openinterp.core.examples import Histogram

    h = Histogram(bin_edges=[0.0, 1.0, 2.0, 3.0, 4.0], counts=[5, 1, 2, 2])
    assert h.share_at_least(2.0) == pytest.approx(0.4)
    assert h.share_at_least(1.5) == pytest.approx(0.4)  # the straddling bin is left out
    assert h.share_at_least(0.0) == 1.0
    assert Histogram(bin_edges=[0.0, 1.0], counts=[0]).share_at_least(0.5) == 0.0


@pytest.mark.parametrize("handle", ["someone@example.com", "-dash", "has space", "x" * 40])
def test_contributor_is_a_public_handle(submission, handle):
    def with_contributor(name):
        data = submission.model_dump(mode="json") | {"contributor": name}
        return NeuronAuditSubmission.model_validate_json(json.dumps(data))

    assert with_contributor("8W9aG").contributor == "8W9aG"
    with pytest.raises(ValidationError):
        with_contributor(handle)
