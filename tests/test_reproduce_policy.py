"""Which records' dictionaries must be on the allow-list (reproduce_policy.py)."""

from openinterpretability.core.identifiers import FeatureIdentifier
from openinterpretability.reproduce_policy import preset_dictionary


def test_a_record_with_no_dictionary_profile_has_no_dictionary_to_check(record):
    assert record.submission.identifier.kind == "mlp_neuron_pole"
    assert preset_dictionary(record)
    feature = FeatureIdentifier(
        model_name="openai-community/gpt2", layer=8, dictionary="not-a-preset", feature_idx=5
    )
    sub = record.submission.model_copy(update={"identifier": feature, "dictionary_profile": None})
    assert preset_dictionary(record.model_copy(update={"submission": sub}))
