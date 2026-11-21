"""The unit that an identifier names (harness/units.py)."""

import pytest

from openinterpretability.core.profiles import SuperpositionMember, superposition_id
from openinterpretability.core.schemas import (
    FeatureIdentifier,
    HeadIdentifier,
    SuperpositionIdentifier,
)
from openinterpretability.harness.units import unit_of

MEMBERS = [SuperpositionMember(neuron_idx=3, weight=1.0)]


@pytest.mark.parametrize(
    ("ident", "profile"),
    [
        (SuperpositionIdentifier(model_name="tiny", layer=1,
                                 superposition_id=superposition_id(MEMBERS)), "superposition"),
        (HeadIdentifier(model_name="tiny", layer=1, head_idx=0), "head"),
        (FeatureIdentifier(model_name="tiny", layer=1, dictionary="tc", feature_idx=5),
         "dictionary"),
    ],
)  # fmt: skip
def test_a_unit_without_its_profile_has_no_defined_strength(ident, profile):
    with pytest.raises(ValueError, match=f"has no {profile} profile, so its strength is undefined"):
        unit_of(ident)
