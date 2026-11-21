"""The in-memory dictionary of a superposition (harness/superpositions.py)."""

from types import SimpleNamespace

import pytest

from openinterpretability.core.profiles import (
    SuperpositionMember,
    SuperpositionProfile,
    superposition_id,
)
from openinterpretability.core.schemas import SuperpositionIdentifier
from openinterpretability.harness.superpositions import build_dictionary


def test_a_member_past_the_layers_width_is_an_error():
    members = [SuperpositionMember(neuron_idx=n, weight=w) for n, w in ((3, 2.0), (20, 0.5))]
    profile = SuperpositionProfile(members=members, method="circuit", source="test")
    ident = SuperpositionIdentifier(
        model_name="tiny", layer=1, superposition_id=superposition_id(profile.members)
    )
    runner = SimpleNamespace(mlp_width=lambda layer: 16)
    with pytest.raises(ValueError, match="layer 1 has 16 neurons"):
        build_dictionary(runner, ident, profile)
