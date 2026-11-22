"""Unit wiring from the weights (openinterpretability/harness/wiring.py): edge cases."""

from dataclasses import replace

import pytest

from openinterpretability.core.exceptions import ArchitectureNotSupportedError
from openinterpretability.core.schemas import FeatureIdentifier
from openinterpretability.harness.features import scan_feature
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.units import FeatureUnit
from openinterpretability.harness.wiring import _attn_read_point, target_of, weight_wiring
from tests.conftest import FakeTokenizer
from tests.test_feature_audit import TEXTS, spec  # noqa: F401  (fixture)


def test_a_unit_of_an_unknown_kind_has_no_wiring(toy_runner):
    class Unknown:
        pass

    with pytest.raises(TypeError, match="no wiring for Unknown"):
        target_of(toy_runner, Unknown())


def test_a_model_without_attention_norms_has_no_attention_read_point(toy_runner):
    with pytest.raises(ArchitectureNotSupportedError, match="norm before attention"):
        _attn_read_point(toy_runner, 0)


def test_a_latent_on_the_mlp_output_reads_only_its_own_layers_neurons(tiny_gpt2, spec):  # noqa: F811
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    ident = FeatureIdentifier(model_name="tiny-gpt2", layer=1, dictionary="tiny-tc", feature_idx=5)
    on_output = replace(spec, kind="sae", input_site="mlp_out")
    profile = scan_feature(runner, TEXTS, ident, on_output, k=2).profile
    unit = FeatureUnit(ident, profile)
    assert target_of(runner, unit).site == "mlp_out"
    wiring = weight_wiring(runner, unit, k=5)
    assert {e.layer for e in wiring.upstream_neurons} == {1}  # the MLP it reads, only
    assert not wiring.upstream_heads  # no head writes into the MLP's output
