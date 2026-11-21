"""Audit option checks: superposition files and the unit the options name."""

import json

import click
import pytest

from openinterpretability.audit_options import read_superposition, resolve_unit


def test_a_superposition_file_without_a_layer_is_a_bad_parameter(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"model": "m/x"}))
    with pytest.raises(click.BadParameter, match="layer") as caught:
        read_superposition(path, "neuron")
    assert caught.value.param_hint == "--superposition"


def test_head_needs_the_head_unit():
    with pytest.raises(click.UsageError, match="--head needs --unit head"):
        resolve_unit("feature", None, None, 3, None, None)


def test_feature_and_dictionary_need_the_feature_unit():
    with pytest.raises(click.UsageError, match="--feature and --dictionary need --unit feature"):
        resolve_unit("head", None, None, None, 7, None)


def test_options_imply_the_unit_when_none_is_given():
    assert resolve_unit(None, None, None, 2, None, None) == "head"
    assert resolve_unit(None, None, None, None, 4, "d") == "feature"
    assert resolve_unit(None, 5, "positive", None, None, None) == "neuron"
