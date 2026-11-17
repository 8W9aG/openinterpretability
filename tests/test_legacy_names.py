"""The old project name: read only as a fallback, and kept from coming back (issue #4089)."""

import importlib.util
from pathlib import Path

import pytest

from openinterpretability.core import legacy_names as legacy


def test_an_old_variable_is_adopted_when_the_new_one_is_unset():
    env = {"OPENINTERP_SCAN_CACHE": "/old", "OPENINTERP_PAUSE_FILE": "/p",
           "OPENINTERPRETABILITY_PAUSE_FILE": "/new"}  # fmt: skip
    with pytest.warns(DeprecationWarning, match="OPENINTERPRETABILITY_SCAN_CACHE"):
        adopted = legacy.adopt_legacy_env(env)
    assert adopted == ["OPENINTERPRETABILITY_SCAN_CACHE"]
    assert env["OPENINTERPRETABILITY_SCAN_CACHE"] == "/old"
    assert env["OPENINTERPRETABILITY_PAUSE_FILE"] == "/new"  # the new name wins


def test_nothing_to_adopt_warns_nothing(recwarn):
    assert legacy.adopt_legacy_env({"OPENINTERPRETABILITY_X": "1"}) == []
    assert not recwarn.list


def test_the_cache_folder_stays_old_only_on_a_machine_that_has_only_the_old_one(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert legacy.cache_base() == tmp_path / "openinterpretability"
    (tmp_path / "openinterp").mkdir()
    assert legacy.cache_base() == tmp_path / "openinterp"  # the running loops' locks live here
    (tmp_path / "openinterpretability").mkdir()
    assert legacy.cache_base() == tmp_path / "openinterpretability"


def test_the_signing_key_and_keychain_fall_back_to_the_old_names(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir()
    assert legacy.signing_key() == Path("~/.ssh/openinterpretability_signing")
    (tmp_path / ".ssh" / "openinterp_signing").write_text("key")
    assert legacy.signing_key() == Path("~/.ssh/openinterp_signing")
    assert legacy.keychain_services("openinterpretability-xai") == [
        "openinterpretability-xai", "openinterp-xai",
    ]  # fmt: skip
    assert legacy.keychain_services("other") == ["other"]


def test_the_check_allows_only_the_frozen_names_and_places():
    spec = importlib.util.spec_from_file_location("check", "scripts/check_old_name.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    lines = [
        "openinterpretability/x.py:1:import openinterp.cli",
        "docs/a.md:2:set OPENINTERP_SCAN_CACHE",
        "openinterpretability/signing.py:3:NAMESPACE = 'openinterp-record'",
        "openinterpretability/core/run_environment.py:4:    openinterp_version: str",
        "openinterpretability/y.py:5:import openinterpretability",
        "wiki/m/L1/N2/x.json:1:openinterp",
    ]
    assert check.offending(lines) == lines[:2]
