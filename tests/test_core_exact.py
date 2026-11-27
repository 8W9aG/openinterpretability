"""The heavy-job lock, the pause file and the legacy names, at their edges (#6491)."""

from __future__ import annotations

import os
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from openinterpretability.core import heavy_lock, legacy_names, pause


@pytest.mark.parametrize(("value", "on"), [("1", True), ("TRUE", True), ("Yes", True),
                                           ("ON", True), ("", False), ("0", False)])  # fmt: skip
def test_the_lock_is_on_for_the_usual_yes_words_in_any_case(monkeypatch, value, on):
    monkeypatch.setenv(heavy_lock.ENABLE_ENV, value)
    assert heavy_lock.enabled() is on


def test_the_lock_makes_its_folder_and_names_an_unknown_holder(tmp_path, monkeypatch, capsys):
    path = tmp_path / "a" / "b" / "heavy.lock"
    monkeypatch.setenv(heavy_lock.FILE_ENV, str(path))
    monkeypatch.delenv(heavy_lock.HELD_ENV, raising=False)
    calls = []

    def flock(handle, flags):
        calls.append(flags)
        if len(calls) == 1:
            raise BlockingIOError  # held, by a process that wrote no name

    monkeypatch.setattr(heavy_lock.fcntl, "flock", flock)
    handle = heavy_lock._acquire()
    err = capsys.readouterr().err
    assert f"Waiting for the heavy-job lock ({path}), held by another process…" in err
    heavy_lock._release(handle)
    assert heavy_lock.HELD_ENV not in os.environ
    handle = heavy_lock._acquire()
    os.environ.pop(heavy_lock.HELD_ENV)
    heavy_lock._release(handle)  # nothing to remove is fine


def test_the_lock_command_needs_a_command(capsys):
    assert heavy_lock.main([]) == 2
    assert capsys.readouterr().err == (
        "usage: python -m openinterpretability.core.heavy_lock <command> [args …]\n"
    )


def test_legacy_variables_are_adopted_into_the_mapping_given():
    environ = {"OPENINTERP_B": "2", "OPENINTERP_A": "1", "PATH": "/bin",
               "OPENINTERPRETABILITY_C": "3"}  # fmt: skip
    with pytest.warns(DeprecationWarning) as caught:
        adopted = legacy_names.adopt_legacy_env(environ)
    assert sorted(adopted) == ["OPENINTERPRETABILITY_A", "OPENINTERPRETABILITY_B"]
    assert environ["OPENINTERPRETABILITY_A"] == "1" and "OPENINTERPRETABILITY_A" not in os.environ
    assert len(environ) == 6  # PATH and the new name are left alone
    (warning,) = caught
    assert "Read for now: OPENINTERPRETABILITY_A, OPENINTERPRETABILITY_B." in str(warning.message)


def test_no_legacy_variable_means_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert legacy_names.adopt_legacy_env({"PATH": "/bin"}) == []


@pytest.fixture
def pause_path(tmp_path, monkeypatch):
    path = tmp_path / "pause"
    monkeypatch.setenv(pause.ENV, str(path))
    return path


def test_a_pause_says_until_when_in_local_time():
    until = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    note = pause.Pause("why", "me", until.isoformat(), until.isoformat()).describe()
    assert note == (f"New experiments are paused until {until.astimezone():%H:%M}: why (by me). "
                    "Run `open-interpretability resume` to continue sooner.")  # fmt: skip


def test_an_unreadable_pause_file_is_a_default_pause_from_its_time(pause_path):
    pause_path.write_text("not json")
    since = datetime(2026, 1, 2, 3, 0, tzinfo=UTC)
    os.utime(pause_path, (since.timestamp(), since.timestamp()))
    ends = since + timedelta(minutes=pause.DEFAULT_MINUTES)
    assert pause.read(now=since) == pause.Pause("the pause file can't be read", "?",
                                                since.isoformat(), ends.isoformat())  # fmt: skip
    assert pause.read(now=ends) is None and not pause_path.exists()  # it ends at its time


def test_a_pause_that_ends_is_gone_even_when_its_file_cant_be_removed(pause_path, monkeypatch):
    now = datetime(2026, 1, 2, tzinfo=UTC)
    pause.start("why", minutes=1, now=now - timedelta(minutes=5))

    def refuse(self, missing_ok=False):
        raise PermissionError(self)

    monkeypatch.setattr(Path, "unlink", refuse)
    assert pause.read(now=now) is None


def test_legacy_variables_are_read_from_the_process_by_default(monkeypatch):
    environ = {"OPENINTERP_X": "1"}
    monkeypatch.setattr(os, "environ", environ)
    with pytest.warns(DeprecationWarning):
        assert legacy_names.adopt_legacy_env() == ["OPENINTERPRETABILITY_X"]
    assert environ["OPENINTERPRETABILITY_X"] == "1"


def test_the_cache_folder_is_under_the_home_cache_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert legacy_names.cache_base() == tmp_path / ".cache" / "openinterpretability"
