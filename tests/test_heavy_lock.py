"""The opt-in machine-wide lock for heavy jobs (core/heavy_lock.py, issue #3247)."""

import fcntl
import os
import subprocess
import sys
import time

import pytest

from openinterpretability.core import heavy_lock


@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path / "heavy.lock"
    monkeypatch.setenv(heavy_lock.FILE_ENV, str(path))
    monkeypatch.delenv(heavy_lock.HELD_ENV, raising=False)
    monkeypatch.setattr(heavy_lock, "_held", None)
    return path


def is_free(path) -> bool:
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def test_off_by_default_it_takes_nothing(lock, monkeypatch):
    monkeypatch.delenv(heavy_lock.ENABLE_ENV, raising=False)
    with heavy_lock.heavy_job():
        assert not lock.exists()


def test_on_it_holds_the_file_and_names_the_holder(lock, monkeypatch):
    monkeypatch.setenv(heavy_lock.ENABLE_ENV, "1")
    with heavy_lock.heavy_job():
        assert lock.read_text() == f"process {os.getpid()}\n"
        assert os.environ[heavy_lock.HELD_ENV] == str(os.getpid())
        holder = subprocess.run(
            [sys.executable, "-c", f"import fcntl; h=open({str(lock)!r}, 'a+'); "
             "fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)"],
            capture_output=True, check=False,
        )  # fmt: skip
        assert holder.returncode != 0  # another process can't take it
    assert heavy_lock.HELD_ENV not in os.environ
    assert is_free(lock)


def test_a_child_of_the_holder_skips_it(lock, monkeypatch):
    monkeypatch.setenv(heavy_lock.ENABLE_ENV, "1")
    monkeypatch.setenv(heavy_lock.HELD_ENV, "12345")  # as a holder's child sees it
    with heavy_lock.heavy_job():
        assert not lock.exists()


def test_a_second_process_waits_until_the_first_exits(lock, tmp_path):
    # The first holds the lock until the test creates `release`, not for a fixed time. On a
    # busy machine, the second process could start after a fixed hold ended (#6916).
    env = {**os.environ, heavy_lock.ENABLE_ENV: "1", heavy_lock.FILE_ENV: str(lock)}
    env.pop(heavy_lock.HELD_ENV, None)
    release = tmp_path / "release"
    hold = (
        f"import pathlib, time\nwhile not pathlib.Path({str(release)!r}).exists(): time.sleep(0.02)"
    )
    run = [sys.executable, "-m", "openinterpretability.core.heavy_lock"]
    first = subprocess.Popen([*run, sys.executable, "-c", hold], env=env)
    deadline = time.monotonic() + 30
    while not lock.exists() or not lock.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    second = subprocess.Popen(
        [*run, sys.executable, "-c", "pass"], env=env, stderr=subprocess.PIPE, text=True
    )
    assert second.stderr is not None
    assert "Waiting for the heavy-job lock" in second.stderr.readline()
    assert second.poll() is None  # it waits while the first holds the lock
    release.touch()
    assert first.wait(timeout=30) == 0
    assert second.wait(timeout=30) == 0
    second.stderr.close()
    assert is_free(lock)  # released at exit


def test_the_runner_passes_the_exit_code(lock):
    assert heavy_lock.main([sys.executable, "-c", "raise SystemExit(3)"]) == 3
    assert heavy_lock.main([]) == 2
