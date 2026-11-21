"""The heavy-job lock in this process: default file, waiting, whole-process hold, module run."""

import fcntl
import os
import runpy
import sys
import threading

import pytest

from openinterpretability.core import heavy_lock


@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path / "heavy.lock"
    monkeypatch.setenv(heavy_lock.FILE_ENV, str(path))
    monkeypatch.setenv(heavy_lock.ENABLE_ENV, "1")
    monkeypatch.delenv(heavy_lock.HELD_ENV, raising=False)
    monkeypatch.setattr(heavy_lock, "_held", None)
    return path


def test_the_default_lock_file_is_in_the_cache_folder(tmp_path, monkeypatch):
    monkeypatch.delenv(heavy_lock.FILE_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert heavy_lock.lock_file() == tmp_path / "openinterpretability" / "heavy.lock"


def test_a_held_lock_makes_the_job_wait_and_name_the_holder(lock, capsys):
    lock.write_text("process 99\n")
    other = lock.open("a+")
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Closing the other handle releases its lock, so the job can go on.
    timer = threading.Timer(0.2, other.close)
    timer.start()
    try:
        with heavy_lock.heavy_job():
            assert lock.read_text() == f"process {os.getpid()}\n"
    finally:
        timer.join()
    assert f"Waiting for the heavy-job lock ({lock}), held by process 99" in capsys.readouterr().err


def test_hold_for_process_keeps_the_lock_and_takes_it_once(lock, monkeypatch):
    heavy_lock.hold_for_process()
    handle = heavy_lock._held
    try:
        assert handle is not None
        assert lock.read_text() == f"process {os.getpid()}\n"
        heavy_lock.hold_for_process()  # already held: no second lock
        assert heavy_lock._held is handle
    finally:
        heavy_lock._release(handle)
    assert heavy_lock.HELD_ENV not in os.environ


# runpy warns that the module is already imported, which is harmless here.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_running_the_module_exits_with_the_commands_code(lock, monkeypatch):
    monkeypatch.delenv(heavy_lock.ENABLE_ENV)
    monkeypatch.setattr(sys, "argv", ["heavy_lock", sys.executable, "-c", "raise SystemExit(4)"])
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("openinterpretability.core.heavy_lock", run_name="__main__")
    assert exit_.value.code == 4
