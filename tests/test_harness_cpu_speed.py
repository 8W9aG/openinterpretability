"""Setting torch's thread count for this machine (``apply_cpu_threads``)."""

import torch

from openinterpretability.harness import cpu_speed


def test_a_new_thread_count_is_set_and_returned(monkeypatch):
    set_to = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "set_num_threads", set_to.append)
    monkeypatch.setenv(cpu_speed.THREADS_ENV, "3")
    assert cpu_speed.apply_cpu_threads() == 3
    assert set_to == [3]


def test_the_current_thread_count_is_left_alone(monkeypatch):
    set_to = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "set_num_threads", set_to.append)
    monkeypatch.setenv(cpu_speed.THREADS_ENV, "8")
    assert cpu_speed.apply_cpu_threads() is None
    assert set_to == []
