"""Keeping the runner cache a second time keeps the runners it already holds."""

from openinterpretability import runner_cache


def test_keeping_the_cache_again_keeps_the_loaded_runners(monkeypatch):
    loads = []
    load = runner_cache.cached(lambda model: loads.append(model) or object())
    monkeypatch.setattr(runner_cache, "_runners", None)
    runner_cache.keep()
    first = load("gpt2")
    runner_cache.keep()
    assert load("gpt2") is first
    assert loads == ["gpt2"]
