"""Experiment costs (openinterpretability/core/costs.py): pricing, summaries and memory peaks."""

import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from openinterpretability.core import costs, pricing


def test_usage_with_each_request_is_priced_request_by_request():
    at = datetime.fromisoformat("2026-10-28T01:00:00+00:00")
    requests = [
        costs.RequestUsage(at=at, input_tokens=4, output_tokens=300),
        costs.RequestUsage(at=at, input_tokens=6, output_tokens=40),
    ]
    usage = costs.ExternalUsage.of(requests, model="claude-sonnet-5-5")
    fields = usage.call_fields("external/claude-sonnet-5-5")
    assert fields["cost_usd"] == pricing.cost_usd("external/claude-sonnet-5-5", requests)
    assert fields["cost_usd"] is not None and fields["price_table"] == pricing.TABLE
    assert (fields["prompt_tokens"], fields["completion_tokens"]) == (10, 340)


def test_a_summary_names_tokens_but_no_memory_peak_when_none_was_measured():
    run = costs.ExperimentCosts(
        stages=[costs.StageTiming(stage="scan", seconds=120.0)],
        llm_calls=[
            costs.LLMCall(agent="judge", model="m", prompt_tokens=1200, completion_tokens=30)
        ],
    )
    assert run.summary() == "2.0 min (slowest: scan, 2.0 min) · 1 LLM call · 1,200 + 30 tokens"
    stages_only = costs.ExperimentCosts(stages=run.stages)
    assert stages_only.summary() == "2.0 min (slowest: scan, 2.0 min)"


def test_without_the_resource_module_the_peak_memory_is_unknown(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)  # as on Windows: the import fails
    rss, _ = costs._peak_memory_mb()
    assert rss is None


def _fake_torch(*, cuda: bool, mps: bool, peak: int = 0):
    """A torch module that reports CUDA and MPS as given; CI has neither device."""
    calls = []
    cuda_api = SimpleNamespace(
        is_available=lambda: cuda,
        is_initialized=lambda: cuda,
        max_memory_allocated=lambda: peak,
        reset_peak_memory_stats=lambda: calls.append("reset"),
    )
    mps_api = SimpleNamespace(
        current_allocated_memory=lambda: 0, empty_cache=lambda: calls.append("empty")
    )
    backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    return SimpleNamespace(cuda=cuda_api, mps=mps_api, backends=backends), calls


def test_a_cuda_stage_records_its_peak_and_resets_it_for_the_next(monkeypatch):
    torch, calls = _fake_torch(cuda=True, mps=False, peak=3 * 2**20)
    monkeypatch.setitem(sys.modules, "torch", torch)
    costs.reset()
    with costs.stage("scan"):
        pass
    (timing,) = costs.measured().stages
    assert timing.peak_cuda_mb == 3
    assert calls == ["reset", "reset"]  # at the start, and after the stage for the next one
    costs.reset()


def test_a_machine_without_mps_or_cuda_releases_and_resets_nothing(monkeypatch):
    torch, calls = _fake_torch(cuda=False, mps=False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    costs.reset()
    costs.add_stage("scan", 1.0)
    assert costs.measured().stages[0].peak_cuda_mb is None and calls == []
    costs.reset()


def test_usage_of_skips_values_that_are_not_numbers():
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=None, prompt_tokens_details=None)
    response = SimpleNamespace(usage=usage, _hidden_params={"response_cost": 0.5})
    assert costs.usage_of([response, response]) == {
        "attempts": 2,
        "prompt_tokens": 14,
        "cost_usd": 1.0,
    }


def test_a_response_without_a_reported_cost_is_priced_by_litellm(monkeypatch):
    import litellm

    response = SimpleNamespace(usage=None, _hidden_params={})
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.125)
    assert costs._cost(response) == 0.125

    def unpriced(completion_response):
        raise ValueError("This model isn't mapped yet")

    monkeypatch.setattr(litellm, "completion_cost", unpriced)
    assert costs._cost(response) is None


@pytest.mark.parametrize("response", [object(), SimpleNamespace(_hidden_params=None)])
def test_a_response_that_is_not_litellms_has_no_cost(response, monkeypatch):
    import litellm

    def unpriced(completion_response):
        raise TypeError("not a ModelResponse")

    monkeypatch.setattr(litellm, "completion_cost", unpriced)  # no pricing request
    assert costs._cost(response) is None
