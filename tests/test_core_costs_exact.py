"""Cost records: model checks, call fields and peaks, at their edges (#6491)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openinterpretability.agents.base import MODEL_ALIASES_ENV
from openinterpretability.core import costs, pricing

HAIKU = "external/claude-haiku-5-5"  # its long-prompt tier starts above 100,000 tokens


def test_totals_only_usage_is_priced_at_the_average_prompt_of_its_requests():
    usage = costs.ExternalUsage(
        requests=2,
        input_tokens=150_000,
        output_tokens=1_000,
        cache_write_5m_tokens=1_000,
        cache_write_1h_tokens=500,
    )
    # 151,500 prompt tokens over 2 requests is 75,750 each: the short-prompt tier.
    assert usage.call_fields(HAIKU) == {
        "prompt_tokens": 151_500,
        "completion_tokens": 1_000,
        "cached_tokens": 0,
        "cache_write_tokens": 1_500,
        "cost_usd": 0.015725,
        "price_table": pricing.TABLE,
    }


@pytest.mark.parametrize("requests", [0, 1])
def test_totals_of_one_or_no_request_use_the_whole_prompt_for_the_tier(requests):
    usage = costs.ExternalUsage(requests=requests, input_tokens=150_000)
    assert usage.call_fields(HAIKU) == {
        "prompt_tokens": 150_000,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.075,
        "price_table": pricing.TABLE,
    }


def _check(monkeypatch, capsys, reported, model="external/claude-sonnet-5-5", aliases=None):
    if aliases is None:
        monkeypatch.delenv(MODEL_ALIASES_ENV, raising=False)
    else:
        monkeypatch.setenv(MODEL_ALIASES_ENV, aliases)
    usage = costs.ExternalUsage(model=reported)
    costs._check_model(usage, model, Path("ev.usage.json"))
    out = capsys.readouterr()
    assert out.out == ""
    return out.err


def test_a_different_reported_model_warns_with_the_exact_message(monkeypatch, capsys):
    err = _check(monkeypatch, capsys, "claude-haiku-5-5")
    assert err == (
        "warning: ev.usage.json was answered by claude-haiku-5-5, but "
        "external/claude-sonnet-5-5 is being recorded and priced.\n"
    )


def test_the_recorded_model_among_several_reported_ones_does_not_warn(monkeypatch, capsys):
    assert _check(monkeypatch, capsys, "claude-haiku-5-5,claude-sonnet-5-5") == ""


def test_the_same_reported_model_does_not_warn(monkeypatch, capsys):
    assert _check(monkeypatch, capsys, "anthropic/claude-sonnet-5-5") == ""


def test_no_reported_model_does_not_warn(monkeypatch, capsys):
    assert _check(monkeypatch, capsys, None) == ""


def test_a_model_alias_setting_silences_the_warning(monkeypatch, capsys):
    assert _check(monkeypatch, capsys, "claude-haiku-5-5", aliases="proxy=claude") == ""


def test_litellm_prices_the_response_itself(monkeypatch):
    import litellm

    response = SimpleNamespace(usage=None, _hidden_params={})
    monkeypatch.setattr(
        litellm,
        "completion_cost",
        lambda completion_response: 0.25 if completion_response is response else 9.0,
    )
    assert costs._cost(response) == 0.25


def _cuda_torch(peak: int):
    cuda = SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True, max_memory_allocated=lambda: peak
    )
    return SimpleNamespace(cuda=cuda)


def test_the_cuda_peak_is_read_from_an_initialised_cuda_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _cuda_torch(5 * 2**20))
    assert costs._cuda_peak_mb() == 5


def test_without_torch_the_cuda_peak_is_unknown(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert costs._cuda_peak_mb() is None
