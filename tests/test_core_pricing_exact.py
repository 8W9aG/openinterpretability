"""Prices, costs and the cost ledger, at their edges (#6491)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

import pytest

from openinterpretability.core.ledger import (
    DrawEntry,
    load_ledger,
    resolve,
    summary,
    write_entry,
)
from openinterpretability.core.pricing import (
    PRICES,
    Price,
    anthropic,
    cost_usd,
    deepseek_peak,
    price_key,
)


@dataclass(frozen=True)
class T:
    """One request's tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    at: datetime | None = None


def test_deepseek_peak_hours_on_a_weekday_and_a_weekend() -> None:
    monday = datetime(2026, 11, 23, tzinfo=UTC)
    peak = [h for h in range(24) if deepseek_peak(monday + timedelta(hours=h))]
    assert peak == [1, 2, 3, 6, 7, 8, 9]
    saturday = datetime(2026, 11, 28, tzinfo=UTC)
    assert [h for h in range(24) if deepseek_peak(saturday + timedelta(hours=h))] == []


def test_the_tier_counts_every_cache_token_in_the_prompt() -> None:
    haiku = PRICES["claude-haiku-5-5"][0][1]
    t = T(
        input_tokens=10,
        cache_read_tokens=60_000,
        cache_write_5m_tokens=30_000,
        cache_write_1h_tokens=20_000,
    )
    # 110 010 prompt tokens > 100 000: the long-prompt rates (0.50 in, 0.05 read).
    expected = (10 * 0.50 + 60_000 * 0.05 + 30_000 * 0.625 + 20_000 * 1.0) / 1e6
    assert haiku.of(t) == pytest.approx(expected)


def test_an_explicit_prompt_size_reaches_a_nested_tier() -> None:
    top = Price(3.0, 0.0, 0.0)
    middle = Price(2.0, 0.0, 0.0, long_prompt=(1000, top))
    price = Price(1.0, 0.0, 0.0, long_prompt=(10, middle))
    t = T(input_tokens=5)
    assert price.of(t) == pytest.approx(5 * 1.0 / 1e6)
    assert price.of(t, 500) == pytest.approx(5 * 2.0 / 1e6)
    assert price.of(t, 2000) == pytest.approx(5 * 3.0 / 1e6)


def test_anthropic_derives_the_cache_write_prices() -> None:
    long = Price(9.0, 9.0, 9.0)
    assert anthropic(4.0, 20.0, 0.2, long_prompt=(7, long)) == Price(
        4.0, 20.0, 0.2, cache_write_5m=5.0, cache_write_1h=8.0, long_prompt=(7, long)
    )


@pytest.mark.parametrize(
    ("model", "key"),
    [
        ("external/anthropic/claude-opus-5-5[1m]", "claude-opus-5-5"),
        ("a/b/c", "c"),
        ("external/Claude-Opus-5", "claude-opus-5"),
        ("deepseek-flash", "deepseek-flash"),
        ("x/claude-3-5-sonnet-20241022", "grok-build-0.1"),
    ],
)
def test_price_key(model: str, key: str) -> None:
    assert price_key(model) == key


def test_cost_without_a_time_is_priced_today() -> None:
    assert cost_usd("claude-opus-5-5", [T(input_tokens=1_000_000)]) == 4.0


def test_cost_with_a_time_in_another_zone_uses_its_utc_day() -> None:
    # 2026-12-31T20:00-05:00 is 2027-01-01 in UTC: the newer Gemini row applies.
    at = datetime(2026, 12, 31, 20, tzinfo=timezone(timedelta(hours=-5)))
    assert cost_usd("gemini-3.8-flash", [T(input_tokens=1_000_000, at=at)]) == 1.5


def _entry(at: datetime, seed: int, **kw: object) -> DrawEntry:
    return DrawEntry(timestamp=at, seed=seed, kind="explore", model="m", **kw)  # type: ignore[arg-type]


def test_write_entry_writes_a_second_file_into_an_existing_day(tmp_path) -> None:
    at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    first = write_entry(_entry(at, 1), tmp_path)
    second = write_entry(_entry(at, 2), tmp_path)
    assert first.parent == second.parent
    assert second.relative_to(tmp_path).as_posix() == "ledger/20260102/20260102T030405Z-2.json"
    assert second.read_bytes() == _entry(at, 2).model_dump_json().encode() + b"\n"


def test_load_ledger_sorts_by_time_not_by_path(tmp_path) -> None:
    # 10:00+05:00 is 05:00 UTC, earlier than 08:00 UTC, but its path sorts later.
    early = _entry(datetime(2026, 1, 1, 10, tzinfo=timezone(timedelta(hours=5))), 1)
    late = _entry(datetime(2026, 1, 1, 8, tzinfo=UTC), 2)
    write_entry(early, tmp_path)
    write_entry(late, tmp_path)
    assert [e.seed for e in load_ledger(tmp_path)] == [1, 2]


def test_load_ledger_without_a_ledger_is_empty(tmp_path) -> None:
    assert load_ledger(tmp_path) == []


def test_resolve_matches_a_slug_at_exactly_its_time() -> None:
    at = datetime(2026, 1, 1, tzinfo=UTC)
    entry = _entry(at, 1, slug="m/x")
    assert resolve(entry, [("m/x", None, at, "VERIFIED")], []) == ("record", "VERIFIED")


def test_resolve_takes_the_earliest_matching_record() -> None:
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = t1 + timedelta(hours=1)
    entry = _entry(t1, 1, audit="--random --seed 7")
    records = [("m/a", 7, t2, "A"), ("m/b", 7, t1, "B")]
    assert resolve(entry, records, []) == ("record", "B")


def test_summary_counts_forced_poles() -> None:
    at = datetime(2026, 1, 1, tzinfo=UTC)
    resolved = [
        (_entry(at, 1, pole_forced=True), "record", "VERIFIED"),
        (_entry(at, 2, pole_forced=False), "none", None),
        (_entry(at, 3), "stopped", None),
    ]
    assert summary(resolved) == {  # type: ignore[arg-type]
        "entries": 3,
        "outcomes": {"record": 1, "stopped": 1, "none": 1},
        "explore_entries": 3,
        "explore_verified": 1,
        "verified_share_of_all_explore": pytest.approx(1 / 3),
        "verified_share_of_finished_explore": 1.0,
        "pole_forced": 1,
    }
