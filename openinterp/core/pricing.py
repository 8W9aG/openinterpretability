"""List prices for token usage nobody priced, e.g. Claude Code subagents answering a prompt.

LiteLLM prices its own calls. An external answerer's tokens (``external/<name>``) are priced
here instead, one request at a time, and each priced call records :data:`TABLE`, since list
prices change. When they do, add a dated row rather than editing one, and bump the table's
date: older records keep the table they were priced with. A model missing from the table,
or a request older than its first row, stays unpriced, never guessed.

Not priced: cache storage billed per hour (Gemini, and Z.ai once its free period ends),
and DeepSeek's off-peak rate on Chinese public holidays (peak rates are charged).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol

# When every row below was last checked against its vendor's pricing page:
#   Anthropic, first-party API          2026-10-06
#   DeepSeek   api-docs.deepseek.com    2026-10-28
#   Google     ai.google.dev            2026-10-28
#   Z.ai       docs.z.ai                2026-10-28
#   xAI        docs.x.ai                2026-10-28
#   Meta       OpenRouter's listing     2026-10-28 (Meta publishes no Contributor rates)
TABLE = "2026-10-28"

ANTHROPIC_WRITE_5M = 1.25  # × input price
ANTHROPIC_WRITE_1H = 2.0


class Tokens(Protocol):
    """One request's tokens; ``input_tokens`` excludes cache reads and writes."""

    at: datetime | None
    input_tokens: int
    cache_read_tokens: int
    cache_write_5m_tokens: int
    cache_write_1h_tokens: int
    output_tokens: int


def deepseek_peak(at: datetime) -> bool:
    """01:00–04:00 and 06:00–10:00 UTC, Monday to Friday."""
    at = at.astimezone(UTC)
    return at.weekday() < 5 and (1 <= at.hour < 4 or 6 <= at.hour < 10)


@dataclass(frozen=True)
class Price:
    """USD per million tokens. Cache writes cost the input price unless set."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float | None = None
    cache_write_1h: float | None = None
    long_prompt: tuple[int, Price] | None = None  # (prompt tokens above which, price)
    peak: Callable[[datetime], bool] | None = None  # when every rate is doubled

    def of(self, t: Tokens, prompt: float | None = None) -> float:
        """``prompt`` overrides the prompt size that picks the tier."""
        if prompt is None:
            prompt = t.input_tokens + t.cache_read_tokens + t.cache_write_5m_tokens
            prompt += t.cache_write_1h_tokens
        if self.long_prompt and prompt > self.long_prompt[0]:
            return self.long_prompt[1].of(t, prompt)
        write_5m = self.input if self.cache_write_5m is None else self.cache_write_5m
        write_1h = self.input if self.cache_write_1h is None else self.cache_write_1h
        usd = (
            t.input_tokens * self.input
            + t.output_tokens * self.output
            + t.cache_read_tokens * self.cache_read
            + t.cache_write_5m_tokens * write_5m
            + t.cache_write_1h_tokens * write_1h
        )
        if self.peak and t.at and self.peak(t.at):
            usd *= 2
        return usd / 1e6


def anthropic(input: float, output: float, cache_read: float, **kw) -> Price:
    return Price(
        input,
        output,
        cache_read,
        cache_write_5m=input * ANTHROPIC_WRITE_5M,
        cache_write_1h=input * ANTHROPIC_WRITE_1H,
        **kw,
    )


ALWAYS = date.min

# model → [(first day the price applies, price)], oldest first
PRICES: dict[str, list[tuple[date, Price]]] = {
    "claude-fable-5-1": [(ALWAYS, anthropic(10.0, 50.0, 0.25))],
    "claude-mythos-5-1": [(ALWAYS, anthropic(10.0, 50.0, 0.25))],
    "claude-fable-5": [(ALWAYS, anthropic(10.0, 50.0, 1.0))],
    "claude-opus-5-5": [(ALWAYS, anthropic(4.0, 20.0, 0.20))],
    "claude-opus-5": [(ALWAYS, anthropic(5.0, 25.0, 0.50))],
    "claude-opus-4-8": [(ALWAYS, anthropic(5.0, 25.0, 0.50))],
    "claude-opus-4-7": [(ALWAYS, anthropic(5.0, 25.0, 0.50))],
    "claude-opus-4-6": [(ALWAYS, anthropic(5.0, 25.0, 0.50))],
    "claude-sonnet-5-5": [(ALWAYS, anthropic(2.0, 10.0, 0.20))],
    "claude-sonnet-5": [(ALWAYS, anthropic(2.0, 10.0, 0.20))],
    "claude-sonnet-4-6": [(ALWAYS, anthropic(3.0, 15.0, 0.30))],
    "claude-haiku-5-5": [
        (ALWAYS, anthropic(0.10, 0.50, 0.01, long_prompt=(100_000, anthropic(0.50, 2.50, 0.05))))
    ],
    "claude-haiku-4-5": [(ALWAYS, anthropic(1.0, 5.0, 0.10))],
    # DeepSeek-V4.1-Flash. Older requests were billed at earlier prices, which aren't listed.
    "deepseek-flash": [(date(2026, 9, 10), Price(0.15, 0.60, 0.003, peak=deepseek_peak))],
    "gemini-3.8-flash": [
        (ALWAYS, Price(0.75, 3.75, 0.075)),
        (date(2027, 1, 1), Price(1.50, 7.50, 0.15)),
    ],
    "glm-5.3-flash": [(ALWAYS, Price(0.15, 0.50, 0.03))],
    "glm-4.7-flash": [(ALWAYS, Price(0.0, 0.0, 0.0))],
    "grok-build-0.1": [
        (ALWAYS, Price(1.0, 2.0, 0.20, long_prompt=(199_999, Price(2.0, 4.0, 0.40))))
    ],
}
PRICES["grok-build"] = PRICES["grok-build-0.1"]

# Every version of a family, e.g. muse-spark-1.3-contributor, at one price. Muse Spark is
# priced at its Contributor tier (Meta may train on the prompts), which these runs use.
FAMILIES: dict[str, list[tuple[date, Price]]] = {
    "muse-spark": [(ALWAYS, Price(0.10, 0.20, 0.002))],
}


def price_key(model: str) -> str:
    """``external/anthropic/claude-opus-5-5[1m]`` → ``claude-opus-5-5``."""
    name = model.removeprefix("external/").rsplit("/", 1)[-1]
    return re.sub(r"\[[^\]]*\]$", "", name).strip().lower()


def prices_of(model: str) -> list[tuple[date, Price]] | None:
    key = price_key(model)
    if key in PRICES:
        return PRICES[key]
    return next((rows for f, rows in FAMILIES.items() if key.startswith(f + "-")), None)


def price_on(model: str, day: date) -> Price | None:
    rows = prices_of(model) or []
    return next((p for since, p in reversed(rows) if since <= day), None)


def cost_usd(
    model: str, requests: Iterable[Tokens], *, prompt: float | None = None
) -> float | None:
    """What these requests cost at list price, or ``None`` when any of them has no price
    (an unlisted model, or a request before the model's first row). A request with no
    time is priced as of today. ``prompt`` sets the prompt size that picks a tier, for
    totals over several requests."""
    total = 0.0
    for t in requests:
        price = price_on(model, (t.at or datetime.now(UTC)).astimezone(UTC).date())
        if price is None:
            return None
        total += price.of(t, prompt)
    return round(total, 6)
