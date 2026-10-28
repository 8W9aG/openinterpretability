"""List prices for token usage nobody priced, e.g. Claude Code subagents answering a prompt.

LiteLLM prices its own calls. An external answerer's tokens (``external/<name>``) are priced
here instead, and each priced call records :data:`TABLE`, since list prices change. When
they do, update the rows and the date together: older records keep the table they were
priced with. A model missing from the table stays unpriced, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Anthropic first-party API list prices, as of this date.
TABLE = "anthropic-2026-10-06"

CACHE_WRITE_5M = 1.25  # × input price
CACHE_WRITE_1H = 2.0


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float


PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(10.0, 50.0, 0.25),
    "claude-mythos-5-1": Price(10.0, 50.0, 0.25),
    "claude-fable-5": Price(10.0, 50.0, 1.0),
    "claude-opus-5-5": Price(4.0, 20.0, 0.20),
    "claude-opus-5": Price(5.0, 25.0, 0.50),
    "claude-opus-4-8": Price(5.0, 25.0, 0.50),
    "claude-opus-4-7": Price(5.0, 25.0, 0.50),
    "claude-opus-4-6": Price(5.0, 25.0, 0.50),
    "claude-sonnet-5-5": Price(2.0, 10.0, 0.20),
    "claude-sonnet-5": Price(2.0, 10.0, 0.20),
    "claude-sonnet-4-6": Price(3.0, 15.0, 0.30),
    # Prompts of up to 100K tokens; longer ones cost 5× and are priced at this rate too.
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01),
    "claude-haiku-4-5": Price(1.0, 5.0, 0.10),
}


def price_key(model: str) -> str:
    """``external/anthropic/claude-opus-5-5[1m]`` → ``claude-opus-5-5``."""
    name = model.removeprefix("external/").rsplit("/", 1)[-1]
    return re.sub(r"\[[^\]]*\]$", "", name).strip().lower()


def price_of(model: str) -> Price | None:
    return PRICES.get(price_key(model))


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_5m_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
) -> float | None:
    """What these tokens cost at list price, or ``None`` for a model not in the table.
    ``input_tokens`` excludes cache reads and writes."""
    p = price_of(model)
    if p is None:
        return None
    total = (
        input_tokens * p.input
        + output_tokens * p.output
        + cache_read_tokens * p.cache_read
        + cache_write_5m_tokens * p.input * CACHE_WRITE_5M
        + cache_write_1h_tokens * p.input * CACHE_WRITE_1H
    )
    return round(total / 1e6, 6)
