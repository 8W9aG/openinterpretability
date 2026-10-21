"""What an experiment cost to run: wall time per stage and usage per LLM call.

Reported, never gated. Each command measures its own stages in a per-process collector
(:func:`stage`, :func:`add_call`); the files it writes carry the costs forward
(evidence → submission → record, and score task → scores → record), so a record holds
the cost of every step of the experiment that produced it. With ``OPENINTERP_COSTS_LOG``
set, every command also appends what it measured to that JSON-lines ledger, which
``openinterp costs`` summarises.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import Field

from openinterp.core.strict import StrictModel

LEDGER_ENV = "OPENINTERP_COSTS_LOG"


class StageTiming(StrictModel):
    stage: str = Field(
        min_length=1,
        examples=["model_load", "scan", "lenses", "proposer", "probe_measurement", "gates"],
    )
    seconds: float = Field(ge=0.0)


class LLMCall(StrictModel):
    """One structured LLM request, counting every retry. Token counts and cost are what
    LiteLLM reported, and ``None`` when it reported none or an external agent answered."""

    agent: str = Field(min_length=1, examples=["proposer", "reviewer", "judge"])
    model: str = Field(min_length=1)
    external: bool = Field(
        default=False,
        description="Answered through a prompt file; the stage's time then runs from the "
        "prompt being written to the answer being written.",
    )
    attempts: int = Field(default=1, ge=0)
    prompt_chars: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0, description="Prompt tokens cached.")
    cost_usd: float | None = Field(default=None, ge=0.0)


class ExperimentCosts(StrictModel):
    """Wall time and LLM usage of the run that produced a record. Reported, never gated."""

    stages: list[StageTiming] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.stages)

    @property
    def cost_usd(self) -> float | None:
        known = [c.cost_usd for c in self.llm_calls if c.cost_usd is not None]
        return sum(known) if known else None

    def tokens(self) -> tuple[int, int]:
        """(prompt, completion) tokens over the calls that reported them."""
        return (
            sum(c.prompt_tokens or 0 for c in self.llm_calls),
            sum(c.completion_tokens or 0 for c in self.llm_calls),
        )

    def summary(self) -> str:
        """One line for experiment pages, e.g. ``24.5 min (scan 13.9) · 4 LLM calls · $0.31``."""
        by_stage: dict[str, float] = {}
        for s in self.stages:
            by_stage[s.stage] = by_stage.get(s.stage, 0.0) + s.seconds
        parts = [f"{self.seconds / 60:.1f} min"]
        if by_stage:
            slowest = max(by_stage, key=by_stage.__getitem__)
            parts[0] += f" (slowest: {slowest}, {by_stage[slowest] / 60:.1f} min)"
        if self.llm_calls:
            n = len(self.llm_calls)
            parts.append(f"{n} LLM call{'s' * (n != 1)}")
        prompt, completion = self.tokens()
        if prompt or completion:
            parts.append(f"{prompt:,} + {completion:,} tokens")
        if self.cost_usd is not None:
            parts.append(f"${self.cost_usd:.2f}")
        return " · ".join(parts)


def merge(*parts: ExperimentCosts | None) -> ExperimentCosts | None:
    parts = tuple(p for p in parts if p is not None)
    if not parts:
        return None
    return ExperimentCosts(
        stages=[s for p in parts for s in p.stages],
        llm_calls=[c for p in parts for c in p.llm_calls],
    )


# This process's measurements. Commands are short-lived, so module state is enough; the
# CLI resets it when each command starts.
_stages: list[StageTiming] = []
_calls: list[LLMCall] = []


def reset() -> None:
    _stages.clear()
    _calls.clear()


def add_stage(name: str, seconds: float) -> None:
    _stages.append(StageTiming(stage=name, seconds=round(max(seconds, 0.0), 3)))


def add_call(call: LLMCall) -> None:
    _calls.append(call)


@contextmanager
def stage(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        add_stage(name, time.perf_counter() - start)


def laps() -> Callable[[str], None]:
    """A timer: each ``lap(name)`` records the time since the previous lap (or start)."""
    last = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal last
        now = time.perf_counter()
        add_stage(name, now - last)
        last = now

    return lap


def measured() -> ExperimentCosts:
    """What this process has measured so far."""
    return ExperimentCosts(stages=list(_stages), llm_calls=list(_calls))


def add_external(agent: str, model: str, prompt: Path, answer: Path) -> None:
    """Record a prompt answered from a file: its size, and the time between the prompt
    and the answer being written (``None`` stage when the prompt file is gone)."""
    if prompt.exists() and answer.exists():
        add_stage(agent, answer.stat().st_mtime - prompt.stat().st_mtime)
    chars = len(prompt.read_text()) if prompt.exists() else None
    add_call(LLMCall(agent=agent, model=model, external=True, prompt_chars=chars))


def usage_of(responses: Iterable[object]) -> dict:
    """Token counts and cost summed over raw LiteLLM responses (one per attempt)."""
    totals: dict = {"attempts": 0}
    for response in responses:
        totals["attempts"] += 1
        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        values = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "cached_tokens": getattr(details, "cached_tokens", None),
            "cost_usd": _cost(response),
        }
        for key, value in values.items():
            if isinstance(value, int | float):
                totals[key] = totals.get(key, 0) + value
    return totals


def _cost(response: object) -> float | None:
    hidden = getattr(response, "_hidden_params", None) or {}
    if hidden.get("response_cost") is not None:
        return float(hidden["response_cost"])
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response))
    except Exception:  # an unpriced model, or not a LiteLLM response
        return None


def append_ledger(command: str, seconds: float) -> None:
    """Append this process's measurements to ``$OPENINTERP_COSTS_LOG``, if set."""
    path = os.environ.get(LEDGER_ENV)
    if not path:
        return
    entry = {"command": command, "seconds": round(seconds, 3), **measured().model_dump()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
