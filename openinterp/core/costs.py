"""What an experiment cost to run: wall time per stage and usage per LLM call.

Reported, never gated. Each command measures its own stages in a per-process collector
(:func:`stage`, :func:`add_call`). The files that it writes carry the costs forward:
evidence → submission → record, and score task → scores → record. So a record holds the
cost of every step of the experiment that produced it.

When ``OPENINTERP_COSTS_LOG`` is set, every command also appends its measurements to
that JSON-lines ledger. ``open-interpretability costs`` summarises the ledger.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import Field

from openinterp.core import pricing
from openinterp.core.strict import StrictModel

LEDGER_ENV = "OPENINTERP_COSTS_LOG"


class StageTiming(StrictModel):
    """Wall-clock seconds that one stage of a command took, and the memory it reached."""

    stage: str = Field(
        min_length=1,
        examples=["model_load", "scan", "lenses", "proposer", "probe_measurement", "gates"],
    )
    seconds: float = Field(ge=0.0)
    peak_rss_mb: int | None = Field(
        default=None,
        ge=0,
        description="The process's peak resident memory so far, when the stage ended. It "
        "never falls, so the stage that raises it set the peak. On Apple silicon it leaves "
        "out MPS memory. None in records from before it was measured.",
    )
    peak_cuda_mb: int | None = Field(
        default=None,
        ge=0,
        description="Peak CUDA memory that PyTorch allocated during the stage. None without CUDA.",
    )


class LLMCall(StrictModel):
    """One structured LLM request, counting every retry.

    Token counts and cost are what
    LiteLLM reported, or, for an external answerer, what its usage file says (priced from
    :mod:`openinterp.core.pricing`). ``None`` when nobody reported them.
    """

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
    cache_write_tokens: int | None = Field(
        default=None, ge=0, description="Prompt tokens written to the cache."
    )
    cost_usd: float | None = Field(default=None, ge=0.0)
    price_table: str | None = Field(
        default=None,
        description="List-price table cost_usd was computed from (pricing.TABLE); None when "
        "the provider reported the cost.",
    )


class RequestUsage(StrictModel):
    """One API request's tokens. ``input_tokens`` excludes cache reads and writes."""

    at: datetime | None = Field(default=None, description="When it was made, if known.")
    input_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_5m_tokens: int = Field(default=0, ge=0)
    cache_write_1h_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ExternalUsage(RequestUsage):
    """Tokens an external answerer used.

    ``open-interpretability usage`` writes it next to the answer as ``<answer>.usage.json``. It
    holds totals, plus each request when known,
    since prices can depend on a request's size and time. ``cost_usd`` is for a provider
    that reports its own cost; otherwise the requests are priced at list price for the
    model the command records.
    """

    model: str | None = Field(default=None, description="The model the transcript reports.")
    requests: int = Field(default=0, ge=0)
    per_request: list[RequestUsage] = Field(default_factory=list)
    cost_usd: float | None = Field(default=None, ge=0.0)
    source: str | None = Field(default=None, examples=["claude-code:agent-a1b2c3"])
    forbidden_reads: list[str] = Field(
        default_factory=list,
        description="Blocked paths that a reviewer's transcript shows it read "
        "(openinterp.core.isolation). `review --probes` refuses a probe set with any.",
    )

    @classmethod
    def of(cls, requests: list[RequestUsage], **kw) -> ExternalUsage:
        """Totals over ``requests``, keeping each one."""
        keys = [k for k in RequestUsage.model_fields if k != "at"]
        totals = {k: sum(getattr(r, k) for r in requests) for k in keys}
        return cls(requests=len(requests), per_request=requests, **totals, **kw)

    def call_fields(self, model: str) -> dict:
        """Token, cost and price-table fields of an :class:`LLMCall` answered by ``model``."""
        writes = self.cache_write_5m_tokens + self.cache_write_1h_tokens
        fields = {
            "prompt_tokens": self.input_tokens + self.cache_read_tokens + writes,
            "completion_tokens": self.output_tokens,
            "cached_tokens": self.cache_read_tokens,
            "cache_write_tokens": writes,
        }
        if self.cost_usd is not None:
            return {**fields, "cost_usd": self.cost_usd}
        if self.per_request:
            cost = pricing.cost_usd(model, self.per_request)
        else:  # totals only: made now, each request with the average prompt
            prompt = fields["prompt_tokens"] / max(self.requests, 1)
            cost = pricing.cost_usd(model, [self], prompt=prompt)
        table = pricing.TABLE if cost is not None else None
        return {**fields, "cost_usd": cost, "price_table": table}


class ExperimentCosts(StrictModel):
    """Wall time and LLM usage of the run that produced a record. Reported, never gated."""

    stages: list[StageTiming] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)

    @property
    def seconds(self) -> float:
        """Total seconds over every stage."""
        return sum(s.seconds for s in self.stages)

    @property
    def cost_usd(self) -> float | None:
        """Total cost of the calls that report one, or ``None`` when none does."""
        known = [c.cost_usd for c in self.llm_calls if c.cost_usd is not None]
        return sum(known) if known else None

    def tokens(self) -> tuple[int, int]:
        """(prompt, completion) tokens over the calls that reported them."""
        return (
            sum(c.prompt_tokens or 0 for c in self.llm_calls),
            sum(c.completion_tokens or 0 for c in self.llm_calls),
        )

    def summary(self) -> str:
        """One line for experiment pages.

        For example ``24.5 min (scan 13.9) · peak 5.4 GB RAM · 4 LLM calls · $0.31``. The
        peak is the largest of the commands that made the record, each in its own process.
        """
        by_stage: dict[str, float] = {}
        for s in self.stages:
            by_stage[s.stage] = by_stage.get(s.stage, 0.0) + s.seconds
        parts = [f"{self.seconds / 60:.1f} min"]
        if by_stage:
            slowest = max(by_stage, key=by_stage.__getitem__)
            parts[0] += f" (slowest: {slowest}, {by_stage[slowest] / 60:.1f} min)"
        peak = max((s.peak_rss_mb for s in self.stages if s.peak_rss_mb), default=None)
        if peak:
            parts.append(f"peak {peak / 1024:.1f} GB RAM")
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
    """Join the costs of several commands of one experiment, or ``None`` when all are ``None``."""
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
    """Clear this process's measurements, so that one command does not count another's."""
    _stages.clear()
    _calls.clear()


def add_stage(name: str, seconds: float) -> None:
    """Record that stage ``name`` took ``seconds``, rounded to the millisecond, and its memory.

    The CUDA peak is reset for the next stage, and MPS's cache is released.
    """
    rss, cuda = _peak_memory_mb()
    timing = StageTiming(
        stage=name, seconds=round(max(seconds, 0.0), 3), peak_rss_mb=rss, peak_cuda_mb=cuda
    )
    _stages.append(timing)
    _reset_cuda_peak()
    _release_mps_cache()


def _peak_memory_mb() -> tuple[int | None, int | None]:
    """(peak resident memory of this process, peak CUDA allocation since the last reset)."""
    rss = None
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = round(peak / 2**20) if sys.platform == "darwin" else round(peak / 2**10)
    except (ImportError, OSError):  # no resource module (Windows)
        pass
    return rss, _cuda_peak_mb()


def _cuda_peak_mb() -> int | None:
    torch = sys.modules.get("torch")  # never import torch just to ask
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**20)


def _release_mps_cache() -> None:
    """Give memory that PyTorch's MPS allocator keeps for reuse back to the system.

    On Apple silicon the GPU uses the same memory as the CPU. The allocator keeps every
    freed block, so over an audit's stages it held 2.8 GB above the weights for
    Qwen2.5-0.5B. Each stage needs different blocks, so the next stage loses little.
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.backends.mps.is_available():
        return
    if torch.mps.current_allocated_memory():  # a CPU run never touches Metal
        torch.mps.empty_cache()


def _reset_cuda_peak() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()


def add_call(call: LLMCall) -> None:
    """Record one LLM request of this process."""
    _calls.append(call)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time the ``with`` block as stage ``name``, also when it raises."""
    _reset_cuda_peak()
    start = time.perf_counter()
    try:
        yield
    finally:
        add_stage(name, time.perf_counter() - start)


def laps() -> Callable[[str], None]:
    """A timer: each ``lap(name)`` records the time since the previous lap (or start)."""
    _reset_cuda_peak()
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


def usage_path(answer: Path) -> Path:
    """Where an answer's usage goes: ``ev.hypothesis.json`` → ``ev.hypothesis.usage.json``."""
    return answer.with_name(answer.name.removesuffix(".json") + ".usage.json")


def add_external(agent: str, model: str, prompt: Path, answer: Path) -> None:
    """Record a prompt answered from a file.

    Records the prompt's size and the time between the prompt and the answer being
    written (no stage when the prompt file is gone). Also records the answerer's tokens
    and cost when ``usage_path(answer)`` exists.
    """
    if prompt.exists() and answer.exists():
        add_stage(agent, answer.stat().st_mtime - prompt.stat().st_mtime)
    chars = len(prompt.read_text()) if prompt.exists() else None
    fields = {}
    if (usage_file := usage_path(answer)).exists():
        usage = ExternalUsage.model_validate_json(usage_file.read_text())
        _check_model(usage, model, usage_file)
        fields = usage.call_fields(model)
    add_call(LLMCall(agent=agent, model=model, external=True, prompt_chars=chars, **fields))


def _check_model(usage: ExternalUsage, model: str, usage_file: Path) -> None:
    """Warn when the transcript names a different model from the recorded one.

    No warning when ``$OPENINTERP_MODEL_ALIASES`` says a proxy renamed the model. The
    recorded model is the one priced.
    """
    from openinterp.agents.base import MODEL_ALIASES_ENV

    reported = usage.model and {pricing.price_key(m) for m in usage.model.split(",")}
    if (
        reported
        and pricing.price_key(model) not in reported
        and not os.environ.get(MODEL_ALIASES_ENV)
    ):
        print(
            f"warning: {usage_file} was answered by {usage.model}, but {model} is being "
            "recorded and priced.",
            file=sys.stderr,
        )


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
            "cache_write_tokens": getattr(details, "cache_creation_tokens", None),
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


ROLE_ORDER = ("proposer", "reviewer", "judge")


def by_agent(
    experiments: Iterable[tuple[str | None, ExperimentCosts | None]],
    name: Callable[[str], str] = str,
) -> list[dict]:
    """LLM cost per experiment for each (role, model), from (verdict status, costs) pairs.

    An experiment counts as priced for a role and model when every one of its calls there
    has a cost. ``mean_usd`` averages over priced experiments, and ``usd_per_verified``
    divides their total by how many of them were VERIFIED. ``name`` maps a call's model to
    the name it is grouped under (e.g. ``openinterp.core.agent_names.canonical_agent``).
    """
    groups: dict[tuple[str, str], dict] = {}
    for status, run in experiments:
        calls: dict[tuple[str, str], list[LLMCall]] = {}
        for call in run.llm_calls if run else ():
            calls.setdefault((call.agent, name(call.model)), []).append(call)
        for key, cs in calls.items():
            g = groups.setdefault(key, {"experiments": 0, "priced": 0, "usd": 0.0, "verified": 0})
            g["experiments"] += 1
            if all(c.cost_usd is not None for c in cs):
                g["priced"] += 1
                g["usd"] += sum(c.cost_usd for c in cs)
                g["verified"] += status == "VERIFIED"
    rows = [
        {
            "agent": agent,
            "model": model,
            **g,
            "mean_usd": g["usd"] / g["priced"] if g["priced"] else None,
            "usd_per_verified": g["usd"] / g["verified"] if g["verified"] else None,
        }
        for (agent, model), g in groups.items()
    ]
    order = {r: i for i, r in enumerate(ROLE_ORDER)}
    return sorted(rows, key=lambda r: (order.get(r["agent"], len(order)), -r["priced"], r["model"]))
