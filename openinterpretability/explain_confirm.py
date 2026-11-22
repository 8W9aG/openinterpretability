"""Ask before ``explain`` spends LLM tokens (#5830).

``explain --run``, ``--complete`` and a narrated ``--level`` call LLMs. A run to a high
target, with no ``--run N`` cap, can take up to ``--max-attempts`` proposer and reviewer
turns for each planned unit. So before the first call, ``explain`` says what will run,
estimates the tokens and the cost, and asks. ``--yes`` skips the question. With no
terminal to ask on, and no ``--yes``, it refuses rather than wait for an answer that never
comes.

The estimate is the registry's own: the mean tokens and cost of the model's newest
experiments that record their LLM calls (``provenance.costs``).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click

from openinterpretability.core.identifiers import model_slug
from openinterpretability.core.paths import experiments_dir

NARRATED = ("long", "paragraph")  # the levels whose narration an LLM writes
RECENT = 200  # the newest records whose LLM calls make the estimate


@dataclass(frozen=True)
class Estimate:
    """The mean tokens and cost of one experiment, from ``records`` of them."""

    tokens: float
    usd: float | None  # None when no call reported a cost
    records: int


def _calls(path: Path) -> list[dict]:
    """The LLM calls that the experiment record at ``path`` reports, or none.

    Args:
        path: An experiment record.

    Returns:
        Its ``provenance.costs.llm_calls``, or an empty list.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    provenance = (data.get("submission") or {}).get("provenance") or {}
    return list((provenance.get("costs") or {}).get("llm_calls") or [])


def estimate(wiki_dir: Path, model: str, recent: int = RECENT) -> Estimate | None:
    """The mean tokens and cost per experiment over ``model``'s newest records with LLM calls.

    Record files are named by their UTC time, so the newest are last by name.

    Args:
        wiki_dir: The registry.
        model: The model's Hub name.
        recent: How many records with LLM calls to average.

    Returns:
        The estimate, or None when no record reports an LLM call.

    Examples:
        >>> import tempfile
        >>> wiki = Path(tempfile.mkdtemp())
        >>> estimate(wiki, "a/m") is None
        True
    """
    folder = experiments_dir(wiki_dir) / model_slug(model)
    paths = sorted(folder.rglob("*.json"), key=lambda p: p.name, reverse=True)
    totals = []
    for path in paths:
        if calls := _calls(path):
            tokens = sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0)
                         for c in calls)  # fmt: skip
            costs = [c["cost_usd"] for c in calls if c.get("cost_usd") is not None]
            totals.append((tokens, sum(costs) if costs else None))
        if len(totals) == recent:
            break
    if not totals:
        return None
    usd = [u for _, u in totals if u is not None]
    return Estimate(sum(t for t, _ in totals) / len(totals),
                    sum(usd) / len(usd) if usd else None, len(totals))  # fmt: skip


def warning(experiments: int, attempts: int, narrate: bool, found: Estimate | None) -> list[str]:
    """The lines that say what will call an LLM, what it may cost, and how to stop it.

    Args:
        experiments: The planned experiments that the run may take, 0 for no run.
        attempts: The most turns for each (``--max-attempts``).
        narrate: Whether a narrator writes the long form.
        found: The registry's :func:`estimate`, or None.

    Returns:
        The warning, one line per item.

    Examples:
        >>> for line in warning(4, 5, True, Estimate(20_000, 0.05, 30)):
        ...     print(line)
        This calls LLMs: up to 4 experiments, at most 5 turns each, and the narrator.
        Estimate: 20,000 tokens and $0.05 an experiment (the mean of 30 records).
        So 80,000 to 400,000 tokens, and $0.20 to $1.00.
        To spend less: --run N to cap the experiments, a lower --target, or Ctrl-C at any time.
    """
    parts = []
    if experiments:
        parts.append(f"up to {experiments} experiments, at most {attempts} turns each")
    if narrate:
        parts.append("the narrator")
    lines = [f"This calls LLMs: {', and '.join(parts)}."]
    if found is None or not experiments:
        lines.append("Estimate: none from the registry yet. This can use a lot of tokens.")
    else:
        each = f"{found.tokens:,.0f} tokens" + (f" and ${found.usd:.2f}" if found.usd else "")
        lines.append(f"Estimate: {each} an experiment (the mean of {found.records} records).")
        low, high = experiments, experiments * attempts
        total = f"So {found.tokens * low:,.0f} to {found.tokens * high:,.0f} tokens"
        if found.usd:
            total += f", and ${found.usd * low:.2f} to ${found.usd * high:.2f}"
        lines.append(total + ".")
    lines.append("To spend less: --run N to cap the experiments, a lower --target, or Ctrl-C "
                 "at any time.")  # fmt: skip
    return lines


def confirm(lines: list[str], yes: bool, tty: bool | None = None,
            ask: Callable[..., bool] = click.confirm) -> None:  # fmt: skip
    """Print ``lines``, then ask to go on, unless ``yes``.

    Args:
        lines: The :func:`warning`.
        yes: ``--yes``: go on without asking.
        tty: Whether stdin is a terminal; None reads it.
        ask: Asks the question; ``click.confirm`` by default.

    Raises:
        click.UsageError: There is no terminal to ask on, and no ``--yes``.
        click.Abort: The answer is no.

    Examples:
        >>> confirm(["This calls LLMs: the narrator."], yes=True)  # says it, goes on
        >>> confirm(["x"], yes=False, tty=False)
        Traceback (most recent call last):
        click.exceptions.UsageError: there is no terminal to ask on. Add --yes (-y) to go on.
    """
    for line in lines:
        click.echo(line, err=True)
    if yes:
        return
    if not (sys.stdin.isatty() if tty is None else tty):
        raise click.UsageError("there is no terminal to ask on. Add --yes (-y) to go on.")
    if not ask("Continue?", default=False, err=True):
        raise click.Abort()
