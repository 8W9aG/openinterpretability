"""``openinterp costs``: summarise what experiments cost, from ledgers of commands or from
the registry's records; ``openinterp usage``: record what an external answerer used."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import click

from openinterp.cli import cli
from openinterp.core.paths import experiments_dir


def read_ledger(paths) -> list[dict]:
    entries = []
    for path in paths:
        if not path.exists():
            continue  # an iteration that ran no openinterp command
        for line in path.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def read_session(path: Path) -> dict | None:
    """The result object of ``claude -p --output-format json``, or ``None``."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def summarise(entries: list[dict], sessions: list[dict]) -> str:
    commands: dict[str, list[float]] = defaultdict(list)
    stages: dict[str, list[float]] = defaultdict(list)
    for e in entries:
        commands[e["command"]].append(e["seconds"])
        for s in e.get("stages", []):
            stages[s["stage"]].append(s["seconds"])
    lines = _table("Commands", commands) + _table("Stages", stages)
    calls: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        for c in e.get("llm_calls", []):
            calls[(c["agent"], c["model"])].append(c)
    if calls:
        lines.append("LLM calls:")
        for (agent, model), cs in sorted(calls.items()):
            lines.append(f"  {agent:<9} {model:<34} ×{len(cs):<3} {_usage(cs)}")
    if sessions:
        lines.append(f"Driver: {_session(sessions)}")
    total = sum(e["seconds"] for e in entries)
    known = [c["cost_usd"] for cs in calls.values() for c in cs if c.get("cost_usd") is not None]
    known += [s["total_cost_usd"] for s in sessions if s.get("total_cost_usd") is not None]
    cost = f", ${sum(known):.2f} reported" if known else ""
    took = f"{total / 60:.1f} min" if total >= 60 else f"{total:.1f} s"
    lines.append(f"Total: {took} in openinterp commands{cost}")
    return "\n".join(lines)


def _table(title: str, groups: dict[str, list[float]]) -> list[str]:
    rows = sorted(groups.items(), key=lambda kv: -sum(kv[1]))
    head = [f"{title} (wall time, s):"] if rows else []
    return head + [f"  {name:<22} ×{len(secs):<3} {sum(secs):9.1f}" for name, secs in rows]


def _usage(calls: list[dict]) -> str:
    def total(key):
        values = [c[key] for c in calls if c.get(key) is not None]
        return sum(values) if values else None

    parts = []
    if (chars := total("prompt_chars")) is not None:
        parts.append(f"{chars / 1000:.0f}k prompt chars")
    prompt, completion = total("prompt_tokens"), total("completion_tokens")
    if prompt is not None or completion is not None:
        cached = total("cached_tokens")
        hit = f" ({cached:,} cached)" if cached else ""
        parts.append(f"{prompt or 0:,}{hit} + {completion or 0:,} tokens")
    if (cost := total("cost_usd")) is not None:
        parts.append(f"${cost:.3f}")
    if any(c.get("external") and c.get("prompt_tokens") is None for c in calls):
        parts.append("external without a usage file: no token counts")
    return ", ".join(parts)


def _session(sessions: list[dict]) -> str:
    usage: dict[str, int] = defaultdict(int)
    for s in sessions:
        for key, value in (s.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] += value
    minutes = sum(s.get("duration_ms") or 0 for s in sessions) / 60000
    turns = sum(s.get("num_turns") or 0 for s in sessions)
    read, written = usage["cache_read_input_tokens"], usage["cache_creation_input_tokens"]
    tokens = (
        f"{usage['input_tokens'] + read + written:,} input ({read:,} cache reads) + "
        f"{usage['output_tokens']:,} output tokens"
    )
    costs = [s["total_cost_usd"] for s in sessions if s.get("total_cost_usd") is not None]
    cost = f", ${sum(costs):.2f}" if costs else ""
    n = len(sessions)
    return f"{n} session{'s' * (n != 1)}, {minutes:.1f} min, {turns} turns, {tokens}{cost}"


def registry_costs(wiki_dir: Path) -> str:
    """Cost per experiment by role and model, over the records in ``wiki_dir``."""
    from openinterp.builder.compiler import load_records
    from openinterp.core.costs import by_agent

    records = load_records(experiments_dir(wiki_dir))
    rows = by_agent(
        (
            r.verdict.status if r.verdict else None,
            r.submission.provenance and r.submission.provenance.costs,
        )
        for r in records
    )
    if not rows:
        return f"No LLM calls recorded in {len(records)} records."

    def usd(v):
        return f"${v:.3f}" if v is not None else "–"

    lines = [
        f"LLM cost per experiment ({len(records)} records; priced = every call has a cost):",
        f"  {'role':<9} {'model':<40} {'priced':>11} {'mean':>8} {'per VERIFIED':>13}",
    ]
    for r in rows:
        priced = f"{r['priced']}/{r['experiments']}"
        lines.append(
            f"  {r['agent']:<9} {r['model']:<40} {priced:>11} {usd(r['mean_usd']):>8} "
            f"{usd(r['usd_per_verified']):>13}"
        )
    return "\n".join(lines)


@cli.command()
@click.argument("ledgers", nargs=-1, type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--session",
    "sessions",
    multiple=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="`claude -p --output-format json` output of the driving session (repeatable).",
)
@click.option(
    "--wiki-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Summarise the registry's records instead: LLM cost per experiment by role and "
    "model, mean and per VERIFIED result.",
)
def costs(ledgers, sessions, wiki_dir):
    """Summarise wall time and LLM usage from $OPENINTERP_COSTS_LOG ledgers, or the
    registry's LLM costs by role and model (--wiki-dir).

    Set OPENINTERP_COSTS_LOG=<file> and every openinterp command appends what it measured
    there: its wall time, its stages and its LLM calls. Records hold the same numbers for
    their own experiment, in `submission.provenance.costs`. Reported, never gated.
    """
    if wiki_dir is not None:
        if ledgers or sessions:
            raise click.UsageError("--wiki-dir summarises records; drop the ledgers.")
        click.echo(registry_costs(wiki_dir))
        return
    if not ledgers and not sessions:
        raise click.UsageError("Pass at least one ledger, --session file or --wiki-dir.")
    found = [s for s in (read_session(p) for p in sessions) if s is not None]
    click.echo(summarise(read_ledger(ledgers), found))


@cli.command()
@click.argument("agent")
@click.option(
    "--answer",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The answer the subagent wrote, e.g. exchange/ev.hypothesis.json.",
)
def usage(agent, answer):
    """Record a Claude Code subagent's token usage next to the answer it wrote.

    AGENT is the agent ID Claude Code reports when the subagent finishes, or the path of
    its transcript (~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl).
    Writes <answer>.usage.json, which `audit --hypothesis`, `review --probes` and
    `score --task` read to record the call's tokens and cost, priced at list price for
    the model they record (after $OPENINTERP_MODEL_ALIASES, so a proxied model is priced
    as the model that actually answered).
    """
    from openinterp.core.costs import usage_path
    from openinterp.core.transcripts import find_transcript, usage_from_transcript

    try:
        transcript = find_transcript(agent)
    except FileNotFoundError as exc:
        raise click.BadParameter(str(exc), param_hint="AGENT") from exc
    found = usage_from_transcript(transcript)
    out = usage_path(answer)
    out.write_text(found.model_dump_json(indent=2) + "\n")
    tokens = found.input_tokens + found.cache_read_tokens
    tokens += found.cache_write_5m_tokens + found.cache_write_1h_tokens
    click.echo(
        f"{found.model}: {found.requests} requests, {tokens:,} input "
        f"({found.cache_read_tokens:,} cache reads) + {found.output_tokens:,} output tokens. "
        f"Wrote {out}"
    )
