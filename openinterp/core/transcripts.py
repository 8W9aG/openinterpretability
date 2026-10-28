"""Token usage of a Claude Code subagent, read from its transcript.

Claude Code keeps each subagent's transcript at
``~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl``, one JSON line per
message part. Every part of one API response repeats that response's ``usage``, and the
last part carries the final output count, so each response is counted once, from its
last line.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from openinterp.core.costs import ExternalUsage, RequestUsage


def _time(value) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def claude_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def find_transcript(agent: str) -> Path:
    """``agent`` is a transcript path, or the agent ID Claude Code reports when a subagent
    finishes (``a1b2c3…``, with or without the ``agent-`` prefix)."""
    path = Path(agent).expanduser()
    if path.suffix == ".jsonl" and path.is_file():
        return path
    name = f"agent-{agent.removeprefix('agent-')}.jsonl"
    found = sorted(
        claude_dir().glob(f"projects/*/*/subagents/{name}"), key=lambda p: p.stat().st_mtime
    )
    if not found:
        raise FileNotFoundError(f"No Claude Code transcript {name} under {claude_dir()}.")
    return found[-1]  # the newest, if a resumed session copied it


def usage_from_transcript(path: Path) -> ExternalUsage:
    last: dict[str, tuple] = {}  # API response id → (time, its last line's message)
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a line cut off by a crash
        message = entry.get("message") if entry.get("type") == "assistant" else None
        if not isinstance(message, dict) or not message.get("usage"):
            continue
        if message.get("model", "").startswith("<"):  # e.g. "<synthetic>" error stubs
            continue
        at = _time(entry.get("timestamp"))
        last[message.get("id") or entry.get("uuid")] = (at, message)

    requests, models = [], {}
    for at, message in last.values():
        u = message["usage"]
        write_1h = (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
        requests.append(
            RequestUsage(
                at=at,
                input_tokens=u.get("input_tokens") or 0,
                cache_read_tokens=u.get("cache_read_input_tokens") or 0,
                cache_write_5m_tokens=max(
                    (u.get("cache_creation_input_tokens") or 0) - write_1h, 0
                ),
                cache_write_1h_tokens=write_1h,
                output_tokens=u.get("output_tokens") or 0,
            )
        )
        model = message.get("model") or "unknown"
        models[model] = models.get(model, 0) + 1
    return ExternalUsage.of(
        requests,
        model=",".join(sorted(models, key=lambda m: -models[m])) or None,
        source=f"claude-code:{path.stem}",
    )
