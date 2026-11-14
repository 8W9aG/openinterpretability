"""Token usage of a Claude Code subagent, read from its transcript.

Claude Code keeps each subagent's transcript at
``~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl``, one JSON line per
message part. Every part of one API response repeats that response's ``usage``. The
last part carries the final output count. So we count each response once, from its
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
    """Claude Code's config directory: ``$CLAUDE_CONFIG_DIR``, else ``~/.claude``."""
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def find_transcript(agent: str) -> Path:
    """Return the transcript file of ``agent``.

    ``agent`` is a transcript path, or the agent ID that Claude Code reports when a
    subagent finishes (``a1b2c3…``, with or without the ``agent-`` prefix).
    """
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
    """Tokens per API request in a Claude Code transcript, for ``openinterp usage``.

    Claude Code writes one line per content block, so a request can have many lines. The
    last line of each request holds its final usage.
    """
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


def forbidden_reads(path: Path) -> tuple[list[str], list[str]]:
    """Blocked paths that the reviewer in transcript ``path`` read, and tried to read.

    The check is :func:`openinterp.core.isolation.forbidden`, for each tool call in the
    transcript. The first list has the calls that completed. The second list has the
    calls that failed, e.g. because the reviewer's hook denied them, so no content was
    read. Each list is in transcript order, without duplicates.
    """
    from openinterp.core.isolation import TOOLS, forbidden

    calls: dict[str, tuple[str, bool]] = {}  # tool_use id → (blocked path, completed)
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        content = (entry.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        cwd = entry.get("cwd") or "."
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") in TOOLS:
                hit = forbidden(block["name"], block.get("input") or {}, cwd, cwd)
                if hit is not None:
                    calls[block.get("id")] = (hit, True)
            elif (
                block.get("type") == "tool_result"
                and block.get("tool_use_id") in calls
                and block.get("is_error")
            ):
                calls[block["tool_use_id"]] = (calls[block["tool_use_id"]][0], False)
    read = [hit for hit, completed in calls.values() if completed]
    tried = [hit for hit, completed in calls.values() if not completed]
    return list(dict.fromkeys(read)), list(dict.fromkeys(tried))
