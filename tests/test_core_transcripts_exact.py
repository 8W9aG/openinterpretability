"""Transcript usage, transcript lookup and corpus names, at their edges (#6491)."""

import json
import os
from datetime import UTC, datetime

import pytest

from openinterpretability.core.corpora import CorpusSource
from openinterpretability.core.transcripts import (
    claude_dir,
    find_transcript,
    forbidden_reads,
    usage_from_transcript,
)


def _write(path, lines):
    path.write_text("\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines) + "\n")
    return path


def _reply(id_, usage, model="claude-x", **entry):
    message = {
        "usage": usage,
        **({"id": id_} if id_ else {}),
        **({"model": model} if model else {}),
    }
    return {"type": "assistant", "message": message, **entry}


def test_claude_dir_is_dot_claude_in_home_without_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert claude_dir() == tmp_path / ".claude"


def _subagent(root, session, name, mtime):
    path = root / "projects" / "p" / session / "subagents" / name
    path.parent.mkdir(parents=True)
    path.write_text("")
    os.utime(path, (mtime, mtime))
    return path


def test_find_transcript_takes_the_newest_copy_with_or_without_the_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    newest = _subagent(tmp_path, "s1", "agent-abc.jsonl", 2_000_000)
    _subagent(tmp_path, "s2", "agent-abc.jsonl", 1_000_000)
    assert find_transcript("agent-abc") == newest
    assert find_transcript("abc") == newest


def test_an_existing_file_that_is_not_jsonl_is_an_agent_id(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    path = tmp_path / "notes.txt"
    path.write_text("")
    with pytest.raises(FileNotFoundError) as error:
        find_transcript(str(path))
    assert str(error.value) == (
        f"No Claude Code transcript agent-{path}.jsonl under {tmp_path / 'config'}."
    )


def test_usage_reads_exact_tokens_time_and_models(tmp_path):
    lines = [
        "{cut off",
        {"type": "assistant", "message": {"model": "<synthetic>", "usage": {"input_tokens": 9}}},
        {"type": "user", "message": {"id": "u", "model": "claude-x", "usage": {"input_tokens": 9}}},
        _reply("m1", {"output_tokens": 5}, model="claude-a", timestamp="2026-01-02T03:04:05Z"),
        _reply(None, {"input_tokens": 2}, model=None, uuid="u1"),
        _reply(
            None, {"input_tokens": 4, "cache_read_input_tokens": 3}, model="claude-b", uuid="u2"
        ),  # fmt: skip
        _reply(
            "m2",
            {
                "cache_creation_input_tokens": 10,
                "output_tokens": 1,
                "cache_creation": {"ephemeral_1h_input_tokens": 4},
            },
            model="claude-b",
        ),  # fmt: skip
    ]
    usage = usage_from_transcript(_write(tmp_path / "agent-z.jsonl", lines))
    got = [r.model_dump() for r in usage.per_request]
    zero = dict.fromkeys(
        ["input_tokens", "cache_read_tokens", "cache_write_5m_tokens", "cache_write_1h_tokens",
         "output_tokens"], 0)  # fmt: skip
    at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert got == [
        {**zero, "at": at, "output_tokens": 5},
        {**zero, "at": None, "input_tokens": 2},
        {**zero, "at": None, "input_tokens": 4, "cache_read_tokens": 3},
        {
            **zero,
            "at": None,
            "cache_write_5m_tokens": 6,
            "cache_write_1h_tokens": 4,
            "output_tokens": 1,
        },  # fmt: skip
    ]
    assert (usage.model, usage.requests, usage.source) == (
        "claude-b,claude-a,unknown",
        4,
        "claude-code:agent-z",
    )


def test_usage_of_an_empty_transcript_has_no_model(tmp_path):
    usage = usage_from_transcript(_write(tmp_path / "agent-e.jsonl", ["{}"]))
    assert (usage.model, usage.requests) == (None, 0)


def test_forbidden_reads_skip_bad_lines_other_blocks_and_use_dot_without_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    read = {"file_path": "submissions/s.json"}
    lines = [
        "{cut off",
        {
            "message": {
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "t0",
                        "name": "Read",
                        "input": {"file_path": "submissions/other.json"},
                    }
                ]
            }
        },  # fmt: skip
        {"message": {"content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": read}]}},
    ]
    assert forbidden_reads(_write(tmp_path / "agent-r.jsonl", lines)) == (
        ["submissions/s.json"],
        [],
    )


def _source(**kw):
    return CorpusSource(
        dataset="d/x", files="*.parquet", documents=3, seed=7, max_chars=9, sha256="0", **kw
    )


def test_corpus_source_names_the_short_revision_and_chat_template():
    source = _source(revision="0123456789abcdef", chat_template="m/chat")
    assert str(source) == "d/x@01234567 (*.parquet, 3 docs, seed 7, m/chat's chat template)"
