"""Claude Code transcripts: odd timestamps, paths given directly and odd content."""

import json

from openinterpretability.core.transcripts import (
    find_transcript,
    forbidden_reads,
    usage_from_transcript,
)


def test_a_transcript_path_is_used_as_it_is(tmp_path):
    path = tmp_path / "anything.jsonl"
    path.write_text("")
    assert find_transcript(str(path)) == path


def test_a_request_with_a_bad_timestamp_has_no_time(tmp_path):
    usage = {"input_tokens": 3, "output_tokens": 7}
    entry = {"type": "assistant", "timestamp": "yesterday",
             "message": {"id": "m1", "model": "claude-x", "usage": usage}}  # fmt: skip
    path = tmp_path / "agent-a.jsonl"
    path.write_text(json.dumps(entry) + "\n")
    (request,) = usage_from_transcript(path).per_request
    assert request.at is None and (request.input_tokens, request.output_tokens) == (3, 7)


def test_forbidden_reads_skip_text_content_and_blocks_that_are_not_objects(tmp_path):
    (tmp_path / "submissions").mkdir()
    use = {"type": "tool_use", "id": "t0", "name": "Read",
           "input": {"file_path": "submissions/s.json"}}  # fmt: skip
    lines = [
        {"type": "user", "cwd": str(tmp_path), "message": {"content": "plain text"}},
        {"type": "assistant", "cwd": str(tmp_path), "message": {"content": ["text", use]}},
    ]
    path = tmp_path / "agent-b.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert forbidden_reads(path) == (["submissions/s.json"], [])
