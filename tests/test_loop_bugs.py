"""Bugs that an experiment loop files (openinterpretability/loop_bugs.py, #3261), with a fake gh."""

import json
import subprocess

import pytest
from click.testing import CliRunner

from openinterpretability import loop_bugs
from openinterpretability.cli import cli

ERROR = 'Traceback:\n  File "/Users/alice/dev/x.py", line 3\nRuntimeError: verify failed at 42\n'


def _fake(prefix: str, body: str) -> str:
    """A token-shaped string, joined at run time so the secret scanner sees no token."""
    return prefix + body


def test_scrub_removes_secrets_emails_and_home_paths():
    tokens = [
        _fake("ghp" + "_", "abcdefghijklmnopqrstuvwxyz0123"),
        _fake("hf" + "_", "abcdefghijklmnopqrstuv"),
        _fake("sk" + "-ant-", "abcdefghijklmnop1234"),
        _fake("AK" + "IA", "ABCDEFGHIJKLMNOP"),
    ]
    text = (
        " ".join(tokens) + " HF_TOKEN=abc123 Authorization: Bearer abcdefghijklmnop "
        "alice@example.com /Users/alice/dev/repo /home/bob/x"
    )
    out = loop_bugs.scrub(text)
    for secret in ("ghp_", "hf_abc", "sk-ant", "abc123", "abcdefghijklmnop", "alice@", "alice",
                   "bob", "AKIA"):  # fmt: skip
        assert secret not in out, secret
    assert "HF_TOKEN=<redacted>" in out and "~/dev/repo" in out and "~/x" in out


def test_one_bug_keeps_its_fingerprint_across_paths_numbers_and_options():
    a = loop_bugs.fingerprint("uv run open-interpretability verify --a 1", ERROR)
    b = loop_bugs.fingerprint(
        "uv run --quiet open-interpretability verify --b 2",
        ERROR.replace("42", "7").replace("alice", "carol"),
    )
    other = loop_bugs.fingerprint("uv run open-interpretability audit", ERROR)
    assert a == b != other and len(a) == 12


class FakeGh:
    """Records gh calls; the issue list holds ``existing`` issues."""

    def __init__(self, existing=()):
        self.calls, self.existing = [], list(existing)

    def __call__(self, args, **_):
        self.calls.append(args)
        out = ""
        if args[1:3] == ["issue", "list"]:
            out = json.dumps(self.existing)
        elif args[1:3] == ["issue", "create"]:
            out = "https://github.com/o/r/issues/99\n"
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")


COMMAND = "uv run open-interpretability verify --x /Users/a/b"
REPORT = loop_bugs.Report("verify fails", COMMAND, ERROR, "experiment/x", "M/L1/N2-positive")


def test_a_new_bug_is_filed_with_labels_scrubbed_text_and_its_fingerprint(monkeypatch):
    gh = FakeGh()
    monkeypatch.setattr(loop_bugs.subprocess, "run", gh)
    assert loop_bugs.file(REPORT).startswith("filed https://github.com/o/r/issues/99")
    create = gh.calls[-1]
    body = create[create.index("--body") + 1]
    assert "/Users/" not in body and "alice" not in body
    assert f"{loop_bugs.MARKER}: {loop_bugs.fingerprint(REPORT.command, REPORT.error)}" in body
    assert body.startswith("**Loop bug**") and "```text" in body
    assert all(label in create for label in loop_bugs.LABELS)


def test_the_same_bug_again_becomes_a_comment(monkeypatch):
    mark = loop_bugs.fingerprint(REPORT.command, REPORT.error)
    gh = FakeGh([{"number": 7, "body": f"text <!-- {loop_bugs.MARKER}: {mark} -->"}])
    monkeypatch.setattr(loop_bugs.subprocess, "run", gh)
    assert loop_bugs.file(REPORT).startswith("commented on #7")
    assert gh.calls[-1][1:4] == ["issue", "comment", "7"]
    assert not any(c[1:3] == ["issue", "create"] for c in gh.calls)


def test_dry_run_posts_nothing(monkeypatch):
    def refuse(*_, **__):
        pytest.fail("gh was called")

    monkeypatch.setattr(loop_bugs.subprocess, "run", refuse)
    result = CliRunner().invoke(
        cli, ["loop-bug", "--title", "t", "--command", "uv run x", "--dry-run"], input=ERROR
    )
    assert result.exit_code == 0 and "would post" in result.output and "alice" not in result.output
