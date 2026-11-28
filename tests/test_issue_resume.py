"""Finding the issues that an agent must resume itself (scripts/issue_resume.py, #7221)."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import issue_resume  # noqa: E402


def _gh(monkeypatch, payload):
    """Make ``gh`` return ``payload`` as JSON, and keep each command it gets."""
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    monkeypatch.setattr(issue_resume.subprocess, "run", run)
    return calls


def _pr(title, body, login="m", fork=False):
    return {"title": title, "body": body, "author": {"login": login}, "isCrossRepository": fork}


def test_open_pr_mentions_joins_each_open_prs_title_and_body(monkeypatch):
    calls = _gh(monkeypatch, [_pr("Fix #5", "Part of #6"), _pr("t", None)])
    text = issue_resume.open_pr_mentions("o/r", "m")
    assert issue_resume.has_open_pr(5, text) and issue_resume.has_open_pr(6, text)
    assert not issue_resume.has_open_pr(7, text)
    assert calls[0][:3] == ["gh", "pr", "list"] and "open" in calls[0]
    assert "author,isCrossRepository" in calls[0][-1]


def test_only_the_maintainers_own_prs_count_as_mentions(monkeypatch):
    pulls = [_pr("a", "Part of #5", login="stranger"), _pr("b", "Part of #6", fork=True),
             _pr("c", "Part of #7"), {"title": "d", "body": "#8", "author": None}]  # fmt: skip
    _gh(monkeypatch, pulls)
    text = issue_resume.open_pr_mentions("o/r", "m")
    assert [n for n in (5, 6, 7, 8) if issue_resume.has_open_pr(n, text)] == [7]


def test_view_issue_reads_the_issue_with_its_state(monkeypatch):
    calls = _gh(monkeypatch, {"number": 9, "state": "OPEN"})
    assert issue_resume.view_issue("o/r", 9) == {"number": 9, "state": "OPEN"}
    assert calls[0][:4] == ["gh", "issue", "view", "9"] and "state" in calls[0][-1]


def test_a_closed_claimed_issue_is_not_resumed():
    states = {"1": {"state": "In Progress"}}
    found = issue_resume.own_stalled(states, [1], "", lambda n: {"state": "CLOSED"},
                                     lambda n: False)  # fmt: skip
    assert found == []
