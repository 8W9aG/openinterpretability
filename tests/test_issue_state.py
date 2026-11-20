"""Which issues an agent may work on (scripts/issue_state.py)."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "issue_state", Path(__file__).parents[1] / "scripts" / "issue_state.py"
)
issue_state = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(issue_state)


def _stub(monkeypatch, author, title="Something", thumbs=(), labels=(), edits=()):
    def gh_api(path):
        if path.endswith("/reactions?per_page=100"):
            when = "2026-11-10T00:00:00Z"
            return [{"content": "+1", "user": {"login": u}, "created_at": when} for u in thumbs]
        return {
            "user": {"login": author},
            "title": title,
            "reactions": {"+1": len(thumbs)},
            "labels": [{"name": n} for n in labels],
        }

    monkeypatch.setattr(issue_state, "gh_api", gh_api)
    monkeypatch.setattr(issue_state, "edits", lambda number: sorted(edits))


@pytest.mark.parametrize("author", ["8W9aG", "dependabot[bot]", "github-advanced-security[bot]"])
def test_trusted_authors_are_eligible(monkeypatch, author):
    _stub(monkeypatch, author)
    assert issue_state.eligibility("1")[0]


@pytest.mark.parametrize("title", ["Nightly reproduction failed", "Deploy registry failed on main"])
def test_workflow_issue_needs_a_workflows_title(monkeypatch, title):
    _stub(monkeypatch, "github-actions[bot]", title=title)
    assert issue_state.eligibility("1")[0]
    _stub(monkeypatch, "github-actions[bot]", title="Something else")
    assert not issue_state.eligibility("1")[0]


def test_outside_issue_needs_the_maintainers_thumbs_up(monkeypatch):
    _stub(monkeypatch, "someone")
    assert not issue_state.eligibility("1")[0]
    _stub(monkeypatch, "someone", thumbs=["another-person"])
    assert not issue_state.eligibility("1")[0]
    _stub(monkeypatch, "someone", thumbs=["another-person", "8W9aG"])
    assert issue_state.eligibility("1") == (True, "opened by someone, thumbs up from 8W9aG")


def test_an_edit_by_someone_else_after_the_thumbs_up_needs_a_new_approval(monkeypatch):
    before, after = "2026-11-09T00:00:00Z", "2026-11-11T00:00:00Z"
    _stub(monkeypatch, "someone", thumbs=["8W9aG"], edits=[(before, "someone")])
    assert issue_state.eligibility("1")[0]
    _stub(monkeypatch, "someone", thumbs=["8W9aG"], edits=[(after, "8W9aG")])
    assert issue_state.eligibility("1")[0]
    _stub(monkeypatch, "someone", thumbs=["8W9aG"], edits=[(before, "8W9aG"), (after, "someone")])
    ok, why = issue_state.eligibility("1")
    assert not ok and "edited by someone after" in why


def test_titles_are_read_from_the_workflows():
    assert issue_state.automatic_titles() == {
        "Nightly reproduction failed",
        "Deploy registry failed on main",
        "Posting to X failed on main",
    }


def test_in_progress_is_refused_for_an_ineligible_issue(monkeypatch):
    _stub(monkeypatch, "someone")
    monkeypatch.setattr("sys.argv", ["issue_state.py", "7", "In Progress"])
    with pytest.raises(SystemExit, match="Refused"):
        issue_state.main()


def test_proposed_issues_wait_for_triage(monkeypatch):
    _stub(monkeypatch, "8W9aG", labels=["proposed", "area: pipeline"])
    allowed, reason = issue_state.eligibility("1")
    assert not allowed and "triaged" in reason


def test_release_moves_only_closed_issues(monkeypatch, capsys):
    monkeypatch.setattr(
        issue_state, "milestone_issues", lambda tag: [("1", "CLOSED"), ("2", "OPEN")]
    )
    moved = []
    monkeypatch.setattr(issue_state, "set_state", lambda n, s: moved.append((n, s)) or f"#{n}: {s}")
    assert issue_state.release("v1")
    assert moved == [("1", "Released")]
    assert "#2: still open" in capsys.readouterr().out
    assert issue_state.release("v1", dry_run=True) and len(moved) == 1


def test_release_reports_a_failure(monkeypatch):
    monkeypatch.setattr(issue_state, "milestone_issues", lambda tag: [("1", "CLOSED")])

    def fail(n, s):
        raise RuntimeError("no such state")

    monkeypatch.setattr(issue_state, "set_state", fail)
    assert not issue_state.release("v1")
