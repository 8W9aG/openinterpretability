"""Choosing the next issue (scripts/next_issue.py)."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("next_issue", SCRIPTS / "next_issue.py")
next_issue = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(next_issue)
ME = {"login": next_issue.issue_state.MAINTAINER}
OPEN_BLOCKERS = next_issue.open_blockers  # the real one, before the fixture replaces it


@pytest.fixture(autouse=True)
def no_comments(monkeypatch):
    """No test reads GitHub: every issue has no comments and no blockers."""
    monkeypatch.setattr(next_issue, "comments_of", lambda number: [])
    monkeypatch.setattr(next_issue, "open_blockers", lambda number: [])


def _issue(number, area, prio=None, created="2026-11-01", extra=()):
    names = [f"area: {area}", *extra] + ([f"priority: {prio}"] if prio else [])
    return {
        "number": number,
        "title": f"t{number}",
        "createdAt": created,
        "labels": [{"name": n} for n in names],
    }


def test_linear_priority_wins_and_the_label_fills_a_gap():
    issue = _issue(1, "site", "low")
    assert next_issue.priority(issue, {"priority": 2}) == 2
    assert next_issue.priority(issue, {"priority": 0}) == 4
    assert next_issue.priority(_issue(2, "site"), {}) == next_issue.NO_PRIORITY


def test_rank_orders_by_priority_then_spreads_across_areas():
    issues = [
        _issue(1, "lit", "high", "2026-11-01"),
        _issue(2, "lit", "high", "2026-11-02"),
        _issue(3, "site", "high", "2026-11-03"),
        _issue(4, "pipe", "urgent", "2026-11-04"),
    ]
    ranked = next_issue.rank(issues, {}, Counter({"site": 1}))
    assert [i["number"] for i in ranked] == [4, 1, 3, 2]


def test_started_and_proposed_issues_are_skipped():
    issues = [_issue(1, "a"), _issue(2, "a"), _issue(3, "a", extra=("proposed",))]
    states = {"1": {"state": "In Progress"}, "2": {"state": "Todo"}}
    assert [i["number"] for i in next_issue.unstarted(issues, states)] == [2]


def test_pick_skips_ineligible_issues(monkeypatch):
    monkeypatch.setattr(next_issue.issue_state, "eligibility", lambda n: (n != "1", ""))
    best, others = next_issue.pick([_issue(1, "a"), _issue(2, "a"), _issue(3, "a")])
    assert best["number"] == 2 and [i["number"] for i in others] == [3]


def test_choose_pulls_from_the_backlog_when_the_release_has_nothing(monkeypatch):
    backlog = [_issue(7, "models", "urgent"), _issue(8, "site", "medium")]
    monkeypatch.setattr(next_issue, "next_milestone", lambda: "v9.9.9")
    monkeypatch.setattr(next_issue, "linear_states", dict)
    monkeypatch.setattr(next_issue, "open_issues", lambda m: [] if m else backlog)
    monkeypatch.setattr(next_issue, "gh_json", lambda *a: [])
    monkeypatch.setattr(next_issue.issue_state, "eligibility", lambda n: (True, ""))
    result = next_issue.choose()
    assert result["pulled"] and result["pick"]["number"] == 8  # area: models is held back


def test_unprioritised_finds_issues_with_neither_source():
    issues = [_issue(1, "a", "low"), _issue(2, "a"), _issue(3, "a")]
    states = {"3": {"priority": 2}}
    assert [i["number"] for i in next_issue.unprioritised(issues, states)] == [2]


def test_set_priority_sets_the_label_and_linear(monkeypatch):
    calls, queries = [], []
    monkeypatch.setattr(next_issue.subprocess, "run", lambda args, **k: calls.append(args))

    def linear(query, variables):
        queries.append(variables)
        return {"attachmentsForURL": {"nodes": [{"issue": {"id": "L1"}}]}}

    monkeypatch.setattr(next_issue.issue_state, "linear", linear)
    assert next_issue.set_priority("7", "high") == "#7: priority: high"
    assert calls[0][-1] == "priority: high" and queries[-1] == {"id": "L1", "p": 2}


def _comments(*bodies):
    return [{"author": ME, "body": b} for b in bodies]


def test_a_reply_after_an_agent_comment_is_an_answer():
    asked = _comments("<!-- linear-linkback -->", "**Plan**\n\n…", "**Deviation**\n\nA, B or C?")
    assert next_issue.waiting_answer(asked) is None  # still waiting for the maintainer
    assert next_issue.waiting_answer([*asked, {"author": ME, "body": "2"}]) == "2"
    acted = [
        *asked,
        {"author": ME, "body": "2"},
        {"author": ME, "body": "**Deviation**\n\nThe maintainer chose 2."},
    ]
    assert next_issue.waiting_answer(acted) is None  # an agent has acted on it


def test_a_comment_with_no_agent_comment_before_it_is_not_an_answer():
    assert next_issue.waiting_answer(_comments("Working on this.")) is None
    assert next_issue.waiting_answer(_comments("a note from the maintainer")) is None
    assert next_issue.waiting_answer([]) is None
    merged = _comments("**Shipped** in #1.", "Merged in #1 as abc.")
    assert next_issue.waiting_answer(merged) is None


def test_an_issue_whose_newest_comment_is_a_question_waits():
    question = {"author": ME, "body": "**Question**\n\nWhich way?"}
    assert next_issue.asked([{"author": ME, "body": "**Plan**"}, question])
    assert not next_issue.asked([question, {"author": ME, "body": "2"}])  # answered
    assert not next_issue.asked([])


def test_pick_skips_an_issue_that_waits_on_an_answer(monkeypatch):
    waiting = {"1": [{"author": ME, "body": "**Question** which?"}]}
    monkeypatch.setattr(next_issue, "comments_of", lambda n: waiting.get(n, []))
    monkeypatch.setattr(next_issue.issue_state, "eligibility", lambda n: (True, ""))
    best, others = next_issue.pick([_issue(1, "a"), _issue(2, "a")])
    assert best["number"] == 2 and others == []


def test_a_release_of_waiting_issues_falls_through_to_the_backlog(monkeypatch):
    release, backlog = [_issue(5, "pipe", "high")], [_issue(8, "site", "medium")]
    monkeypatch.setattr(next_issue, "next_milestone", lambda: "v9.9.9")
    monkeypatch.setattr(next_issue, "linear_states", dict)
    monkeypatch.setattr(next_issue, "open_issues", lambda m: release if m else backlog)
    monkeypatch.setattr(next_issue, "gh_json", lambda *a: [])
    monkeypatch.setattr(next_issue.issue_state, "eligibility", lambda n: (True, ""))
    monkeypatch.setattr(
        next_issue,
        "comments_of",
        lambda n: [{"author": ME, "body": "**Question**"}] if n == "5" else [],
    )
    result = next_issue.choose()
    assert result["pulled"] and result["pick"]["number"] == 8


def test_only_the_maintainers_login_can_answer_or_hold_an_issue():
    """Anyone can comment on a public issue: an outsider's text is never the maintainer's."""
    outsider = {"login": "someone-else"}
    question = {"author": ME, "body": "**Question**\n\nWhich way?"}
    assert next_issue.waiting_answer([question, {"author": outsider, "body": "2"}]) is None
    assert next_issue.waiting_answer([question, {"author": ME, "body": "2"}]) == "2"
    assert not next_issue.asked([{"author": outsider, "body": "**Question** hold it"}])
    # An outsider's '**' comment after the maintainer's reply doesn't hide the reply.
    hidden = [question, {"author": ME, "body": "1"}, {"author": outsider, "body": "**Plan**"}]
    assert next_issue.waiting_answer(hidden) == "1"


def test_pick_skips_an_issue_with_an_open_blocker(monkeypatch):
    """An open "blocked by" issue holds a candidate back, and says which issue (#3842)."""
    monkeypatch.setattr(next_issue.issue_state, "eligibility", lambda n: (True, ""))
    monkeypatch.setattr(next_issue, "open_blockers", lambda n: [9] if n == "1" else [])
    blocked: dict = {}
    best, others = next_issue.pick([_issue(1, "site"), _issue(2, "site")], blocked)
    assert best["number"] == 2 and others == [] and blocked == {"1": [9]}


def test_open_blockers_keeps_only_open_ones(monkeypatch):
    nodes = [{"number": 9, "state": "OPEN"}, {"number": 8, "state": "CLOSED"}]
    data = {"data": {"repository": {"issue": {"blockedBy": {"nodes": nodes}}}}}
    monkeypatch.setattr(next_issue, "gh_json", lambda *a: data)
    assert OPEN_BLOCKERS("1") == [9]


def test_muse_issues_are_never_picked_ranked_or_answered(monkeypatch):
    issues = [_issue(1, "a"), _issue(2, "a", extra=("muse",))]
    assert [i["number"] for i in next_issue.unstarted(issues, {})] == [1]
    assert [i["number"] for i in next_issue.unprioritised(issues, {})] == [1]
    replied = _comments("**Question**\n\nWhich?", "1")
    views = {
        n: i | {"state": "OPEN", "comments": replied}
        for n, i in (("1", issues[0]), ("2", issues[1]))
    }
    monkeypatch.setattr(next_issue, "gh_json", lambda *args: dict(views[args[2]]))
    states = {"1": {"state": "In Progress"}, "2": {"state": "In Progress"}}
    assert [i["number"] for i in next_issue.answered(states)] == [1]
