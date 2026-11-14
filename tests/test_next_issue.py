"""Choosing the next issue (scripts/next_issue.py)."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("next_issue", SCRIPTS / "next_issue.py")
next_issue = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(next_issue)


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
