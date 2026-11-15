"""Tests for the priority and workstream rules in scripts/linear_sync.py."""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "linear_sync", Path(__file__).parents[1] / "scripts" / "linear_sync.py"
)
linear_sync = importlib.util.module_from_spec(_SPEC)
sys.modules["linear_sync"] = linear_sync  # dataclasses look the module up by name
_SPEC.loader.exec_module(linear_sync)

WORKSTREAMS = {"Why neurons rest", "Testing"}


def _issue(**kw):
    base = dict(
        url="u",
        linear_id="l",
        linear_priority=0,
        linear_project=None,
        gh_labels=[],
        roadmap_item="item",
        roadmap_priority=None,
        roadmap_workstream=None,
    )
    return linear_sync.Linked(**{**base, **kw})


def test_in_sync_issue_needs_nothing():
    issue = _issue(
        linear_priority=2,
        linear_project="Testing",
        gh_labels=["priority: high"],
        roadmap_priority="High",
        roadmap_workstream="Testing",
    )
    assert linear_sync.plan(issue, WORKSTREAMS).empty()


def test_linear_priority_wins_over_label():
    issue = _issue(linear_priority=1, gh_labels=["priority: low", "bug"])
    act = linear_sync.plan(issue, WORKSTREAMS)
    assert act.add_labels == ["priority: urgent"]
    assert act.remove_labels == ["priority: low"]
    assert act.linear_priority is None
    assert act.roadmap_priority == "Urgent"


def test_label_sets_missing_linear_priority():
    act = linear_sync.plan(_issue(gh_labels=["priority: medium"]), WORKSTREAMS)
    assert act.linear_priority == 3
    assert act.add_labels == [] and act.remove_labels == []


def test_label_case_is_normalised():
    act = linear_sync.plan(_issue(gh_labels=["priority: Medium"]), WORKSTREAMS)
    assert act.linear_priority == 3
    assert act.add_labels == ["priority: medium"] and act.remove_labels == ["priority: Medium"]


def test_roadmap_workstream_sets_missing_linear_project():
    act = linear_sync.plan(_issue(roadmap_workstream="Why neurons rest"), WORKSTREAMS)
    assert act.linear_project == "Why neurons rest"
    assert act.roadmap_workstream is None


def test_linear_project_wins_and_unknown_workstream_is_not_set():
    issue = _issue(linear_project="Testing", roadmap_workstream="Why neurons rest")
    act = linear_sync.plan(issue, WORKSTREAMS)
    assert act.linear_project is None and act.roadmap_workstream == "Testing"
    act = linear_sync.plan(_issue(linear_project="New project"), WORKSTREAMS)
    assert act.roadmap_workstream is None


def test_issue_missing_from_roadmap_is_added():
    assert linear_sync.plan(_issue(roadmap_item=None), WORKSTREAMS).add_to_roadmap


def test_no_priority_anywhere_leaves_labels_alone():
    act = linear_sync.plan(_issue(gh_labels=["bug"]), WORKSTREAMS)
    assert act.linear_priority is None and act.add_labels == [] and act.roadmap_priority is None


def test_released_early_without_a_published_milestone():
    published = {"v0.0.19"}
    assert linear_sync.released_early("v0.0.20", published)
    assert linear_sync.released_early(None, published)
    assert not linear_sync.released_early("v0.0.19", published)


def test_unrelease_moves_only_issues_whose_release_is_not_published(monkeypatch):
    issues = [("a", "OPE-1", "u1"), ("b", "OPE-2", "u2"), ("c", "OPE-3", "u3")]
    monkeypatch.setattr(linear_sync, "released_issues", lambda: issues)
    monkeypatch.setattr(linear_sync, "published_releases", lambda: {"v0.0.19"})
    milestones = {"u1": "v0.0.19", "u2": "v0.0.20", "u3": None}
    monkeypatch.setattr(linear_sync, "github_milestones", lambda: milestones)
    monkeypatch.setattr(linear_sync.issue_state, "state_id", lambda name: f"id-{name}")
    calls = []
    monkeypatch.setattr(linear_sync, "linear", lambda q, v=None: calls.append(v))
    assert linear_sync.unrelease(dry=False) == 2
    assert calls == [
        {"id": "b", "i": {"stateId": "id-Done"}},
        {"id": "c", "i": {"stateId": "id-Done"}},
    ]
    calls.clear()
    assert linear_sync.unrelease(dry=True) == 2
    assert calls == []
