"""The read isolation of agents and circuit decisions, at their edges (#6491)."""

from __future__ import annotations

import io
import json
import os
import sys

import pytest

from openinterpretability.core import isolation
from openinterpretability.core.circuit_records import (
    CircuitMeasures,
    CircuitPolicy,
    decide,
    null_threshold,
)

MESSAGE = (
    "Reviewer isolation (docs/PROTOCOL.md §2): {hit} is blocked. Records, "
    "submissions, exchange files other than prompts and probe sets, logs and the "
    "built site hold probe activations. Read only your prompt and the skills."
)


def _hook(monkeypatch, capsys, event: dict) -> str:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    assert isolation.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out


def _real(path) -> str:
    return os.path.realpath(path)


def test_the_exchange_directory_itself_is_blocked_when_its_name_ends_as_a_prompt(tmp_path):
    # The exchange directory resolves to a directory with a reviewer file's name end.
    target = tmp_path / "x.prompt.md"
    target.mkdir()
    (tmp_path / "exchange").symlink_to(target)
    root = isolation._resolve(tmp_path, tmp_path)
    assert isolation.blocked_file(isolation._resolve("exchange", root), root) is True
    assert isolation.blocked_file(isolation._resolve("exchange/a.prompt.md", root), root) is False


def test_a_search_from_a_missing_directory_that_holds_a_blocked_one_is_blocked(tmp_path):
    root = tmp_path / "project"  # it does not exist, and its name is not a reviewer file
    assert isolation.blocked_search(root, root) is True


def test_a_search_of_one_blocked_file_is_blocked(tmp_path):
    (tmp_path / "exchange").mkdir()
    log = tmp_path / "exchange" / "review.log"
    log.write_text("x")
    root = isolation._resolve(tmp_path, tmp_path)
    assert isolation.blocked_search(isolation._resolve(log, root), root) is True
    hit = isolation.forbidden("Grep", {"path": str(log)}, tmp_path, tmp_path)
    assert hit == _real(log)


def test_an_edit_of_a_blocked_file_is_forbidden(tmp_path):
    path = str(tmp_path / "logs" / "run.log")
    assert isolation.forbidden("Edit", {"file_path": path}, tmp_path, tmp_path) == path
    assert isolation.forbidden("edit", {"file_path": path}, tmp_path, tmp_path) is None


def test_the_hook_takes_the_root_from_the_project_directory(tmp_path, monkeypatch, capsys):
    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    project.mkdir()
    elsewhere.mkdir()
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))
    path = str(project / "exchange" / "review.log")
    event = {"cwd": str(elsewhere), "tool_name": "Read", "tool_input": {"file_path": path}}
    out = _hook(monkeypatch, capsys, event)
    reason = MESSAGE.format(hit=path)
    decision = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }
    assert out == json.dumps({"hookSpecificOutput": decision}) + "\n"


def test_the_hook_falls_back_to_the_working_directory_as_root(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    path = str(tmp_path / "logs" / "a.log")
    event = {"cwd": str(tmp_path), "tool_name": "Write", "tool_input": {"file_path": path}}
    out = _hook(monkeypatch, capsys, event)
    assert json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"] == (
        MESSAGE.format(hit=path)
    )


def test_the_hook_allows_an_event_with_no_tool_name(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    event = {"cwd": str(tmp_path), "tool_input": {"file_path": str(tmp_path / "logs" / "a")}}
    assert _hook(monkeypatch, capsys, event) == ""


def _measures(necessity: float, nulls: list[float], negatives: dict[str, float]):
    return CircuitMeasures(
        holdout_faithfulness=1.0,
        holdout_necessity=necessity,
        null_necessities=nulls,
        negative_necessities=negatives,
    )


@pytest.mark.parametrize(
    ("nulls", "quantile", "expected"),
    [
        ([], 0.5, 0.0),
        ([2.0, 1.0], 0.5, 1.0),
        ([2.0, 1.0], 0.1, 1.0),
        ([2.0, 1.0], 1.0, 2.0),
        ([2.0, 1.0], 2.0, 2.0),
        ([3.0, 1.0, 2.0], 0.7, 2.0),
    ],
)
def test_the_null_threshold_is_the_quantile_clamped_to_the_list(nulls, quantile, expected):
    assert null_threshold(nulls, quantile) == expected


def test_a_negative_task_at_the_limit_fails():
    policy = CircuitPolicy(max_negative_necessity=0.2, min_specificity=0.5)
    status, gates = decide(_measures(0.5, [0.1], {"a": 0.2}), policy)
    assert (status, gates) == (
        "NEEDS_REVISION",
        {"necessity_floor": True, "beats_null": True, "specificity": False},
    )
    status, gates = decide(_measures(0.5, [0.1], {"a": -0.19}), policy)
    assert status == "VERIFIED"


def test_no_negative_tasks_gives_no_specificity():
    policy = CircuitPolicy(min_specificity=0.0)
    status, gates = decide(_measures(0.5, [0.1], {}), policy)
    assert (status, gates["specificity"]) == ("VERIFIED", True)
    status, gates = decide(_measures(0.5, [0.1], {}), CircuitPolicy(min_specificity=0.5))
    assert (status, gates["specificity"]) == ("NEEDS_REVISION", False)


def test_specificity_is_the_share_of_negative_tasks_that_pass():
    policy = CircuitPolicy(min_specificity=0.6)
    _, gates = decide(_measures(0.5, [0.1], {"a": 0.0, "b": 0.9}), policy)
    assert gates["specificity"] is False
    policy = CircuitPolicy(min_specificity=0.8)
    negatives = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.9}
    _, gates = decide(_measures(0.5, [0.1], negatives), policy)
    assert gates["specificity"] is True


def test_the_necessity_floor_includes_its_limit_and_the_null_excludes_its_own():
    policy = CircuitPolicy(min_necessity=0.25)
    status, gates = decide(_measures(0.25, [0.1], {"a": 0.0}), policy)
    assert (status, gates["necessity_floor"]) == ("VERIFIED", True)
    status, gates = decide(_measures(0.5, [0.5], {"a": 0.0}), policy)
    assert (status, gates) == (
        "REJECTED",
        {"necessity_floor": True, "beats_null": False, "specificity": True},
    )


def test_beating_the_null_alone_is_rejected():
    policy = CircuitPolicy(min_necessity=0.25)
    status, gates = decide(_measures(0.2, [0.1], {"a": 0.9}), policy)
    assert (status, gates) == (
        "REJECTED",
        {"necessity_floor": False, "beats_null": True, "specificity": False},
    )
