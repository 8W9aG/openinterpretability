"""Which issues an agent may work on (scripts/issue_state.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "issue_state", Path(__file__).parents[1] / "scripts" / "issue_state.py"
)
issue_state = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(issue_state)
claims = sys.modules["claims"]  # issue_state imports it from scripts/


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


def test_only_the_maintainers_issues_or_approved_ones_may_carry_the_muse_label(monkeypatch):
    _stub(monkeypatch, "8W9aG")
    assert issue_state.muse_eligibility("1") == (True, "opened by 8W9aG")
    for bot in ("dependabot[bot]", "github-actions[bot]"):  # eligible for work, not for Muse
        _stub(monkeypatch, bot, title="Nightly reproduction failed")
        assert not issue_state.muse_eligibility("1")[0]
    _stub(monkeypatch, "someone", thumbs=["8W9aG"])
    assert issue_state.muse_eligibility("1")[0]
    after = "2026-11-11T00:00:00Z"
    _stub(monkeypatch, "someone", thumbs=["8W9aG"], edits=[(after, "someone")])
    assert not issue_state.muse_eligibility("1")[0]


def test_the_muse_check_exits_1_for_an_issue_that_may_not_have_the_label(monkeypatch, capsys):
    _stub(monkeypatch, "someone")
    monkeypatch.setattr(issue_state.sys, "argv", ["issue_state.py", "--muse-check", "#7"])
    with pytest.raises(SystemExit) as stop:
        issue_state.main()
    assert stop.value.code == 1
    out = capsys.readouterr().out
    assert "#7: NOT eligible (opened by someone, with no thumbs up from 8W9aG)" in out


def test_the_muse_label_workflow_runs_the_check_on_labelled_issues():
    text = (Path(__file__).parents[1] / ".github/workflows/muse-label.yml").read_text()
    assert '--muse-check "$NUMBER"' in text and "--remove-label muse" in text
    assert "contains(github.event.issue.labels.*.name, 'muse')" in text


@pytest.fixture
def agents(tmp_path, monkeypatch):
    """Two agents' clones of one bare remote, which holds the claims (#6725)."""
    remote = tmp_path / "remote.git"
    claims.git(["init", "--quiet", "--bare", str(remote)])
    clones = []
    for name in ("a", "b"):
        clone = tmp_path / name
        claims.git(["clone", "--quiet", str(remote), str(clone)])
        for key, value in (("user.name", name), ("user.email", f"{name}@example.com")):
            claims.git(["config", key, value], str(clone))
        clones.append(str(clone))
    monkeypatch.delenv(claims.AGENT_ENV, raising=False)
    return clones


def test_only_one_of_two_agents_wins_a_claim(agents):
    a, b = agents
    won, branch = claims.claim("issue-7", cwd=a)
    assert won and branch == "claim/issue-7"
    won, holder = claims.claim("issue-7", cwd=b)
    assert not won and holder.startswith(f"Claim issue-7 by {claims.agent_id(a)} at ")
    # The winner can't claim it twice either: the claim is the branch, not the agent.
    assert not claims.claim("issue-7", cwd=a)[0]
    assert claims.claim_holder("issue-8", cwd=b) is None


def test_each_reply_is_its_own_claim_and_unclaim_drops_them_all(agents):
    a, b = agents
    assert claims.claim(claims.reply_key("7", None), cwd=a)[0]
    reply = claims.reply_key("7", "20261125T080102Z")
    assert reply == "issue-7-reply-20261125T080102Z"
    assert claims.claim(reply, cwd=b)[0]
    assert not claims.claim(reply, cwd=a)[0]
    assert claims.claim("issue-70", cwd=a)[0]  # another issue, kept by unclaim
    assert sorted(claims.unclaim("7", cwd=a)) == [
        "claim/issue-7",
        "claim/issue-7-reply-20261125T080102Z",
    ]
    assert claims.unclaim("7", cwd=a) == []
    assert claims.claim_holder("issue-70", cwd=a) is not None


def test_a_push_that_fails_for_another_reason_raises(agents, tmp_path):
    with pytest.raises(RuntimeError, match="could not push claim/issue-7"):
        claims.claim("issue-7", remote=str(tmp_path / "missing.git"), cwd=agents[0])


def test_agent_ids_name_the_checkout_unless_set(agents, monkeypatch):
    a, b = agents
    assert claims.agent_id(a) != claims.agent_id(b)
    assert claims.agent_id(a) == claims.agent_id(a)
    assert claims.agent_id(a).startswith("agent-") and len(claims.agent_id(a)) == 14
    monkeypatch.setenv(claims.AGENT_ENV, "agent-laptop")
    assert claims.agent_id(a) == "agent-laptop"


def test_a_reply_time_must_be_letters_and_digits():
    with pytest.raises(ValueError, match="letters and digits"):
        claims.reply_key("7", "../main")


def _claims(monkeypatch, won=True, eligible=True):
    keys, states = [], []
    monkeypatch.setenv("LINEAR_API_KEY", "key")

    def claim(key):
        keys.append(key)
        return (True, f"claim/{key}") if won else (False, "Claim by agent-other")

    monkeypatch.setattr(issue_state, "claim", claim)
    monkeypatch.setattr(issue_state, "eligibility", lambda n: (eligible, "why"))
    monkeypatch.setattr(issue_state, "agent_id", lambda cwd=None: "agent-me")
    monkeypatch.setattr(issue_state, "set_state", lambda n, s: states.append((n, s)) or s)
    return keys, states


@pytest.mark.parametrize(
    ("argv", "key"),
    [(["--claim", "7"], "issue-7"), (["--claim", "#7", "--reply", "20261125T080102Z"],
      "issue-7-reply-20261125T080102Z")],
)  # fmt: skip
def test_a_won_claim_sets_in_progress(monkeypatch, argv, key):
    keys, states = _claims(monkeypatch)
    monkeypatch.setattr("sys.argv", ["issue_state.py", *argv])
    with pytest.raises(SystemExit) as exit_:
        issue_state.main()
    assert exit_.value.code == 0 and keys == [key] and states == [("7", "In Progress")]


def test_a_lost_claim_exits_3_and_sets_no_state(monkeypatch, capsys):
    keys, states = _claims(monkeypatch, won=False)
    assert issue_state.claim_issue("7") == claims.CLAIMED == 3
    assert (
        states == []
        and "claimed by another agent (Claim by agent-other)" in capsys.readouterr().out
    )


def test_an_ineligible_issue_is_never_claimed(monkeypatch):
    keys, states = _claims(monkeypatch, eligible=False)
    assert issue_state.claim_issue("7") == 1 and keys == [] and states == []


def test_a_release_claim_names_the_previous_tag(monkeypatch, capsys):
    keys, _ = _claims(monkeypatch)
    monkeypatch.setattr("sys.argv", ["issue_state.py", "--claim-release", "v0.5.0"])
    with pytest.raises(SystemExit) as exit_:
        issue_state.main()
    assert exit_.value.code == 0 and keys == ["release-after-v0.5.0"]
    _claims(monkeypatch, won=False)
    assert issue_state.claim_release("v0.5.0") == 3
    assert "Work an issue" in capsys.readouterr().out


def test_whoami_and_unclaim_print(monkeypatch, capsys):
    monkeypatch.setattr(issue_state, "agent_id", lambda cwd=None: "agent-me")
    monkeypatch.setattr(issue_state, "unclaim", lambda n: [])
    for argv in (["--whoami"], ["--unclaim", "7"]):
        monkeypatch.setattr("sys.argv", ["issue_state.py", *argv])
        issue_state.main()
    assert capsys.readouterr().out == "agent-me\n#7: no claim branch\n"


def test_a_claim_with_no_linear_key_holds_and_sets_no_state(monkeypatch, capsys):
    keys, states = _claims(monkeypatch)
    monkeypatch.delenv("LINEAR_API_KEY")
    assert issue_state.claim_issue("7") == 0 and keys == ["issue-7"] and states == []
    assert "no LINEAR_API_KEY" in capsys.readouterr().out
