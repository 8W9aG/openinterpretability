"""A draw's claim never collides, and a session that ran nothing uses no attempt (#6066)."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from openinterpretability import experiment_loop as el
from openinterpretability import loop_sessions as ls
from openinterpretability.draw_steps import claim_branch
from openinterpretability.explanation_targets import TargetedDraws

START = datetime(2026, 11, 24, 1, 0, tzinfo=UTC)


def test_two_follow_ups_of_a_kind_on_a_unit_get_two_branches():
    first = claim_branch("revise", "m-L1-N2-positive", START)
    second = claim_branch("revise", "m-L1-N2-positive", START.replace(minute=7))
    assert first != second and first.startswith("experiment/m-L1-N2-positive-revise-")
    assert claim_branch("revise", "k").startswith("experiment/k-revise-")  # the clock's time


NO_BRANCH = lambda slug, since: None  # noqa: E731  (the session left no branch)


def _options():
    kept = []
    return SimpleNamespace(pending=SimpleNamespace(keep=lambda: kept.append(True))), kept


def test_a_session_with_no_line_that_landed_a_pr_counts():
    options, kept = _options()
    asked = []
    found = lambda slug, since: asked.append((slug, since)) or ("https://x/pull/9", True)  # noqa: E731
    result = ls._result(None, Path("log"), options, "m/L1/N2-positive", START, found)
    assert (result.url, result.merged, result.verdict) == ("https://x/pull/9", True, "unknown")
    assert result.slug == "m/L1/N2-positive" and asked == [("m/L1/N2-positive", START)]
    assert kept == [True]


def test_a_session_that_ran_nothing_names_its_unit():
    options, _ = _options()
    with pytest.raises(ls.NothingRan) as raised:
        ls._result("none claimed", Path("log"), options, "m/L1/N2-positive", START,
                   lambda slug, since: None, NO_BRANCH)  # fmt: skip
    assert raised.value.nothing_ran == "m/L1/N2-positive"
    assert "ran no experiment" in str(raised.value)


def test_landed_pr_reads_github_and_skips_older_or_other_prs(monkeypatch):
    rows = ('[{"headRefName": "experiment/m-L1-N2-positive-revise-1123", "createdAt": '
            '"2026-11-23T00:00:00Z", "url": "old", "state": "MERGED"}, '
            '{"headRefName": "experiment/m-L1-N2-positive-revise-1124", "createdAt": '
            '"2026-11-24T02:00:00Z", "url": "new", "state": "OPEN"}]')  # fmt: skip
    monkeypatch.setattr(ls.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=rows))  # fmt: skip
    assert ls.landed_pr("m/L1/N2-positive", START) == ("new", False)
    monkeypatch.setattr(ls.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))  # fmt: skip
    assert ls.landed_pr("m/L1/N2-positive", START) is None


def test_the_loop_gives_back_the_attempt_of_a_session_that_ran_nothing(tmp_path, monkeypatch):
    draws = TargetedDraws([], reviewer="r")
    draws.attempts, draws.started = {"m/L1/N2-positive": 2}, 3
    loop = el.Loop(options=SimpleNamespace(), log_dir=tmp_path, plan_next=draws,
                   claims=lambda: [], sleep=lambda s: None)  # fmt: skip
    assert loop._failed(ls.NothingRan("ran nothing", "m/L1/N2-positive"), tmp_path / "l") is None
    assert draws.attempts["m/L1/N2-positive"] == 1 and draws.started == 2 and loop.failures == 1
    loop._failed(el.LoopError("a real failure"), tmp_path / "l")  # keeps its attempt
    assert draws.attempts["m/L1/N2-positive"] == 1
    draws.unstart("never-drawn")  # no attempt to give back
    assert draws.started == 2
    draws.unstart("m/L1/N2-positive")  # its last attempt goes too
    assert draws.attempts["m/L1/N2-positive"] == 0 and draws.started == 1


def test_the_summary_says_which_targets_the_run_never_reached():
    a = SimpleNamespace(slug="a", label="L1/N1+")
    b = SimpleNamespace(slug="b", label="L1/N2+")
    c = SimpleNamespace(slug="c", label="L1/N3+")
    draws = TargetedDraws([a, b, c], reviewer="r")
    draws.attempts, draws.claimed = {"a": 1}, {"b"}
    assert draws.summary() == ("L1/N1+ waiting on an open PR; L1/N2+ waiting on an open PR; "
                               "L1/N3+ not reached")  # fmt: skip


def test_a_known_units_claim_has_no_note():
    from openinterpretability.draw_steps import _claim
    from openinterpretability.planner import Candidate

    c = Candidate(kind="revise", model="m", weight=1.0, rationale="r", commands=[], slug="m/L1/N2")
    assert _claim(c, "m-L1-N2", 7, known=True).note == ""
    assert "isn't known" in _claim(c, "m-L1-N2", 7, known=False).note


def test_rate_limits_back_off_one_wait_at_a_time_then_stop(tmp_path, capsys):
    waits = []
    loop = el.Loop(options=SimpleNamespace(), log_dir=tmp_path, plan_next=None,
                   claims=lambda: [], sleep=waits.append)  # fmt: skip
    limit = el.LoopError("proposer failed")
    limit.__cause__ = RuntimeError("RateLimitError: 429")
    assert [loop._failed(limit, tmp_path / "l") for _ in el.BACKOFF_MINUTES] == [None] * 3
    assert waits == [m * 60 for m in el.BACKOFF_MINUTES] and loop.backoffs == 3
    assert "Rate limited; waiting 1 min" in capsys.readouterr().out
    assert loop._failed(limit, tmp_path / "l") == 1  # the waits are used up


def test_the_loops_failure_lines_reach_the_console_at_once(tmp_path, monkeypatch):
    # The loop runs for hours, so each line is flushed, not left in a buffer.
    flushes = []
    out = SimpleNamespace(write=lambda text: None, flush=lambda: flushes.append(True))
    monkeypatch.setattr("sys.stdout", out)
    loop = el.Loop(options=SimpleNamespace(), log_dir=tmp_path, plan_next=None,
                   claims=lambda: [], sleep=lambda s: None)  # fmt: skip
    limit = el.LoopError("proposer failed")
    limit.__cause__ = RuntimeError("RateLimitError: 429")
    loop._failed(limit, tmp_path / "l")
    loop._failed(el.LoopError("a real failure"), tmp_path / "l")
    assert len(flushes) == 2


def test_a_result_line_gives_the_pr_and_whether_it_merged():
    options, kept = _options()
    line = "m/L1/N2-positive VERIFIED https://x/pull/3 merged extra"
    result = ls._result(line, Path("log"), options, "", None)
    assert (result.slug, result.verdict, result.url, result.merged) == (
        "m/L1/N2-positive", "VERIFIED", "https://x/pull/3", True)  # fmt: skip
    opened = ls._result(line.replace("merged", "open"), Path("log"), options, "", None)
    assert opened.merged is False


def test_a_long_none_line_still_means_the_session_ran_nothing():
    options, _ = _options()
    with pytest.raises(ls.NothingRan):
        ls._result("none claim taken by #6064", Path("log"), options, "m/L1/N2-positive", START,
                   lambda slug, since: None, NO_BRANCH)  # fmt: skip


def test_a_session_with_no_unit_or_start_looks_for_no_pr():
    options, _ = _options()
    asked = []
    find = lambda slug, since: asked.append(slug) or ("u", True)  # noqa: E731
    strand = lambda slug, since: asked.append(since) or ("b", True)  # noqa: E731
    with pytest.raises(ls.NothingRan):
        ls._result(None, Path("log"), options, "", START, find, strand)
    with pytest.raises(ls.NothingRan):
        ls._result(None, Path("log"), options, "m/L1/N2-positive", None, find, strand)
    assert asked == []


def test_landed_pr_asks_github_for_every_recent_pr(monkeypatch):
    calls = []
    rows = ('[{"headRefName": "experiment/m-L1-N2-positive-revise-1124", '
            '"createdAt": "2026-11-24T02:00:00Z", "url": "u", "state": "MERGED"}]')  # fmt: skip

    def run(args, **kw):
        calls.append((args, kw))
        return SimpleNamespace(returncode=calls[-1] and len(calls) - 1, stdout=rows)

    monkeypatch.setattr(ls.subprocess, "run", run)
    assert ls.landed_pr("m/L1/N2-positive", START) == ("u", True)
    args, kw = calls[0]
    assert args == ["gh", "pr", "list", "--state", "all", "--limit", "100",
                    "--json", "headRefName,createdAt,url,state"]  # fmt: skip
    assert kw == {"capture_output": True, "text": True, "check": False}
    assert ls.landed_pr("m/L1/N2-positive", START) is None  # gh failed: its output is ignored


def test_a_session_runs_the_draws_steps_on_its_unit(tmp_path, monkeypatch):
    plan = SimpleNamespace(choice=SimpleNamespace(kind="revise", model="m"))
    seen = {}
    monkeypatch.setattr(el, "reset", lambda: None)
    monkeypatch.setattr(el, "draw", lambda options, plan_next, claims: plan)
    monkeypatch.setattr(el, "draw_slug", lambda choice: choice and "m/L1/N2-positive")
    monkeypatch.setattr(el, "steps", lambda *a: seen.setdefault("steps", a) and [])
    monkeypatch.setattr("openinterpretability.draw_steps.render", lambda steps: "STEPS")
    monkeypatch.setattr(ls, "result_line", lambda text: None)
    monkeypatch.setattr(ls, "session_text", lambda result: "")
    monkeypatch.setattr(ls, "_result", lambda *a: seen.setdefault("result", a))
    session = SimpleNamespace(run=lambda prompt, cwd, *a: seen.update(prompt=prompt, cwd=cwd))
    applied = SimpleNamespace(applied=lambda: __import__("contextlib").nullcontext())
    sessions = SimpleNamespace(
        session=session, log_dir=tmp_path / "logs" / "loop"
    )  # made with its parents
    options = SimpleNamespace(pending=applied, corpus=None, sessions=sessions)
    clock = datetime(2026, 11, 24, 3, 4, 5, 678901, tzinfo=UTC)

    class Clock:
        @staticmethod
        def now(tz=None):
            return clock if tz is UTC else clock.replace(hour=22)  # not UTC: another hour

    monkeypatch.setattr(ls, "datetime", Clock)
    ls.session_iteration(options, None, None)
    assert seen["steps"][1:] == ("corpus.jsonl", None, None, True)  # a fresh corpus by default
    assert seen["result"][1] == tmp_path / "logs/loop/20261124T030405678901Z.log"  # in UTC
    assert seen["cwd"] == Path.cwd()
    assert "a revise draw on m/L1/N2-positive" in seen["prompt"] and "STEPS" in seen["prompt"]
    assert seen["result"][3] == "m/L1/N2-positive" and seen["result"][4] is not None


def test_the_session_prompt_asks_for_one_plain_command_per_call():
    # #6145: a prefix rule matches only a command that starts with it.
    prompt = ls.session_prompt("m/L1/N2-positive", "revise", "1. Claim")
    assert "one command per Bash call" in prompt
    assert all(f"`{shape}`" in prompt for shape in ("cd", "&&", ";", "env"))
    assert "pipes" in prompt and "(#6145)" in prompt


BRANCH = "experiment/m-L1-N2-positive-revise-1124010203"


def _git(stamps, added):
    """A fake ``git``: two remote branches, each with its last commit's time and added files."""
    calls, fetched = [], []

    def git(*args):
        calls.append(args)
        if args[0] == "ls-remote":
            return "".join(f"abc\trefs/heads/{b}\n" for b in stamps)
        if args[0] == "fetch":
            fetched.append(args[-1])
            return ""
        if args[0] == "log":
            return stamps[fetched[-1]] + "\n"
        return added[fetched[-1]]

    return git, calls


def test_a_branch_pushed_since_the_start_is_found_with_its_record():
    old = "experiment/m-L1-N2-positive"
    git, calls = _git(
        {old: "2026-11-23T00:00:00+00:00", BRANCH: "2026-11-24T01:03:00+00:00"},
        {old: "", BRANCH: "wiki/experiments/m/L1/N2-positive/r.json\nr.json.sig"},
    )
    assert ls.stranded_branch("m/L1/N2-positive", START, git) == (BRANCH, True)
    assert calls[0] == ("ls-remote", "--heads", "origin",
                        "refs/heads/experiment/m-L1-N2-positive*")  # fmt: skip
    assert ("fetch", "--quiet", "origin", old) in calls
    assert ("log", "-1", "--format=%cI", "FETCH_HEAD") in calls
    assert calls[-1] == ("diff", "--name-only", "--diff-filter=A", "origin/main...FETCH_HEAD")


def test_a_branch_with_only_its_claim_has_no_record():
    for added in ("wiki/ledger/m.jsonl", "wiki/experiments/m/L1/N2-positive/r.json.sig",
                  "docs/wiki/experiments/r.json"):  # fmt: skip
        git, _ = _git({BRANCH: "2026-11-24T01:02:00+00:00"}, {BRANCH: added})
        assert ls.stranded_branch("m/L1/N2-positive", START, git) == (BRANCH, False)


def test_an_older_or_unreadable_branch_is_not_the_sessions():
    git, _ = _git({BRANCH: "2026-11-24T00:59:59+00:00"}, {BRANCH: ""})
    assert ls.stranded_branch("m/L1/N2-positive", START, git) is None
    git, _ = _git({BRANCH: "2026-11-24T01:00:00+00:00"}, {BRANCH: ""})  # pushed at the start
    assert ls.stranded_branch("m/L1/N2-positive", START, git) == (BRANCH, False)
    git, _ = _git({BRANCH: ""}, {BRANCH: ""})
    assert ls.stranded_branch("m/L1/N2-positive", START, git) is None
    git, _ = _git({}, {})
    assert ls.stranded_branch("m/L1/N2-positive", START, git) is None


def test_a_stranded_record_is_a_loop_error_that_names_the_branch():
    options, _ = _options()
    strand = lambda slug, since: (BRANCH, True)  # noqa: E731
    with pytest.raises(el.LoopError) as raised:
        ls._result(None, Path("s.log"), options, "m/L1/N2-positive", START, NO_BRANCH, strand)
    assert not isinstance(raised.value, ls.NothingRan)  # the experiment ran: the attempt counts
    expected = (f"the session pushed a record on {BRANCH} but opened no PR: "
                f"open it with `gh pr create --head {BRANCH}`; see s.log")  # fmt: skip
    assert str(raised.value) == expected


def test_a_stranded_claim_gives_back_the_attempt_and_names_the_branch():
    options, _ = _options()
    seen = []
    strand = lambda slug, since: seen.append((slug, since)) or (BRANCH, False)  # noqa: E731
    with pytest.raises(ls.NothingRan) as raised:
        ls._result("none paused", Path("s.log"), options, "m/L1/N2-positive", START, NO_BRANCH,
                   strand)  # fmt: skip
    assert seen == [("m/L1/N2-positive", START)]
    assert raised.value.nothing_ran == "m/L1/N2-positive"
    assert str(raised.value) == (
        f"the session ran no experiment (none paused); it left {BRANCH} with "
        f"only its claim: `git push origin --delete {BRANCH}`; see s.log"
    )
    with pytest.raises(ls.NothingRan) as raised:
        ls._result(None, Path("s.log"), options, "m/L1/N2-positive", START, NO_BRANCH, NO_BRANCH)
    assert str(raised.value) == "the session ran no experiment (no RESULT line); see s.log"


def test_git_gives_its_output_or_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ls._git("--version").startswith("git version")
    assert ls._git("log", "-1") == ""  # not a repository: git fails


def test_git_runs_git_and_drops_a_failed_commands_output(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=len(calls) - 1, stdout="out")

    monkeypatch.setattr(ls.subprocess, "run", run)
    assert ls._git("ls-remote", "origin") == "out"  # exit 0
    assert ls._git("log") == ""  # exit 1: its output is dropped
    assert calls[0] == (["git", "ls-remote", "origin"], {"capture_output": True, "text": True})
