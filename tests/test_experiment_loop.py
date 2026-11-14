"""``openinterp loop`` (issue #3246): a temporary git repo, a stubbed ``gh`` and fake commands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterp import draw_steps, runner_cache
from openinterp import experiment_loop as el
from openinterp.planner import Candidate, Plan
from openinterp.revision import previous_attempt

SLUG = "toy/model/L0/N1-positive"
URL = "https://github.com/o/r/pull/7"

GH = """#!{python}
import json, os, sys
args = sys.argv[1:]
with open(os.environ["GH_LOG"], "a") as log:
    log.write(" ".join(args) + "\\n")
files = json.loads(os.environ.get("GH_FILES", '["wiki/x.json"]'))
if args[:2] == ["pr", "view"] and "url" in args:
    print("{url}")
elif args[:2] == ["pr", "view"]:
    print(json.dumps({{"state": "OPEN", "isDraft": False, "mergeable": "MERGEABLE",
                      "files": [{{"path": p}} for p in files], "headRefOid": "abc"}}))
elif args[:2] == ["repo", "view"]:
    print(os.environ.get("GH_PERMISSION", "WRITE"))
elif args[:2] == ["pr", "checks"]:
    sys.exit(int(os.environ.get("GH_CHECKS_EXIT", "0")))
"""


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A worktree whose origin is a bare repo, with a stub ``gh`` first on PATH."""
    origin, work, bin_dir = tmp_path / "origin.git", tmp_path / "work", tmp_path / "bin"
    _git("init", "--bare", "-b", "main", str(origin))
    _git("clone", str(origin), str(work))
    for key, value in (("user.name", "t"), ("user.email", "t@example.com")):
        _git("config", key, value, cwd=work)
    (work / "wiki").mkdir()
    (work / "wiki" / "README.md").write_text("registry\n")
    _git("add", ".", cwd=work)
    _git("commit", "-m", "init", cwd=work)
    _git("push", "origin", "HEAD:main", cwd=work)
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(GH.format(python=sys.executable, url=URL))
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path('/usr/bin')}:{Path('/bin')}")
    monkeypatch.setenv("GH_LOG", str(tmp_path / "gh.log"))
    monkeypatch.setenv("OPENINTERP_STOP_FILE", str(tmp_path / "stop"))
    monkeypatch.chdir(work)
    return SimpleNamespace(origin=origin, work=work, gh_log=tmp_path / "gh.log", tmp=tmp_path)


def _plan(kind="explore", record=None):
    choice = Candidate(
        kind,
        "toy/model",
        1.0,
        "test",
        [],
        slug=None if kind == "explore" else SLUG,
        record=record,
        audit="--model toy/model --random --seed 1",
    )
    return Plan(seed=1, choice=choice, explored=kind == "explore", candidates=[])


@pytest.fixture
def fakes(monkeypatch):
    """Replace the in-process commands: ``verify`` writes a record file, the rest log."""
    ran: list[list[str]] = []

    def openinterp(args):
        ran.append(args)
        if args[0] in ("verify", "reevaluate"):
            out = Path("wiki/experiments") / SLUG.replace("/", "--", 1) / "rec.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}")
        if args[0] == "verify" and "--boom" in args:
            raise el.LoopError("verify failed")

    monkeypatch.setattr(el, "openinterp", openinterp)
    monkeypatch.setattr(el, "draw_slug", lambda choice: SLUG)
    monkeypatch.setattr(draw_steps, "draw_slug", lambda choice: SLUG)
    monkeypatch.setattr(el, "verdict_of", lambda kind, key: ("VERIFIED", f"Explore {key}"))
    return ran


def _loop(plan=None, **kw):
    options = el.Options("api/proposer", kw.pop("reviewer", "api/reviewer"), corpus="c.jsonl")
    return el.Loop(
        options,
        Path("../logs"),
        plan_next=lambda *a, **k: plan or _plan(),
        claims=lambda: [],
        sleep=lambda s: None,
        **kw,
    )


def test_one_iteration_claims_runs_posts_and_lands(repo, fakes, capsys):
    assert _loop(max_experiments=1).run() == 0
    assert f"RESULT: {SLUG} VERIFIED {URL} merged" in capsys.readouterr().out
    gh = repo.gh_log.read_text()
    assert "pr create --draft" in gh and "pr ready" in gh
    assert "pr merge https://github.com/o/r/pull/7 --squash --match-head-commit abc" in gh
    assert [a[0] for a in fakes] == ["audit", "verify", "wiki", "check"]
    assert "--agent-model" in fakes[0] and "api/reviewer" in fakes[1]
    log = subprocess.run(
        ["git", "log", "--format=%s"], capture_output=True, text=True, cwd=repo.work
    ).stdout
    assert log.split("\n")[:2] == [
        "Explore toy-model-L0-N1-positive",
        "Claim toy-model-L0-N1-positive",
    ]
    branches = subprocess.run(
        ["git", "branch", "--list", "experiment/*"], capture_output=True, text=True, cwd=repo.origin
    ).stdout
    assert branches == ""  # deleted after the merge, so a later draw can claim it again


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({"GH_FILES": '["wiki/x.json", "openinterp/cli.py"]'}, "outside wiki/"),
        ({"GH_PERMISSION": "READ"}, "permission is READ"),
        ({"GH_CHECKS_EXIT": "1"}, "checks didn't pass"),
    ],
)
def test_a_pr_that_may_not_merge_is_left_open(repo, fakes, monkeypatch, capsys, env, reason):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert _loop(max_experiments=1).run() == 0
    assert capsys.readouterr().out.strip().endswith(" open")
    gh = repo.gh_log.read_text()
    assert "pr merge" not in gh and reason in gh


def test_a_failed_step_closes_the_pr_and_counts(repo, fakes, monkeypatch, capsys):
    monkeypatch.setattr(el, "steps", _steps_with("boom"))
    assert _loop(max_experiments=5, max_failures=2).run() == 1
    out = capsys.readouterr().out
    assert out.count("RESULT: none verify failed") == 2
    assert "pr close" in repo.gh_log.read_text()


def _steps_with(extra):
    real = el.steps

    def steps(*args, **kwargs):
        out = real(*args, **kwargs)
        review = next(s for s in out if any(" verify " in c for c in s.commands))
        review.commands[0] += f" --{extra}"
        return out

    return steps


def test_a_review_of_the_reviewers_own_record_is_drawn_again(monkeypatch):
    plans = iter([_plan("replicate", "own.json"), _plan("replicate", "other.json")])
    proposers = {"own.json": "api/reviewer", "other.json": "api/proposer"}
    monkeypatch.setattr(el, "proposer_of", proposers.get)
    options = el.Options("api/proposer", "api/reviewer")
    plan = el.draw(options, lambda *a, **k: next(plans), lambda: [])
    assert plan.choice.record == "other.json"
    with pytest.raises(el.LoopError, match="twice"):
        el.draw(options, lambda *a, **k: _plan("replicate", "own.json"), lambda: [])


def test_rate_limits_back_off_then_stop(repo, monkeypatch, capsys):
    waits = []

    def limited(*args):
        raise el.LoopError("proposer failed") from RuntimeError("RateLimitError: 429")

    monkeypatch.setattr(el, "iteration", limited)
    loop = _loop(max_experiments=1)
    loop.sleep = waits.append
    assert loop.run() == 1
    assert waits == [60, 300, 900]
    assert "rate limited after 3 waits" in capsys.readouterr().out


def test_the_stop_file_stops_before_any_draw(repo, monkeypatch, capsys):
    (repo.tmp / "stop").touch()
    monkeypatch.setattr(el, "iteration", lambda *a: pytest.fail("drew after the stop file"))
    assert _loop().run() == 0
    assert "Stopping" in capsys.readouterr().out


def test_merge_blockers():
    view = {"files": [{"path": "wiki/a.json"}], "isDraft": False, "mergeable": "MERGEABLE"}
    assert el.merge_blockers(view, "WRITE") == []
    assert el.merge_blockers(view | {"isDraft": True}, "ADMIN") == ["it is a draft"]
    assert len(el.merge_blockers(view | {"mergeable": "CONFLICTING"}, "")) == 2


def test_external_models_are_refused():
    from openinterp.cli import cli

    result = CliRunner().invoke(cli, ["loop", "--proposer", "external/x", "--reviewer", "a/b"])
    assert result.exit_code == 2 and "external/x" in result.output


def test_result_lines():
    assert el.Result(None, "paused").line() == "RESULT: none paused"
    assert el.Result("s", "REJECTED", "u").line() == "RESULT: s REJECTED u open"


def test_the_runner_cache_is_off_until_kept(monkeypatch):
    loads = []
    load = runner_cache.cached(lambda model, device: loads.append(model) or object())
    monkeypatch.setattr(runner_cache, "_runners", None)
    assert load("m", "cpu") is not load("m", "cpu")
    runner_cache.keep()
    assert load("m", "cpu") is load("m", "cpu") and len(loads) == 3
    monkeypatch.setattr(runner_cache, "_runners", None)


def test_a_revision_sees_the_failed_gates_and_wrong_probes():
    probe = SimpleNamespace(
        kind="hard_negative",
        measured=True,
        passed=False,
        expected_active=False,
        activation=0.9,
        prompt="the bank of a river",
    )
    ok = SimpleNamespace(
        kind="positive_holdout",
        measured=True,
        passed=True,
        expected_active=True,
        activation=1.0,
        prompt="a bank loan",
    )
    verdict = SimpleNamespace(
        status="NEEDS_REVISION",
        adversarial_test_results=[probe, ok],
        metrics=SimpleNamespace(gates={"specificity": False, "x": True}),
    )
    record = SimpleNamespace(
        verdict=verdict, submission=SimpleNamespace(natural_language_explanation="Fires on 'bank'.")
    )
    brief = previous_attempt(record)
    assert "Fires on 'bank'." in brief and "Failed gates: specificity." in brief
    assert "the bank of a river" in brief and "a bank loan" not in brief
    assert "predicted inactive, activation 0.90" in brief


def test_is_rate_limit():
    assert el.is_rate_limit(RuntimeError("Error code: 429"))
    assert not el.is_rate_limit(RuntimeError("bad JSON"))
    assert json.dumps(el.BACKOFF_MINUTES) == "[1, 5, 15]"
