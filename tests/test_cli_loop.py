"""The `open-interpretability loop` and `loop-bug` commands, with a fake git and gh."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from openinterpretability import cli_loop, loop_bugs, runner_cache
from openinterpretability.cli import cli
from openinterpretability.core.heavy_lock import ENABLE_ENV


class FakeGit:
    """Answers `git` like a clone at ``repo``, and keeps each call."""

    def __init__(self, repo: Path, fail: str | None = None):
        self.repo, self.fail, self.calls = repo, fail, []

    def __call__(self, args, **_):
        self.calls.append(tuple(args[1:]))
        if args[1] == self.fail:
            return SimpleNamespace(returncode=128, stdout="", stderr=" fatal: no remote \n")
        out = str(self.repo / ".git") if args[1] == "rev-parse" else ""
        return SimpleNamespace(returncode=0, stdout=out + "\n", stderr="")


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    """A repo and its existing loop worktree, a stop file, and state the loop changes."""
    repo, worktree = tmp_path / "repo", tmp_path / "repo-loop"
    worktree.mkdir(parents=True)
    git = FakeGit(repo)
    monkeypatch.setattr(cli_loop.subprocess, "run", git)
    stop = tmp_path / "stop"
    stop.touch()  # the loop stops before its first draw
    monkeypatch.setenv("OPENINTERPRETABILITY_STOP_FILE", str(stop))
    monkeypatch.setenv(ENABLE_ENV, "0")  # restored after the test
    monkeypatch.setattr(runner_cache, "_runners", None)
    monkeypatch.chdir(tmp_path)  # the loop moves into its worktree; this moves back after
    return SimpleNamespace(repo=repo, worktree=worktree, git=git, stop=stop)


def test_the_loop_runs_in_its_worktree_with_the_lock_and_the_runner_cache(loop_env, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("text\n")
    args = ["loop", "--proposer", "a/m", "--reviewer", "a/m", "--worktree", str(loop_env.worktree),
            "--one-heavy-job", "--corpus", str(corpus)]  # fmt: skip
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "warning: a/m both proposes and reviews" in result.output
    assert f"Stopping: {loop_env.stop} exists." in result.output
    assert Path.cwd() == loop_env.worktree.resolve()
    assert os.environ[ENABLE_ENV] == "1" and runner_cache._runners == {}
    assert loop_env.git.calls == [("rev-parse", "--path-format=absolute", "--git-common-dir")]


def test_run_loop_hands_the_caller_its_loop(loop_env, tmp_path):
    out = {}
    corpus = tmp_path / "c.txt"
    corpus.write_text("x")
    code = cli_loop.run_loop("a/p", "b/r", None, corpus=corpus, worktree=loop_env.worktree,
                             out=out, until_verified=True, max_turns=2)  # fmt: skip
    loop = out["loop"]
    assert code == 0 and loop.log_dir == loop_env.repo / "logs" / "loop"
    assert loop.options.corpus == str(corpus.resolve())
    assert (loop.options.proposer, loop.options.reviewer) == ("a/p", "b/r")
    assert os.environ[ENABLE_ENV] == "0"  # only --one-heavy-job turns the lock on


def test_a_missing_worktree_is_added_at_origin_main_beside_the_repo(loop_env):
    # The default is <repo>-loop beside the repo. It exists, so git adds nothing.
    assert cli_loop.loop_worktree() == (loop_env.repo, loop_env.worktree.resolve())
    assert len(loop_env.git.calls) == 1
    loop_env.git.calls.clear()
    missing = loop_env.repo.parent / "elsewhere"
    assert cli_loop.loop_worktree(missing) == (loop_env.repo, missing.resolve())
    assert loop_env.git.calls[1:] == [
        ("fetch", "--quiet", "origin", "main"),
        ("worktree", "add", "--detach", str(missing.resolve()), "origin/main"),
    ]


def test_a_failed_git_command_stops_with_its_error(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_loop.subprocess, "run", FakeGit(tmp_path, fail="rev-parse"))
    with pytest.raises(click.ClickException, match="git rev-parse failed: fatal: no remote$"):
        cli_loop.loop_worktree()


def test_an_unknown_provider_is_a_usage_error():
    result = CliRunner().invoke(cli, ["loop", "--provider", "no-such-provider"])
    assert result.exit_code == 2
    assert "unknown provider 'no-such-provider'" in result.output


def test_loop_bug_reports_a_gh_failure(monkeypatch):
    def gh(*_, **__):
        raise subprocess.CalledProcessError(1, ["gh"], stderr="  HTTP 401: Bad credentials \n")

    monkeypatch.setattr(loop_bugs.subprocess, "run", gh)
    result = CliRunner().invoke(
        cli, ["loop-bug", "--title", "t", "--command", "uv run x"], input="Traceback: boom"
    )
    assert result.exit_code == 1
    assert "gh failed: HTTP 401: Bad credentials" in result.output
