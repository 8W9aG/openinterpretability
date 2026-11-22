"""What open PRs claim, read with `gh` (planner_claims.py)."""

import subprocess

from openinterpretability import planner_claims


def test_no_gh_means_no_claims(monkeypatch):
    monkeypatch.setattr(planner_claims.shutil, "which", lambda name: None)
    assert planner_claims.open_pr_claims() == []


def test_a_failed_gh_call_means_no_claims(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 30)

    monkeypatch.setattr(planner_claims.shutil, "which", lambda name: "/usr/bin/gh")
    monkeypatch.setattr(planner_claims.subprocess, "run", fail)
    assert planner_claims.open_pr_claims() == []
