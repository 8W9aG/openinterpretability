"""Pending experiments on the home and model pages (builder/pending.py, #3283)."""

import subprocess

import pytest

from openinterpretability.builder import pending
from openinterpretability.builder.compiler import compile_site


def _pr(branch, number=7, fork=False, draft=False):
    url = f"https://github.com/o/r/pull/{number}"
    return {"number": number, "url": url, "headRefName": branch, "isCrossRepository": fork,
            "isDraft": draft, "createdAt": "2026-11-16T22:09:36Z"}  # fmt: skip


def test_a_branch_names_its_unit_pole_and_kind():
    p = pending.parse(_pr("experiment/Qwen--Qwen2.5-0.5B-L15-N1365-positive-reevaluate"))
    assert (p.model_slug, p.layer, p.unit, p.pole, p.kind) == (
        "Qwen--Qwen2.5-0.5B", 15, "N1365", "positive", "reevaluate",
    )  # fmt: skip
    assert p.unit_href == "models/Qwen--Qwen2.5-0.5B/L15/N1365.html" and p.opened == "2026-11-16"
    head = pending.parse(_pr("experiment/EleutherAI--pythia-70m-L2-H4"))
    assert (head.unit, head.pole, head.kind) == ("H4", None, "explore")


def test_forks_and_other_branches_are_not_listed():
    assert pending.parse(_pr("experiment/openai-community--gpt2-L3-N7-positive", fork=True)) is None
    assert pending.parse(_pr("site/pending-3283")) is None


def test_the_home_and_model_pages_list_the_pending_experiments(tmp_path, record):
    from tests.test_compiler import experiment

    records = [experiment(record, idx=1, status="VERIFIED", counts=(10, 0, 5, 0))]
    slug = record.submission.identifier.model_name.replace("/", "--")
    own = pending.parse(_pr(f"experiment/{slug}-L3-N1-positive-revise", draft=True))
    other = pending.parse(_pr("experiment/other--model-L2-H4", number=9))
    compile_site(records, tmp_path / "a", pending=[own, other])
    home = (tmp_path / "a" / "index.html").read_text(encoding="utf-8")
    assert "Pending experiments" in home and "running (draft)" in home
    assert "pull/7" in home and "pull/9" in home and "revise" in home
    model = (tmp_path / "a" / "models" / slug / "index.html").read_text(encoding="utf-8")
    assert "Pending experiments" in model and "pull/7" in model and "pull/9" not in model
    assert not (tmp_path / "a" / "pending.html").exists()
    compile_site(records, tmp_path / "b")
    assert "Pending experiments" not in (tmp_path / "b" / "index.html").read_text(encoding="utf-8")


def test_without_gh_nothing_is_fetched(monkeypatch):
    monkeypatch.setattr(pending.shutil, "which", lambda _: None)
    assert pending.open_experiments() == []


@pytest.mark.parametrize(
    "error",
    [subprocess.TimeoutExpired("gh", 60), subprocess.CalledProcessError(1, "gh"), OSError("gh")],
)
def test_a_gh_call_that_fails_or_times_out_lists_nothing(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(pending.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(pending.subprocess, "run", fail)
    assert pending.open_experiments() == []
