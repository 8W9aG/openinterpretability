"""A draft pull request starts no CI work, and never shows a required check as passed (#4690)."""

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
REQUIRED = ("Lint and tests", "Experiment records", "Reproduce records")  # branch protection
TYPES = ["opened", "synchronize", "reopened", "ready_for_review"]


def _load(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text())
    return data | {"on": data.get("on", data.get(True))}  # YAML 1.1 reads "on" as True


def test_pull_request_runs_start_on_ready_for_review_too():
    for name in ("ci.yml", "secret-scan.yml", "codeql.yml"):
        trigger = _load(name)["on"]["pull_request"]
        assert trigger["types"] == TYPES, name


def test_every_job_that_a_pull_request_starts_skips_a_draft():
    for name in ("ci.yml", "secret-scan.yml", "codeql.yml"):
        for job_id, job in _load(name)["jobs"].items():
            condition = str(job.get("if", ""))
            starts_alone = "needs" not in job or "!cancelled()" in condition
            push_only = "github.event_name == 'push'" in condition
            if starts_alone and not push_only:
                assert "!github.event.pull_request.draft" in condition, f"{name}: {job_id}"


def test_auto_merge_waits_for_the_run_that_leaving_draft_starts():
    # A draft run's skipped jobs report success under the required names. So on
    # ready_for_review, auto-merge waits for the new CI run before it turns itself on.
    steps = _load("auto-merge.yml")["jobs"]["enable"]["steps"]
    names = [s.get("name", "") for s in steps]
    wait = names.index("Wait for the CI run that leaving draft starts")
    assert wait < next(i for i, n in enumerate(names) if n.startswith("Auto-merge if"))
    assert steps[wait]["if"] == "github.event.action == 'ready_for_review'"
    assert "actions/workflows/ci.yml/runs" in steps[wait]["run"]
    names = [str(job["name"]) for job in _load("ci.yml")["jobs"].values()]
    assert set(REQUIRED) <= set(names)  # static, so a skipped job still reports its name


def test_a_draft_time_run_cannot_cancel_the_ready_run():
    # #6179: the push and ready_for_review runs arrive a second apart. A shared group with
    # cancel-in-progress let the push's skipped run cancel the real one.
    for name in ("ci.yml", "codeql.yml"):
        group = _load(name)["concurrency"]
        assert group["group"].endswith("-${{ github.event.pull_request.draft }}"), name
        assert group["cancel-in-progress"] is True, name
