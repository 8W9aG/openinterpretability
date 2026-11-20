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


def test_a_draft_shows_no_required_check_as_passed():
    jobs = _load("ci.yml")["jobs"].values()
    names = [str(job["name"]) for job in jobs]
    for check in REQUIRED:
        (name,) = [n for n in names if f"'{check}'" in n]
        assert name.startswith("${{ github.event.pull_request.draft")
        assert f"'{check} (draft, skipped)' || '{check}'" in name
