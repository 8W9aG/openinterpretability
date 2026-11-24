"""CI runs mutation testing on other authors' pull requests, not the maintainer's (#5857)."""

from pathlib import Path

import yaml

CI = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"


def _job() -> dict:
    return yaml.safe_load(CI.read_text())["jobs"]["mutation"]


def test_the_job_skips_the_maintainer_drafts_and_wiki_only_prs():
    job = _job()
    condition = job["if"]
    assert job["needs"] == "code-changes"
    assert "needs.code-changes.outputs.code == 'true'" in condition
    assert "!github.event.pull_request.draft" in condition
    # The repository's owner is the maintainer's login, so no copy of it is needed here.
    assert "github.event.pull_request.user.login != github.repository_owner" in condition
    assert "8W9aG" not in CI.read_text().split("  mutation:")[1].split("\n  accessibility:")[0]


def test_the_job_mutates_the_changed_files_and_fails_on_a_survivor():
    steps = _job()["steps"]
    run = next(s for s in steps if s.get("name") == "Mutate the changed package files")
    assert 'make mutation BASE="origin/$BASE_REF"' in run["run"]
    assert "set -o pipefail" in run["run"]  # tee would hide the script's exit status
    summary = next(s for s in steps if s.get("name") == "Surviving mutants in the job summary")
    assert "GITHUB_STEP_SUMMARY" in summary["run"] and summary["if"] == "${{ !cancelled() }}"
    assert _job()["permissions"] == {"contents": "read"}


def test_the_maintainers_prs_must_record_a_local_run():
    job = yaml.safe_load(CI.read_text())["jobs"]["mutation-ran"]
    assert "github.event.pull_request.user.login == github.repository_owner" in job["if"]
    assert "needs.code-changes.outputs.code == 'true'" in job["if"]
    step = job["steps"][-1]
    assert "github.event.pull_request.body" in step["env"]["BODY"]  # through env only
    assert "${{" not in step["run"]
    assert "**Mutation testing:** No mutant survived" in step["run"]
