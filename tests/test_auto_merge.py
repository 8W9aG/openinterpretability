"""The auto-merge rules of ``.github/workflows/auto-merge.yml``.

The test runs the workflow's own scripts with a stand-in ``gh`` on ``PATH``. The stand-in
answers from a list of the PR's files and from this repository's files, as the contents API
would at the base commit. It writes each ``gh pr merge`` call to a log.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "auto-merge.yml"
FAKE_GH = """#!{python}
import json, os, sys
args = sys.argv[1:]
state = json.load(open(os.environ["FAKE_STATE"]))
if args[:2] == ["pr", "merge"]:
    with open(os.environ["FAKE_LOG"], "a") as log:
        log.write(" ".join(args) + "\\n")
elif args[:2] == ["pr", "view"] and "changedFiles" in args:
    print(state["changed_files"])
elif args[:2] == ["pr", "view"]:
    print("false")
elif args[0] == "api" and "/contents/" in args[1]:
    path = args[1].split("/contents/")[1].split("?")[0]
    sys.stdout.write(state["contents"][path])
elif args[0] == "api":
    query = args[args.index("--jq") + 1]
    for f in state["files"]:
        if "select" in query:
            if f["status"] != "added":
                print(f["filename"])
        else:
            print(f["filename"])
            if f.get("previous_filename"):
                print(f["previous_filename"])
"""


def _script(start: str) -> str:
    """The run script of the step whose name starts with ``start``."""
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["enable"]["steps"]
    return next(s["run"] for s in steps if s.get("name", "").startswith(start))


def _env() -> dict:
    """The job's environment, with the values that GitHub fills in left out."""
    return {k: v for k, v in yaml.safe_load(WORKFLOW.read_text())["jobs"]["enable"]["env"].items()
            if not str(v).startswith("${{")}  # fmt: skip


REPO = "8W9aG/openinterpretability"
MAINTAINER = "the-owner"  # github.repository_owner, which GitHub fills in
ROOT = WORKFLOW.parents[2]
RULE = "openinterpretability/core/record_kinds.py"


def _contents(pyproject: str | None = None) -> dict:
    """The files that the contents API gives at the base commit: this repository's own."""
    text = (ROOT / "pyproject.toml").read_text() if pyproject is None else pyproject
    return {"pyproject.toml": text, RULE: (ROOT / RULE).read_text()}


def _run(tmp_path, author, sender, files, changed_files=None, head_repo=REPO, contents=None):
    gh = tmp_path / "bin" / "gh"
    gh.parent.mkdir(exist_ok=True)
    gh.write_text(FAKE_GH.format(python=sys.executable))
    gh.chmod(0o755)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"files": files, "changed_files": changed_files or len(files),
                                 "contents": contents or _contents()}))  # fmt: skip
    log, github_env = tmp_path / "merge.log", tmp_path / "github.env"
    log.write_text("")
    github_env.write_text("")
    env = {
        **os.environ, **{k: str(v) for k, v in _env().items()},
        "PATH": f"{gh.parent}{os.pathsep}{os.environ['PATH']}",
        "FAKE_STATE": str(state), "FAKE_LOG": str(log),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "GITHUB_ENV": str(github_env),
        "REPO": REPO, "PR": "7", "HEAD_SHA": "abc", "BASE_SHA": "base",
        "MAINTAINER": MAINTAINER, "AUTHOR": author, "SENDER": sender, "HEAD_REPO": head_repo,
    }  # fmt: skip
    # The first step reads the rule, and passes its function on through GITHUB_ENV.
    subprocess.run(["bash", "-c", _script("Read the project's rule")], env=env, cwd=tmp_path,
                   check=True)  # fmt: skip
    env |= dict(line.split("=", 1) for line in github_env.read_text().splitlines() if line)
    subprocess.run(["bash", "-c", _script("Auto-merge")], env=env, cwd=tmp_path, check=True)
    return log.read_text()


ADDED_WIKI = [{"filename": "wiki/experiments/m/L0/N1-positive/r.json", "status": "added"}]
CODE = [{"filename": "openinterpretability/cli.py", "status": "modified"}]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def test_the_maintainer_is_the_owner_and_the_rule_comes_from_the_config():
    env = yaml.safe_load(WORKFLOW.read_text())["jobs"]["enable"]["env"]
    assert env["MAINTAINER"] == "${{ github.repository_owner }}"
    assert f'auto_merge_rule = "{RULE}:reproducible"' in (ROOT / "pyproject.toml").read_text()


def test_the_maintainers_own_pr_auto_merges(tmp_path):
    assert "--auto" in _run(tmp_path, MAINTAINER, MAINTAINER, CODE)


def test_the_maintainers_pr_that_someone_else_pushed_gets_the_rule(tmp_path):
    assert "--auto" not in _run(tmp_path, MAINTAINER, "someone", CODE)
    assert "--auto" in _run(tmp_path, MAINTAINER, "someone", ADDED_WIKI)


def test_with_no_rule_only_the_maintainer_and_dependabot_auto_merge(tmp_path):
    # The template's case: [tool.labinacan] names no auto_merge_rule.
    bare = '[project]\nname = "a-project"\n\n[tool.labinacan]\nname = "A Project"\n'
    contents = _contents(bare)
    assert "--auto" not in _run(tmp_path, "someone", "someone", ADDED_WIKI, contents=contents)
    assert "--auto" in _run(tmp_path, MAINTAINER, MAINTAINER, CODE, contents=contents)
    summary = (tmp_path / "summary.md").read_text()
    assert "Auto-merge on: opened by" in summary


def test_the_rule_is_read_at_the_base_commit_with_no_checkout():
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["enable"]["steps"]
    assert not any("uses" in s for s in steps)  # nothing of the PR is checked out
    read = _script("Read the project's rule")
    assert "contents/pyproject.toml?ref=$BASE_SHA" in read
    assert "contents/${rule%%:*}?ref=$BASE_SHA" in read


def test_dependabot_updates_auto_merge_when_dependabot_pushed(tmp_path):
    assert "--auto" in _run(tmp_path, "dependabot[bot]", "dependabot[bot]", CODE)


def test_a_dependabot_pr_that_someone_else_pushed_waits(tmp_path):
    assert "--auto" not in _run(tmp_path, "dependabot[bot]", "someone", CODE)


def test_a_dependabot_pr_from_a_fork_waits(tmp_path):
    assert "--auto" not in _run(tmp_path, "dependabot[bot]", "dependabot[bot]", CODE,
                                head_repo="someone/openinterpretability")  # fmt: skip


def test_an_experiment_from_anyone_auto_merges(tmp_path):
    assert "--auto" in _run(tmp_path, "someone", "someone", ADDED_WIKI)


def test_a_pr_with_more_files_than_the_api_lists_waits(tmp_path):
    assert "--auto" not in _run(tmp_path, "someone", "someone", ADDED_WIKI, changed_files=3001)


def test_a_code_change_from_anyone_waits(tmp_path):
    assert "--auto" not in _run(tmp_path, "someone", "someone", CODE)


def test_a_name_is_checked_whole_not_split_on_its_spaces(tmp_path):
    # " wiki/…" is a folder named " wiki", outside wiki/. Split on whitespace, it would read
    # as a record path and auto-merge.
    spaced = [{"filename": " wiki/experiments/m/L0/N1-positive/r.json", "status": "added"}]
    assert "--auto" not in _run(tmp_path, "someone", "someone", spaced)
    assert "--auto" in _run(tmp_path, "someone", "someone", ADDED_WIKI)
