"""The auto-merge rules of ``.github/workflows/auto-merge.yml``.

The test runs the workflow's own script with a stand-in ``gh`` on ``PATH``. The stand-in
answers from a list of the PR's files and writes each ``gh pr merge`` call to a log.
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


def _script() -> str:
    """The run script of the step that turns auto-merge on."""
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["enable"]["steps"]
    return next(s["run"] for s in steps if "Auto-merge" in s["name"])


def _env() -> dict:
    """The job's environment, with the values that GitHub fills in left out."""
    return {k: v for k, v in yaml.safe_load(WORKFLOW.read_text())["jobs"]["enable"]["env"].items()
            if not str(v).startswith("${{")}  # fmt: skip


REPO = "8W9aG/openinterpretability"


def _run(tmp_path, author, sender, files, changed_files=None, head_repo=REPO):
    gh = tmp_path / "bin" / "gh"
    gh.parent.mkdir(exist_ok=True)
    gh.write_text(FAKE_GH.format(python=sys.executable))
    gh.chmod(0o755)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"files": files, "changed_files": changed_files or len(files)}))
    log = tmp_path / "merge.log"
    log.write_text("")
    env = {
        **os.environ, **{k: str(v) for k, v in _env().items()},
        "PATH": f"{gh.parent}{os.pathsep}{os.environ['PATH']}",
        "FAKE_STATE": str(state), "FAKE_LOG": str(log),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "REPO": "8W9aG/openinterpretability", "PR": "7", "HEAD_SHA": "abc",
        "AUTHOR": author, "SENDER": sender, "HEAD_REPO": head_repo,
    }  # fmt: skip
    subprocess.run(["bash", "-c", _script()], env=env, cwd=tmp_path, check=True)
    return log.read_text()


ADDED_WIKI = [{"filename": "wiki/experiments/m/L0/N1-positive/r.json", "status": "added"}]
CODE = [{"filename": "openinterpretability/cli.py", "status": "modified"}]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


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
