"""Workflows give the scripts labinacan from PyPI, at the version that pyproject pins (#7097)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"
_SPEC = importlib.util.spec_from_file_location("labinacan_pin", SCRIPTS / "labinacan_pin.py")
labinacan_pin = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(labinacan_pin)


def _pyproject(tmp_path: Path, deps: list[str]) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text("[project]\ndependencies = [" + ", ".join(f'"{d}"' for d in deps) + "]\n")
    return path


def test_the_pin_drops_the_extras_and_keeps_the_version(tmp_path):
    deps = ["click>=8", "labinacan[r2,pii]==0.1.3", "torch"]
    assert labinacan_pin.pin(_pyproject(tmp_path, deps)) == "labinacan==0.1.3"
    assert labinacan_pin.pin(_pyproject(tmp_path, ["Labinacan >=0.2"])) == "labinacan>=0.2"


def test_a_project_with_no_labinacan_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="names no labinacan dependency"):
        labinacan_pin.pin(_pyproject(tmp_path, ["click"]))


def test_the_repos_own_pin_is_its_dependency():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'"labinacan\[[^\]]*\]==([0-9.]+)"', text).group(1)
    assert labinacan_pin.pin() == f"labinacan=={version}"


def _needs_labinacan() -> set[str]:
    """The scripts that import labinacan, directly or through another script."""
    imports = {}
    for path in SCRIPTS.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        imports[path.stem] = set(re.findall(r"^\s*(?:from|import) (\w+)", text, re.M))
    needs = {name for name, found in imports.items() if "labinacan" in found}
    grew = True
    while grew:
        more = {name for name, found in imports.items() if found & needs} - needs
        needs |= more
        grew = bool(more)
    return needs


def test_no_workflow_runs_a_script_that_needs_labinacan_with_a_bare_python():
    needs = _needs_labinacan()
    assert {"issue_state", "linear_sync", "claims"} <= needs
    bare = []
    for workflow in WORKFLOWS.glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        if "ci-install.sh" in text or "uv sync" in text:
            continue  # the job installs the project, labinacan with it
        # The runner's interpreter is python3. Under `uv run --with`, a script runs as python.
        for name in re.findall(r"(?<![\w-])python3 scripts/(\w+)\.py", text):
            if name in needs:
                bare.append(f"{workflow.name}: scripts/{name}.py")
    assert bare == []
