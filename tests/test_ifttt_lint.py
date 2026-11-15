"""IFTTT lint (scripts/ifttt_lint.py, issue #3161)."""

import importlib.util
import subprocess
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ifttt_lint", Path(__file__).parents[1] / "scripts" / "ifttt_lint.py"
)
ifttt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ifttt)

BLOCK = "a\n# LINT.IfChange\nMODELS = x\n# LINT.ThenChange(b.txt)\nz\n"


def repo(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "a.txt").write_text(BLOCK)
    (tmp_path / "b.txt").write_text("MODELS = x\n")
    git("add", ".")
    git("commit", "-qm", "base")
    return git


def check(tmp_path):
    diff = subprocess.run(
        ["git", "diff", "-U0", "HEAD"], cwd=tmp_path, capture_output=True, text=True
    )
    return ifttt.problems(ifttt.changed_lines(diff.stdout), tmp_path)


def test_a_block_changed_without_its_target_fails(tmp_path):
    repo(tmp_path)
    (tmp_path / "a.txt").write_text(BLOCK.replace("MODELS = x", "MODELS = y"))
    (problem,) = check(tmp_path)
    assert "a.txt:2-4" in problem and "b.txt" in problem


def test_a_block_changed_with_its_target_passes(tmp_path):
    repo(tmp_path)
    (tmp_path / "a.txt").write_text(BLOCK.replace("MODELS = x", "MODELS = y"))
    (tmp_path / "b.txt").write_text("MODELS = y\n")
    assert check(tmp_path) == []


def test_a_change_outside_the_block_passes(tmp_path):
    repo(tmp_path)
    (tmp_path / "a.txt").write_text(BLOCK.replace("z\n", "zz\n"))
    assert check(tmp_path) == []


def test_a_line_removed_inside_the_block_counts(tmp_path):
    repo(tmp_path)
    (tmp_path / "a.txt").write_text(BLOCK.replace("MODELS = x\n", ""))
    assert len(check(tmp_path)) == 1


def test_a_target_that_does_not_exist_fails(tmp_path):
    repo(tmp_path)
    (tmp_path / "a.txt").write_text(BLOCK.replace("b.txt", "gone.txt").replace("z\n", "zz\n"))
    assert any("doesn't exist" in p for p in check(tmp_path))
