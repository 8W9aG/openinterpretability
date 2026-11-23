"""The wrapper that runs mutmut on a branch's changed files (#5856)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "mutation.py"
    spec = importlib.util.spec_from_file_location("mutation_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A repository with a package file, a script and a test, on a base commit."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.org")
    _git(tmp_path, "config", "user.name", "T")
    pkg = tmp_path / "openinterpretability" / "core"
    pkg.mkdir(parents=True)
    (pkg / "stats.py").write_text("X = 1\n")
    (pkg / "gone.py").write_text("Y = 1\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tool.py").write_text("Z = 1\n")
    (tmp_path / "wiki" / "draws").mkdir(parents=True)
    (tmp_path / "wiki" / "draws" / "a.json").write_text("{}")
    (tmp_path / "wiki" / "LICENSE.md").write_text("L")
    (tmp_path / "wiki" / "experiments").mkdir()
    (tmp_path / "wiki" / "experiments" / "b.json").write_text("{}")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def test_changed_files_are_the_branchs_package_files_only(repo):
    script = _script()
    _git(repo, "checkout", "-qb", "work")
    (repo / "openinterpretability" / "core" / "stats.py").write_text("X = 2\n")
    _git(repo, "commit", "-qam", "committed change")
    (repo / "openinterpretability" / "core" / "gone.py").unlink()  # deleted: not mutated
    (repo / "openinterpretability" / "core" / "new.py").write_text("W = 1\n")  # untracked
    (repo / "scripts" / "tool.py").write_text("Z = 2\n")  # outside the package
    assert script.changed_files("main", repo) == ["openinterpretability/core/new.py",
                                                  "openinterpretability/core/stats.py"]  # fmt: skip


def test_copied_skips_the_package_tests_and_experiment_records(repo):
    got = _script().copied(repo)
    assert "openinterpretability" not in got and "wiki/experiments" not in got
    assert "scripts" in got and "wiki" not in got
    assert got.index("wiki/draws") < got.index("wiki/LICENSE.md")  # folders before files


def test_imports_finds_the_module_however_a_test_imports_it():
    script = _script()
    module = "openinterpretability.core.stats"
    assert script.imports("import openinterpretability.core.stats as s", module)
    assert script.imports("from openinterpretability.core import (\n    schemas,\n    stats,\n)",
                          module)  # fmt: skip
    assert not script.imports("from openinterpretability.core import statsmodels", module)
    assert not script.imports("from openinterpretability.agents import stats", module)


def test_tests_for_selects_the_tests_that_import_a_changed_file(tmp_path):
    script = _script()
    (tmp_path / "test_a.py").write_text("from openinterpretability.core import stats\n")
    (tmp_path / "test_b.py").write_text("from openinterpretability.core import schemas\n")
    (tmp_path / "helper.py").write_text("import openinterpretability.core.stats\n")  # not a test
    got = script.tests_for(["openinterpretability/core/stats.py"], tmp_path)
    assert got == [str(tmp_path / "test_a.py")]


def _fake_mutmut(monkeypatch, results: str, calls: list):
    """Stand-ins for mutmut's configuration and command line."""
    configuration = types.ModuleType("mutmut.configuration")
    configuration._config_reader = lambda: lambda key, default: f"file:{key}"
    configuration.reset_config = lambda: calls.append("reset")
    main = types.ModuleType("mutmut.__main__")

    def cli(args, standalone_mode):
        calls.append(args)
        if args == ["results"]:
            print(results)

    main.cli = cli
    package = types.ModuleType("mutmut")
    monkeypatch.setitem(sys.modules, "mutmut", package)
    monkeypatch.setitem(sys.modules, "mutmut.configuration", configuration)
    monkeypatch.setitem(sys.modules, "mutmut.__main__", main)
    return configuration


def test_override_replaces_only_the_three_keys(monkeypatch):
    calls: list = []
    configuration = _fake_mutmut(monkeypatch, "", calls)
    _script().override(["a.py"], ["tests/test_a.py"], ["scripts"])
    read = configuration._config_reader()
    assert read("only_mutate", []) == ["a.py"] and read("also_copy", []) == ["scripts"]
    assert read("pytest_add_cli_args_test_selection", []) == ["tests/test_a.py"]
    assert read("source_paths", []) == "file:source_paths" and calls == ["reset"]


def test_main_fails_on_a_surviving_mutant(monkeypatch, capsys):
    calls: list = []
    _fake_mutmut(monkeypatch, "    m__mutmut_1: survived\n    m__mutmut_2: killed", calls)
    assert _script().main([]) == 1
    assert calls == [["run"], ["results"]]
    assert capsys.readouterr().out.strip() == "m__mutmut_1: survived"


def test_main_passes_with_no_survivor(monkeypatch, capsys):
    _fake_mutmut(monkeypatch, "", [])
    assert _script().main([]) == 0
    assert "No mutant survived." in capsys.readouterr().out


def test_main_with_a_base_mutates_the_changed_files(monkeypatch, capsys):
    calls: list = []
    configuration = _fake_mutmut(monkeypatch, "", calls)
    script = _script()
    monkeypatch.setattr(script, "changed_files", lambda base: ["openinterpretability/x.py"])
    monkeypatch.setattr(script, "tests_for", lambda paths: ["tests/test_x.py"])
    monkeypatch.setattr(script, "copied", lambda: ["scripts"])
    assert script.main(["--base", "origin/main"]) == 0
    assert "Mutating 1 files with 1 test files" in capsys.readouterr().out
    assert configuration._config_reader()("only_mutate", []) == ["openinterpretability/x.py"]
    script.main(["--base", "origin/main", "show", "m"])  # a command runs under the selection
    assert calls[-1] == ["show", "m"]


def test_main_with_no_change_does_nothing(monkeypatch, capsys):
    calls: list = []
    _fake_mutmut(monkeypatch, "", calls)
    script = _script()
    monkeypatch.setattr(script, "changed_files", lambda base: [])
    assert script.main(["--base", "origin/main"]) == 0
    assert calls == [] and "nothing to mutate" in capsys.readouterr().out
