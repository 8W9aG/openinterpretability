"""Commands run outside a clone read the project's settings (planner._config, #3693)."""

from pathlib import Path

from click.testing import CliRunner

from openinterpretability import planner

REPO_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_another_directory_reads_the_projects_settings(tmp_path, monkeypatch):
    expected = planner.configured_models(REPO_PYPROJECT)
    windows = planner.configured_windows(REPO_PYPROJECT)
    assert expected and windows  # the repository configures both
    monkeypatch.chdir(tmp_path)
    assert planner.configured_models() == expected
    assert planner.configured_windows() == windows
    assert planner.configured_licenses() == planner.configured_licenses(REPO_PYPROJECT)


def test_another_projects_pyproject_does_not_hide_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\n')
    assert planner.configured_models() == planner.configured_models(REPO_PYPROJECT)
    (tmp_path / "pyproject.toml").write_text('[tool.openinterpretability]\nmodels = ["a/x"]\n')
    assert planner.configured_models() == ("a/x",)  # a table here wins, as in a clone


def test_an_install_reads_the_copy_it_holds(tmp_path, monkeypatch):
    packaged = tmp_path / "data" / "pyproject.toml"
    packaged.parent.mkdir()
    packaged.write_text('[tool.openinterpretability]\nmodels = ["b/y"]\n')
    monkeypatch.setattr(planner, "PACKAGED_PYPROJECT", packaged)
    monkeypatch.setattr(planner, "CLONE_PYPROJECT", tmp_path / "missing.toml")
    monkeypatch.chdir(tmp_path)
    assert planner.configured_models() == ("b/y",)
    monkeypatch.setattr(planner, "PACKAGED_PYPROJECT", tmp_path / "missing.toml")
    assert planner.configured_models() == ()


def test_the_wheel_ships_the_settings():
    import tomllib

    hatch = tomllib.loads(REPO_PYPROJECT.read_text())["tool"]["hatch"]["build"]["targets"]
    shipped = hatch["wheel"]["force-include"]["pyproject.toml"]
    package = Path(planner.__file__).parent
    assert shipped == "openinterpretability/data/pyproject.toml"
    assert package / "data" / "pyproject.toml" == planner.PACKAGED_PYPROJECT


def test_commands_that_work_anywhere_start_outside_the_repository(tmp_path, monkeypatch):
    from openinterpretability.cli import cli

    monkeypatch.chdir(tmp_path)
    for command in (["--help"], ["show", "--help"], ["provider", "list"], ["usage", "--help"]):
        result = CliRunner().invoke(cli, command)
        assert result.exit_code == 0, (command, result.output)
