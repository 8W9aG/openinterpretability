from click.testing import CliRunner

from openinterpretability.cli import cli
from tests.test_checks import page_of, repo  # noqa: F401  (a fixture)


def test_in_github_actions_problems_are_annotations_and_go_in_the_step_summary(
    repo,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    page = page_of(repo)
    page.unlink()
    result = CliRunner().invoke(cli, ["check"])
    assert result.exit_code == 1
    assert "::error file=" in result.output and "has no page" in result.output
    text = summary.read_text()
    assert text.startswith("### Experiment check\n\n")
    assert "1 problem(s)." in text and "has no page" in text


def test_outside_github_actions_problems_go_to_stderr_and_no_summary_is_written(
    repo,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    summary = tmp_path / "summary.md"
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    page_of(repo).unlink()
    result = CliRunner().invoke(cli, ["check"])
    assert result.exit_code == 1
    assert "error: " in result.stderr and "has no page" in result.stderr
    assert "::error" not in result.output
    assert not summary.exists()


def test_in_github_actions_without_a_step_summary_only_annotations_are_written(
    repo,  # noqa: F811
    monkeypatch,
):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    page_of(repo).unlink()
    result = CliRunner().invoke(cli, ["check"])
    assert result.exit_code == 1
    assert "::error file=" in result.stdout and "1 problem(s)." in result.stdout
