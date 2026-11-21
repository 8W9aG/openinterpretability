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
