"""``explain`` reuses the best explanation of a prompt before it runs or records (#4503)."""

from datetime import UTC, datetime

from click.testing import CliRunner

from openinterpretability.explain_reuse import decide
from tests.test_explanation_records import _record, _write

MODEL, PROMPT = "openai-community/gpt2", "John gave a drink to"


def _wiki(tmp_path, *coverages_and_levels):
    wiki = tmp_path / "wiki"
    for day, (coverage, narrated) in enumerate(coverages_and_levels, start=1):
        r = _record(datetime(2026, 11, day, tzinfo=UTC))
        _write(
            wiki,
            r.model_copy(
                update={"prompt_coverage": coverage, "narration": r.narration if narrated else None}
            ),
        )
    return wiki


def test_a_prompt_with_no_explanation_runs(tmp_path):
    reuse = decide(tmp_path / "wiki", MODEL, PROMPT, -1, 0.2, "mechanism")
    assert reuse.action == "run" and reuse.previous is None and reuse.fields["reused"] is False


def test_a_short_explanation_is_built_on(tmp_path):
    wiki = _wiki(tmp_path, (0.1, False), (0.05, False))
    reuse = decide(wiki, MODEL, PROMPT, -1, 0.2, "mechanism")
    assert reuse.action == "run" and reuse.previous == "20261101T000000Z"  # the best, not newest
    assert "Builds on" in reuse.message and "reached 10.0%" in reuse.message


def test_one_that_reaches_the_target_at_the_level_stops_the_command(tmp_path):
    wiki = _wiki(tmp_path, (0.3, True))
    reuse = decide(wiki, MODEL, PROMPT, -1, 0.3, "long")
    assert reuse.action == "done" and "Already explained" in reuse.message
    assert "/models/openai-community--gpt2/explanations/" in reuse.message  # its link


def test_one_that_reaches_the_target_without_the_level_only_adds_it(tmp_path):
    wiki = _wiki(tmp_path, (0.3, False))
    reuse = decide(wiki, MODEL, PROMPT, -1, 0.3, "long")
    assert reuse.action == "level" and reuse.fields == {"previous": "20261101T000000Z",
                                                        "reused": True}  # fmt: skip
    assert decide(wiki, MODEL, PROMPT, -1, 0.3, "mechanism").action == "done"


def test_the_command_makes_no_new_explanation_when_one_reaches_the_target(tmp_path, monkeypatch):
    from openinterpretability import cli_explain
    from openinterpretability.cli import cli

    wiki = _wiki(tmp_path, (0.5, False))
    before = sorted(wiki.rglob("*.json"))
    loaded = []
    monkeypatch.setattr(cli_explain, "_load_runner", lambda *a, **k: loaded.append(1))
    args = ["explain", PROMPT, "--model", MODEL, "--wiki-dir", str(wiki), "--target", "0.4",
            "--record", "--no-land"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    assert "Already explained" in out.output and not loaded  # not even the model loads
    assert sorted(wiki.rglob("*.json")) == before
