"""``explain --run``: the run's target and the agents it hands to the loop."""

import json
from types import SimpleNamespace

from click.testing import CliRunner

from openinterpretability import explain as ex
from openinterpretability import providers
from openinterpretability.explain_run import _reachable
from tests.test_explain import PROMPT, TEXTS


def test_an_unreachable_target_stops_at_the_reach_and_names_waiting_circuits(capsys):
    reach = {"now": 0.1, "reach": 0.4, "target": 0.8, "experiments": None}
    planned = SimpleNamespace(blocked=0.05)
    assert _reachable(reach, planned, ["C1 (a task to write)"]) == 0.4
    out = capsys.readouterr().out
    assert "Not run: circuit C1 (a task to write)" in out
    assert "The target 80% is out of reach, so the run stops at 40.0%" in out


def test_a_run_with_only_a_provider_takes_the_providers_model(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    seen = {}

    def run_loop(proposer, reviewer, plan_next, out=None):
        seen.update(proposer=proposer, reviewer=reviewer)
        return 0

    monkeypatch.setattr("openinterpretability.cli_loop.run_loop", run_loop)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(tmp_path / "wiki"), "--edges", "0", "--run", "1",
            "--provider", "anthropic"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    model = providers.load("anthropic").litellm_model
    assert model and seen == {"proposer": model, "reviewer": model}
