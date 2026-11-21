"""The audit command's option checks and its proposer call (cli_evidence.py)."""

from pathlib import Path
from types import SimpleNamespace

import click
import pytest

import openinterpretability.cli  # noqa: F401  (cli_evidence imports from it)
from openinterpretability.agents.proposer import ProposerAgent
from openinterpretability.cli_evidence import (
    ask_proposer,
    check_scan,
    check_steps,
    load_superseded,
)


def test_a_superseded_file_that_is_not_a_record_is_refused(tmp_path):
    path = tmp_path / "r.json"
    path.write_text('{"not": "a record"}')
    with pytest.raises(click.BadParameter, match="not an audit record"):
        load_superseded(path)


def test_a_superseded_record_needs_a_verdict(tmp_path, record):
    path = tmp_path / "r.json"
    path.write_text(record.model_copy(update={"verdict": None}).model_dump_json())
    with pytest.raises(click.BadParameter, match="has no verdict yet"):
        load_superseded(path)
    path.write_text(record.model_dump_json())
    assert load_superseded(path) == record


def test_a_scan_and_a_reused_scan_do_not_go_together():
    ev, hyp = Path("ev.json"), Path("ev.hypothesis.json")
    with pytest.raises(click.UsageError, match="--evidence reuses a scan. Pick one."):
        check_steps(ev, hyp, Path("out.json"), None, "external/proposer")


def test_a_scan_needs_a_model_unless_it_reuses_evidence():
    with pytest.raises(click.UsageError, match="--model is required unless --evidence"):
        check_scan(None, None, None, None)
    check_scan(Path("ev.json"), None, None, None)  # the evidence names the model


def test_the_proposer_gets_every_part_of_the_evidence(monkeypatch, capsys):
    asked = []

    def propose(self, *args, **kwargs):
        asked.append((self.model, args, kwargs))
        return "draft"

    monkeypatch.setattr(ProposerAgent, "propose", propose)
    evidence = SimpleNamespace(
        identifier="ident", top_examples=["ex"], vocabulary_lens="lens", confounds="conf",
        correlated_neurons=["n"], head_profile=None, input_attribution="attr",
        dictionary_profile=None, first_token_floor=0.0, superposition_profile="sp",
        manifold="mf",
    )  # fmt: skip
    agent, draft = ask_proposer("openai/gpt-test", evidence, True, "related text")
    assert draft == "draft" and agent.model == "openai/gpt-test"
    assert "Asking openai/gpt-test for a hypothesis" in capsys.readouterr().err
    ((model, args, kwargs),) = asked
    assert args == ("ident", ["ex"], "lens", "conf", ["n"], None, "attr", None, False, True,
                    "related text")  # fmt: skip
    assert kwargs == {"superposition_profile": "sp", "manifold": "mf"}


def test_a_one_step_audit_asks_the_proposer_and_writes_the_submission(
    tmp_path, monkeypatch, toy_runner
):
    from click.testing import CliRunner

    from openinterpretability.agents.proposer import CounterfactualDraft, HypothesisDraft
    from openinterpretability.cli import cli
    from openinterpretability.core.schemas import NeuronAuditSubmission
    from tests.test_agents import TEXTS

    toy_runner.model.config = SimpleNamespace()
    monkeypatch.setattr("openinterpretability.cli._load_runner", lambda *a, **k: toy_runner)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.txt").write_text("\n".join(TEXTS))
    draft = HypothesisDraft(
        explanation="Fires on repeated letters.",
        activating_tokens=["a"],
        mechanism="retrospective",
        counterfactuals=[
            CounterfactualDraft(
                base_prompt=b, counterfactual_prompt=c, expected_delta=0.5, rationale="r"
            )
            for b, c in [("aaaa", "abcd"), ("zzzz", "zyxw"), ("bbbb", "bcde")]
        ],
    )
    asked = []
    monkeypatch.setattr(
        ProposerAgent, "propose", lambda self, ident, *a, **k: asked.append(ident) or draft
    )
    args = ["audit", "--model", "toy", "--layer", "1", "--neuron", "2", "--corpus", "corpus.txt",
            "--agent-model", "openai/gpt-test", "--out", "sub.json"]  # fmt: skip
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "Asking openai/gpt-test for a hypothesis" in result.output
    sub = NeuronAuditSubmission.model_validate_json((tmp_path / "sub.json").read_text())
    assert sub.submitting_agent_model == "openai/gpt-test"
    assert sub.natural_language_explanation == "Fires on repeated letters."
    assert asked == [sub.identifier] and (sub.identifier.layer, sub.identifier.neuron_idx) == (1, 2)
