"""``open-interpretability answers``: the commands of one answer experiment (#4778)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from click.testing import CliRunner

from openinterpretability import cli_answers
from openinterpretability import explain as ex
from openinterpretability.cli import cli
from openinterpretability.core import answer_records as ar
from tests.test_answer_records import TEXTS, _claim, _measures, _probes, _record

MODEL = "EleutherAI/pythia-70m"


def _cli(*args):
    return CliRunner().invoke(cli, [str(a) for a in args], catch_exceptions=False)


@pytest.fixture
def runner(tiny_gpt2, tmp_path, monkeypatch):
    """The tiny GPT-2 in place of a downloaded model, with means on a 4-token window."""
    from openinterpretability.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    gpt2 = NeuronHookRunner(tiny_gpt2.double(), FakeTokenizer())
    monkeypatch.setattr(cli_answers, "_load_runner", lambda *a, **k: gpt2)
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    return gpt2


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    return path


def _write_claim(tmp_path, **kw):
    path = tmp_path / "claim-1.json"
    path.write_text(_claim(**kw).model_dump_json())
    return path


def _write_probes(tmp_path):
    path = tmp_path / "answer-1.probes.json"
    path.write_text(_probes().model_dump_json())
    return path


def _store(wiki, record):
    path = ar.record_path(wiki, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json())


def test_propose_prompt_writes_the_first_attempts_prompt(tmp_path):
    out = tmp_path / "work"
    result = _cli("answers", "propose-prompt", "abcd", "--model", MODEL, "--answer", "e",
                  "--out-dir", out, "--wiki-dir", tmp_path / "wiki")  # fmt: skip
    prompt = out / "proposer-1.prompt.md"
    assert result.output.strip() == f"attempt 1: {prompt} (the claim goes to {out}/claim-1.json)"
    assert str(out / "claim-1.json") in prompt.read_text()


def test_propose_prompt_needs_an_answer_when_no_explanation_is_recorded(tmp_path):
    result = CliRunner().invoke(cli, ["answers", "propose-prompt", "abcd", "--model", MODEL,
                                      "--wiki-dir", str(tmp_path)])  # fmt: skip
    assert result.exit_code == 2
    assert "no explanation of this prompt is recorded: pass --answer" in result.output


def test_propose_prompt_stops_after_a_verified_claim(tmp_path):
    _store(tmp_path, _record())  # the default measures pass every gate
    result = CliRunner().invoke(cli, ["answers", "propose-prompt", "abcd", "--model", MODEL,
                                      "--answer", "e", "--wiki-dir", str(tmp_path)])  # fmt: skip
    assert result.exit_code == 2
    assert "no more attempts on this prompt: a claim is VERIFIED" in result.output


def test_propose_prompt_stops_at_the_attempt_cap(tmp_path):
    for i in range(ar.MAX_ATTEMPTS):
        when = datetime(2026, 11, 18, i, tzinfo=UTC)
        _store(tmp_path, _record(_measures(patching=0.0), when, attempt=i + 1))
    result = CliRunner().invoke(cli, ["answers", "propose-prompt", "abcd", "--model", MODEL,
                                      "--answer", "e", "--wiki-dir", str(tmp_path)])  # fmt: skip
    assert result.exit_code == 2
    assert f"no more attempts on this prompt: {ar.MAX_ATTEMPTS} claims" in result.output


class _Tokenizers:
    """``AutoTokenizer`` with the character tokenizer, so no tokenizer is downloaded."""

    @staticmethod
    def from_pretrained(name, revision=None):
        from tests.conftest import FakeTokenizer

        return FakeTokenizer()


def test_check_passes_a_claim_with_one_answer_token_and_matched_lengths(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizers)
    result = _cli("answers", "check", "--claim", _write_claim(tmp_path))
    assert result.output == "ok\n"


def test_check_lists_each_problem_and_exits_with_one(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizers)
    claim = _write_claim(tmp_path, answer="ef", corrupted="abc")
    result = CliRunner().invoke(cli, ["answers", "check", "--claim", str(claim)])
    assert result.exit_code == 1
    assert result.output.splitlines() == [
        "problem: the answer 'ef' is not one token",
        "problem: the corrupted prompt is 3 tokens, the prompt 4",
    ]


def test_review_prompt_writes_beside_the_claim_and_names_the_probes_file(tmp_path):
    claim = _write_claim(tmp_path)
    result = _cli("answers", "review-prompt", "--claim", claim)
    out, probes = tmp_path / "reviewer-1.prompt.md", tmp_path / "answer-1.probes.json"
    assert result.output.strip() == f"{out} (the probes go to {probes})"
    assert str(probes) in out.read_text()


def test_review_writes_the_record_and_remeasure_writes_a_new_one(runner, corpus, tmp_path):
    wiki = tmp_path / "wiki"
    claim, probes = _write_claim(tmp_path), _write_probes(tmp_path)
    result = _cli("answers", "review", "--claim", claim, "--probes", probes, "--corpus", corpus,
                  "--mean-tokens", 200, "--seed", 3, "--wiki-dir", wiki)  # fmt: skip
    (path,) = (wiki / ar.ANSWERS_WIKI).rglob("*.json")
    record = ar.AnswerRecord.model_validate_json(path.read_text())
    v = record.verdict
    assert v.measures.null_seed == 3 and v.measures.route_space is not None
    assert result.output.strip() == (
        f"{v.status} (failed gates: {', '.join(v.failed) or 'none'}) → {path}"
    )
    # The re-measure keeps the claim, the probes and the seed, and names the old record.
    again = _cli("answers", "remeasure", "--record", path, "--corpus", corpus,
                 "--mean-tokens", 200, "--wiki-dir", tmp_path / "again")  # fmt: skip
    (new_path,) = (tmp_path / "again" / ar.ANSWERS_WIKI).rglob("*.json")
    new = ar.AnswerRecord.model_validate_json(new_path.read_text())
    assert new.remeasures == record.id and new.verdict.measures.null_seed == 3
    assert new.claim == record.claim and new.probes == record.probes
    assert again.output.strip().startswith(f"{new.verdict.status} (was {v.status}; failed gates:")
    assert again.output.strip().endswith(f"→ {new_path}")


def test_review_refuses_a_claim_that_is_not_the_next_attempt(runner, corpus, tmp_path):
    claim, probes = _write_claim(tmp_path, attempt=2), _write_probes(tmp_path)
    result = CliRunner().invoke(cli, ["answers", "review", "--claim", str(claim), "--probes",
                                      str(probes), "--corpus", str(corpus), "--wiki-dir",
                                      str(tmp_path / "wiki")])  # fmt: skip
    assert result.exit_code == 2
    assert "the claim is attempt 2; the next is 1" in result.output


def test_review_turns_a_measuring_error_into_a_usage_error(runner, corpus, tmp_path):
    claim, probes = _write_claim(tmp_path, corrupted="abc"), _write_probes(tmp_path)
    result = CliRunner().invoke(cli, ["answers", "review", "--claim", str(claim), "--probes",
                                      str(probes), "--corpus", str(corpus), "--mean-tokens",
                                      "200", "--wiki-dir", str(tmp_path / "wiki")])  # fmt: skip
    assert result.exit_code == 2
    assert "the corrupted prompt is 3 tokens" in result.output
    assert not (tmp_path / "wiki").exists()


def test_space_fits_the_models_space_and_records_the_corpus(runner, corpus, tmp_path, monkeypatch):
    import hashlib

    from openinterpretability import embedding_space as es

    monkeypatch.setattr(es, "WINDOW", 8)
    monkeypatch.chdir(tmp_path)  # the space goes to data/embedding-space/ under the cwd
    result = _cli("answers", "space", "--model", "tiny/model", "--corpus", corpus,
                  "--points", 60)  # fmt: skip
    path = tmp_path / "data" / "embedding-space" / "tiny--model.json"
    assert result.output.strip() == f"60 points × 3 layers → {es.path_for('tiny/model')}"
    saved = json.loads(path.read_text())
    assert saved["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert saved["points"] == 60 and saved["window"] == 8 and saved["layers"] == 3
