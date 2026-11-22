"""`open-interpretability costs` and `usage` (openinterpretability/cli_costs.py)."""

import json

from click.testing import CliRunner

from openinterpretability.cli import cli
from openinterpretability.cli_costs import read_ledger, read_session, summarise
from openinterpretability.core.costs import ExperimentCosts, LLMCall
from openinterpretability.core.schemas import ExperimentProvenance
from tests.test_isolation import _session, repo  # noqa: F401  (a fixture)


def _invoke(*args):
    return CliRunner().invoke(cli, [str(a) for a in args])


def test_a_missing_ledger_is_empty_and_blank_lines_are_skipped(tmp_path):
    ledger = tmp_path / "costs.jsonl"
    ledger.write_text('{"command": "audit", "seconds": 1.0}\n\n   \n')
    assert read_ledger([tmp_path / "missing.jsonl", ledger]) == [
        {"command": "audit", "seconds": 1.0}
    ]


def test_a_session_file_that_is_missing_unreadable_or_not_an_object_is_none(tmp_path):
    assert read_session(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{cut off")
    assert read_session(tmp_path / "bad.json") is None
    (tmp_path / "list.json").write_text("[1, 2]")
    assert read_session(tmp_path / "list.json") is None


def test_a_summary_without_llm_calls_or_prompt_sizes(tmp_path):
    entries = [{"command": "scan", "seconds": 120.0, "stages": [{"stage": "scan", "seconds": 90}]}]
    text = summarise(entries, [])
    assert "LLM calls:" not in text and "Driver:" not in text
    assert "Total: 2.0 min in openinterpretability commands" in text
    # A call that reported tokens but not its prompt's size.
    call = {"agent": "judge", "model": "m", "prompt_tokens": 10, "completion_tokens": 2}
    text = summarise([{"command": "score", "seconds": 1.0, "llm_calls": [call]}], [])
    assert "judge" in text and "10 + 2 tokens" in text and "prompt chars" not in text


def test_a_session_counts_only_whole_number_usage(tmp_path):
    session = {"duration_ms": 60000, "num_turns": 2, "usage": {"input_tokens": 5,
               "output_tokens": 7, "service_tier": "standard",
               "server_tool_use": {"web_search_requests": 0}}}  # fmt: skip
    text = summarise([], [session])
    assert "Driver: 1 session, 1.0 min, 2 turns, 5 input (0 cache reads) + 7 output tokens" in text


def test_costs_needs_an_input_unless_it_prints_dollars():
    result = _invoke("costs")
    assert result.exit_code != 0 and "Pass at least one ledger" in result.output
    result = _invoke("costs", "--usd")
    assert result.exit_code == 0 and result.output == "0.0000\n"


def test_costs_in_dollars_count_priced_calls_and_name_the_unpriced(tmp_path):
    ledger = tmp_path / "costs.jsonl"
    calls = [{"agent": "proposer", "model": "m", "cost_usd": 0.25}]
    ledger.write_text(json.dumps({"command": "audit", "seconds": 1.0, "llm_calls": calls}) + "\n")
    result = CliRunner().invoke(cli, ["costs", str(ledger), "--usd"])
    assert result.exit_code == 0 and result.output == "0.2500\n"  # nothing unpriced to name

    calls.append({"agent": "reviewer", "model": "m"})
    ledger.write_text(json.dumps({"command": "audit", "seconds": 1.0, "llm_calls": calls}) + "\n")
    result = CliRunner().invoke(cli, ["costs", str(ledger), "--usd"])
    assert result.exit_code == 0
    assert "0.2500" in result.output and "1 unpriced calls or sessions" in result.output


def _with_costs(record, calls, status="VERIFIED"):
    provenance = ExperimentProvenance(
        dtype="float32", corpus="c", n_snippets=1, tokens_scanned=1, selection="random",
        harness_version="t", costs=ExperimentCosts(llm_calls=calls),
    )  # fmt: skip
    sub = record.submission.model_copy(update={"provenance": provenance})
    verdict = record.verdict.model_copy(update={"status": status})
    return record.model_copy(update={"submission": sub, "verdict": verdict})


def test_registry_costs_by_role_and_model(record, tmp_path):
    from openinterpretability.builder.wiki import archive_record

    wiki = tmp_path / "wiki"
    (wiki / "experiments").mkdir(parents=True)
    result = _invoke("costs", "--wiki-dir", wiki)
    assert result.exit_code == 0 and "No LLM calls recorded in 0 records." in result.output

    archive_record(_with_costs(record, [LLMCall(agent="proposer", model="p", cost_usd=0.5),
                                        LLMCall(agent="reviewer", model="r")]), wiki)  # fmt: skip
    result = _invoke("costs", "--wiki-dir", wiki)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("LLM cost per experiment (1 records;")
    proposer = next(line for line in lines if line.strip().startswith("proposer"))
    reviewer = next(line for line in lines if line.strip().startswith("reviewer"))
    assert proposer.split()[1:] == ["p", "1/1", "$0.500", "$0.500"]
    assert reviewer.split()[1:] == ["r", "0/1", "–", "–"]  # unpriced: no mean


def test_registry_costs_take_no_ledgers(tmp_path):
    ledger = tmp_path / "costs.jsonl"
    ledger.write_text("")
    result = _invoke("costs", ledger, "--wiki-dir", tmp_path)
    assert result.exit_code != 0 and "--wiki-dir summarises records" in result.output


def test_usage_names_the_blocked_paths_a_reviewer_tried(repo):  # noqa: F811
    transcript = _session(
        repo / "agent-r.jsonl",
        [
            ("Read", {"file_path": "exchange/k/review.prompt.md"}, False),
            ("Read", {"file_path": "submissions/s.json"}, True),  # denied by the hook
        ],
    )
    probes = repo / "exchange/k/review.probes.json"
    probes.write_text("{}")
    result = _invoke("usage", transcript, "--answer", probes)
    assert result.exit_code == 0, result.output
    assert "The reviewer tried to read blocked paths: submissions/s.json" in result.output
    assert "will refuse" not in result.output  # a denied read saw nothing
    assert (repo / "exchange/k/review.probes.usage.json").exists()
