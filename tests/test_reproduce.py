"""Reproducing a record catches edited numbers, results and verdicts."""

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterp.agents.reviewer import evaluate_submission
from openinterp.cli import cli
from openinterp.core.schemas import AuditRecord
from openinterp.reproduce import reproduce_record, rotating_sample
from tests.test_agents import SMALL, _probes, _submission


@pytest.fixture
def draft():
    from openinterp.agents.proposer import CounterfactualDraft, HypothesisDraft

    return HypothesisDraft(
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


@pytest.fixture
def ident():
    from openinterp.core.schemas import NeuronIdentifier

    return NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")


@pytest.fixture
def record(ident, toy_runner, draft):
    sub = _submission(ident, toy_runner, draft, honest=True)
    probes = _probes(toy_runner, ident, sub, honest=True)
    verdict = evaluate_submission(sub, toy_runner, probes, "external/reviewer", SMALL)
    return AuditRecord(submission=sub, verdict=verdict)


def edit_result(record, index, **update):
    results = list(record.verdict.adversarial_test_results)
    results[index] = results[index].model_copy(update=update)
    verdict = record.verdict.model_copy(update={"adversarial_test_results": results})
    return record.model_copy(update={"verdict": verdict})


def test_honest_record_reproduces(record, toy_runner):
    report = reproduce_record(record, toy_runner)
    assert report.ok, report.mismatches
    assert report.compared > len(record.verdict.adversarial_test_results)
    assert record.verdict.environment.dtype == "float32"  # pinned at review time


def test_edited_probe_activation_is_caught(record, toy_runner):
    tampered = edit_result(
        record, 0, activation=record.verdict.adversarial_test_results[0].activation + 1.0
    )
    report = reproduce_record(tampered, toy_runner)
    assert not report.ok and "activation recorded" in report.mismatches[0]


def test_flipped_result_and_status_are_caught(record, toy_runner):
    first = record.verdict.adversarial_test_results[0]
    tampered = edit_result(record, 0, passed=not first.passed)
    tampered = tampered.model_copy(
        update={"verdict": tampered.verdict.model_copy(update={"status": "REJECTED"})}
    )
    messages = " ".join(reproduce_record(tampered, toy_runner).mismatches)
    assert "passed=" in messages and "status recorded REJECTED" in messages


def test_edited_scan_evidence_is_caught(record, toy_runner):
    sub = record.submission
    ex = sub.top_examples[0]
    louder = ex.model_copy(update={"activations": [a + 1.0 for a in ex.activations]})
    tampered = record.model_copy(
        update={
            "submission": sub.model_copy(
                update={"top_examples": [louder, *sub.top_examples[1:]], "max_activation": 99.0}
            )
        }
    )
    messages = " ".join(reproduce_record(tampered, toy_runner).mismatches)
    assert "top example 1: activations differ" in messages
    assert "is not the top example's peak" in messages


def test_corpus_sample_is_redrawn(record, toy_runner):
    from openinterp.core.schemas import ExperimentProvenance
    from openinterp.corpus import sample_tokens

    docs = [e.text for e in record.submission.top_examples] + ["padding text"] * 3
    _, sample = sample_tokens(toy_runner.tokenizer, docs, 10**6, 0)
    prov = ExperimentProvenance(
        dtype="float32", corpus="c", corpus_sha256="abc", n_snippets=len(docs),
        tokens_scanned=sample.tokens, selection="manual", harness_version="t",
        corpus_sample=sample,
    )  # fmt: skip
    sub = record.submission.model_copy(update={"provenance": prov})
    rec = record.model_copy(update={"submission": sub})

    def check(texts, sha="abc", r=rec):
        return reproduce_record(r, toy_runner, corpus=(texts, sha))

    assert check(docs).ok
    assert check(docs[:-1]).mismatches == [f"corpus sample recorded {sample}, re-drawn "
        f"{sample_tokens(toy_runner.tokenizer, docs[:-1], 10**6, 0)[1]}"]  # fmt: skip
    other = check(docs, sha="other")
    assert other.ok and "not the scan's" in other.warnings[-1]
    # A sample that can't have produced the top examples: one document, not theirs.
    _, lone = sample_tokens(toy_runner.tokenizer, ["padding text"], 10**6, 0)
    fake = prov.model_copy(update={"corpus_sample": lone})
    r = rec.model_copy(update={"submission": sub.model_copy(update={"provenance": fake})})
    assert "top example 1 is not in the corpus sample" in check(["padding text"], r=r).mismatches


def test_other_protocol_versions_are_refused(record, toy_runner):
    old_policy = record.verdict.policy.model_copy(update={"protocol_version": "0.4"})
    old = record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"policy": old_policy})}
    )
    (message,) = reproduce_record(old, toy_runner).mismatches
    assert "protocol 0.4" in message and "re-evaluate" in message


def test_cli_exit_code(tmp_path, monkeypatch, record, toy_runner):
    toy_runner.model.config = SimpleNamespace()
    loads = []
    monkeypatch.setattr(
        "openinterp.cli._load_runner",
        lambda name, device, dtype, revision=None, attn=None: (
            loads.append((name, dtype)) or toy_runner
        ),
    )
    good = tmp_path / "good.json"
    good.write_text(record.model_dump_json())
    bad = tmp_path / "bad.json"
    bad.write_text(edit_result(record, 0, passed=False).model_dump_json())

    result = CliRunner().invoke(cli, ["reproduce", str(good)])
    assert result.exit_code == 0, result.output
    assert "REPRODUCED" in result.output and loads == [("toy", "float32")]

    result = CliRunner().invoke(cli, ["reproduce", str(good), str(bad)])
    assert result.exit_code == 1 and "FAILED" in result.output
    assert "1/2 record(s) reproduced" in result.output
    assert len(loads) == 2  # one load per invocation, shared across same-model records


def test_cli_time_budget_leaves_later_records_unreached(tmp_path, monkeypatch, record, toy_runner):
    import time

    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: toy_runner)
    clock = iter([0.0, 0.0, 120 * 60.0])  # start, before the first record, before the second
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(record.model_dump_json())
    bad.write_text(edit_result(record, 0, passed=False).model_dump_json())
    args = ["reproduce", str(good), str(bad), "--time-budget", "90"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output  # the edited record was never started
    assert f"REPRODUCED: {good}" in result.output and f"NOT REACHED: {bad}" in result.output
    assert "1/1 record(s) reproduced. 1 not reached in the 90-minute time budget." in result.output
    assert "(not reached)" in summary.read_text()


def test_cli_allow_list_refuses_other_models_without_loading(tmp_path, monkeypatch, record):
    monkeypatch.setattr(
        "openinterp.cli._load_runner", lambda *a, **k: pytest.fail("must not load the model")
    )
    path = tmp_path / "r.json"
    path.write_text(record.model_dump_json())
    result = CliRunner().invoke(cli, ["reproduce", str(path), "--allow-model", "Qwen/Qwen2.5-0.5B"])
    assert result.exit_code == 1 and "model toy is not on the allow-list" in result.output


def test_cli_reports_to_github_actions(tmp_path, monkeypatch, record, toy_runner):
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: toy_runner)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    bad = tmp_path / "bad.json"
    bad.write_text(edit_result(record, 0, passed=False).model_dump_json())
    result = CliRunner().invoke(cli, ["reproduce", str(bad)])
    assert result.exit_code == 1
    assert f"::error file={bad}::" in result.output
    assert "### Reproduction" in summary.read_text() and "❌" in summary.read_text()


def test_changed_records_lists_added_records(tmp_path, monkeypatch):
    import subprocess

    from openinterp.checks import changed_records

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    monkeypatch.chdir(tmp_path)
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    old = tmp_path / "wiki" / "experiments" / "m" / "old.json"
    old.parent.mkdir(parents=True)
    old.write_text("{}")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    git("switch", "-q", "-c", "pr")
    (old.parent / "new.json").write_text("{}")
    (old.parent / "new.md").write_text("page")
    git("add", "-A")
    git("commit", "-q", "-m", "add")
    assert changed_records(Path("wiki"), "main") == [Path("wiki/experiments/m/new.json")]


def test_rotating_sample_covers_the_registry_and_is_repeatable():
    paths = [Path(f"wiki/experiments/m/{i}.json") for i in range(10)]
    day = date(2026, 11, 6)
    assert rotating_sample(paths, 3, day) == rotating_sample(list(reversed(paths)), 3, day)
    assert rotating_sample(paths, 3, day) != rotating_sample(paths, 3, day + timedelta(days=1))
    assert len(set(rotating_sample(paths, 3, day))) == 3
    covered = {p for n in range(4) for p in rotating_sample(paths, 3, day + timedelta(days=n))}
    assert covered == set(paths)
    assert sorted(rotating_sample(paths, 11, day)) == sorted(paths)


def _old_protocol(record):
    policy = record.verdict.policy.model_copy(update={"protocol_version": "0.1"})
    return record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"policy": policy})}
    )


def test_measurements_only_compares_an_old_records_activations(record, toy_runner):
    old = _old_protocol(record)
    assert not reproduce_record(old, toy_runner).ok
    report = reproduce_record(old, toy_runner, measurements_only=True)
    assert report.ok, report.mismatches
    assert any("measurements only" in w for w in report.warnings)

    example = old.submission.top_examples[0]
    edited = example.model_copy(update={"activations": [a + 1.0 for a in example.activations]})
    sub = old.submission.model_copy(update={"top_examples": [edited]})
    tampered = old.model_copy(update={"submission": sub})
    assert not reproduce_record(tampered, toy_runner, measurements_only=True).ok


def test_measurements_only_still_runs_the_review_of_a_current_record(record, toy_runner):
    report = reproduce_record(
        edit_result(record, 0, passed=False), toy_runner, measurements_only=True
    )
    assert not report.ok


def _wiki_record(tmp_path, record):
    path = tmp_path / "wiki" / "experiments" / "m" / "r.json"
    path.parent.mkdir(parents=True)
    path.write_text(record.model_dump_json())
    return path


def test_cli_sample_reproduces_signed_and_old_records(tmp_path, monkeypatch, record, toy_runner):
    from openinterp.signing import sign_record
    from tests.test_signing import keypair, signer_line

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: toy_runner)
    path = _wiki_record(tmp_path, _old_protocol(record))
    key = keypair(tmp_path, "maintainer")
    sign_record(path, key)
    signers = tmp_path / "allowed_signers"
    signers.write_text(signer_line("will", key))
    common = ["--allowed-signers", str(signers), "--wiki-dir", "wiki"]

    skipped = CliRunner().invoke(cli, ["reproduce", str(path.relative_to(tmp_path)), *common])
    assert "SIGNED by will" in skipped.output

    args = ["reproduce", "--sample", "5", "--date", "2026-11-06", *common]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "REPRODUCED: wiki/experiments/m/r.json" in result.output
    assert "measurements only" in result.output

    path.write_text(path.read_text() + "\n")  # edited after signing
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1 and "not a valid signature" in result.output


def test_cli_include_signed_runs_the_review(tmp_path, monkeypatch, record, toy_runner):
    from openinterp.signing import sign_record
    from tests.test_signing import keypair, signer_line

    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: toy_runner)
    path = _wiki_record(tmp_path, record)
    key = keypair(tmp_path, "maintainer")
    sign_record(path, key)
    signers = tmp_path / "allowed_signers"
    signers.write_text(signer_line("will", key))
    args = ["reproduce", str(path), "--include-signed", "--allowed-signers", str(signers)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0 and "REPRODUCED" in result.output, result.output


def test_cli_reproduces_a_record_signed_outside_its_keys_window(
    tmp_path, monkeypatch, record, toy_runner
):
    from openinterp.signing import sign_record
    from tests.test_signing import keypair, signer_line

    monkeypatch.setattr("openinterp.cli._load_runner", lambda *a, **k: toy_runner)
    path = _wiki_record(tmp_path, record)
    key = keypair(tmp_path, "maintainer")
    sign_record(path, key)
    signers = tmp_path / "allowed_signers"
    signers.write_text(signer_line("will", key, 'valid-before="20200101Z"'))  # revoked
    args = ["reproduce", str(path), "--allowed-signers", str(signers)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0 and "REPRODUCED" in result.output, result.output
    assert "not valid now" in result.output and "so it is reproduced" in result.output


def test_nightly_workflow_opens_issues_and_keeps_failures_visible():
    root = Path(__file__).parent.parent
    workflow = (root / ".github/workflows/reproduce-nightly.yml").read_text()
    assert "issues: write" in workflow and "if: failure()" in workflow
    assert "shell: bash" in workflow  # pipefail: tee can't hide a failed reproduction
    assert "openinterp reproduce" in workflow and "--sample" in workflow
    assert "--time-budget" in workflow  # a night that draws slow records still finishes
    assert "fetch-depth: 0" in workflow  # signing.landing_times needs main's history
    assert "gh issue create" in workflow and "gh issue comment" in workflow


# -- 16-bit records (issue #2995) ----------------------------------------------


def _pinned(record, dtype):
    env = record.verdict.environment.model_copy(update={"dtype": dtype})
    return record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"environment": env})}
    )


def test_a_16_bit_record_is_reproduced_in_float32_with_a_wider_tolerance():
    from openinterp.half_precision import reproduction_dtype
    from openinterp.reproduce import HALF_TOLERANCE, Tolerance

    assert reproduction_dtype("bfloat16") == "float32" == reproduction_dtype("float16")
    assert reproduction_dtype("float32") == "float32"
    base = Tolerance()
    assert base.for_dtype("float32") is base
    wide = base.for_dtype("bfloat16")
    assert wide.atol == HALF_TOLERANCE.atol and wide.rtol == HALF_TOLERANCE.rtol
    assert Tolerance(atol=1.0).for_dtype("bfloat16").atol == 1.0  # a wider --atol stays


def test_a_bfloat16_record_reproduces_on_float32_weights_with_a_warning(record, toy_runner):
    report = reproduce_record(_pinned(record, "bfloat16"), toy_runner)
    assert report.ok, report.mismatches
    assert any("reproduced in float32; the record pins bfloat16" in w for w in report.warnings)
    float16_on_float32 = reproduce_record(_pinned(record, "float16"), toy_runner)
    assert float16_on_float32.ok


def _at_threshold(record, toy_runner):
    """(index, under, over): a passed hard negative, and values either side of its threshold."""
    results = record.verdict.adversarial_test_results
    i = next(n for n, r in enumerate(results) if r.kind == "hard_negative" and r.passed)
    threshold = record.verdict.policy.activation_threshold * record.submission.max_activation
    return i, round(threshold * 0.99, 6), round(threshold * 1.01, 6)


def _flip(record, toy_runner, *, forge: bool):
    from openinterp.half_precision import rederived
    from openinterp.reproduce import DEFAULT_TOLERANCE, ReproductionReport, _compare_verdicts

    i, under, over = _at_threshold(record, toy_runner)
    # the record, consistent with itself: the probe just under its threshold, so it passed
    old = edit_result(record, i, activation=under).verdict
    if forge:  # a pass that doesn't follow from the recorded number
        old = edit_result(record, i, activation=over).verdict
    new_results = list(old.adversarial_test_results)
    new_results[i] = new_results[i].model_copy(update={"activation": over, "passed": False})
    new = rederived(old.model_copy(update={"adversarial_test_results": new_results}))
    report = ReproductionReport(path=None, record=record)
    tolerance = DEFAULT_TOLERANCE.for_dtype("bfloat16")
    _compare_verdicts(old, new, tolerance, record.submission.max_activation, report, half=True)
    return report


def test_a_16_bit_result_that_flips_at_its_threshold_is_a_warning(record, toy_runner):
    report = _flip(record, toy_runner, forge=False)
    assert report.ok, report.mismatches
    assert any("within 16-bit rounding" in w for w in report.warnings)


def test_a_16_bit_pass_that_its_own_number_contradicts_is_caught(record, toy_runner):
    report = _flip(record, toy_runner, forge=True)
    assert any("does not follow from its number" in m for m in report.mismatches)


def test_a_16_bit_record_whose_status_doesnt_follow_from_its_results_is_caught(record, toy_runner):
    from openinterp.reproduce import DEFAULT_TOLERANCE, ReproductionReport, _compare_verdicts

    i, under, over = _at_threshold(record, toy_runner)
    old = edit_result(record, i, activation=under).verdict
    old = old.model_copy(update={"status": "VERIFIED" if old.status != "VERIFIED" else "REJECTED"})
    new_results = list(old.adversarial_test_results)
    new_results[i] = new_results[i].model_copy(update={"activation": over, "passed": False})
    new = old.model_copy(update={"adversarial_test_results": new_results})
    report = ReproductionReport(path=None, record=record)
    tolerance = DEFAULT_TOLERANCE.for_dtype("bfloat16")
    _compare_verdicts(old, new, tolerance, record.submission.max_activation, report, half=True)
    assert any("status recorded" in m for m in report.mismatches)


def test_a_float32_result_that_flips_still_fails(record, toy_runner):
    from openinterp.reproduce import DEFAULT_TOLERANCE, ReproductionReport, _compare_verdicts

    i, under, over = _at_threshold(record, toy_runner)
    old = edit_result(record, i, activation=under).verdict
    new_results = list(old.adversarial_test_results)
    new_results[i] = new_results[i].model_copy(update={"activation": over, "passed": False})
    new = old.model_copy(update={"adversarial_test_results": new_results})
    report = ReproductionReport(path=None, record=record)
    _compare_verdicts(old, new, DEFAULT_TOLERANCE, record.submission.max_activation, report)
    assert any("reproduced passed=False" in m for m in report.mismatches)


def _edit_tokens(record, count: int, by: float):
    """``record`` with ``count`` non-peak tokens of its first example moved by ``by`` × scale."""
    sub = record.submission
    ex = sub.top_examples[0]
    peak = max(range(len(ex.activations)), key=lambda i: ex.activations[i] or 0.0)
    acts = list(ex.activations)
    for i in [i for i in range(len(acts)) if i != peak and acts[i] is not None][:count]:
        acts[i] = acts[i] + by * sub.max_activation
    edited = ex.model_copy(update={"activations": acts})
    return record.model_copy(
        update={
            "submission": sub.model_copy(update={"top_examples": [edited, *sub.top_examples[1:]]})
        }
    )


def test_a_16_bit_record_may_have_a_few_stray_tokens(record, toy_runner, monkeypatch):
    import openinterp.reproduce

    monkeypatch.setattr(openinterp.reproduce, "HALF_STRAY_TOKENS", 2)  # the toy has 5 tokens
    half = _pinned(record, "bfloat16")
    few = reproduce_record(_edit_tokens(half, 2, -0.2), toy_runner)
    assert few.ok, few.mismatches
    assert any("beyond the 16-bit tolerance" in w for w in few.warnings)
    many = reproduce_record(_edit_tokens(half, 3, -0.2), toy_runner)
    assert any("top example 1: activations differ" in m for m in many.mismatches)
    float32 = reproduce_record(_edit_tokens(record, 1, -0.2), toy_runner)
    assert not float32.ok


def test_a_10_percent_edit_of_a_16_bit_peak_is_caught(record, toy_runner):
    half = _pinned(record, "bfloat16")
    sub = half.submission
    louder = [
        ex.model_copy(update={"max_activation": ex.max_activation * 1.1}) for ex in sub.top_examples
    ]
    tampered = half.model_copy(
        update={"submission": sub.model_copy(update={"top_examples": louder})}
    )
    assert any(
        "max_activation differs" in m for m in reproduce_record(tampered, toy_runner).mismatches
    )
