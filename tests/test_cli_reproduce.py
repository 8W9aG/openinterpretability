"""The reproduce and sign commands' own paths (cli_reproduce.py)."""

from datetime import date
from pathlib import Path

from click.testing import CliRunner

from openinterpretability.cli import cli
from openinterpretability.cli_reproduce import _default_key, _rebuild_hint, _Reproducer
from openinterpretability.core.errata import Erratum, ErratumRule
from tests.test_dead import PROVENANCE


def _reproducer(**options):
    """A reproducer that trusts no signer and may load any model."""
    defaults = {"signers": "", "landed": {}, "tolerance": None, "scan_corpora": {},
                "allowed_models": (), "device": "cpu"}  # fmt: skip
    return _Reproducer(**(defaults | options))


def test_no_records_means_nothing_to_reproduce():
    result = CliRunner().invoke(cli, ["reproduce"])
    assert result.exit_code == 0, result.output
    assert result.output == "No records to reproduce.\n"


def test_a_file_that_is_not_a_record_fails(tmp_path):
    path = tmp_path / "r.json"
    path.write_text('{"submission": {}}')
    assert _reproducer().check(path) == ("FAILED", ["invalid experiment record"], [])


def test_a_record_an_erratum_sets_aside_is_skipped_unless_new(tmp_path, record):
    path = tmp_path / "r.json"
    path.write_text(record.model_dump_json())
    rule = ErratumRule(models=[record.submission.identifier.model_name])
    erratum = Erratum(id="bad-scan", date=date(2026, 11, 1), title="t", bug="b",
                      fixed_in="abcdef1", set_aside=True, rule=rule)  # fmt: skip
    skipped = _reproducer(errata=[erratum]).check(path)
    assert skipped == ("SET ASIDE by erratum bad-scan (reproduction skipped)", [], [])
    new = _reproducer(errata=[erratum], new={path.resolve()}).check(path)
    assert new == ("FAILED", ["a new record can't be set aside (erratum bad-scan)"], [])


def test_a_scan_pinned_to_auto_is_reproduced_in_the_reviews_dtype(record, monkeypatch):
    from openinterpretability.core.run_environment import RunEnvironment

    loads = []
    monkeypatch.setattr(
        "openinterpretability.cli._load_runner",
        lambda model, device, dtype, revision=None, attn=None: loads.append(dtype) or "runner",
    )
    prov = PROVENANCE.model_copy(update={"dtype": "auto"})
    env = RunEnvironment(device="cpu", dtype="bfloat16", openinterp_version="t", torch_version="t")
    record = record.model_copy(
        update={"submission": record.submission.model_copy(update={"provenance": prov}),
                "verdict": record.verdict.model_copy(update={"environment": env})}
    )  # fmt: skip
    assert _reproducer().runner(record, scan=True) == "runner"
    assert loads == ["float32"]  # the review's bfloat16, reproduced in float32


def test_the_default_key_is_the_new_key_file_on_a_new_machine(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))  # no key files of either name
    assert _default_key() == Path("~/.ssh/openinterpretability_signing")


def test_a_record_with_a_known_source_says_how_to_rebuild_it(record):
    from openinterpretability.corpus_rebuild import COMMITTED_SIDECARS, load_sources

    sha, source = next(iter(load_sources(COMMITTED_SIDECARS).items()))
    prov = PROVENANCE.model_copy(update={"corpus_sha256": sha, "corpus_source": source})
    record = record.model_copy(
        update={"submission": record.submission.model_copy(update={"provenance": prov})}
    )
    hint = _rebuild_hint(record, Path("r.json"))
    assert hint == "Rebuild it with `open-interpretability corpus --same-as r.json --out <file>`."


def test_a_key_outside_its_window_signs_nothing(tmp_path, record):
    from tests.test_signing import keypair, signer_line

    path = tmp_path / "r.json"
    path.write_text(record.model_dump_json())
    key = keypair(tmp_path, "maintainer")
    signers = tmp_path / "allowed_signers"
    signers.write_text(signer_line("will", key, 'valid-before="20200101Z"'))  # revoked
    args = ["sign", str(path), "--key", str(key), "--allowed-signers", str(signers)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert f"Did not sign {path}" in result.output and "not valid now" in result.output
    assert not path.with_name("r.json.sig").exists()


def _reproduce_a_non_record(tmp_path):
    path = tmp_path / "r.json"
    path.write_text('{"submission": {}}')
    return path, CliRunner().invoke(cli, ["reproduce", str(path)])


def test_outside_github_actions_no_step_summary_is_written(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    path, result = _reproduce_a_non_record(tmp_path)
    assert result.exit_code == 1
    assert "0/1 record(s) reproduced." in result.output
    assert f"::error file={path}" not in result.output
    assert not summary.exists()


def test_in_github_actions_without_a_step_summary_the_result_is_only_echoed(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    path, result = _reproduce_a_non_record(tmp_path)
    assert result.exit_code == 1
    assert f"::error file={path}::" in result.output
    assert "0/1 record(s) reproduced." in result.output
