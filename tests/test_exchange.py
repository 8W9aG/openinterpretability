"""What papers and analyses read from exchange/ must be published and pinned (#3255)."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
# Imported by name. Ruff sorts a script as first-party only where a folder of its name sits at
# the root. exchange/ is a local, ignored folder, so the sort would differ in CI.
exchange = importlib.import_module("exchange")


@pytest.fixture(autouse=True)
def sample_project(monkeypatch):
    """A project's own outputs and kept-back inputs, as [tool.labinacan.exchange] gives them."""
    monkeypatch.setattr(exchange, "OUTPUTS", {"bench", ".published"})
    monkeypatch.setattr(exchange, "OUTPUTS_FOLDED", {"bench", ".published"})
    monkeypatch.setattr(exchange, "UNPUBLISHED", {"samples/*.raw.jsonl": "the raw corpus"})


def _repo(tmp_path, monkeypatch, text):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(text)
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "add", "."], check=True)


def test_cited_inputs_leave_out_outputs_and_working_folders(tmp_path, monkeypatch):
    text = (
        "Read exchange/spd-logs/A.log and exchange/basis, write exchange/ev.json, "
        "exchange/review.prompt.md and exchange/bench/u.tests.json."
    )
    _repo(tmp_path, monkeypatch, text)
    assert exchange.cited() == {"spd-logs", "basis"}


def test_check_needs_every_cited_input_in_the_pinned_manifest(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, "Read exchange/spd-logs and exchange/samples.")
    assert exchange.missing() == ["spd-logs"]  # samples is partly unpublished on purpose
    (tmp_path / "data").mkdir()
    exchange.write_manifest({"repo": "r", "revision": None, "files": {"spd-logs/A.log": "0" * 64}})
    assert exchange.missing() == []
    assert exchange.read_manifest()["files"] == {"spd-logs/A.log": "0" * 64}  # a round trip


def test_staging_rewrites_the_home_folder_and_leaves_out_raw_files(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, "Read exchange/basis and exchange/samples.")
    (tmp_path / "exchange" / "basis").mkdir(parents=True)
    (tmp_path / "exchange" / "basis" / "s.json").write_text('{"repo": "/home/me/.cache/x"}')
    (tmp_path / "exchange" / "samples").mkdir()
    (tmp_path / "exchange" / "samples" / "year.raw.jsonl").write_text("{}")
    (tmp_path / "exchange" / "samples" / "year.clean.jsonl").write_text("{}")
    manifest = exchange.stage("/home/me")
    assert set(manifest["files"]) == {"basis/s.json", "samples/year.clean.jsonl"}
    staged = (tmp_path / exchange.STAGING / "basis" / "s.json").read_text()
    assert staged == '{"repo": "~/.cache/x"}'
    assert (tmp_path / exchange.STAGING / "README.md").exists()


def test_staging_changes_no_byte_but_the_home_folder(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, "Read exchange/spd-logs.")
    (tmp_path / "exchange" / "spd-logs").mkdir(parents=True)
    raw = b"progress 10%\rprogress 100%\r\nsaved /home/me/run\n"
    (tmp_path / "exchange" / "spd-logs" / "A.log").write_bytes(raw)
    exchange.stage("/home/me")
    staged = (tmp_path / exchange.STAGING / "spd-logs" / "A.log").read_bytes()
    assert staged == raw.replace(b"/home/me", b"~")


def test_prose_like_exchange_dot_cites_nothing(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, "Outputs go to exchange/. and exchange/... and exchange/.hidden.")
    assert exchange.cited() == set()


def test_staging_never_takes_private_or_output_files(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, "Read exchange/basis.")
    (tmp_path / "exchange" / "basis").mkdir(parents=True)
    (tmp_path / "exchange" / "basis" / "s.json").write_text("{}")
    (tmp_path / "exchange" / "basis" / "t.labels.private.jsonl").write_text("{}")
    manifest = exchange.stage("/home/me")
    assert set(manifest["files"]) == {"basis/s.json"}
    assert exchange._excluded("bench/test.tests.json")


def test_staging_refuses_a_name_outside_exchange(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, "")
    (tmp_path / "exchange").mkdir()
    monkeypatch.setattr(exchange, "cited", lambda: {"../docs"})
    try:
        exchange.stage("/home/me")
    except SystemExit as e:
        assert "not a name inside exchange/" in str(e)
    else:
        raise AssertionError("a name outside exchange/ was staged")


def test_a_case_variant_of_an_output_folder_cites_nothing(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, "Tests go to exchange/BENCH and exchange/Bench/u.tests.json.")
    assert exchange.cited() == set()
    assert exchange._excluded("BENCH/u.tests.json")
    assert exchange._excluded("basis/t.labels.PRIVATE.jsonl")
    assert not exchange._citable("REVIEW.prompt.md")
