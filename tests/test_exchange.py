"""What papers and analyses read from exchange/ must be published and pinned (#3255)."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import exchange  # noqa: E402


def _repo(tmp_path, monkeypatch, text):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(text)
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "add", "."], check=True)


def test_cited_inputs_leave_out_outputs_and_working_folders(tmp_path, monkeypatch):
    text = (
        "Read exchange/spd-logs/A.log and exchange/basis, write exchange/ev.json, "
        "exchange/review.prompt.md and exchange/bench/u.probes.json."
    )
    _repo(tmp_path, monkeypatch, text)
    assert exchange.cited() == {"spd-logs", "basis"}


def test_check_needs_every_cited_input_in_the_pinned_manifest(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, "Read exchange/spd-logs and exchange/probe-comparison.")
    assert exchange.missing() == ["spd-logs"]  # probe-comparison is partly unpublished on purpose
    (tmp_path / "data").mkdir()
    exchange.write_manifest({"repo": "r", "revision": None, "files": {"spd-logs/A.log": "0" * 64}})
    assert exchange.missing() == []
    assert exchange.read_manifest()["files"] == {"spd-logs/A.log": "0" * 64}  # a round trip


def test_staging_rewrites_the_home_folder_and_leaves_out_pile_files(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, "Read exchange/basis and exchange/probe-comparison.")
    (tmp_path / "exchange" / "basis").mkdir(parents=True)
    (tmp_path / "exchange" / "basis" / "s.json").write_text('{"repo": "/home/me/.cache/x"}')
    (tmp_path / "exchange" / "probe-comparison").mkdir()
    (tmp_path / "exchange" / "probe-comparison" / "year.pile.jsonl").write_text("{}")
    (tmp_path / "exchange" / "probe-comparison" / "year.fineweb.jsonl").write_text("{}")
    manifest = exchange.stage("/home/me")
    assert set(manifest["files"]) == {"basis/s.json", "probe-comparison/year.fineweb.jsonl"}
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
