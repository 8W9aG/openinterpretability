"""``open-interpretability evidence``: carry slim records' blobs to the dataset (#3254)."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from openinterpretability import evidence_branch as eb
from openinterpretability.cli import cli
from openinterpretability.core import evidence


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv(evidence.DIR_ENV, str(root))
    monkeypatch.setattr(evidence, "_fetch", lambda sha: None)  # never the network
    return root


def _write(record, path, slim=True):
    data = json.loads(record.model_dump_json())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence.split(data) if slim else data), encoding="utf-8")
    return path


def _changed(monkeypatch, paths):
    """The records that ``git diff`` names since the base ref."""
    monkeypatch.setattr("openinterpretability.checks.changed_records", lambda wiki, base: paths)


def test_push_with_no_slim_record_pushes_nothing(record, store, tmp_path, monkeypatch):
    _changed(monkeypatch, [_write(record, tmp_path / "whole.json", slim=False)])
    monkeypatch.setattr(eb, "push", lambda *a: pytest.fail("pushed"))
    out = CliRunner().invoke(cli, ["evidence", "push", "--wiki-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "nothing to push" in out.output


def test_push_sends_the_added_records_blobs_to_the_branch(record, store, tmp_path, monkeypatch):
    slim = _write(record, tmp_path / "wiki/experiments/a/L3/N1-positive/r.json")
    explanation = tmp_path / "wiki/explanations/m/p/e.json"  # left out, so never read
    explanation.parent.mkdir(parents=True)
    explanation.write_text("not json")
    _changed(monkeypatch, [slim, explanation])
    seen = {}

    def push(branch, shas, remote):
        seen.update(branch=branch, shas=set(shas), remote=remote)
        return f"evidence/{branch}"

    def git(args, **kwargs):
        assert args == ["git", "branch", "--show-current"]
        return SimpleNamespace(stdout="experiment/x\n")

    monkeypatch.setattr(eb, "push", push)
    monkeypatch.setattr(subprocess, "run", git)
    out = CliRunner().invoke(
        cli, ["evidence", "push", "--wiki-dir", str(tmp_path), "--remote", "up"]
    )
    assert out.exit_code == 0, out.output
    assert seen["shas"] == eb.record_refs([slim]) and seen["shas"]
    assert (seen["branch"], seen["remote"]) == ("experiment/x", "up")
    assert f"Pushed {len(seen['shas'])} evidence blobs to up/evidence/experiment/x." in out.output


def test_load_reports_the_blobs_it_copied(monkeypatch):
    monkeypatch.setattr(eb, "load_branch", lambda ref: ["a", "b"] if ref == "FETCH_HEAD" else [])
    out = CliRunner().invoke(cli, ["evidence", "load", "--ref", "FETCH_HEAD"])
    assert out.exit_code == 0, out.output
    assert "Loaded 2 evidence blobs from FETCH_HEAD." in out.output


def test_load_all_reads_every_evidence_branch_of_the_remote(monkeypatch):
    fetched = []
    blobs = {"refs/evidence/a": ["1"], "refs/evidence/b": ["2", "3"]}
    monkeypatch.setattr(eb, "fetch_branches", fetched.append)
    monkeypatch.setattr(eb, "_git", lambda *args: "\n".join(blobs).encode())
    monkeypatch.setattr(eb, "load_branch", blobs.__getitem__)
    out = CliRunner().invoke(cli, ["evidence", "load-all", "--remote", "up"])
    assert out.exit_code == 0, out.output
    assert fetched == ["up"]
    assert "Loaded 3 evidence blobs from 2 evidence branches." in out.output


def _publish(monkeypatch, tmp_path, missing):
    from openinterpretability.core.paths import experiments_dir

    root = experiments_dir(tmp_path)
    for name in ["b/r.json", "a/r.json"]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("{}")
    seen = []

    def publish_missing(records):
        seen.extend(records)
        return 2, missing

    monkeypatch.setattr(eb, "fetch_branches", lambda remote: None)
    monkeypatch.setattr(eb, "publish_missing", publish_missing)
    out = CliRunner().invoke(cli, ["evidence", "publish", "--wiki-dir", str(tmp_path)])
    return out, [p.relative_to(root).as_posix() for p in seen]


def test_publish_uploads_the_blobs_of_every_record(monkeypatch, tmp_path):
    out, seen = _publish(monkeypatch, tmp_path, [])
    assert out.exit_code == 0, out.output
    assert seen == ["a/r.json", "b/r.json"]
    assert f"Uploaded 2 evidence blobs to {evidence.EVIDENCE_REPO}." in out.output


def test_publish_fails_when_a_blob_is_on_no_branch(monkeypatch, tmp_path):
    out, _ = _publish(monkeypatch, tmp_path, ["abc"])
    assert out.exit_code == 1
    assert "1 blobs are on no evidence branch: ['abc']" in out.output


def test_verify_passes_when_every_added_record_has_its_evidence(record, store, tmp_path,
                                                                monkeypatch):  # fmt: skip
    _changed(monkeypatch, [_write(record, tmp_path / "r.json")])
    out = CliRunner().invoke(cli, ["evidence", "verify", "--wiki-dir", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "Every added record has its evidence." in out.output


def test_verify_names_each_record_that_lacks_its_evidence(record, tmp_path, monkeypatch):
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "store"))
    monkeypatch.setattr(evidence, "_fetch", lambda sha: None)
    path = _write(record, tmp_path / "r.json")
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "empty"))
    _changed(monkeypatch, [path])
    out = CliRunner().invoke(cli, ["evidence", "verify", "--wiki-dir", str(tmp_path)])
    assert out.exit_code == 1
    assert str(path) in out.output
    assert "1 records lack their evidence" in out.output
