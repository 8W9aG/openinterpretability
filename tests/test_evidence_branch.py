"""Evidence blobs ride on their own branch to the dataset (evidence_branch.py, #3254)."""

import json
import subprocess

import pytest

from openinterpretability import evidence_branch as eb
from openinterpretability.core import evidence

GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    work = tmp_path / "repo"
    work.mkdir()
    subprocess.run([*GIT, "init", "-q"], cwd=work, check=True)
    monkeypatch.chdir(work)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    monkeypatch.setattr(evidence, "_fetch", lambda sha: None)  # never the network
    return work


def _slim_record(record, tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv(evidence.DIR_ENV, str(store))
    slim = evidence.split(json.loads(record.model_dump_json()))
    path = tmp_path / "record.json"
    path.write_text(json.dumps(slim))
    return path, store


def test_blobs_go_to_a_parentless_commit_and_load_back(record, repo, tmp_path, monkeypatch):
    path, _ = _slim_record(record, tmp_path, monkeypatch)
    shas = eb.record_refs([path])
    assert shas  # the record refers to its top examples at least
    commit = eb.blob_commit(shas)
    parents = subprocess.run(["git", "rev-list", "--parents", "-n1", commit],
                             capture_output=True, text=True, check=True).stdout.split()  # fmt: skip
    assert parents == [commit]  # no parent: nothing of main's history comes with it
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit],
                           capture_output=True, text=True, check=True).stdout.split()  # fmt: skip
    assert sorted(names) == sorted(evidence.blob_name(s) for s in shas)
    fresh = tmp_path / "fresh-store"
    assert sorted(eb.load_branch(commit, fresh)) == sorted(shas)
    monkeypatch.setenv(evidence.DIR_ENV, str(fresh))
    assert eb.unresolved([path]) == []


def test_verify_names_a_record_whose_blob_is_missing(record, repo, tmp_path, monkeypatch):
    path, _ = _slim_record(record, tmp_path, monkeypatch)
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "empty"))
    (problem,) = eb.unresolved([path])
    assert str(path) in problem and evidence.EVIDENCE_REPO in problem


def test_upload_creates_the_public_dataset_and_sends_each_blob(record, repo, tmp_path, monkeypatch):
    path, store = _slim_record(record, tmp_path, monkeypatch)
    shas = sorted(eb.record_refs([path]))
    calls = {}

    class FakeApi:
        exists = False

        def repo_exists(self, repo_id, repo_type):
            return self.exists

        def create_repo(self, repo_id, **kwargs):
            calls["create"] = (repo_id, kwargs)

        def upload_folder(self, repo_id, folder_path, **kwargs):
            from pathlib import Path

            calls["files"] = sorted(
                p.relative_to(folder_path).as_posix() for p in Path(folder_path).rglob("*.json")
            )

    assert eb.upload(shas, store, api=FakeApi()) == len(shas)
    assert calls["create"] == (evidence.EVIDENCE_REPO, {"repo_type": "dataset", "private": False})
    assert calls["files"] == sorted(evidence.blob_name(s) for s in shas)
    assert eb.upload([], store, api=FakeApi()) == 0
    calls.clear()
    existing = FakeApi()
    existing.exists = True  # a token with write access to the dataset only can't create it
    assert eb.upload(shas, store, api=existing) == len(shas)
    assert "create" not in calls


def test_a_draw_pushes_its_evidence_before_its_branch():
    from openinterpretability.draw_steps import _post

    commands = _post("k", "explore").commands
    assert commands.index("uv run open-interpretability evidence push") < commands.index(
        "git push && gh pr ready"
    )


def test_publish_uploads_only_what_records_need_and_the_dataset_lacks(
    record, repo, tmp_path, monkeypatch
):
    path, _ = _slim_record(record, tmp_path, monkeypatch)
    shas = sorted(eb.record_refs([path]))
    stray = evidence.put(b'{"not": "referenced"}')  # on the branch, named by no record
    commit = eb.blob_commit([*shas, stray])
    subprocess.run(["git", "update-ref", f"{eb.LOCAL_REFS}some-branch", commit], check=True)
    sent = []

    class FakeApi:
        def repo_exists(self, repo_id, repo_type):
            return True

        def list_repo_files(self, repo_id, repo_type):
            return [evidence.blob_name(shas[0])]  # one blob is already there

        def upload_folder(self, repo_id, folder_path, **kwargs):
            from pathlib import Path

            sent.extend(Path(p).stem for p in Path(folder_path).rglob("*.json"))

    fresh = tmp_path / "fresh-store"
    uploaded, missing = eb.publish_missing([path], fresh, api=FakeApi())
    assert sorted(sent) == shas[1:] and uploaded == len(shas) - 1 and missing == []
    assert stray not in sent


def test_load_all_loads_every_evidence_branch(record, repo, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from openinterpretability.cli import cli

    path, _ = _slim_record(record, tmp_path, monkeypatch)
    shas = eb.record_refs([path])
    subprocess.run(["git", "update-ref", f"{eb.LOCAL_REFS}a", eb.blob_commit(shas)], check=True)
    fresh = tmp_path / "fresh-store"
    monkeypatch.setenv(evidence.DIR_ENV, str(fresh))
    monkeypatch.setattr(eb, "fetch_branches", lambda remote: None)
    result = CliRunner().invoke(cli, ["evidence", "load-all"])
    assert result.exit_code == 0, result.output
    assert f"Loaded {len(shas)} evidence blobs from 1 evidence branches." in result.output
    assert eb.unresolved([path]) == []


def test_a_blob_missing_from_the_dataset_is_read_from_its_evidence_branch(
    record, repo, tmp_path, monkeypatch
):
    # A record merged before the deploy published its blobs (#4530).
    path, store = _slim_record(record, tmp_path, monkeypatch)
    shas = eb.record_refs([path])
    commit = eb.blob_commit(shas)
    subprocess.run(["git", "update-ref", f"{eb.LOCAL_REFS}experiment/x", commit], check=True)
    for sha in shas:
        (store / evidence.blob_name(sha)).unlink()  # gone from the store, and not on the dataset
    monkeypatch.setenv(evidence.BRANCHES_ENV, "1")
    monkeypatch.setattr(evidence, "_BRANCHES", {"fetched": True})  # the branches are already here
    sha = next(iter(shas))
    assert evidence.sha256(evidence.get(sha)) == sha and (store / evidence.blob_name(sha)).exists()


def test_the_fallback_fetches_once_and_a_git_error_is_not_found(
    record, repo, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setenv(evidence.BRANCHES_ENV, "1")
    monkeypatch.setattr(evidence, "_BRANCHES", {})
    monkeypatch.setattr(eb, "fetch_branches", lambda *a: calls.append(1))
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "empty"))
    for _ in range(2):
        with pytest.raises(evidence.EvidenceMissing):
            evidence.get("0" * 64)
    assert calls == [1]  # fetched once per process
    monkeypatch.setattr(evidence, "_BRANCHES", {})
    monkeypatch.setattr(eb, "fetch_branches", lambda *a: (_ for _ in ()).throw(OSError("no git")))
    with pytest.raises(evidence.EvidenceMissing):
        evidence.get("1" * 64)  # a git error is "not found", not a crash


def test_only_experiment_records_are_checked_for_blobs(monkeypatch):
    # An explanation record has no evidence blobs, and isn't an AuditRecord (#4793).
    from pathlib import Path

    from openinterpretability import cli_evidence_store

    added = [Path("wiki/experiments/m/L1/N2-positive/1.json"),
             Path("wiki/explanations/m/abc123/20261119T013311Z.json")]  # fmt: skip
    monkeypatch.setattr("openinterpretability.checks.changed_records", lambda *a: added)
    assert cli_evidence_store._added(Path("wiki"), "main") == added[:1]
