"""Evidence blobs ride on their own branch to the dataset (openinterp/evidence_branch.py, #3254)."""

import json
import subprocess

import pytest

from openinterp import evidence_branch as eb
from openinterp.core import evidence

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
    from openinterp.draw_steps import _post

    commands = _post("k", "explore").commands
    assert commands.index("uv run open-interpretability evidence push") < commands.index(
        "git push && gh pr ready"
    )
