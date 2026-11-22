"""The evidence store: repeated puts, the dataset fetch and the branch fallback.

The dataset download and git are replaced by stand-ins, so no network is used.
"""

import huggingface_hub

from openinterpretability import evidence_branch as eb
from openinterpretability.core import evidence


def test_putting_a_stored_blob_again_keeps_the_one_copy(tmp_path):
    sha = evidence.put(b"data", tmp_path)
    path = tmp_path / evidence.blob_name(sha)
    before = path.stat().st_mtime_ns
    assert evidence.put(b"data", tmp_path) == sha
    assert path.stat().st_mtime_ns == before and path.read_bytes() == b"data"
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temporary file left


def test_the_dataset_fetch_reads_the_downloaded_blob(tmp_path, monkeypatch):
    blob = tmp_path / "blob.json"
    blob.write_bytes(b"{}")
    asked = []

    def download(repo, name, repo_type):
        asked.append((repo, name, repo_type))
        return str(blob)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    sha = "ab" + "0" * 62
    assert evidence._fetch(sha) == b"{}"
    assert asked == [(evidence.EVIDENCE_REPO, evidence.blob_name(sha), "dataset")]


def test_a_failed_dataset_fetch_is_none(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    assert evidence._fetch("0" * 64) is None


def test_the_branch_fallback_tries_each_branch_until_one_holds_the_blob(tmp_path, monkeypatch):
    data = b'{"x":1}'
    sha = evidence.sha256(data)
    tried = []

    def load_branch(ref, root, only):
        tried.append(ref)
        if ref == "refs/b":
            evidence.put(data, root)
            return True
        return False

    monkeypatch.setenv(evidence.BRANCHES_ENV, "1")
    monkeypatch.setattr(evidence, "_BRANCHES", {"fetched": True})
    monkeypatch.setattr(eb, "_git", lambda *args: b"refs/a\nrefs/b\n")
    monkeypatch.setattr(eb, "load_branch", load_branch)
    monkeypatch.setattr(evidence, "_fetch", lambda sha: None)
    assert evidence.get(sha, tmp_path) == data
    assert tried == ["refs/a", "refs/b"]


def test_resolve_returns_anything_but_a_record_as_it_is():
    assert evidence.resolve([1, 2]) == [1, 2]
    assert evidence.resolve(None) is None and evidence.resolve("text") == "text"
