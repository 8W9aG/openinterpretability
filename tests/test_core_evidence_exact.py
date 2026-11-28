"""The evidence store's names, lookups and branch reads, at their edges (#6491).

Each store is under ``tmp_path``. The dataset fetch and git are stand-ins, so no test uses
the network, the real store or the real evidence branches.
"""

import json
import re

import pytest

from openinterpretability import evidence_branch as eb
from openinterpretability.core import evidence
from openinterpretability.core.schemas import AuditRecord

SHA = "ab" + "0" * 62


@pytest.fixture
def offline(monkeypatch):
    """No dataset fetch: the default fetch finds nothing."""
    monkeypatch.setattr(evidence, "_fetch", lambda sha: None)


def test_the_store_defaults_to_the_evidence_folder_of_the_cache(tmp_path, monkeypatch):
    monkeypatch.delenv(evidence.DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert evidence.store_dir() == tmp_path / "openinterpretability" / "evidence"


def test_a_blob_is_named_by_the_first_two_characters_of_its_hash():
    assert evidence.blob_name("abcdef") == "ab/abcdef.json"


def test_canonical_bytes_sort_the_keys_drop_the_spaces_and_keep_utf8():
    value = {"b": [1, 2], "a": "é"}
    assert evidence.canonical(value) == '{"a":"é","b":[1,2]}'.encode()


def test_the_branch_fallback_is_off_when_the_variable_is_0(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv(evidence.BRANCHES_ENV, "0")
    monkeypatch.setattr(evidence, "_BRANCHES", {})
    monkeypatch.setattr(eb, "fetch_branches", lambda: calls.append("fetch"))
    monkeypatch.setattr(eb, "_git", lambda *args: calls.append(args) or b"")
    assert evidence._from_branches(SHA, tmp_path) is None
    assert calls == []


def test_the_branch_fallback_lists_the_local_refs_and_asks_for_the_one_blob(tmp_path, monkeypatch):
    data = b'{"x":1}'
    sha = evidence.sha256(data)
    gits, loads = [], []

    def git(*args):
        gits.append(args)
        return b"refs/evidence/a\n"

    def load_branch(ref, root, only):
        loads.append((ref, root, only))
        evidence.put(data, root)
        return [sha]

    monkeypatch.setenv(evidence.BRANCHES_ENV, "1")
    monkeypatch.setattr(evidence, "_BRANCHES", {"fetched": True})
    monkeypatch.setattr(eb, "_git", git)
    monkeypatch.setattr(eb, "load_branch", load_branch)
    assert evidence._from_branches(sha, tmp_path) == data
    assert gits == [("for-each-ref", "--format=%(refname)", eb.LOCAL_REFS)]
    assert loads == [("refs/evidence/a", tmp_path, {sha})]


def test_get_asks_the_branches_for_the_given_store(tmp_path, monkeypatch, offline):
    asked = []
    monkeypatch.setattr(evidence, "_from_branches", lambda sha, root: asked.append(root))
    with pytest.raises(evidence.EvidenceMissing):
        evidence.get(SHA, tmp_path)
    assert asked == [tmp_path]


def test_a_missing_blob_names_the_given_store(tmp_path, monkeypatch, offline):
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "default"))
    monkeypatch.setenv(evidence.BRANCHES_ENV, "0")
    root = tmp_path / "given"
    message = f"evidence blob {SHA} is not in {root} or {evidence.EVIDENCE_REPO}"
    with pytest.raises(evidence.EvidenceMissing, match=f"^{re.escape(message)}$"):
        evidence.get(SHA, root)


def test_a_fetched_blob_is_stored_in_the_given_store(tmp_path, monkeypatch):
    default = tmp_path / "default"
    monkeypatch.setenv(evidence.DIR_ENV, str(default))
    data = b"[1]"
    sha = evidence.sha256(data)
    root = tmp_path / "given"
    assert evidence.get(sha, root, fetch=lambda s: data) == data
    assert (root / evidence.blob_name(sha)).read_bytes() == data
    assert not default.exists()


def test_split_leaves_a_reference_as_it_is(tmp_path):
    ref = {evidence.REF_KEY: SHA}
    data = {"submission": {"top_examples": ref}}
    assert evidence.split(data, tmp_path) == data
    assert not tmp_path.joinpath(SHA[:2]).exists()


def test_split_moves_the_second_field_when_the_first_parent_is_missing(tmp_path):
    scores = [{"score": 1}]
    out = evidence.split({"verdict": {"explanation_scores": scores}}, tmp_path)
    sha = evidence.sha256(evidence.canonical(scores))
    assert out == {"verdict": {"explanation_scores": {evidence.REF_KEY: sha}}}


def test_resolve_reads_the_given_store(tmp_path, monkeypatch, offline):
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "default"))
    monkeypatch.setenv(evidence.BRANCHES_ENV, "0")
    root = tmp_path / "given"
    sha = evidence.put(b"[2]", root)
    data = {"submission": {"top_examples": {evidence.REF_KEY: sha}}}
    assert evidence.resolve(data, root) == {"submission": {"top_examples": [2]}}


def test_resolve_uses_the_given_fetch(tmp_path, monkeypatch, offline):
    monkeypatch.setenv(evidence.BRANCHES_ENV, "0")
    sha = evidence.sha256(b"[3]")
    data = {"verdict": {"explanation_scores": {evidence.REF_KEY: sha}}}
    out = evidence.resolve(data, tmp_path, fetch=lambda s: b"[3]")
    assert out == {"verdict": {"explanation_scores": [3]}}


def _lax(record):
    """The record's JSON with an int field written as a string, which only lax mode takes."""
    data = json.loads(record.model_dump_json())
    data["submission"]["model_num_layers"] = "12"
    return data


def test_model_validate_passes_its_options_on_for_a_whole_record(record):
    loaded = AuditRecord.model_validate(_lax(record), strict=False)
    assert loaded.submission.model_num_layers == 12


def test_model_validate_passes_its_options_on_for_a_slim_record(
    record, tmp_path, monkeypatch, offline
):
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path))
    slim = evidence.split(_lax(record))
    loaded = AuditRecord.model_validate(slim, strict=False)
    assert loaded.submission.model_num_layers == 12
    assert loaded.submission.top_examples == record.submission.top_examples
