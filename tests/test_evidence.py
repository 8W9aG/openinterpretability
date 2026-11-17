"""Slim records: bulky evidence in a content-addressed store (issue #3254)."""

import json

import pytest

from openinterpretability.builder.wiki import archive_record
from openinterpretability.core import evidence
from openinterpretability.core.schemas import AuditRecord


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    monkeypatch.setenv(evidence.DIR_ENV, str(root))
    monkeypatch.setattr(evidence, "_fetch", lambda sha: None)  # never the network
    return root


def test_a_slim_record_loads_as_the_whole_one(record, store):
    whole = json.loads(record.model_dump_json())
    slim = evidence.split(whole)
    assert evidence.is_ref(slim["submission"]["top_examples"])
    assert len(json.dumps(slim)) < len(json.dumps(whole))
    loaded = AuditRecord.model_validate_json(json.dumps(slim))
    assert loaded.model_dump_json() == record.model_dump_json()
    assert AuditRecord.model_validate(slim).model_dump_json() == record.model_dump_json()


def test_a_whole_record_is_untouched(record, store):
    whole = json.loads(record.model_dump_json())
    assert evidence.resolve(whole) is whole
    assert not store.exists()  # nothing was stored


def test_a_blob_with_the_wrong_hash_is_refused(record, store):
    slim = evidence.split(json.loads(record.model_dump_json()))
    sha = slim["submission"]["top_examples"][evidence.REF_KEY]
    (store / evidence.blob_name(sha)).write_bytes(b"[]")  # tampered
    with pytest.raises(evidence.EvidenceMissing, match="wrong hash"):
        evidence.resolve(slim)


def test_a_missing_blob_names_the_store_and_the_dataset(record, store, tmp_path):
    slim = evidence.split(json.loads(record.model_dump_json()), root=tmp_path / "elsewhere")
    with pytest.raises(evidence.EvidenceMissing, match=evidence.EVIDENCE_REPO):
        evidence.resolve(slim)


def test_a_fetched_blob_is_checked_and_kept(record, store, tmp_path):
    other = tmp_path / "remote"
    slim = evidence.split(json.loads(record.model_dump_json()), root=other)
    sha = slim["submission"]["top_examples"][evidence.REF_KEY]
    fetched = evidence.get(sha, fetch=lambda s: (other / evidence.blob_name(s)).read_bytes())
    assert evidence.sha256(fetched) == sha and (store / evidence.blob_name(sha)).exists()


def test_archive_record_writes_slim_unless_turned_off(record, store, tmp_path, monkeypatch):
    whole_path = archive_record(record, tmp_path / "wiki-a")  # the tests' default is "0"
    assert evidence.REF_KEY not in whole_path.read_text()
    monkeypatch.delenv(evidence.OFFLOAD_ENV)  # the real default
    assert evidence.offload_enabled()
    slim_path = archive_record(record, tmp_path / "wiki-b")
    assert evidence.REF_KEY in slim_path.read_text()
    loaded = AuditRecord.model_validate_json(slim_path.read_text())
    assert loaded.model_dump_json() == record.model_dump_json()
