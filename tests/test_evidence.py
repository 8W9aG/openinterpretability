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


def test_the_planner_skips_a_record_whose_evidence_is_missing(record, store, tmp_path, capsys):
    # A record whose evidence isn't on the dataset yet must not stop the planner (#4530).
    # The site build and the checks still fail on it.
    from openinterpretability.builder.wiki import load_archive_paths
    from openinterpretability.planner import load_states

    wiki = tmp_path / "wiki"
    path = archive_record(record, wiki)
    slim = evidence.split(json.loads(path.read_text()), root=tmp_path / "elsewhere")
    path.write_text(json.dumps(slim))
    with pytest.raises(evidence.EvidenceMissing):
        load_archive_paths(wiki)
    assert load_archive_paths(wiki, skip_missing=True) == []
    assert load_states(wiki) == {} and "skipping" in capsys.readouterr().err


# The scan's evidence for one unit (openinterpretability/evidence.py).


def _gpt2_runner(tiny_gpt2):
    from openinterpretability.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def test_a_dead_neuron_stops_the_scan(tiny_gpt2, tmp_path):
    import torch

    from openinterpretability.evidence import ScanRequest, Unauditable, scan_evidence

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text": "abcabc"}\n{"text": "hello there"}\n')
    with torch.no_grad():
        tiny_gpt2.transformer.h[1].mlp.c_fc.bias[5] = -1000.0
    request = ScanRequest(model_name="openai-community/gpt2", layer=1, neuron=5,
                          pole="positive", corpus=corpus, dtype="float32")  # fmt: skip
    with pytest.raises(Unauditable, match="is dead: no input can make it fire") as stop:
        scan_evidence(_gpt2_runner(tiny_gpt2), request)
    assert stop.value.draw.outcome == "dead" and stop.value.draw.proven_dead


def test_a_chosen_unit_needs_its_layer_and_index(tiny_gpt2):
    import click

    from openinterpretability.evidence import ScanRequest, pick_unit

    request = ScanRequest(model_name="m", unit="head", layer=1)
    with pytest.raises(click.UsageError, match="--layer and --head are required"):
        pick_unit(_gpt2_runner(tiny_gpt2), request, None)


def test_a_silent_feature_gets_no_weight_check(tiny_gpt2):
    from openinterpretability.core.schemas import FeatureIdentifier
    from openinterpretability.evidence import dead_unit
    from tests.test_dead import PROVENANCE

    ident = FeatureIdentifier(model_name="m", layer=1, dictionary="d", feature_idx=3)
    stop = dead_unit(_gpt2_runner(tiny_gpt2), ident, PROVENANCE)
    assert stop.message == f"{ident} never activates on this corpus; try a larger --corpus."
    assert (
        stop.draw.outcome == "dead" and not stop.draw.proven_dead and stop.draw.dead_bound is None
    )


def test_a_document_longer_than_the_context_is_refused(tiny_gpt2):
    import click

    from openinterpretability.evidence import check_context

    runner = _gpt2_runner(tiny_gpt2)
    check_context(runner, ["a" * 64])  # tiny-gpt2 reads 64 positions
    with pytest.raises(click.UsageError, match="has 65 tokens but the model reads at most 64"):
        check_context(runner, ["a" * 65])


def test_a_streamed_model_scans_in_larger_batches(tiny_gpt2, capsys):
    import torch

    from openinterpretability.evidence import _scan_batch_tokens
    from openinterpretability.harness import memory
    from openinterpretability.harness.scan import BATCH_TOKENS

    runner = _gpt2_runner(tiny_gpt2)
    assert _scan_batch_tokens(runner) == BATCH_TOKENS
    assert capsys.readouterr().err == ""
    # One weight left on the meta device stands for a layer that stays on disk.
    tiny_gpt2.register_parameter("on_disk", torch.nn.Parameter(torch.empty(1, device="meta")))
    tokens = _scan_batch_tokens(runner)
    have = memory.available_bytes()
    assert tokens == memory.stream_batch_tokens(tiny_gpt2.config, torch.float32, have,
                                                BATCH_TOKENS)  # fmt: skip
    assert tokens > BATCH_TOKENS
    assert f"Streaming: scanning {tokens:,} tokens per batch." in capsys.readouterr().err
