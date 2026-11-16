"""The explanation-score backfill (scripts/score_backfill.py, issue #3499)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "score_backfill", Path(__file__).parents[1] / "scripts" / "score_backfill.py"
)
backfill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backfill)

RECORD = "wiki/experiments/Qwen--Qwen2.5-0.5B/L0/N3019-negative/20261104T201612Z.json"


def test_each_record_has_its_own_task_folder_and_fixed_seed():
    assert backfill.task_dir(RECORD) == Path(
        "exchange/score-backfill/Qwen--Qwen2.5-0.5B/L0/N3019-negative/20261104T201612Z"
    )
    assert backfill.seed_of(RECORD) == backfill.seed_of(RECORD)
    assert backfill.seed_of(RECORD) != backfill.seed_of(RECORD.replace("201612", "201613"))


def test_a_record_with_no_corpus_hash_has_no_corpus():
    assert backfill.corpus_for({"corpus_sha256": None, "record": RECORD}, {}) is None


def test_a_corpus_hash_that_is_not_hex_has_no_corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside" / "dummy.jsonl"
    outside.parent.mkdir()
    outside.write_text('{"text": "x"}\n')
    (tmp_path / backfill.WORK / "corpora").mkdir(parents=True)
    for sha in ("../../../outside/dummy", str(outside.with_suffix("")), "A" * 64):
        assert backfill.corpus_for({"corpus_sha256": sha, "record": RECORD}, {}) is None


def test_a_cached_corpus_with_another_hash_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backfill, "CLI", ["false"])
    sha = "0" * 64
    cached = backfill.WORK / "corpora" / f"{sha}.jsonl"
    cached.parent.mkdir(parents=True)
    cached.write_text('{"text": "not this corpus"}\n')
    assert backfill.corpus_for({"corpus_sha256": sha, "record": RECORD}, {}) is None
    assert not cached.exists()
    good = cached.parent / "good.jsonl"
    good.write_text('{"text": "this corpus"}\n')
    named = good.rename(cached.parent / f"{backfill.sha256(good)}.jsonl")
    row = {"corpus_sha256": named.stem, "record": RECORD}
    assert backfill.corpus_for(row, {}) == named  # a matching file is kept, with no rebuild
