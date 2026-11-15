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
