"""Answer-record summaries and checks, at their edges (#6491)."""

import codecs
from pathlib import Path

from openinterpretability.checks import CheckReport
from openinterpretability.core import answer_records as ar
from tests.test_answer_records import _measures, _record


def test_the_summary_has_each_field_and_rounds_necessity_to_4_places():
    probes = {f"q{i}": i < 5 for i in range(10)}  # half: NEEDS_REVISION
    record = _record(_measures(route_cut=2.0, all_cut=3.0, probes=probes), attempt=2)
    assert ar.summary(record) == {
        "status": "NEEDS_REVISION",
        "attempt": 2,
        "route": ["L0/H1", "L1/N3+"],
        "sources": ["b"],
        "roles": {},
        "claim": "b moves to the end.",
        "failed_gates": ["probes"],
        "necessity": 0.6667,
        "sufficiency": 0.7,
        "patching": 0.8,
        "reviewer_prompts_as_predicted": "5 of 10",
    }


def _check(tmp_path, scope=None):
    report = CheckReport()
    ar.check_answers(tmp_path, report, scope)
    return {p.path.name: p.message for p in report.problems}


def test_check_gives_the_first_validation_error_and_the_decode_error(tmp_path):
    folder = tmp_path / ar.ANSWERS_WIKI / "m" / "p"
    folder.mkdir(parents=True)
    (folder / "bad.json").write_text('{"claim": 5}')
    (folder / "binary.json").write_bytes(b"\xff\xfe\x00")
    assert _check(tmp_path) == {
        "bad.json": "not a valid answer record: Input should be an object",
        "binary.json": "not a valid answer record: 'utf-8' codec can't decode byte 0xff"
        " in position 0: invalid start byte",
    }


def test_check_reads_only_the_files_in_its_scope(tmp_path):
    folder = tmp_path / ar.ANSWERS_WIKI / "m" / "p"
    folder.mkdir(parents=True)
    (folder / "in.json").write_text('{"claim": 5}')
    (folder / "out.json").write_text('{"claim": 5}')
    found = _check(tmp_path, scope={folder / "in.json"})
    assert found == {"in.json": "not a valid answer record: Input should be an object"}


def test_check_reads_records_as_utf_8(tmp_path, monkeypatch):
    record = _record()
    path = ar.record_path(tmp_path, record)
    path.parent.mkdir(parents=True)
    path.write_text(record.model_dump_json(), encoding="utf-8")
    encodings = []
    read_text = Path.read_text

    def spy(self, *args, **kwargs):
        encodings.append(kwargs.get("encoding"))
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    assert _check(tmp_path) == {}
    assert [codecs.lookup(e).name for e in encodings] == ["utf-8"]  # never the locale's
