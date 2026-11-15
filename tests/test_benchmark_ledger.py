"""The benchmark ledger (scripts/benchmark_ledger.py, issue #4011)."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_ledger", Path(__file__).parents[1] / "scripts" / "benchmark_ledger.py"
)
ledger = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ledger)

MACHINE = {"platform": "Linux-6.8-x86_64", "processor": "x86_64", "cpus": 4,
           "python": "3.12", "torch": "2.14", "device": "cpu"}  # fmt: skip


def _run(scan_mb=1000, wall_s=60.0, machine=MACHINE):
    memory = {"metric": "rss", "models": {"gpt2": {"scan_mb": scan_mb, "scan_s": 30.0}}}
    perf = {"machine": machine, "workloads": {"scan": {"wall_s": wall_s, "cpu_s": 100.0,
                                                       "peak_mb": 900.0}}}  # fmt: skip
    return memory, perf


def _row(**kwargs):
    return ledger.row(*_run(**kwargs), "abc", "1", datetime(2026, 11, 15, tzinfo=UTC))


def test_a_row_flattens_every_measure_and_names_the_runner():
    r = _row()
    assert r["measures"] == {
        "memory/gpt2/scan_mb": 1000, "memory/gpt2/scan_s": 30.0,
        "perf/scan/wall_s": 60.0, "perf/scan/cpu_s": 100.0, "perf/scan/peak_mb": 900.0,
    }  # fmt: skip
    assert r["runner"] == {"system": "Linux", "arch": "x86_64", "cpus": 4, "device": "cpu"}
    assert r["date"] == "2026-11-15T00:00:00Z"


def test_check_waits_for_history_then_compares_with_the_median():
    history = [_row(), _row()]
    regressions, notes = ledger.check(_row(scan_mb=5000), history)
    assert regressions == [] and "not compared yet" in notes[0]
    history = [_row(scan_mb=m) for m in (1000, 1010, 990)]
    assert ledger.check(_row(scan_mb=1240), history)[0] == []  # within 1.25x of 1000
    (line,) = ledger.check(_row(scan_mb=1300), history)[0]
    assert line.startswith("memory/gpt2/scan_mb: 1300 against a median of 1000")
    assert ledger.check(_row(wall_s=89.0), history)[0] == []  # times get 1.5x
    (line,) = ledger.check(_row(wall_s=91.0), history)[0]
    assert "perf/scan/wall_s" in line


def test_another_runner_isnt_compared():
    other = {**MACHINE, "cpus": 16}
    history = [_row(machine=other) for _ in range(5)]
    regressions, notes = ledger.check(_row(scan_mb=99999), history)
    assert regressions == [] and "0 earlier runs" in notes[0]


def test_append_adds_one_sorted_line(tmp_path):
    memory, perf = _run()
    (tmp_path / "m.json").write_text(json.dumps(memory))
    (tmp_path / "p.json").write_text(json.dumps(perf))
    out = tmp_path / "ledger.jsonl"
    args = ["--memory", str(tmp_path / "m.json"), "--perf", str(tmp_path / "p.json")]
    for _ in range(2):
        assert ledger.main(["append", *args, "--ledger", str(out), "--commit", "c"]) == 0
    rows = ledger.read(out)
    assert len(rows) == 2 and rows[0]["commit"] == "c"
    assert ledger.main(["check", *args, "--ledger", str(out)]) == 0
