"""The performance benchmark reads both `time` formats and flags regressions (issue #3265)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import perf_bench  # noqa: E402

BSD = """
       58.40 real       120.10 user        33.58 sys
          1700000000  maximum resident set size
          1476853760  peak memory footprint
"""

GNU = """
\tUser time (seconds): 120.10
\tSystem time (seconds): 33.58
\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.50
\tMaximum resident set size (kbytes): 1442240
"""


def test_reads_bsd_time_and_prefers_the_footprint():
    got = perf_bench.parse_time(BSD)
    assert got["wall_s"] == 58.4 and got["cpu_s"] == pytest.approx(153.68)
    assert got["peak_mb"] == pytest.approx(1476853760 / 2**20)  # not the resident set


def test_reads_gnu_time():
    got = perf_bench.parse_time(GNU)
    assert got["wall_s"] == pytest.approx(3600 + 2 * 60 + 3.5)
    assert got["peak_mb"] == pytest.approx(1442240 / 1024)
    assert perf_bench.parse_time(GNU.replace("1:02:03.50", "2:03.50"))["wall_s"] == 123.5


def test_no_measurements_is_an_error():
    with pytest.raises(ValueError, match="no time measurements"):
        perf_bench.parse_time("Command exited with non-zero status 1")


def _run(machine="m", **workloads):
    return {"machine": {"device": machine}, "workloads": workloads}


def test_a_measure_past_its_tolerance_is_a_regression():
    before = {"wall_s": 10.0, "cpu_s": 20.0, "peak_mb": 1000.0}
    base = _run(scan=before)
    assert perf_bench.compare(_run(scan={**before, "wall_s": 14.9}), base) == []
    (line,) = perf_bench.compare(_run(scan={**before, "wall_s": 15.1}), base)
    assert line.startswith("scan wall_s: 15.1 against 10")
    assert perf_bench.compare(_run(scan={**before, "peak_mb": 1190.0}), base) == []
    (line,) = perf_bench.compare(_run(scan={**before, "peak_mb": 1210.0}), base)
    assert "peak_mb" in line and "past 1.2x" in line
    # A workload the baseline lacks (such as --build) is not compared.
    assert perf_bench.compare(_run(scan=before, build=before), base) == []


def test_another_machine_is_named():
    before = {"wall_s": 10.0, "cpu_s": 20.0, "peak_mb": 1000.0}
    (line,) = perf_bench.compare(_run("mps", scan=before), _run("cpu", scan=before))
    assert line == "machine differs from the baseline's (device)"


def test_workloads_write_only_under_the_output_dir(tmp_path):
    jobs = perf_bench.workloads(tmp_path, "cpu", build=True)
    assert set(jobs) == {"scan", "reproduce", "build"}
    written = [a for args in jobs.values() for a in args if a.startswith(str(tmp_path))]
    assert len(written) == 2  # the evidence file and the site
    assert Path(perf_bench.RECORD).exists()
