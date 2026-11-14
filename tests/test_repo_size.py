"""The size report behind docs/analyses/repo-size.md (issue #3254)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from repo_size import MB, budget, packed_bytes, report, tracked  # noqa: E402

LS_TREE = (
    "100644 blob aaa 2097152\twiki/experiments/a.json\n"
    "100644 blob bbb 1048576\twiki/experiments/a.md\n"
    "100644 blob ccc 1024\tREADME.md\n"
    "160000 commit ddd       -\tvendor/sub\n"
)


def test_tracked_sums_by_top_directory_and_skips_submodules():
    dirs = tracked(LS_TREE)
    assert dirs == {"wiki": {"bytes": 3 * MB, "files": 2}, ".": {"bytes": 1024, "files": 1}}


def test_packed_size_is_in_kib():
    assert packed_bytes("count: 3\nsize: 10\nsize-pack: 2038\nsize-garbage: 0\n") == 2048 * 1024


def test_over_budget_is_named_and_under_is_not(tmp_path):
    dirs = tracked(LS_TREE)
    assert report(dirs, 0, {"tracked_mb": 10, "files": 10})["over_budget"] == []
    over = report(dirs, 0, {"tracked_mb": 2, "files": 2})["over_budget"]
    assert over == ["tracked 3 MB > 2 MB", "3 files > 2"]


def test_budget_reads_pyproject_over_the_defaults(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text("[tool.openinterp.size_budget]\ntracked_mb = 5\n")
    assert budget(path) == {"tracked_mb": 5, "files": 20000}
    assert budget(tmp_path / "missing.toml")["tracked_mb"] == 1000
