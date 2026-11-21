"""One list of the record kinds that CI reproduces, read everywhere (#4792)."""

import subprocess
import sys
from pathlib import Path

from openinterpretability.core.record_kinds import RECORD_KINDS, REPRODUCIBLE, reproducible

ROOT = Path(__file__).resolve().parents[1]


def test_running_the_module_prints_the_reproducible_prefixes():
    out = subprocess.run(
        [sys.executable, str(ROOT / "openinterpretability/core/record_kinds.py")],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert out.split() == list(REPRODUCIBLE) == ["wiki/experiments/", "wiki/explanations/"]


def test_the_workflows_read_the_list_not_their_own():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    merge = (ROOT / ".github/workflows/auto-merge.yml").read_text()
    assert "python3 openinterpretability/core/record_kinds.py" in ci
    assert "-- wiki/experiments |" not in ci
    assert "sparse-checkout: openinterpretability/core/record_kinds.py" in merge
    assert "from record_kinds import reproducible" in merge
    assert "^wiki/experiments/" not in merge


def test_every_wiki_record_kind_is_listed_and_documented():
    wiki = {p.name for p in (ROOT / "wiki").iterdir() if p.is_dir()}
    listed = {k.prefix.split("/")[1] for k in RECORD_KINDS}
    assert wiki <= listed | {"pending"}, f"unlisted record folders: {wiki - listed}"
    protocol = (ROOT / "docs/PROTOCOL.md").read_text()
    assert all(f"`{k.prefix}`" in protocol for k in RECORD_KINDS)
    assert reproducible("wiki/experiments/m/L1/N2-positive/1.json")
    assert reproducible("wiki/explanations/m/abc/1.json")  # since #4793
    assert not reproducible("wiki/circuits/m/abc/1.json")


def test_the_module_run_as_a_script_in_this_process_prints_the_prefixes(capsys):
    # The subprocess test above is not measured by coverage, so run it here too.
    import runpy

    runpy.run_path(str(ROOT / "openinterpretability/core/record_kinds.py"), run_name="__main__")
    assert capsys.readouterr().out.split() == list(REPRODUCIBLE)
