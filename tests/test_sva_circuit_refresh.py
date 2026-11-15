"""``scripts/sva_circuit.py --refresh`` recomputes only the registry columns (issue #3656)."""

import importlib.util
import json
from pathlib import Path

from click.testing import CliRunner

_SPEC = importlib.util.spec_from_file_location(
    "sva_circuit", Path(__file__).parents[1] / "scripts" / "sva_circuit.py"
)
sva = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sva)


def test_refresh_keeps_the_curve_and_ranking_and_reads_the_registry_again(tmp_path, monkeypatch):
    curve = {"table": "curve", "basis": "attribution", "size": 10, "faithfulness": 0.1}
    old = [
        {"table": "neuron", "rank": 2, "layer": 3, "neuron_idx": 7, "attribution": -0.5,
         "experiments": 0, "verified_poles": [], "mentions_number": False, "slugs": []},
        {"table": "neuron", "rank": 1, "layer": 0, "neuron_idx": 9, "attribution": 1.5,
         "experiments": 0, "verified_poles": [], "mentions_number": False, "slugs": []},
    ]  # fmt: skip
    out = tmp_path / "sva.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in [curve, *old]))
    found = {"slug": "m-L0-N9-positive", "pole": "positive", "status": "VERIFIED",
             "unit_status": "VERIFIED", "explanation": "Fires on plural nouns."}  # fmt: skip
    registry = {(0, 9): [found]}
    monkeypatch.setattr(sva, "registry_rows", lambda site_dir, model: registry)

    result = CliRunner().invoke(
        sva.main, ["--site-dir", str(tmp_path), "--out", str(out), "--refresh"]
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0] == curve
    first, second = rows[1:]
    assert (first["rank"], first["layer"], first["neuron_idx"]) == (1, 0, 9)
    assert first["verified_poles"] == ["positive"] and first["mentions_number"] is True
    assert first["slugs"] == ["m-L0-N9-positive"] and first["attribution"] == 1.5
    assert (second["rank"], second["experiments"], second["verified_poles"]) == (2, 0, [])
