"""The repo's literature map (labinacan.literature_map, #4577 and #5798)."""

import json
from pathlib import Path

from labinacan.literature_map import MIN_COMPACT
from labinacan.site.cluster_map import SLOTS

MAP = Path(__file__).parents[1] / "docs" / "literature" / "map.json"


def test_the_repo_map_has_compact_regions_one_per_colour_and_no_queued_region():
    """#5798: at most one region per colour, every one named, queued papers in topics."""
    data = json.loads(MAP.read_text())
    regions = data["regions"]
    assert len(regions) <= SLOTS
    assert not any("queued" in r["name"].lower() for r in regions)
    assert data["compactness"] >= MIN_COMPACT
    numbers = {r["number"] for r in regions}
    assert {q["region"] for q in data["queued"]} <= numbers
    assert all({int(n) for n in r["touches"]} <= numbers - {r["number"]} for r in regions)
