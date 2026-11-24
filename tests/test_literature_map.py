"""The literature map's regions (scripts/literature_map.py, #4577 and #5798)."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "literature_map", Path(__file__).parents[1] / "scripts" / "literature_map.py"
)
lm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lm)

MAP = Path(__file__).parents[1] / "docs" / "literature" / "map.json"


def test_the_knn_graph_keeps_each_pair_once_with_its_best_cosine():
    vectors = np.eye(3)[[0, 0, 1, 2]].astype(float)  # papers 0 and 1 are the same text
    edges = lm.knn_edges(vectors)
    assert edges[(0, 1)] == 1.0
    assert all(i < j for i, j in edges) and all(w >= 0 for w in edges.values())


def test_a_community_with_too_few_reviews_spreads_over_its_neighbours():
    # Papers 0–5 are reviews in communities 0 and 1. Queued papers 6 and 7 are community 2,
    # with no reviews, and each is tied to a different community.
    labels = [0, 0, 0, 1, 1, 1, 2, 2]
    edges = {(0, 6): 0.9, (3, 7): 0.9, (6, 7): 0.5}
    assert lm.merge_small(labels, edges, reviewed=6) == [0, 0, 0, 1, 1, 1, 0, 1]


def test_a_community_tied_to_nothing_joins_the_biggest():
    labels = [0, 0, 0, 1, 1, 1, 1, 2]
    assert lm.merge_small(labels, {}, reviewed=7) == [0, 0, 0, 1, 1, 1, 1, 1]


def test_a_paper_tied_only_inside_waits_for_its_neighbour_to_move():
    labels = [0, 0, 0, 1, 1]  # 3 and 4 are queued; only 3 is tied outside
    assert lm.merge_small(labels, {(0, 3): 0.8, (3, 4): 0.9}, reviewed=3) == [0] * 5


def test_one_community_is_left_as_it_is():
    assert lm.merge_small([0, 0, 0], {}, reviewed=1) == [0, 0, 0]


def test_a_region_that_smoothing_left_without_reviews_joins_its_nearest():
    # #5810: smoothing moved every review out of a community, which left it with no name.
    import numpy as np

    points = np.array(
        [[0.0, 0], [0.1, 0], [0.2, 0], [5, 0], [5.1, 0], [5.2, 0], [0.3, 0], [5.3, 0]]
    )
    labels = [0, 0, 0, 1, 1, 1, 2, 2]  # region 2 holds only the two queued papers
    assert lm.absorb_small(points, labels, reviewed=6) == [0, 0, 0, 1, 1, 1, 0, 1]
    assert lm.absorb_small(points, [0] * 8, reviewed=1) == [0] * 8


def test_a_region_no_section_can_name_joins_its_nearest(monkeypatch):
    # #6030: a small region whose sections bigger regions took is merged, not named "Region n".
    import numpy as np

    points = np.array([[0.0, 0], [0.1, 0], [5, 0], [5.1, 0], [0.2, 0]])
    seen = []

    def regions(points, labels, names, reviewed):
        seen.append(list(labels))
        rows = [
            {"number": n, "name": "A" if n == 1 else f"Region {n}"} for n in sorted(set(labels))
        ]
        return list(labels), rows

    monkeypatch.setattr(lm, "regions", regions)
    region_of, rows = lm.named_regions(points, [1, 1, 1, 1, 2], [], reviewed=4)
    assert region_of == [1, 1, 1, 1, 1] and seen == [[1, 1, 1, 1, 2], [1, 1, 1, 1, 1]]
    order = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])  # each point's points, nearest first
    assert lm.merge_region(order, [1, 2, 2], 2) == [1, 1, 1]


def test_smoothing_gives_a_stray_point_its_neighbours_region():
    rng = np.random.default_rng(0)
    left, right = rng.normal(0, 0.05, (10, 2)), rng.normal(1, 0.05, (10, 2))
    points = np.vstack([left, right])
    labels = [0] * 10 + [1] * 10
    labels[3] = 1  # a point of the left cloud labelled as the right
    assert lm.compactness(points, labels) == 0.95
    smoothed = lm.smooth(points, labels)
    assert smoothed == [0] * 10 + [1] * 10
    assert lm.compactness(points, smoothed) == 1.0


def test_regions_touch_where_their_points_are_neighbours():
    points = np.array([[0, 0], [0, 0.1], [0.2, 0], [5, 5], [5, 5.1], [5.2, 5]] * 2, dtype=float)
    labels = [1, 1, 1, 2, 2, 2, 1, 1, 3, 2, 2, 2]
    touches = lm.touching(points, labels)
    assert touches[1][3] > 0 and touches[3][1] == touches[1][3]


def test_a_tie_names_the_region_after_both_sections():
    taken = set()
    assert lm.region_name(Counter({"B": 2, "A": 2, None: 4}), taken) == "A · B"
    assert lm.region_name(Counter({"A": 3}), taken) == ""
    assert taken == {"A", "B"}


def test_the_issue_template_is_not_embedded():
    lead = "**Bears on:** the reviewer. Write a short not-applicable page."
    assert lm.plain_lead(lead) == "the reviewer."


def test_the_repo_map_has_compact_regions_one_per_colour_and_no_queued_region():
    """#5798: at most one region per colour, every one named, queued papers in topics."""
    from openinterpretability.builder.cluster_map import SLOTS

    data = json.loads(MAP.read_text())
    regions = data["regions"]
    assert len(regions) <= SLOTS
    assert not any("queued" in r["name"].lower() for r in regions)
    assert data["compactness"] >= lm.MIN_COMPACT
    numbers = {r["number"] for r in regions}
    assert {q["region"] for q in data["queued"]} <= numbers
    assert all({int(n) for n in r["touches"]} <= numbers - {r["number"]} for r in regions)


def test_the_report_warns_when_the_regions_are_not_compact(capsys, tmp_path):
    data = {"reviews": [1], "queued": [], "regions": [1, 2], "compactness": 0.5,
            "layout": "umap seed 3"}  # fmt: skip
    lm.report(data, tmp_path / "map.json")
    out = capsys.readouterr()
    assert "1 reviews, 0 queued, 2 regions, compactness 0.5 (umap seed 3)" in out.out
    assert "Compactness is under" in out.err


def test_the_most_compact_seed_wins_and_a_tie_keeps_the_earlier():
    """#6529: one new page can reshuffle one seed's regions, so every seed is tried."""
    scores = {4577: 0.7, 1: 0.9, 2: 0.9, 3: 0.8}
    tried = []

    def build(vectors, names, reviewed, seed):
        tried.append((vectors, names, reviewed, seed))
        return seed, "points", "regions", "rows", scores[seed]

    best = lm.most_compact("v", ["a"], 1, seeds=(4577, 1, 2, 3), build=build)
    assert best == (1, "points", "regions", "rows", 0.9)
    assert tried == [("v", ["a"], 1, seed) for seed in (4577, 1, 2, 3)]
    assert lm.SEEDS[0] == 4577 and len(set(lm.SEEDS)) == len(lm.SEEDS)  # today's seed first
