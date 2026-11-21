"""The route figure of an answer experiment (openinterpretability/builder/route_figures.py)."""

import re
from types import SimpleNamespace

from openinterpretability.builder.route_figures import route_figure
from openinterpretability.core.answer_records import RouteReading


def test_a_traced_edge_between_units_is_drawn_and_a_weak_one_is_left_out():
    claim = SimpleNamespace(answer=" Paris", route=["L0/N1+", "L1/H2", "L1/N3-"], roles={})
    reading = RouteReading(
        tokens=["The", " capital"],
        top=" Paris",
        answer_log_prob=-0.5,
        route_share=None,
        strength={},
        attribution={},
        edges=[
            {"source": "L0/N1+", "target": "L1/H2", "weight": -1.0},
            {"source": "L0/N1+", "target": "L1/N3-", "weight": 0.01},  # under MIN_EDGE
        ],
    )
    drawn = route_figure("route-0", claim, reading, reading)
    edges = re.findall(r'<path class="(edge[^"]*)"', drawn.svg)
    assert edges == ["edge dashed rl1"]  # the push-down edge into layer 1, dashed
