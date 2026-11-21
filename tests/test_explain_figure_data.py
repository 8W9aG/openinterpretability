"""The figure data of an explanation: a head's reading, and the groups and hubs (#4479)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from openinterpretability import explain as ex
from openinterpretability.explain_figure_data import _groups, figure_data, head_strength
from tests.test_explain import PROMPT, gpt2, gpt2_means  # noqa: F401  (fixtures)


def test_a_heads_reading_is_its_outputs_distance_from_its_mean(gpt2, gpt2_means):  # noqa: F811
    result = ex.explain(gpt2, "tiny", PROMPT, gpt2_means, {}, top=50, edges=0)
    head = next(u for u in result.poles if u.pole == ex.HEAD)
    result.poles = [head, *[u for u in result.poles if u is not head]]  # the head is read
    planned = SimpleNamespace(blocked_units=[])
    data = figure_data(gpt2, gpt2_means, result, {}, planned)
    reading = data.reading[head.label]
    assert len(reading) == len(data.tokens)
    encoded = gpt2.encode(PROMPT)
    with torch.no_grad():
        _, _, outputs = ex._captured(gpt2, encoded, True)
    out = outputs[head.layer][0].float()
    width = out.shape[-1] // gpt2.n_heads
    cols = slice(head.neuron * width, (head.neuron + 1) * width)
    expected = head_strength(out[:, cols], gpt2_means.heads[head.layer], cols)
    assert reading == pytest.approx([round(float(v), 4) for v in expected], abs=1e-4)
    assert data.coverage is None and data.random_draws == []


def _pole(layer, neuron, hub=None):
    return ex.PoleAttribution(layer, neuron, "positive", 1.0, 1.0, status="VERIFIED", hub=hub)


def test_groups_hold_their_top_units_and_hubs_name_their_center():
    top, other = _pole(1, 2, hub={"hub": "quote hub"}), _pole(2, 5)
    group = SimpleNamespace(kind="circuit", name="C1", status="VERIFIED")
    empty = SimpleNamespace(kind="circuit", name="C2", status="REJECTED")
    unknown = SimpleNamespace(kind="superposition", name="S9", status="VERIFIED")
    result = SimpleNamespace(poles=[top, other], groups=[group, empty, unknown])
    defs = [SimpleNamespace(kind="circuit", name="C1", units=[top.key, (0, 0, "positive")]),
            SimpleNamespace(kind="circuit", name="C2", units=[(0, 0, "positive")])]  # fmt: skip
    assert _groups(result, defs) == [
        {"name": "C1", "kind": "circuit", "status": "VERIFIED", "members": ["L1/N2+"]},
        {"name": "quote hub", "kind": "hub", "status": "VERIFIED", "members": ["L1/N2+"]},
    ]
