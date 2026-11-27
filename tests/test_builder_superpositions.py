"""Superpositions on the site (openinterpretability/builder/superpositions.py)."""

from types import SimpleNamespace

from openinterpretability.builder.superpositions import members_of


def test_a_superposition_whose_newest_record_has_no_profile_has_no_members(record):
    sub = record.submission.model_copy(update={"superposition_profile": None})
    newest = SimpleNamespace(id="b", record=record.model_copy(update={"submission": sub}))
    older = SimpleNamespace(id="a", record=record)
    unit = SimpleNamespace(experiments=[older, newest], layer=0, profile="unset")
    assert members_of(SimpleNamespace(slug="m"), unit) == []
    assert unit.profile is None


def _layer_with(n_members: int, monkeypatch):
    """A model with one superposition of ``n_members`` members, and no other unit."""
    from openinterpretability.builder import superpositions as sp
    from openinterpretability.builder.tree import LayerNode, ModelNode, UnitNode

    unit = UnitNode("m/x", 3, 0, "s.html", kind="superposition")
    layer = LayerNode("m/x", 3, "l.html", units=[unit])
    model = ModelNode("m/x", "m--x", 4, "m.html", layers=[layer])
    found = [sp.Member(i, float(n_members - i), (n_members - i) / n_members, f"n{i}.html")
             for i in range(n_members)]  # fmt: skip
    monkeypatch.setattr(sp, "members_of", lambda m, u: found)
    sp.add_members([model])
    return layer, unit


def test_a_dense_direction_gives_pages_to_its_largest_members_only(monkeypatch):
    from openinterpretability.builder.superpositions import MEMBER_PAGES

    layer, unit = _layer_with(4864, monkeypatch)
    assert unit.member_count == 4864
    assert [m.neuron_idx for m in unit.members] == list(range(MEMBER_PAGES))
    assert sorted(n.idx for n in layer.members) == list(range(MEMBER_PAGES))  # pages
    assert all(len(n.part_of) == 1 for n in layer.members)


def test_a_sparse_superposition_keeps_every_member(monkeypatch):
    layer, unit = _layer_with(16, monkeypatch)
    assert unit.member_count == 16 and len(unit.members) == len(layer.members) == 16
