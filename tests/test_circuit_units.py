"""Registry circuit pages and hub facts (builder/circuit_units.py, #4287)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from openinterpretability.builder.circuit_units import (
    CONTESTED,
    CircuitData,
    ExperimentView,
    _kept,
    family,
    lineage_of,
    task_standing,
    tasks_of,
)
from openinterpretability.builder.compiler import compile_site
from openinterpretability.circuit_ids import load_events
from openinterpretability.core.circuit_records import CircuitRecord
from openinterpretability.hubs import NormRow
from tests.test_circuits import MODEL, at, wiring

REAL = Path(__file__).parents[1] / "wiki/circuits/openai-community--gpt2/C4/20261116T232532Z.json"
WHEN = datetime(2026, 11, 16, 12, tzinfo=UTC)


def _href(layer, idx):
    return f"models/{MODEL}/L{layer}/N{idx}.html"


def _record(status="VERIFIED", *, minutes=0, proposer="p", reviewer="r", members=(), task="t",
            supersedes=None, model="Qwen/Qwen2.5-0.5B"):  # fmt: skip
    real = CircuitRecord.model_validate_json(REAL.read_text())
    task_ = real.hypothesis.task.model_copy(update={"name": task, "description": "A task."})
    hyp = real.hypothesis.model_copy(update={
        "model_name": model, "circuit_id": "C1", "task": task_, "submitting_agent_model": proposer,
        "cluster_members": list(members), "supersedes": supersedes,
        "explanation": "They copy."})  # fmt: skip
    when = WHEN + timedelta(minutes=minutes)
    verdict = real.verdict.model_copy(update={
        "status": status, "reviewer_agent_model": reviewer, "timestamp": when})  # fmt: skip
    return real.model_copy(update={"hypothesis": hyp, "verdict": verdict})


def _view(record, name):
    return ExperimentView(record, f"wiki/circuits/{MODEL}/C1/{name}.json")


def test_independent_reviews_outweigh_self_reviews_and_a_tie_is_contested():
    own = _view(_record("REJECTED", proposer="a", reviewer="a"), "1")
    other = _view(_record("VERIFIED", proposer="a", reviewer="b"), "2")
    assert task_standing([own, other]) == "VERIFIED"  # the self-review doesn't count
    split = [other, _view(_record("REJECTED", proposer="a", reviewer="c"), "3")]
    assert task_standing(split) == CONTESTED
    assert task_standing([]) is None


def test_a_superseded_record_does_not_count():
    old = _view(_record("REJECTED"), "20261116T120000Z")
    new = _view(_record("VERIFIED", minutes=5, supersedes="20261116T120000Z"), "20261116T120500Z")
    assert task_standing([old, new]) == "VERIFIED"


def test_an_experiment_is_stale_when_members_changed():
    then = [_href(2, 0), _href(2, 1), _href(2, 2)]
    now = frozenset([_href(2, 0), _href(2, 1), _href(3, 117)])
    (task,) = tasks_of([("wiki/x.json", _record(members=then))], now)
    (exp,) = task.experiments
    assert exp.stale and exp.joined == ["L3/N117"] and exp.left == ["L2/N2"]
    (same,) = tasks_of([("wiki/x.json", _record(members=sorted(now)))], now)
    assert not same.experiments[0].stale


def test_lineage_merges_and_sub_circuits():
    a, b, c = _href(1, 0), _href(1, 1), _href(1, 2)
    events = [
        {"date": "d1", "event": "new", "id": "C1", "members": [a, b]},
        {"date": "d1", "event": "new", "id": "C2", "members": [c]},
        {"date": "d2", "event": "grow", "id": "C1", "members": [a, b, c]},
        {"date": "d2", "event": "merge", "id": "C1", "from": ["C2"]},
        {"date": "d2", "event": "gone", "id": "C2"},
    ]
    steps, stability = lineage_of("C1", events)
    assert [s.event for s in steps] == ["new", "grow", "merge"] and steps[1].joined == ["L1/N2"]
    assert steps[2].others == ["C2"] and stability == 2 / 3
    assert family("C1", events) == (["C2"], []) and family("C2", events) == ([], ["C1"])
    assert set(_kept(events, load_events(events))) == {"C1", "C2"}  # C2 keeps its page


def _hub_site(record):
    spoke = wiring("activations", down=[("mlp_neuron", 12, 1, 0.1)])
    return [
        at(record, 1, 5, weights=wiring("weights", down=[("mlp_neuron", 3, 117, 0.4)])),
        at(record, 3, 117, activations=spoke),
        *(at(record, 2, i, activations=spoke) for i in range(9)),
        at(record, 12, 1),  # a hub: 10 units are wired into it
    ]


def test_the_site_has_a_page_per_registry_circuit_and_hub_facts(tmp_path, record):
    members = [_href(1, 5), _href(3, 117), _href(2, 0)]
    lineage = [{"date": "2026-11-16", "event": "new", "id": "C1", "members": members}]
    stale = _record("NEEDS_REVISION", members=[_href(1, 5), _href(3, 117), _href(2, 8)])
    hub = f"{MODEL}/L12/N1/HUB"
    norm = NormRow(hub=hub, date="2026-11-16", norm=9.5, layer_median=2.0, percentile=0.99)
    data = CircuitData(lineage, [(f"wiki/circuits/{MODEL}/C1/x.json", stale)], {hub: norm})
    compile_site(_hub_site(record), tmp_path, circuits={MODEL: data})
    site = tmp_path / "models" / MODEL
    page = (site / "circuits" / "C1.html").read_text()
    assert "circuit C1" in page and "NEEDS REVISION" in page and "Stale." in page
    assert 'Joined: <span class="font-mono">L2/N0</span>' in page and "L2/N8" in page
    assert f"models/{MODEL}/L1/N5.html" in page  # a member links to its unit page
    index = (site / "circuits.html").read_text()
    assert "circuits/C1.html" in index and "no task hypothesis" not in index
    assert "Wiring groups" in index
    hub_page = (site / "hubs" / "L12-N1.html").read_text()
    assert "Output norm 9.500" in hub_page and "99%" in hub_page
    assert "Standing and artefact checks" in hub_page and "circuits/C1.html" in hub_page


def test_revisions_link_both_ways_by_name_or_path():
    from openinterpretability.builder.circuit_units import link_revisions

    old = _view(_record("REJECTED"), "20261116T120000Z")
    by_name = _view(_record(minutes=5, supersedes="20261116T120000Z"), "20261116T120500Z")
    by_path = _view(_record(minutes=9, supersedes=by_name.source), "20261116T120900Z")
    lone = _view(_record(minutes=1, supersedes="missing"), "20261116T120100Z")
    link_revisions([old, by_name, by_path, lone])
    assert by_name.supersedes is old and old.superseded_by == [by_name]
    assert by_path.supersedes is by_name and by_name.superseded_by == [by_path]
    assert lone.supersedes is None and old.supersedes is None and not lone.superseded_by


def test_a_record_that_names_itself_does_not_supersede_itself():
    from openinterpretability.builder.circuit_units import link_revisions

    me = _view(_record(supersedes="20261116T120000Z"), "20261116T120000Z")
    link_revisions([me])
    assert me.supersedes is None and me.superseded_by == []


def test_each_experiment_gets_its_page_from_its_record_name():
    page = lambda name: f"page/{name}.html"  # noqa: E731
    (task,) = tasks_of([(f"wiki/circuits/{MODEL}/C1/20261116T120000Z.json", _record())],
                       frozenset(), page)  # fmt: skip
    assert task.experiments[0].href == "page/20261116T120000Z.html"
    (bare,) = tasks_of([("wiki/x.json", _record())], frozenset())
    assert bare.experiments[0].href == ""


def test_each_circuit_experiment_has_a_page_that_the_circuit_table_links(tmp_path, record):
    from openinterpretability.builder.hrefs import circuit_experiment_href

    members = [_href(1, 5), _href(3, 117), _href(2, 0)]
    lineage = [{"date": "2026-11-16", "event": "new", "id": "C1", "members": members}]
    old = _record("REJECTED", members=[_href(1, 5), _href(3, 117), _href(2, 8)], reviewer="p")
    new = _record("VERIFIED", minutes=5, members=members, supersedes="20261116T120000Z")
    roles = {"members": ["L1/N5", "L9/N1"], "roles": {"L1/N5": "copies the name"}}
    new = new.model_copy(update={"hypothesis": new.hypothesis.model_copy(update=roles)})
    sources = [
        f"wiki/circuits/{MODEL}/C1/{n}.json" for n in ("20261116T120000Z", "20261116T120500Z")
    ]
    data = CircuitData(lineage, [(sources[0], old), (sources[1], new)], {})
    compile_site(_hub_site(record), tmp_path, circuits={MODEL: data})
    circuit = (tmp_path / "models" / MODEL / "circuits" / "C1.html").read_text()
    pages = [
        circuit_experiment_href(MODEL, "C1", n) for n in ("20261116T120000Z", "20261116T120500Z")
    ]
    assert all(f'href="../../../{p}"' in circuit for p in pages)  # the table links each page
    assert "current" in circuit and "superseded" in circuit and "self-reviewed" in circuit
    stale = (tmp_path / pages[0]).read_text()
    assert "Stale:" in stale and "joined" in stale and "L2/N8" in stale
    assert "Superseded" in stale and "20261116T120500Z.html" in stale
    assert "self-reviewed" in stale and "✓</span> necessity floor" in stale
    assert f"blob/main/{sources[0]}" in stale  # the raw record
    current = (tmp_path / pages[1]).read_text()
    assert "on its current members" in current and "Revises" in current
    assert "independent review" in current and "Task pairs: " in current
    assert f'href="../../../../models/{MODEL}/L1/N5.html">L1/N5</a>' in current  # its page
    assert "copies the name" in current and "L9/N1</a>" not in current  # no page: no link
    assert '../C1.html"' in current or "circuits/C1.html" in current  # back to the circuit


def test_circuit_pages_remove_personal_data_from_the_record(tmp_path, record):
    members = [_href(1, 5)]
    lineage = [{"date": "2026-11-16", "event": "new", "id": "C1", "members": members}]
    leaky = _record(members=members)
    leaky = leaky.model_copy(update={"hypothesis": leaky.hypothesis.model_copy(
        update={"explanation": "Mail jane.roe@realmail.co.uk about it."})})  # fmt: skip
    data = CircuitData(lineage, [(f"wiki/circuits/{MODEL}/C1/20261116T120000Z.json", leaky)], {})
    compile_site(_hub_site(record), tmp_path, circuits={MODEL: data})
    site = tmp_path / "models" / MODEL / "circuits"
    for page in (site / "C1.html", site / "C1" / "20261116T120000Z.html"):
        text = page.read_text()
        assert "jane.roe@realmail.co.uk" not in text and "[removed]" in text


def _model(record):
    from openinterpretability.builder.tree import build_tree

    chain = [
        at(record, 1, 5, weights=wiring("weights", down=[("mlp_neuron", 3, 117, 0.4)])),
        at(record, 3, 117),
        at(record, 6, 2),  # in no circuit
    ]
    (model,) = build_tree(chain, [])
    return model


def _recorder(common):
    calls = []

    def render(href, template, **kw):
        calls.append((href, template, kw))

    render.common = common
    return render, calls


def test_render_circuits_renders_each_page_and_redirects_every_other_name(record):
    from openinterpretability.builder.circuit_pages import render_circuits
    from openinterpretability.builder.circuits import circuits_and_hubs
    from openinterpretability.builder.hrefs import circuit_href, hub_href

    model = _model(record)
    found, hubs = circuits_and_hubs(model, {})
    render, calls = _recorder({})  # no circuit_data: no registry circuits
    pages = render_circuits(render, model, found, hubs)
    by_href = {href: (template, kw) for href, template, kw in calls}
    index = by_href[f"models/{MODEL}/circuits.html"]
    assert index[0] == "circuits.html" and index[1]["n_units"] == 2
    page = circuit_href(MODEL, "L1-N5")
    assert by_href[page][0] == "circuit.html"
    moved = by_href[circuit_href(MODEL, "L3-N117")]  # a member: to its circuit's page
    assert moved == ("redirect.html", {"target": page})
    alone = by_href[hub_href(MODEL, "L6-N2")]  # in nothing: to its own page
    assert alone == ("redirect.html", {"target": f"models/{MODEL}/L6/N2.html"})
    assert by_href[circuit_href(MODEL, "L6-N2")] == alone
    assert pages[f"models/{MODEL}/L3/N117.html"] == page


def test_render_circuits_renders_when_any_kind_of_circuit_exists(record):
    from openinterpretability.builder.circuit_pages import render_circuits
    from openinterpretability.builder.circuits import circuits_and_hubs

    model = _model(record)
    found, hubs = circuits_and_hubs(model, {})
    render, calls = _recorder({})
    render_circuits(render, model, [], [])
    assert calls == []  # nothing to show
    render_circuits(render, model, found, [])
    assert calls  # a wiring circuit alone is enough
    lineage = [{"date": "d", "event": "new", "id": "C1", "members": [_href(1, 5)]}]
    render, calls = _recorder({"circuit_data": {MODEL: CircuitData(lineage, [], {})}})
    render_circuits(render, model, [], [])
    assert (f"models/{MODEL}/circuits/C1.html", "circuit_unit.html") in [c[:2] for c in calls]


def test_registry_circuits_sort_by_number_with_their_wiring_and_family(record):
    from openinterpretability.builder.circuit_units import circuit_views
    from openinterpretability.builder.circuits import circuits_and_hubs

    model = _model(record)
    (found,), _ = circuits_and_hubs(model, {})
    a, b, c = _href(1, 5), _href(3, 117), _href(6, 2)
    lineage = [
        {"date": "d1", "event": "new", "id": "C10", "members": [c]},
        {"date": "d1", "event": "new", "id": "C2", "members": [a, b]},
        {"date": "d1", "event": "new", "id": "C3", "members": [c]},
        {"date": "d2", "event": "merge", "id": "C10", "from": ["C3"]},
        {"date": "d2", "event": "gone", "id": "C3"},
    ]
    stray = found.edges[0].__class__(found.edges[0].source, model.layers[-1].units[0],
                                     found.edges[0].methods)  # fmt: skip
    views = circuit_views(model, CircuitData(lineage, [], {}), [*found.edges, stray])
    assert [v.id for v in views] == ["C2", "C3", "C10"]  # by number, not by text
    c2 = views[0]
    kept = [(e.source.href, e.target.href) for e in c2.wiring.edges]
    assert kept == [(a, b)]  # the stray edge leaves the circuit, so it is left out
    assert views[2].subs == ["C3"] and views[1].parents == ["C10"]


def test_tasks_list_experiments_newest_first_with_their_description_and_standing():
    old = (f"wiki/circuits/{MODEL}/C1/20261116T120000Z.json", _record("REJECTED"))
    new = (f"wiki/circuits/{MODEL}/C1/20261116T120500Z.json", _record("VERIFIED", minutes=5))
    (task,) = tasks_of([old, new], frozenset())
    assert [e.id for e in task.experiments] == ["20261116T120500Z", "20261116T120000Z"]
    (same,) = tasks_of([new, old], frozenset())
    assert [e.id for e in same.experiments] == ["20261116T120500Z", "20261116T120000Z"]
    assert task.description == "A task." and task.standing == CONTESTED
    # Newest first by time, not by name: here the newer record's name sorts first.
    named = [("wiki/z.json", _record("REJECTED")), ("wiki/a.json", _record(minutes=5))]
    (by_time,) = tasks_of(named, frozenset())
    assert [e.source for e in by_time.experiments] == ["wiki/a.json", "wiki/z.json"]
