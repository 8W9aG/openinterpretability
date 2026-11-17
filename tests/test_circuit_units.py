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
