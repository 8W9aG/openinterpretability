"""Circuit ids and their lineage (openinterpretability/circuit_ids.py, #4284)."""

from openinterpretability import circuit_ids as ci

DAY = "2026-11-16"


def _events(result):
    return [(e["event"], e["id"]) for e in result.events]


def test_new_clusters_get_new_ids_and_a_rerun_records_nothing():
    first = ci.update({}, [frozenset("abc"), frozenset("xyz")], DAY)
    assert _events(first) == [("new", "C1"), ("new", "C2")]
    again = ci.update(first.circuits, [frozenset("abc"), frozenset("xyz")], DAY, first.circuits)
    assert again.events == [] and again.circuits == first.circuits


def test_a_circuit_that_grows_keeps_its_id():
    result = ci.update({"C1": frozenset("abcd")}, [frozenset("abcde")], DAY)
    assert result.circuits == {"C1": frozenset("abcde")} and _events(result) == [("grow", "C1")]


def test_a_split_keeps_the_id_on_the_larger_part_and_names_the_others():
    result = ci.update({"C1": frozenset("abcdefgh")}, [frozenset("abcde"), frozenset("fgh")], DAY)
    assert result.circuits["C1"] == frozenset("abcde")
    split = next(e for e in result.events if e["event"] == "split")
    assert split["id"] == "C1" and split["into"] == ["C2"]


def test_a_merge_keeps_one_id_and_names_its_sources():
    old = {"C1": frozenset("abcd"), "C2": frozenset("efg")}
    result = ci.update(old, [frozenset("abcdefg")], DAY)
    assert set(result.circuits) == {"C1"}
    merge = next(e for e in result.events if e["event"] == "merge")
    assert merge["id"] == "C1" and merge["from"] == ["C2"]
    assert ("gone", "C2") in _events(result)


def test_an_id_is_never_used_twice(tmp_path):
    path = tmp_path / "m.jsonl"
    first = ci.update({}, [frozenset("abc")], DAY)
    ci.append(path, first.events)
    gone = ci.update(first.circuits, [], DAY, {"C1"})
    ci.append(path, gone.events)
    circuits, ever = ci.load(path)
    assert circuits == {} and ever == {"C1"}
    again = ci.update(circuits, [frozenset("abc")], DAY, ever)
    assert set(again.circuits) == {"C2"}


def test_the_file_round_trips(tmp_path):
    path = ci.path_for("m", tmp_path)
    result = ci.update({}, [frozenset("abc"), frozenset("def")], DAY)
    ci.append(path, result.events)
    assert ci.load(path) == (result.circuits, set(result.circuits))


def test_clusters_are_communities_of_three_or_more_wired_units(record):
    from openinterpretability.builder.tree import build_tree
    from tests.test_circuits import at, wiring

    strong = [("mlp_neuron", 3, 117, 2.0), ("mlp_neuron", 6, 2, 2.0)]
    records = [
        # A strong triangle L1/N5, L3/N117, L6/N2, then a zero-weight edge to L9/N9
        at(record, 1, 5, activations=wiring("activations", down=strong)),
        at(record, 3, 117, weights=wiring("weights", down=[("mlp_neuron", 6, 2, 2.0)])),
        at(record, 6, 2, weights=wiring("weights", down=[("mlp_neuron", 9, 9, 0.0)])),
        at(record, 9, 9),
        at(record, 12, 1),  # wired to nothing
    ]
    (model,) = build_tree(records, [])
    page = "models/Qwen--Qwen2.5-0.5B/{}.html"
    assert ci.clusters(model, {}) == [
        frozenset(page.format(u) for u in ("L1/N5", "L3/N117", "L6/N2"))
    ]
    pair = build_tree(records[:2], [])[0]  # two units are too few for a circuit
    assert ci.clusters(pair, {}) == []


def test_edges_that_touch_a_hub_make_no_cluster(record):
    from openinterpretability.builder.tree import build_tree
    from tests.test_circuits import at, wiring

    spoke = wiring("activations", down=[("mlp_neuron", 12, 1, 0.1)])
    records = [*(at(record, 2, i, activations=spoke) for i in range(10)), at(record, 12, 1)]
    (model,) = build_tree(records, [])
    assert ci.clusters(model, {}) == []  # every edge touches the hub L12/N1


def test_appending_no_events_writes_no_file(tmp_path):
    path = ci.path_for("m", tmp_path / "circuits")
    ci.append(path, [])
    assert not path.parent.exists()
