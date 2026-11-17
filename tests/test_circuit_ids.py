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
