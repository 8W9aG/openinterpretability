"""The copy family's prompts, splits and review prompt (scripts/copy_family.py, #5232)."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "copy_family.py"
spec = importlib.util.spec_from_file_location("copy_family", SCRIPT)
family = importlib.util.module_from_spec(spec)
sys.modules["copy_family"] = family
spec.loader.exec_module(family)


def _row(first, second, kind="wrong_copy", shared=False):
    return {"prompt": f"A {first} … a {second}", "first": first, "second": second,
            "kind": kind, "shared_number": shared}  # fmt: skip


def test_every_ordered_pair_of_different_subjects_is_a_candidate():
    cands = family.candidates()
    n = len(family.facts())
    assert len(cands) == n * (n - 1)
    week = next(c for c in cands if c["first"] == "week" and c["second"] == "year")
    assert week["prompt"] == "A week has seven days and a year has twelve"
    assert (week["copied"], week["right"]) == (" days", " months")
    dozen = next(c for c in cands if c["first"] == "dozen" and c["second"] == "year")
    assert dozen["shared_number"] and not week["shared_number"]
    assert next(c for c in cands if c["first"] == "hour")["prompt"].startswith("An hour")


def test_a_train_prompt_holds_out_its_reverse_and_shared_numbers_are_held_out():
    rows = [_row(a, b) for a, b in [("week", "year"), ("year", "week"), ("hand", "foot"),
                                    ("foot", "hand"), ("cat", "car")]]  # fmt: skip
    rows += [_row("dozen", "year", shared=True), _row("dog", "cat", "right_copy"),
             _row("car", "bus", "right_copy")]  # fmt: skip
    found = family.split(rows, seed=1)
    by = {r["prompt"]: r for r in found}
    for r in found:
        if r["split"] == "train":
            reverse = next(x for x in found if x["reversed_of"] == r["prompt"])
            assert reverse["split"] == "heldout"  # a learned relation, not a memorised answer
    assert by["A dozen … a year"]["split"] == "heldout"
    assert {by["A dog … a cat"]["split"], by["A car … a bus"]["split"]} == {"retain", "heldout"}
    assert by["A cat … a car"]["split"] == "heldout"  # no reverse: it can't be split
    assert [r["split"] for r in family.split([dict(r) for r in rows], seed=1)] == [
        r["split"] for r in found
    ]  # the seed fixes the splits


def test_the_review_prompt_shows_train_prompts_and_no_measure(tmp_path):
    rows = family.split(
        [_row("week", "year"), _row("year", "week"), _row("dog", "cat", "right_copy")]
    )
    for r in rows:
        r.update(table="prompt", right_p=0.03, top=[(" days", 0.7)])
    rows.insert(0, family.metadata(rows))
    text = family.review_prompt(rows, 5, "x/review.prompts.json")
    assert "Write 5 new **failing** prompts" in text and "x/review.prompts.json" in text
    assert "same category" in text  # the confound is named
    assert "0.03" not in text and "0.7" not in text  # no measure
    held = [r["prompt"] for r in rows if r.get("split") == "heldout"]
    assert all(p not in text for p in held)  # only train prompts are shown
