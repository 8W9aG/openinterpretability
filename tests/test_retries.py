"""The counting rules behind docs/analyses/retries.md."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from retries import attempts, shares, union_bound  # noqa: E402


def experiment(status, independent=True):
    return SimpleNamespace(
        record=SimpleNamespace(verdict=SimpleNamespace(status=status)), independent=independent
    )


def test_attempts_stop_at_the_first_verified():
    pole = [experiment("REJECTED"), experiment("VERIFIED"), experiment("VERIFIED")]
    got = attempts(pole)
    assert (got["verified"], got["all"], got["independent"]) == (True, 2, 2)
    assert got["first"] is pole[1]


def test_a_re_measurement_is_an_attempt_but_not_an_independent_one():
    pole = [experiment("NEEDS_REVISION"), experiment("VERIFIED", independent=False)]
    got = attempts(pole)
    assert (got["all"], got["independent"]) == (2, 1)


def test_a_unit_never_verified_spends_all_its_attempts():
    got = attempts([experiment("REJECTED"), experiment("NEEDS_REVISION")])
    assert (got["verified"], got["all"], got["first"]) == (False, 2, None)


def test_per_attempt_share_divides_units_verified_by_attempts_spent():
    rows = [
        {"verified": True, "all": 1, "independent": 1},
        {"verified": True, "all": 3, "independent": 2},
        {"verified": False, "all": 2, "independent": 2},
    ]
    got = shares(rows, records=6, verified_records=3)
    assert (got["units_verified"], got["attempts"], got["independent_attempts"]) == (2, 6, 5)
    assert got["per_attempt"].startswith("33.3%") and got["per_unit"].startswith("66.7%")


def test_union_bound_charges_each_claim_for_its_tries_and_caps_at_one():
    got = union_bound([(0.9, 1), (0.9, 3), (0.2, 2)])
    assert got["expected_false"] == round(0.1 + 0.1 + 0.8, 2)
    assert got["union_bound_false"] == round(0.1 + 0.3 + 1.0, 2)
