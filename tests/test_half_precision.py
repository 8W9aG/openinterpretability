"""Checks of a 16-bit record's dtype and of its recorded probe results."""

from openinterpretability.core.schemas import AdversarialTestResult, ReviewPolicy
from openinterpretability.half_precision import compare_dtype, recorded_pass
from openinterpretability.reproduce import ReproductionReport

POLICY = ReviewPolicy()


def probe(**fields):
    base = dict(kind="hard_negative", prompt="p", passed=True, expected_active=False,
                activation=0.0)  # fmt: skip
    return AdversarialTestResult(**base | fields)


def test_a_dtype_the_pin_does_not_allow_fails(record):
    report = ReproductionReport(None, record)
    compare_dtype("record", "float32", "bfloat16", report)
    assert report.mismatches == ["loaded bfloat16 weights; the record pins float32"]
    assert report.warnings == []


def test_a_16_bit_pin_reproduced_in_float32_only_warns(record):
    report = ReproductionReport(None, record)
    compare_dtype("scan", "bfloat16", "float32", report)
    assert report.ok and "the scan pins bfloat16" in report.warnings[0]


def test_an_unmeasured_probe_or_no_scale_has_no_recorded_pass():
    assert recorded_pass(probe(measured=False), POLICY, 1.0) is None
    assert recorded_pass(probe(), POLICY, 0.0) is None


def test_a_probe_without_its_number_fails():
    assert recorded_pass(probe(activation=None), POLICY, 1.0) is False
    pair = probe(kind="counterfactual", expected_active=None, activation=None,
                 expected_delta=0.5, observed_delta=None)  # fmt: skip
    assert recorded_pass(pair, POLICY, 1.0) is False


def test_a_probe_with_its_number_follows_the_rule():
    assert recorded_pass(probe(activation=0.0), POLICY, 1.0) is True
    assert recorded_pass(probe(activation=0.9), POLICY, 1.0) is False
