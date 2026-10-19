from openinterp.builder.significance import significance
from tests.test_compiler import experiment


def test_strong_positive(record):
    sig = significance(experiment(record, idx=1, status="VERIFIED", counts=(10, 0, 5, 0)))
    assert sig.direction == "supports" and sig.conclusiveness > 0.9
    assert sig.label == "strong evidence for"


def test_strong_null_is_as_significant_as_strong_positive(record):
    """A well-probed failure is a conclusive result, not a non-result."""
    null = significance(experiment(record, idx=1, status="REJECTED", counts=(5, 10, 1, 9)))
    assert null.direction == "refutes" and null.conclusiveness > 0.9
    assert null.label == "strong evidence against"


def test_thin_evidence_is_inconclusive(record):
    thin = significance(experiment(record, idx=1, status="VERIFIED", counts=(2, 0, 1, 0)))
    many = significance(experiment(record, idx=1, status="VERIFIED", counts=(20, 0, 10, 0)))
    assert thin.conclusiveness < many.conclusiveness
    assert thin.label.startswith(("weak", "moderate"))


def test_deterministic(record):
    rec = experiment(record, idx=1, status="NEEDS_REVISION", counts=(10, 0, 2, 3))
    assert significance(rec) == significance(rec)
    sig = significance(rec)
    assert (sig.holdouts.passed, sig.holdouts.clusters, sig.holdouts.raw) == (2, 5, 5)
