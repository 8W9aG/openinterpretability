"""The significance memo: a key it does not hold is computed and kept."""

import json

from labinacan.site.disk_memo import DiskMemo

from openinterpretability.builder import significance as sig_module
from openinterpretability.builder.significance import significance
from openinterpretability.core.cache_dirs import build_cache
from openinterpretability.core.schemas import ReviewPolicy
from openinterpretability.core.stats import p_balanced_above
from tests.test_compiler import experiment


def test_empty_memo_computes_p_useful_and_keeps_it(record, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_BUILD_CACHE", str(tmp_path))
    memo = DiskMemo("p-useful-test", b"fresh", build_cache)
    monkeypatch.setattr(sig_module, "P_USEFUL", memo)
    sig = significance(experiment(record, idx=1, status="VERIFIED", counts=(10, 0, 5, 0)))
    key = json.dumps(
        [sig.negatives.passed, sig.negatives.clusters, sig.holdouts.passed, sig.holdouts.clusters]
        + [0.6]
    )
    expected = p_balanced_above(sig.negatives, sig.holdouts, 0.6)
    assert memo.get(key) == expected
    assert sig.p_useful == round(expected, 4)
    memo.save()
    assert DiskMemo("p-useful-test", b"fresh", build_cache).get(key) == expected


# The edges that mutation testing found untested (#6810).


def test_each_strength_starts_at_its_own_line():
    strength = sig_module.evidence_strength
    assert strength(sig_module.STRONG_EVIDENCE) == "strong"
    assert strength(0.8999) == "moderate"
    assert strength(sig_module.MODERATE_EVIDENCE) == "moderate"
    assert strength(0.4999) == "weak"


def test_a_stored_p_useful_is_read_without_sampling(record, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_BUILD_CACHE", str(tmp_path))
    memo = DiskMemo("p-useful-test", b"stored", build_cache)
    monkeypatch.setattr(sig_module, "P_USEFUL", memo)
    monkeypatch.setattr(sig_module, "p_balanced_above", lambda *a: 1 / 0)
    rec = experiment(record, idx=1, status="VERIFIED", counts=(10, 0, 5, 0))
    neg, hold = _counts(rec)
    memo.put(json.dumps([neg.passed, neg.clusters, hold.passed, hold.clusters, 0.6]), 0.123456)
    sig = significance(rec)
    # The stored value, rounded; conclusiveness keeps 4 places, not 5 (0.75309).
    assert (sig.p_useful, sig.conclusiveness, sig.direction) == (0.1235, 0.7531, "refutes")


def _counts(rec):
    """The effective counts that significance() makes its memo key of."""
    policy = rec.verdict.policy or ReviewPolicy()
    rows = rec.verdict.adversarial_test_results

    def one(kind):
        kept = [r for r in rows if r.kind == kind and r.measured]
        return sig_module.effective_counts(
            [r.prompt for r in kept], [r.passed for r in kept], policy.probe_cluster_threshold
        )

    return one("hard_negative"), one("positive_holdout")


def test_an_even_chance_supports(record, monkeypatch):
    monkeypatch.setattr(sig_module, "_p_useful", lambda *a: 0.5)
    sig = significance(experiment(record, idx=1, status="VERIFIED", counts=(10, 0, 5, 0)))
    assert (sig.conclusiveness, sig.direction) == (0.0, "supports")


def test_the_records_own_policy_sets_the_line(record, monkeypatch):
    seen = []
    monkeypatch.setattr(sig_module, "_p_useful", lambda n, h, line: seen.append(line) or 0.5)
    rec = experiment(record, idx=1, status="VERIFIED", counts=(10, 0, 5, 0))
    policy = ReviewPolicy(reject_below_balanced_accuracy=0.7)
    verdict = rec.verdict.model_copy(update={"policy": policy})
    significance(rec.model_copy(update={"verdict": verdict}))
    assert seen == [0.7]


def test_a_record_with_no_verdict_has_no_probes(record, monkeypatch):
    monkeypatch.setattr(sig_module, "_p_useful", lambda *a: 0.5)
    sig = significance(record.model_copy(update={"verdict": None}))
    assert (sig.negatives.clusters, sig.holdouts.clusters) == (0, 0)
