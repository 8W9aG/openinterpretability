"""The significance memo: a key it does not hold is computed and kept."""

import json

from openinterpretability.builder import significance as sig_module
from openinterpretability.builder.disk_memo import DiskMemo
from openinterpretability.builder.significance import significance
from openinterpretability.core.stats import p_balanced_above
from tests.test_compiler import experiment


def test_empty_memo_computes_p_useful_and_keeps_it(record, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_BUILD_CACHE", str(tmp_path))
    memo = DiskMemo("p-useful-test", b"fresh")
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
    assert DiskMemo("p-useful-test", b"fresh").get(key) == expected
