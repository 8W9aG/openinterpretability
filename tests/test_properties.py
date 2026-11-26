"""Properties of the code that decides verdicts, tested with Hypothesis (#3167).

Each property holds for every input that Hypothesis generates, not only for the cases that a
test names. ``derandomize=True`` makes each run try the same examples. So a property never
passes on one run and fails on the next, and a counterexample fails every run.
"""

import subprocess

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from openinterpretability.agents.reviewer import decide, review_metrics
from openinterpretability.core.schemas import AdversarialTestResult, ReviewPolicy
from openinterpretability.core.stats import cluster_texts
from openinterpretability.signing import SignatureError, sign_record, verify_record
from tests.test_compiler import _reviewed_by, _standing, experiment

PROPERTY = settings(derandomize=True, max_examples=60, deadline=None)
# A pytest fixture is made once for all of a test's examples. The tests below only read it.
WITH_FIXTURE = settings(
    PROPERTY, suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=25
)

# Prompts from a few words, so that some share most of their words and cluster.
WORDS = st.sampled_from(["the", "cat", "sat", "on", "a", "mat", "dog", "ran"])
PROMPT = st.lists(WORDS, min_size=2, max_size=5).map(" ".join)
RESULT = st.builds(
    AdversarialTestResult,
    kind=st.sampled_from(["hard_negative", "positive_holdout", "counterfactual"]),
    prompt=PROMPT,
    passed=st.booleans(),
    measured=st.booleans(),
)


def _orders(draw_list):
    """A list, and the same list in an order that Hypothesis chooses."""
    return draw_list.flatmap(lambda items: st.tuples(st.just(items), st.permutations(items)))


@PROPERTY
@given(_orders(st.lists(RESULT, max_size=30)))
def test_the_review_does_not_depend_on_the_order_of_its_test_prompts(pair):
    results, shuffled = pair
    policy = ReviewPolicy()
    metrics = review_metrics(results, policy, independent=True)
    assert review_metrics(shuffled, policy, independent=True) == metrics
    assert decide(review_metrics(shuffled, policy, independent=True), policy) == decide(
        metrics, policy
    )


def _partition(texts: list[str], labels: list[int]) -> set[frozenset[tuple[str, int]]]:
    """The clusters as sets of (text, occurrence) pairs, which no order or numbering changes."""
    seen: dict[str, int] = {}
    keyed = []
    for text in texts:
        seen[text] = seen.get(text, 0) + 1
        keyed.append((text, seen[text]))
    groups: dict[int, set] = {}
    for key, label in zip(keyed, labels, strict=True):
        groups.setdefault(label, set()).add(key)
    return {frozenset(g) for g in groups.values()}


@PROPERTY
@given(
    st.lists(PROMPT, max_size=20).flatmap(
        lambda texts: st.tuples(st.just(texts), st.permutations(texts))
    ),
    st.sampled_from([0.3, 0.5, 0.8, 1.0]),
)
def test_clustering_does_not_depend_on_the_order_of_the_texts(pair, threshold):
    texts, shuffled = pair
    # Equal texts are interchangeable, so the partition is read by text and occurrence.
    left = _partition(texts, cluster_texts(texts, threshold))
    right = _partition(shuffled, cluster_texts(shuffled, threshold))
    assert {frozenset(t for t, _ in g) for g in left} == {frozenset(t for t, _ in g) for g in right}
    assert sorted(map(len, left)) == sorted(map(len, right))


READING = st.tuples(
    st.sampled_from(["VERIFIED", "REJECTED", "NEEDS_REVISION"]),
    st.sampled_from(["anthropic/a", "deepseek/flash", "muse/spark"]),
    st.integers(min_value=1, max_value=48),
)


@WITH_FIXTURE
@given(st.lists(READING, min_size=1, max_size=5, unique_by=lambda r: (r[1], r[2])), st.randoms())
def test_standing_does_not_depend_on_the_order_of_the_records(record, readings, rnd):
    first = experiment(record, idx=117, status="VERIFIED", counts=(10, 0, 5, 0))
    records = [first] + [
        _reviewed_by(first, reviewer, status=status, hours=hours)
        for status, reviewer, hours in readings
    ]
    shuffled = list(records)
    rnd.shuffle(shuffled)
    standing, _ = _standing(records)
    again, _ = _standing(shuffled)
    assert again.status == standing.status
    assert [e.record.verdict for e in again.readings] == [
        e.record.verdict for e in standing.readings
    ]


@pytest.fixture(scope="module")
def signer(tmp_path_factory):
    """One key for every example: making a key per example would take most of the time."""
    folder = tmp_path_factory.mktemp("key")
    key = folder / "maintainer"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "m", "-f", str(key)],
                   check=True)  # fmt: skip
    line = f'will namespaces="openinterp-record" {key.with_suffix(".pub").read_text()}'
    return key, line, folder


@settings(PROPERTY, max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.text(min_size=1, max_size=200), st.data())
def test_signing_then_verifying_always_round_trips(signer, text, data):
    key, line, folder = signer
    record = folder / "r.json"
    record.write_text(text, encoding="utf-8")
    sign_record(record, key)
    assert verify_record(record, line) == "will"
    raw = record.read_bytes()
    at = data.draw(st.integers(min_value=0, max_value=len(raw) - 1))
    changed = raw[:at] + bytes([raw[at] ^ 1]) + raw[at + 1 :]
    record.write_bytes(changed)
    with pytest.raises(SignatureError):
        verify_record(record, line)


def test_the_properties_try_the_same_examples_each_run():
    # No property may pass on one run and fail on the next (#3167).
    for test in (
        test_the_review_does_not_depend_on_the_order_of_its_test_prompts,
        test_clustering_does_not_depend_on_the_order_of_the_texts,
        test_standing_does_not_depend_on_the_order_of_the_records,
        test_signing_then_verifying_always_round_trips,
    ):
        assert test._hypothesis_internal_use_settings.derandomize, test.__name__
