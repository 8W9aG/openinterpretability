"""Concept probes: label files, full sparse curves, and refused one-class token sets."""

import pytest

from openinterpretability.core.probing import ProbeScore, SiteProbes, SparsePoint
from openinterpretability.core.schemas import NeuronIdentifier
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.probing import (
    Window,
    compare,
    neurons_needed,
    read_labels,
    sparse_curve,
)
from tests.conftest import N_LAYERS
from tests.test_probing import OffsetTokenizer, planted


def test_read_labels_skips_blank_lines(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text('{"text": "ab", "spans": [[0, 1]]}\n\n{"text": "cd"}\n')
    assert read_labels(path) == [("ab", [(0, 1)]), ("cd", [])]


def test_a_sparse_curve_within_the_width_has_a_point_for_every_k():
    (x, y), (xt, yt) = planted(), planted(seed=1)
    points, rank = sparse_curve(x, y, xt, yt, neuron=7, l2=1e-3, ks=(1, 3))
    assert rank == 1 and [p.k for p in points] == [1, 3]


def _score(ba):
    return ProbeScore(balanced_accuracy=ba, auroc=ba)


def _dense(ba):
    return SiteProbes(site="mlp_hidden", width=4, logistic=_score(ba), mass_mean=_score(0.5),
                      logistic_cosine=0.0, mass_mean_cosine=0.0)  # fmt: skip


def test_neurons_needed_is_the_first_k_near_the_dense_probe_or_none():
    points = [SparsePoint(k=k, score=_score(ba), includes_neuron=True)
              for k, ba in ((1, 0.6), (4, 0.96))]  # fmt: skip
    assert neurons_needed(points, _dense(0.96)) == 4
    assert neurons_needed(points[:1], _dense(0.96)) is None


def test_compare_refuses_training_tokens_of_one_class(toy_model, submission):
    runner = NeuronHookRunner(toy_model, OffsetTokenizer())
    ident = NeuronIdentifier(model_name="toy", layer=N_LAYERS - 1, neuron_idx=3, pole="positive")
    sub = submission.model_copy(update={"identifier": ident})
    train = [Window([1, 2, 3, 4], [False] * 4, 0)]
    test = [Window([5, 6, 7], [False, True, False], 1)]
    with pytest.raises(ValueError, match=r"the training tokens hold only one class: \(0, 3\)"):
        compare(runner, sub, train, test, max_positive=10, max_negative=10, seed=0)
