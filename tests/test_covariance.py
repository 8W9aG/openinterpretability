import pytest
import torch

from openinterp.core.schemas import ReviewPolicy
from openinterp.core.stats import (
    EffectiveCounts,
    balanced_accuracy,
    cluster_texts,
    effective_counts,
    expected_false_discoveries,
    p_balanced_above,
    rank_columns,
    spearman,
    spearman_columns,
)
from openinterp.harness.confounds import ConfoundAccumulator, baseline_predictions
from openinterp.harness.covariance import CovarianceAccumulator, PartnerSearch, joint_ablation
from openinterp.harness.hooks import pole_strength
from openinterp.harness.scan import cut_pieces, plan_batches, scan_neuron
from tests.conftest import D_MLP, FakeTokenizer

TEXTS = ["abcde", "hello world", "zzzz", "quoted 'text'", "mixed Case 123", "aaaa bbbb"]


# -- probe statistics ------------------------------------------------------------


def test_near_duplicates_cluster():
    texts = [
        "Many cities, including Paris and Rome, banned cars.",
        "Many cities, including London and Rome, banned cars.",
        "She is as tall as her older brother.",
    ]
    assert cluster_texts(texts, 0.5) == [0, 0, 1]
    counts = effective_counts(texts, [True, False, True], 0.5)
    assert (counts.raw, counts.clusters, counts.passed) == (3, 2, 1.5)
    assert counts.rate == 0.75


def test_effective_counts_shrink_confidence():
    many = EffectiveCounts(raw=10, clusters=10, passed=10)
    clustered = EffectiveCounts(raw=10, clusters=2, passed=2)
    holdouts = EffectiveCounts(raw=5, clusters=5, passed=5)
    assert p_balanced_above(many, holdouts, 0.6) > p_balanced_above(clustered, holdouts, 0.6)
    assert balanced_accuracy(many, EffectiveCounts(0, 0, 0)) == 1.0  # missing side ignored


def test_expected_false_discoveries():
    assert expected_false_discoveries([0.9, 0.95, 0.5]) == pytest.approx(0.65)


# -- covariance --------------------------------------------------------------------


def test_pearson_accumulator_matches_corrcoef():
    torch.manual_seed(0)
    acts = torch.randn(200, 6)
    acts[:, 3] = 2 * acts[:, 0] + 0.1 * torch.randn(200)
    target = acts[:, 0]
    acc = CovarianceAccumulator(6)
    for chunk in torch.split(torch.arange(200), 37):  # streaming, uneven batches
        acc.update(target[chunk], acts[chunk])
    expected = torch.corrcoef(torch.cat([target[None], acts.T]))[0, 1:]
    torch.testing.assert_close(acc.correlations().float(), expected, atol=1e-5, rtol=1e-4)


def test_ranks_average_ties_and_spearman_is_rank_based():
    x = torch.tensor([0.0, 0.0, 3.0, 1.0, 0.0])
    assert rank_columns(x[:, None])[:, 0].tolist() == [1.0, 1.0, 4.0, 3.0, 1.0]
    a = torch.arange(1.0, 21.0)
    assert spearman(a, a**3) == pytest.approx(1.0)  # monotonic, far from linear
    rho = spearman_columns(a, torch.stack([a.exp(), -a], dim=1))
    torch.testing.assert_close(rho, torch.tensor([1.0, -1.0]))


def _planted(n=6000, seed=0):
    """Target fires on ~3% of tokens. Neuron 3's positive pole and neuron 4's negative
    pole fire with it; neuron 5 has huge outliers on the target's three strongest tokens."""
    g = torch.Generator().manual_seed(seed)
    acts = torch.randn(n, 8, generator=g) * 0.1
    fires = torch.rand(n, generator=g) < 0.03
    target = torch.where(fires, 1 + torch.rand(n, generator=g).exp(), torch.zeros(n))
    acts[fires, 3] += 2.0
    acts[fires, 4] -= 2.0
    peaks = target.topk(3).indices
    acts[peaks, 5] = 500.0  # "massive activations" on three tokens dominate Pearson
    return target, acts


def test_partner_search_finds_coactivating_poles():
    target, acts = _planted()
    search = PartnerSearch(8, reservoir=4096, top_k=1024)
    for chunk in torch.split(torch.arange(target.numel()), 500):
        search.update(target[chunk], acts[chunk])
    partners = search.partners(exclude=0, active_threshold=0.25 * float(target.max()), k=4)
    top = {(p.neuron_idx, p.pole) for p in partners[:2]}
    assert top == {(3, "positive"), (4, "negative")}
    assert all(p.coactive_tokens > 50 for p in partners[:2])

    outlier = next(
        p
        for p in search.partners(0, 0.25 * float(target.max()), k=16)
        if (p.neuron_idx, p.pole) == (5, "positive")
    )
    # Pearson is inflated by three values; the rank correlation is not...
    assert outlier.pearson > 0.1 and outlier.pearson > 5 * abs(outlier.spearman)
    # ...and a handful of co-active tokens (some by chance) doesn't make it a partner:
    # its lift's lower confidence bound stays below the policy minimum.
    assert outlier.coactivation_lift_lower < ReviewPolicy().min_partner_lift
    assert all(p.coactivation_lift_lower > 10 for p in partners[:2])


def test_reservoir_is_bounded_and_deterministic():
    target, acts = _planted()

    def run():
        search = PartnerSearch(8, reservoir=256, top_k=64, seed=7)
        for chunk in torch.split(torch.arange(target.numel()), 333):
            search.update(target[chunk], acts[chunk])
        return search

    a, b = run(), run()
    assert a.res_target.numel() == 256 and a.top_target.numel() == 64
    assert torch.equal(a.res_target, b.res_target)
    assert a.top_target.min() >= torch.sort(target, descending=True).values[63]


# -- confounds -----------------------------------------------------------------------


def test_token_identity_r2_and_baseline():
    acc = ConfoundAccumulator()
    ids = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
    strengths = torch.tensor([5.0, 0.0, 0.0] * 3)  # fires only on token 1
    acc.update(ids, torch.arange(9) % 3, strengths)
    profile = acc.finalize(None, min_count=3)
    assert profile.token_r2 == pytest.approx(1.0)
    assert profile.trigger_tokens[0].token_id == 1
    assert profile.frequency_correlation == 0.0  # all tokens equally frequent

    tok = FakeTokenizer()  # char-level: token id = 1 + ord(ch) % 15
    with_trigger = next(chr(c) for c in range(97, 123) if 1 + c % 15 == 1)
    preds = baseline_predictions(profile, tok, [with_trigger, "b"], threshold=1.0)
    assert preds["token identity"] == [True, False]


def test_scan_gathers_everything(toy_runner):
    scan = scan_neuron(toy_runner, TEXTS, 1, 2, "positive", n_correlated=4)
    assert scan.examples and len(scan.values) == sum(len(t) for t in TEXTS)
    assert len(scan.correlated) == 4 and all(c.neuron_idx != 2 for c in scan.correlated)
    lifts = [c.coactivation_lift_lower for c in scan.correlated]
    assert lifts == sorted(lifts, reverse=True)
    assert 0 <= scan.confounds.token_r2 <= 1 and 0 <= scan.confounds.position_r2 <= 1
    assert len(scan.confounds.position_means) == max(len(t) for t in TEXTS)

    # The reservoir holds every token here, so Spearman equals a direct computation.
    enc = toy_runner.encode(TEXTS)
    _, cache = toy_runner.run_with_cache(enc, [1])
    acts = cache[1][enc["attention_mask"].bool()]
    target = pole_strength(acts[:, 2], "positive")
    top = scan.correlated[0]
    partner = pole_strength(acts[:, top.neuron_idx], top.pole)
    assert top.spearman == pytest.approx(spearman(target, partner), abs=1e-4)


def test_windows_cover_each_document_once():
    tok = FakeTokenizer()
    pieces = list(cut_pieces(tok, ["a" * 10, "b" * 3], 4, 3))
    assert [(p.doc, p.start, p.end, p.fresh) for p in pieces] == [
        (0, 0, 4, None), (0, 3, 7, 3), (0, 6, 10, 3), (1, 0, 3, None),
    ]  # fmt: skip
    assert [p.text for p in cut_pieces(tok, ["abc"], None, None)] == ["abc"]
    with pytest.raises(ValueError, match="stride"):
        list(cut_pieces(tok, ["abc"], 4, 5))


def test_windowed_scan_counts_each_token_once(toy_runner):
    texts = ["the quick brown fox jumps over the lazy dog", *TEXTS]
    scan = scan_neuron(toy_runner, texts, 1, 2, "positive", window_tokens=8, stride=5)
    assert len(scan.values) == sum(len(t) for t in texts)
    assert scan.windows == len(list(cut_pieces(toy_runner.tokenizer, texts, 8, 5)))

    # Each window's own tokens, first window whole and later ones by their fresh tail.
    expected = []
    for piece in cut_pieces(toy_runner.tokenizer, texts, 8, 5):
        acts = toy_runner.neuron_activations(piece.text, 1, 2, "positive")[0].tolist()
        expected += acts[-piece.fresh :] if piece.fresh else acts
    assert scan.values == pytest.approx(expected)

    # Examples are windows that re-measure as recorded, and never overlap.
    assert all(len(ex.tokens) <= 8 for ex in scan.examples)
    for ex in scan.examples:
        acts = toy_runner.neuron_activations(ex.text, 1, 2, "positive")[0].tolist()
        assert ex.activations == pytest.approx(acts, abs=1e-5)
    peaks = [ex.max_activation for ex in scan.examples]
    assert peaks == sorted(peaks, reverse=True)

    # One document read in heavily overlapping windows: the examples still don't overlap.
    doc = texts[0]
    scan = scan_neuron(toy_runner, [doc], 1, 2, "positive", window_tokens=8, stride=2)
    assert len(scan.examples) > 1 and sum(len(ex.tokens) for ex in scan.examples) <= len(doc)


def test_batches_fill_a_token_budget_in_order():
    assert plan_batches([5, 3, 4, 10, 2], 15) == [3, 1, 1]  # 3×5; 10 alone; then 2
    assert plan_batches([20, 1], 8) == [1, 1]  # a piece longer than the budget goes alone
    assert plan_batches([], 8) == []


def test_a_scan_does_not_depend_on_its_batches(toy_runner):
    texts = ["the quick brown fox jumps over the lazy dog", *TEXTS]
    one = scan_neuron(toy_runner, texts, 1, 2, "positive", batch_tokens=1)
    many = scan_neuron(toy_runner, texts, 1, 2, "positive", batch_tokens=10_000)
    assert one.values == pytest.approx(many.values, abs=1e-5)
    peaks = [ex.max_activation for ex in many.examples]  # tied peaks may swap texts
    assert [ex.max_activation for ex in one.examples] == pytest.approx(peaks, abs=1e-5)


def test_joint_ablation(toy_runner):
    result = joint_ablation(toy_runner, TEXTS[:3], 1, 2, "positive", partners=[0, 5])
    assert result.partners == [0, 5]
    assert min(result.kl_single, result.kl_partners, result.kl_joint) >= 0
    assert result.kl_joint > 0  # ablating several neurons moves the output
    none = joint_ablation(toy_runner, TEXTS[:3], 1, 2, "positive", partners=[])
    assert none.kl_partners == 0.0 and D_MLP > 2


# -- registry-level dependence --------------------------------------------------------


def test_reused_probes_are_not_independent(record):
    from openinterp.builder.tree import build_tree, false_discovery
    from tests.test_compiler import experiment

    original = experiment(record, idx=1, status="VERIFIED", counts=(10, 0, 5, 0))
    remeasured = original.model_copy(
        update={
            "verdict": original.verdict.model_copy(
                update={"timestamp": original.verdict.timestamp.replace(hour=18)}
            )
        }
    )
    (model,) = build_tree([original, remeasured])
    (neuron,) = model.layers[0].neurons
    reuse = [e for e in neuron.experiments if not e.independent]
    from openinterp.builder.wiki import experiment_id

    assert len(reuse) == 1
    assert reuse[0].id == experiment_id(remeasured)
    assert reuse[0].reuses == experiment_id(original)

    fdr = false_discovery([model])
    assert fdr["claims"] == 1  # one claim per neuron pole, however often re-measured
    assert 0 <= fdr["expected_false"] < 0.5
