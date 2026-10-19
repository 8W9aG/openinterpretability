import math
from types import SimpleNamespace

import pytest
import torch

from openinterp.core.exceptions import ShapeMismatchError
from openinterp.harness.attribution import attribution_patching, top_attributed_neurons
from openinterp.harness.hooks import NeuronHookRunner, replace_pole
from openinterp.harness.lookahead import measure_lookahead, output_token_projection
from openinterp.harness.patcher import (
    ActivationPatcher,
    logit_direction_metric,
    metric_change,
    next_token_kl,
    token_logit_metric,
)
from tests.conftest import D_MLP

CLEAN = torch.tensor([[1, 5, 7, 2, 9]])
CORRUPT = torch.tensor([[1, 5, 3, 2, 4]])


def neuron(layer=1, idx=4, pole="positive"):
    return SimpleNamespace(layer=layer, neuron_idx=idx, pole=pole)


@pytest.fixture
def runner(toy_model):
    return NeuronHookRunner(toy_model)


def find_active_neuron(runner, ids, layer, pole):
    strengths = runner.neuron_activations(ids, layer, slice(None), None)  # [1, seq, d_mlp]
    signed = strengths if pole == "positive" else -strengths
    return int(signed.clamp(min=0).sum(dim=(0, 1)).argmax())


def test_self_patch_is_identity(runner):
    result = ActivationPatcher(runner).patch(CLEAN, CLEAN, neuron())
    assert result.metric_delta == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(result.logit_delta, torch.zeros_like(result.logit_delta), atol=1e-6)
    assert math.isnan(result.recovery)  # no source/target gap to recover


def test_denoise_injects_clean_activation(runner):
    """The patched run's activation equals the clean activation on the chosen pole."""
    n = neuron(layer=0, idx=find_active_neuron(runner, CLEAN, 0, "positive"))
    _, clean_cache = runner.run_with_cache(CLEAN, [0])
    _, corrupt_cache = runner.run_with_cache(CORRUPT, [0])

    seen = {}
    patcher = ActivationPatcher(runner)
    original_hooks = runner.hooks

    def spying_hooks(interventions=None, output_interventions=None):
        if interventions and 0 in interventions:
            fn = interventions[0]

            def spy(act):
                out = fn(act)
                if out is not None:  # recorder hooks return None; the injector returns a tensor
                    seen["act"] = out
                return out

            interventions = {0: spy}
        return original_hooks(interventions, output_interventions)

    runner.hooks = spying_hooks
    result = patcher.denoise(CLEAN, CORRUPT, n)
    runner.hooks = original_hooks

    expected = replace_pole(
        corrupt_cache[0][..., n.neuron_idx], clean_cache[0][..., n.neuron_idx], "positive"
    )
    torch.testing.assert_close(seen["act"][..., n.neuron_idx], expected)
    # Other neurons untouched.
    others = [i for i in range(D_MLP) if i != n.neuron_idx]
    torch.testing.assert_close(seen["act"][..., others], corrupt_cache[0][..., others])
    assert result.baseline_metric != result.patched_metric


def test_recovery_bookkeeping(runner):
    patcher = ActivationPatcher(runner)
    clean_logits = runner.run_with_interventions(CLEAN)
    metric = token_logit_metric(int(clean_logits[0, -1].argmax()))
    result = patcher.denoise(CLEAN, CORRUPT, neuron(layer=2), metric=metric)
    assert result.reference_metric == pytest.approx(float(metric(clean_logits)))
    assert result.recovery == pytest.approx(
        (result.patched_metric - result.baseline_metric)
        / (result.reference_metric - result.baseline_metric)
    )


def test_noise_is_reverse_direction(runner):
    patcher = ActivationPatcher(runner)
    n = neuron()
    metric = token_logit_metric(3)
    noised = patcher.noise(CLEAN, CORRUPT, n, metric=metric)
    reverse = patcher.patch(CORRUPT, CLEAN, n, metric=metric)
    assert noised.patched_metric == reverse.patched_metric
    assert noised.baseline_metric == pytest.approx(
        float(metric(runner.run_with_interventions(CLEAN)))
    )


def test_shape_mismatch(runner):
    with pytest.raises(ShapeMismatchError):
        ActivationPatcher(runner).patch(CLEAN, CLEAN[:, :3], neuron())
    result = ActivationPatcher(runner).patch(CLEAN, CLEAN[:, :3], neuron(), positions="last")
    assert math.isfinite(result.patched_metric)


def test_freeze_downstream_blocks_later_mlps(runner, toy_model):
    """With downstream outputs frozen, patching the last layer equals patching with freezing off."""
    patcher = ActivationPatcher(runner)
    n_last = neuron(layer=2)
    assert patcher.patch(
        CLEAN, CORRUPT, n_last, freeze_downstream=True
    ).patched_metric == pytest.approx(patcher.patch(CLEAN, CORRUPT, n_last).patched_metric)
    # For an early layer, freezing removes the indirect path through later MLPs.
    n_first = neuron(layer=0, idx=find_active_neuron(runner, CLEAN, 0, "positive"))
    frozen = patcher.patch(CLEAN, CORRUPT, n_first, freeze_downstream=True)
    free = patcher.patch(CLEAN, CORRUPT, n_first)
    assert frozen.patched_metric != pytest.approx(free.patched_metric)


def test_resample_ablation_averages(runner):
    sources = [CORRUPT, torch.tensor([[2, 2, 2, 2, 2]])]
    result = ActivationPatcher(runner).resample_ablation(CLEAN, sources, neuron())
    assert len(result.per_source) == 2
    assert result.mean_metric_delta == pytest.approx(
        sum(r.metric_delta for r in result.per_source) / 2
    )


def test_attribution_matches_exact_patching_on_last_layer(runner):
    """At the final MLP, logits are linear in the activation, so attribution is exact."""
    clean_logits = runner.run_with_interventions(CLEAN)
    metric = token_logit_metric(int(clean_logits[0, -1].argmax()))
    scores = attribution_patching(runner, CLEAN, CORRUPT, [2], metric=metric)
    assert scores[2].shape == (D_MLP,)

    patcher = ActivationPatcher(runner)
    for idx in range(D_MLP):
        # Swap both poles = the whole activation, matching unpoled attribution.
        pos = patcher.denoise(CLEAN, CORRUPT, neuron(2, idx, "positive"), metric=metric)
        exact = pos.metric_delta
        neg = patcher.denoise(CLEAN, CORRUPT, neuron(2, idx, "negative"), metric=metric)
        exact += neg.metric_delta
        assert float(scores[2][idx]) == pytest.approx(exact, abs=1e-4)


def test_attribution_pole_split_sums(runner):
    pos = attribution_patching(runner, CLEAN, CORRUPT, [0, 1], pole="positive")
    neg = attribution_patching(runner, CLEAN, CORRUPT, [0, 1], pole="negative")
    both = attribution_patching(runner, CLEAN, CORRUPT, [0, 1])
    for layer in (0, 1):
        torch.testing.assert_close(pos[layer] + neg[layer], both[layer], atol=1e-5, rtol=1e-4)
    top = top_attributed_neurons(both, k=3)
    assert len(top) == 3 and abs(top[0][2]) >= abs(top[-1][2])


def test_attribution_leaves_no_param_grads(runner, toy_model):
    attribution_patching(runner, CLEAN, CORRUPT, [1])
    assert all(p.grad is None for p in toy_model.parameters())


def test_attribution_on_gpt2(tiny_gpt2):
    scores = attribution_patching(NeuronHookRunner(tiny_gpt2), CLEAN, CORRUPT, [0, 1])
    assert scores[0].shape == (128,) and torch.isfinite(scores[1]).all()


def test_output_token_projection(runner, toy_model):
    top = output_token_projection(runner, 2, 0, "positive", k=4)
    direct = toy_model.lm_head.weight @ toy_model.model.layers[2].mlp.down_proj.weight[:, 0]
    assert [t for t, _ in top] == direct.topk(4).indices.tolist()
    neg = output_token_projection(runner, 2, 0, "negative", k=4)
    assert [t for t, _ in neg] == (-direct).topk(4).indices.tolist()


def test_measure_lookahead(runner):
    profile = measure_lookahead(runner, CLEAN, 2, 0, "positive")
    assert profile.kl_divergence >= 0
    assert isinstance(profile.is_lookahead, bool)


def test_logit_direction_metric_on_tokens_is_mean_minus_vocab_mean():
    logits = torch.randn(2, 3, 16)
    weights = torch.zeros(16)
    weights[[4, 9]] = 0.5
    expected = pytest.approx(
        float((logits[:, 1, [4, 9]].mean(-1) - logits[:, 1].mean(-1)).mean()), abs=1e-6
    )
    metric = logit_direction_metric(weights, 1)
    assert float(metric(logits)) == expected
    assert float(metric(logits + 3.0)) == expected  # a shared logit offset changes nothing


@pytest.mark.parametrize("freeze", [False, True])
@pytest.mark.parametrize("positions", ["all", "last"])
def test_patch_many_matches_patch_one_at_a_time(runner, freeze, positions):
    metric = token_logit_metric(3)
    indices = [0, 4, 7, 11]
    batched = ActivationPatcher(runner).patch_many(
        CLEAN,
        CORRUPT,
        1,
        indices,
        "positive",
        effect=metric_change(metric),
        positions=positions,
        freeze_downstream=freeze,
        batch_size=3,
    )
    single = [
        ActivationPatcher(runner)
        .patch(
            CLEAN,
            CORRUPT,
            neuron(idx=i),
            metric=metric,
            positions=positions,
            freeze_downstream=freeze,
        )
        .metric_delta
        for i in indices
    ]
    assert batched == pytest.approx(single, abs=1e-5)


def test_next_token_kl_is_zero_without_change_and_positive_with_it(runner):
    logits = runner.run_with_interventions(CLEAN)
    assert next_token_kl(2)(logits, logits) == pytest.approx(0.0, abs=1e-6)
    shifted = logits.clone()
    shifted[:, 2, 0] += 5.0
    assert next_token_kl(2)(shifted, logits) > 0
