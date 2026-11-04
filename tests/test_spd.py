"""A Stochastic Parameter Decomposition of the down projection, read as a dictionary.

scripts/train_spd.py writes each subcomponent as two latents. Read back through the
harness, they must give exactly SPD's projection with every subcomponent masked by its
causal importance.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from openinterp.agents.proposer import format_dictionary_profile
from openinterp.core.schemas import DictionaryProfile
from openinterp.harness.dictionaries import load_dictionary
from openinterp.harness.hooks import NeuronHookRunner
from tests.conftest import VOCAB, FakeTokenizer

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))  # train_spd imports train_transcoders
_SPEC = importlib.util.spec_from_file_location("train_spd", SCRIPTS / "train_spd.py")
train_spd = sys.modules["train_spd"] = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_spd)
spd_model = sys.modules["spd_model"]

LAYER, COMPONENTS = 1, 24
DOWN, UP = (LAYER, "down"), (LAYER, "up")


def _settings(out: Path, gate: str, **changes):
    options = dict(
        model_name="tiny-gpt2", revision=None, layers="all", matrices="down", components=COMPONENTS,
        d_gate=4, gate=gate, init="random", steps=1, batch_rows=4, seq_len=16, lr=1e-3,
        faith_warmup_steps=0,
        coeff_faith=1.0, coeff_stoch=1.0, coeff_layerwise=1.0, coeff_ci_masked=1.0,
        coeff_ci_output=1.0, coeff_imp=1e-3, center=True, p_start=2.0, p_end=1.0,
        p_anneal=1.0, imp_beta=0.5, imp_eps=1e-3, leaky_alpha=0.01, grad_clip=1.0,
        gate_grad_clip=1.0, eval_every=1, alive_rows=4, seed=0, device="cpu",
        scan_corpus=out, out=out, resume=None,
    )  # fmt: skip
    return train_spd.Settings(**options | changes)


@pytest.fixture(params=["component", "layer"])
def exported(tiny_gpt2, tmp_path, request):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    cfg = _settings(tmp_path, request.param)
    ids = torch.randint(0, VOCAB, (4, 16), generator=torch.Generator().manual_seed(1))
    dec = train_spd.Decomposer(runner, [0, LAYER], cfg)
    try:
        dec.set_means(ids)
        _, hidden, inner, logits = dec.importance(ids)
        stats = {"final": train_spd.evaluate(dec, ids)}
        train_spd.export(dec, cfg, ids, stats)
    finally:
        dec.close()
    part = dec.parts[DOWN]
    profile = DictionaryProfile(
        kind="spd", repo=str(tmp_path), path=f"layer_{LAYER}.safetensors",
        input_site="mlp_hidden", activation="causal_importance",
        n_features=2 * COMPONENTS, density=0.5, tokens=1,
    )  # fmt: skip
    assert part.mean.abs().sum() > 0
    h = hidden[DOWN][:, 1:] + part.mean  # importance() returns them centred
    expected = (inner[DOWN][:, 1:] * logits[DOWN][:, 1:].clamp(0, 1)) @ part.u + part.offset()
    return load_dictionary(profile), profile, h, expected.detach()


def test_latent_pairs_reconstruct_the_ci_masked_projection(exported):
    dictionary, _, h, expected = exported
    assert dictionary.n_features % 2 == 0
    torch.testing.assert_close(dictionary.reconstruct(h), expected, atol=1e-5, rtol=1e-4)


def test_each_pair_has_one_sign_active_per_token(exported):
    dictionary, _, h, _ = exported
    acts = dictionary.encode(h).unflatten(-1, (-1, 2))
    assert not ((acts[..., 0] > 0) & (acts[..., 1] > 0)).any()
    assert (acts > 0).any()


def test_a_latent_alone_reads_as_in_the_whole_dictionary(exported):
    dictionary, _, h, _ = exported
    acts = dictionary.encode(h)
    torch.testing.assert_close(dictionary.feature(h, 3), acts[..., 3])
    picked = torch.tensor([0, 5, 6])
    torch.testing.assert_close(dictionary.feature(h, picked), acts[..., picked])


def test_the_proposer_reads_an_spd_latent_as_a_piece_of_the_weights(exported):
    _, profile, _, _ = exported
    text = format_dictionary_profile(profile)
    assert "parameter decomposition" in text
    assert "causal importance" in text


def test_resume_loads_the_checkpoint_and_continues(tiny_gpt2, tmp_path):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    rows = torch.randint(0, VOCAB, (8, 16), generator=torch.Generator().manual_seed(2)).tolist()
    valid = torch.tensor(rows[:4])
    first = train_spd.Decomposer(runner, [LAYER], _settings(tmp_path, "layer"))
    try:
        first.set_means(valid)
        train_spd.train(first, first.cfg, rows, valid)
    finally:
        first.close()
    checkpoint = tmp_path / "checkpoint.pt"
    cfg = _settings(tmp_path, "layer", steps=2, resume=checkpoint, seed=5)
    resumed = train_spd.Decomposer(runner, [LAYER], cfg)
    try:
        assert train_spd.first_step(resumed, cfg) == 1
        for name, value in first.parts[DOWN].state_dict().items():
            torch.testing.assert_close(resumed.parts[DOWN].state_dict()[name], value)
        train_spd.train(resumed, cfg, rows, valid)
    finally:
        resumed.close()
    assert torch.load(checkpoint)["step"] == 1


@pytest.mark.parametrize("site", [DOWN, UP])
def test_neuron_init_makes_each_subcomponent_one_neuron(tiny_gpt2, tmp_path, site):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    d_mlp = spd_model.linear_weight(runner.down_proj(LAYER))[0].shape[0]
    cfg = _settings(tmp_path, "layer", init="neurons", components=d_mlp, matrices="both")
    dec = train_spd.Decomposer(runner, [LAYER], cfg)
    dec.close()
    part = dec.parts[site]
    neurons = part.v if site == DOWN else part.u  # the side the neurons are on
    torch.testing.assert_close(neurons, torch.eye(d_mlp))
    torch.testing.assert_close(part.delta(), torch.zeros_like(part.w_target))


def test_both_matrices_train_and_report_the_input_projection(tiny_gpt2, tmp_path):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    rows = torch.randint(0, VOCAB, (8, 16), generator=torch.Generator().manual_seed(3)).tolist()
    valid = torch.tensor(rows[:4])
    cfg = _settings(tmp_path, "layer", matrices="both")
    dec = train_spd.Decomposer(runner, [LAYER], cfg)
    try:
        assert set(dec.parts) == {DOWN, UP} and set(dec.acts) == {UP}
        dec.set_means(valid)
        train_spd.train(dec, cfg, rows, valid)
        report = train_spd.evaluate(dec, valid)
    finally:
        dec.close()
    assert set(report["l0"]) == {str(LAYER), f"{LAYER}.up"}
    saved = spd_model.load_parts(torch.load(tmp_path / "checkpoint.pt")["parts"])
    assert set(saved) == {DOWN, UP}


def test_the_input_projection_with_every_mask_on_is_the_model(tiny_gpt2, tmp_path):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    dec = train_spd.Decomposer(runner, [LAYER], _settings(tmp_path, "layer", matrices="up"))
    ids = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(4))
    try:
        dec.set_means(ids)
        target = dec.logits(ids)
        ones = (torch.ones(2, 16, COMPONENTS), torch.ones(2, 16))  # subcomponents + delta = W
        torch.testing.assert_close(dec.logits(ids, {UP: ones}), target, atol=1e-4, rtol=1e-4)
    finally:
        dec.close()


def test_old_checkpoints_key_down_projections_by_layer():
    assert spd_model.load_parts({3: "a", (4, "up"): "b"}) == {(3, "down"): "a", (4, "up"): "b"}
