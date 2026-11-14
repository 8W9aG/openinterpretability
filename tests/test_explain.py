"""`openinterp explain` (openinterp/explain.py, issue #3250), on the conftest's toy model."""

import json

import pytest
import torch
from click.testing import CliRunner

from openinterp import explain as ex
from openinterp.core.identifiers import NeuronIdentifier

TEXTS = ["a b c d e f g h", "h g f e d c b a", "a c e g b d f h"] * 20
PROMPT = "a b c d"


@pytest.fixture
def means(toy_runner, tmp_path, monkeypatch):
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    return ex.neuron_means(toy_runner, "toy", TEXTS, 200)


def test_means_are_cached_and_leave_out_each_windows_first_token(toy_runner, means, tmp_path):
    assert sorted(means) == list(range(toy_runner.n_layers))
    assert len(list((tmp_path / "means").glob("*.pt"))) == 1
    ids = [i for t in TEXTS for i in toy_runner.tokenizer.encode(t)][:200]
    windows = torch.tensor([ids[i : i + 4] for i in range(0, len(ids) - 3, 4)])
    _, cache = toy_runner.run_with_cache(windows, [0])
    assert torch.allclose(means[0], cache[0][:, 1:].reshape(-1, cache[0].shape[-1]).mean(0))


def test_each_contribution_goes_to_the_pole_its_sign_names(toy_runner, means):
    token = int(ex.logits_with(toy_runner, PROMPT, {}, -1).argmax())
    scores, _ = ex.attribute(toy_runner, PROMPT, means, token, -1)
    _, cache = toy_runner.run_with_cache(PROMPT, [0])
    act = cache[0][0, 1:]  # the first token is left out
    for (layer, neuron, pole), _ in scores.items():
        if layer == 0:
            values = act[:, neuron]
            assert (values > 0).any() if pole == "positive" else (values < 0).any()


def test_ablating_a_last_layer_pole_moves_the_logit_by_minus_its_attribution(toy_runner, means):
    # The toy model has no attention, and its last layer reaches the logits linearly, so
    # first-order attribution is exact there.
    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=1000, confirm=1000)
    last = [p for p in result.poles if p.layer == toy_runner.n_layers - 1 and p.ablation_change]
    assert last
    for p in last:
        assert p.ablation_change == pytest.approx(-p.attribution, abs=1e-4)


def test_shares_and_labels(toy_runner, means):
    first = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=3)
    assert first.verified_share == 0.0 and first.joint == {"poles": 0}
    assert all(p.status == "unstudied" and p.explanation is None for p in first.poles)
    top = first.poles[0]
    slug = NeuronIdentifier(
        model_name="toy", layer=top.layer, neuron_idx=top.neuron, pole=top.pole
    ).slug
    labels = {slug: ("VERIFIED", "fires on b", 2.0)}
    result = ex.explain(toy_runner, "toy", PROMPT, means, labels, top=3)
    shown = result.poles[0]
    assert shown.status == "VERIFIED" and shown.explanation == "fires on b"
    assert result.verified_share == pytest.approx(abs(shown.attribution) / result.total)
    assert result.joint["poles"] == 1 and "logit_change" in result.joint
    rejected = ex.explain(toy_runner, "toy", PROMPT, means, {slug: ("REJECTED", "x", 2.0)}, top=3)
    assert rejected.poles[0].explanation is None  # only a VERIFIED explanation is shown


def test_cli(toy_runner, tmp_path, monkeypatch):
    from openinterp.cli import cli

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr("openinterp.cli_explain._load_runner", lambda *a, **k: toy_runner)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    base = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200"]
    args = [*base, "--wiki-dir", str(tmp_path / "wiki"), "--top", "3"]
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    assert "VERIFIED poles hold 0.0%" in out.output and "nothing here is explained" in out.output
    as_json = CliRunner().invoke(cli, [*args, "--json"])
    assert json.loads(as_json.output)["verified_share"] == 0.0
