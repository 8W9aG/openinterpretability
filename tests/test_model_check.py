"""``open-interpretability model-check`` (openinterpretability/model_check.py, #3370)."""

from __future__ import annotations

import pytest
import torch
from click.testing import CliRunner
from torch import nn

from openinterpretability import cli_model_check
from openinterpretability.cli import cli
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.model_check import check, config_findings, layout, mlp_family, smoke
from tests.conftest import D_MODEL, FakeTokenizer, ToyModel

IDS = torch.tensor([1, 2, 3, 4, 5])


def _status(found):
    return {f.item: f.status for f in found}


def test_a_model_with_no_attention_or_norms_fails_and_names_each_gap(toy_runner):
    found = {f.item: f for f in layout(toy_runner)}
    assert found["layers"].detail == "3 at model.layers"
    assert found["MLP family"].detail == "plain, activation not in the config"
    assert found["attention"].status == "fail" and "attention output" in found["attention"].detail
    assert found["norm before the MLP"].status == "fail"
    assert found["final norm"].detail == "no final norm found (FINAL_NORM_PATHS)"


def test_the_smoke_run_is_skipped_when_the_layout_fails(toy_runner):
    items = [f.item for f in check(toy_runner, "toy/model", IDS)]
    assert "ablation" not in items and items[-3:] == ["models", "licence", "CI"]


def test_a_gated_mlp_is_named_gated():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(num_hidden_layers=1, hidden_size=8, intermediate_size=16,
                         num_attention_heads=2, num_key_value_heads=1, vocab_size=16)  # fmt: skip
    assert mlp_family(NeuronHookRunner(Qwen2ForCausalLM(config))) == "gated, silu"


def test_a_real_layout_passes_the_smoke_run_on_text(tiny_gpt2):
    found = smoke(NeuronHookRunner(tiny_gpt2, FakeTokenizer()))
    assert _status(found) == {"ablation": "pass", "final read": "pass"}


def test_a_pole_whose_ablation_leaves_the_logits_fails(toy_model):
    with torch.no_grad():
        toy_model.model.layers[-1].mlp.down_proj.weight.zero_()  # the last MLP writes nothing
    found = smoke(NeuronHookRunner(toy_model), IDS)[0]
    assert found.status == "fail" and found.detail.endswith("leaves the logits")


def test_neurons_that_are_not_finite_fail(toy_model):
    with torch.no_grad():
        toy_model.model.layers[-1].mlp.up_proj.weight.fill_(float("nan"))
    found = smoke(NeuronHookRunner(toy_model), IDS)[0]
    assert (found.item, found.status) == ("neurons", "fail")


class _NormedTwice(ToyModel):
    """A toy model whose final norm runs twice, so its one read is not the residual stream."""

    def __init__(self) -> None:
        super().__init__()
        self.model.norm = nn.LayerNorm(D_MODEL)

    def forward(self, input_ids, attention_mask=None):
        out = super().forward(input_ids)
        self.model.norm(self.model.norm(self.embed(input_ids)))
        return out


def test_the_final_read_fails_without_a_final_norm_or_with_two_reads(toy_runner):
    assert smoke(toy_runner, IDS)[1].detail == "no final norm to read"
    found = smoke(NeuronHookRunner(_NormedTwice()), IDS)[1]
    assert found.status == "fail" and found.detail == f"read widths [8, 8], residual stream {8}"


@pytest.mark.parametrize(
    ("ci", "status"),
    [("env:\n  ALLOWED_MODELS: a/x b/y\n", "pass"), ("env: {}\n", "fail")],
)
def test_the_ci_allow_list_is_read_from_the_workflow(tmp_path, ci, status):
    (tmp_path / "p.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["a/x"]\n'
        '[tool.openinterpretability.licenses."a/x"]\n'
        'license = "mit"\nurl = "https://x"\ngated = false\npublishing_allowed = true\n'
    )
    (tmp_path / "ci.yml").write_text(ci)
    found = config_findings("a/x", tmp_path / "p.toml", tmp_path / "ci.yml")
    assert _status(found) == {"models": "pass", "licence": "pass", "CI": status}


def test_the_command_prints_each_finding_and_exits_1_on_a_failure(monkeypatch, toy_runner):
    monkeypatch.setattr(cli_model_check, "_load_runner", lambda *a: toy_runner)
    result = CliRunner().invoke(cli, ["model-check", "toy/model"])
    assert result.exit_code == 1
    assert "FAIL  final norm: no final norm found" in result.output


def test_the_command_exits_0_when_every_check_passes(monkeypatch, tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    monkeypatch.setattr(cli_model_check, "_load_runner", lambda *a: runner)
    result = CliRunner().invoke(cli, ["model-check", "openai-community/gpt2"])
    assert result.exit_code == 0, result.output
    assert "PASS  ablation" in result.output and "FAIL" not in result.output


def test_a_gpt_neox_block_is_named_parallel():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    config = GPTNeoXConfig(num_hidden_layers=1, hidden_size=8, intermediate_size=16,
                           num_attention_heads=2, vocab_size=16)  # fmt: skip
    found = {f.item: f for f in layout(NeuronHookRunner(GPTNeoXForCausalLM(config)))}
    assert found["block"].detail == "parallel: attention and the MLP read the same input"
    assert found["MLP family"].detail == "plain, gelu"
