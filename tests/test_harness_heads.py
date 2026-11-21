"""Head scans and head patching: the inputs they refuse."""

import pytest
import torch

from openinterpretability.harness.heads import patch_heads, scan_head
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.patcher import metric_change, token_logit_metric
from tests.conftest import FakeTokenizer
from tests.test_head_audit import HEAD

EFFECT = metric_change(token_logit_metric(3))


@pytest.fixture
def runner(tiny_gpt2):
    tiny_gpt2.set_attn_implementation("eager")
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


def test_a_scan_needs_a_tokenizer(tiny_gpt2):
    with pytest.raises(ValueError, match="scan_head requires a tokenizer"):
        scan_head(NeuronHookRunner(tiny_gpt2), ["abc"], HEAD)


def test_a_scan_of_an_empty_corpus_is_refused(runner):
    with pytest.raises(ValueError, match="the corpus has no tokens to scan"):
        scan_head(runner, [], HEAD)


def test_patch_heads_takes_one_source_and_one_target(runner):
    two = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="one source and one target prompt"):
        patch_heads(runner, two, torch.tensor([[1, 2, 3]]), 1, [0], effect=EFFECT)


def test_patch_heads_at_every_position_needs_prompts_of_one_length(runner):
    with pytest.raises(ValueError, match="differ in length; use positions='last'"):
        patch_heads(runner, torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2]]), 1, [0],
                    effect=EFFECT)  # fmt: skip
