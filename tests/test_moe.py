"""The openinterp experts forward (harness/moe.py): the same as eager, and streamed slices."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from openinterp.harness import memory
from openinterp.harness.moe import (
    EXPERTS_IMPLEMENTATION,
    expert_hook,
    experts_implementation,
    experts_modules,
    use_openinterp_experts,
)

_SPEC = importlib.util.spec_from_file_location(
    "moe_experts_check", Path(__file__).parents[1] / "scripts" / "moe_experts_check.py"
)
check = importlib.util.module_from_spec(_SPEC)
sys.modules["moe_experts_check"] = check
_SPEC.loader.exec_module(check)


def _zero(neuron: int, expert: int, seen: dict):
    def hook(act, e, tokens, weights):
        seen["negative"] = seen.get("negative", False) or bool((weights < 0).any())
        if e != expert:
            return act
        act = act.clone()
        act[:, neuron] = 0
        return act

    return hook


@pytest.mark.parametrize("name", list(check.MODELS))
def test_the_forward_matches_eager_and_its_hook_edits_one_neuron(name):
    model = check.tiny(check.MODELS[name])
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        model.set_experts_implementation("eager")
        eager = model(ids).logits
        assert use_openinterp_experts(model)
        ours = model(ids).logits
        seen: dict = {}
        with expert_hook(experts_modules(model)[0], _zero(3, 1, seen)):
            ablated = model(ids).logits
        after = model(ids).logits
    torch.testing.assert_close(ours, eager, atol=1e-5, rtol=1e-5)
    assert not torch.equal(ablated, ours)
    assert torch.equal(after, ours)  # the hook is gone when its block ends
    assert seen["negative"] is False  # the poles rely on non-negative routing weights
    assert experts_implementation(model) == EXPERTS_IMPLEMENTATION


@pytest.fixture
def saved_mixtral(tmp_path):
    model = check.tiny(check.MODELS["Mixtral"])
    model.save_pretrained(tmp_path / "moe")
    return tmp_path / "moe"


def _load(path, mode, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    memory.request(mode)
    try:
        return memory.load_causal_lm(str(path), "cpu", torch.float32, None).eval()
    finally:
        memory.request("auto")


def test_a_streamed_moe_reads_only_its_routed_experts(saved_mixtral, monkeypatch, tmp_path):
    whole = _load(saved_mixtral, "normal", monkeypatch, tmp_path)
    streamed = _load(saved_mixtral, "stream", monkeypatch, tmp_path)
    assert experts_implementation(whole) == experts_implementation(streamed) == "openinterp"
    stacks, slices = [], []
    get, cut = memory.OpenCheckpoint.__getitem__, memory.OpenCheckpoint.slice

    def read_whole(self, key):
        stacks.append(key)
        return get(self, key)

    def read_slice(self, key, index):
        slices.append((key, index))
        return cut(self, key, index)

    monkeypatch.setattr(memory.OpenCheckpoint, "__getitem__", read_whole)
    monkeypatch.setattr(memory.OpenCheckpoint, "slice", read_slice)
    ids = torch.tensor([[5, 9, 2]])  # 3 tokens, 2 experts each: at most 6 of 4 x 2 layers
    with torch.no_grad():
        torch.testing.assert_close(streamed(ids).logits, whole(ids).logits)
        seen: dict = {}  # the hook edits a streamed expert as it does a loaded one
        with expert_hook(experts_modules(streamed)[0], _zero(3, 0, seen)):
            ablated = streamed(ids).logits
        with expert_hook(experts_modules(whole)[0], _zero(3, 0, seen)):
            torch.testing.assert_close(ablated, whole(ids).logits)
    assert not any("experts" in key for key in stacks)  # no expert stack read whole
    read = {(key.split(".experts.")[0], index) for key, index in slices}
    assert 0 < len(read) <= 2 * 6  # only the routed experts of each layer
