"""Memory planning and streamed loads (harness/memory.py): the less common paths."""

from __future__ import annotations

import pytest
import torch

from openinterpretability.harness import memory


@pytest.fixture
def saved_gpt2(tiny_gpt2, tmp_path):
    path = tmp_path / "gpt2"
    tiny_gpt2.save_pretrained(path)
    return path


def test_free_memory_is_free_ram_and_swap_without_a_budget(monkeypatch):
    import psutil

    monkeypatch.delenv(memory.MEMORY_ENV)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("V", (), {"available": 5}))
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("S", (), {"free": 2}))
    assert memory.available_bytes("cpu") == 7


def test_free_memory_on_a_cuda_device_is_the_gpus_free_memory(monkeypatch):
    monkeypatch.delenv(memory.MEMORY_ENV)
    seen = []

    def mem_get_info(device):
        seen.append(device)
        return 123, 456

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    assert memory.available_bytes("cuda:0") == 123
    assert seen == [torch.device("cuda:0")]


def test_free_memory_is_unknown_when_it_cannot_be_read(monkeypatch):
    import psutil

    def fails():
        raise OSError("unsupported platform")

    monkeypatch.delenv(memory.MEMORY_ENV)
    monkeypatch.setattr(psutil, "virtual_memory", fails)
    assert memory.available_bytes("cpu") is None


def test_weights_bytes_is_the_checkpoint_size_without_a_dtype_to_scale(saved_gpt2):
    class NoDtype:
        pass

    size = memory.checkpoint_bytes(saved_gpt2)
    assert size > 0
    assert memory.weights_bytes(saved_gpt2, NoDtype(), torch.bfloat16) == size
    stored = type("C", (), {"torch_dtype": "float32"})()
    assert memory.weights_bytes(saved_gpt2, stored, "auto") == size
    assert memory.weights_bytes(saved_gpt2, stored, torch.bfloat16) == size // 2


def test_streaming_for_a_non_cuda_device_runs_on_the_cpu(saved_gpt2, monkeypatch, tmp_path,
                                                         capsys):  # fmt: skip
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    memory.request("stream")
    try:
        model = memory.load_causal_lm(str(saved_gpt2), "mps", torch.float32, None)
    finally:
        memory.request("auto")
    assert memory.is_streamed(model)
    assert "Streaming runs on the CPU, not mps." in capsys.readouterr().err
    assert memory._budget_device == "cpu"


def test_open_checkpoint_reads_held_tensors_from_the_loader():
    from accelerate.utils import OffloadedWeightsLoader

    held = torch.arange(6.0).view(2, 3)
    opened = memory.OpenCheckpoint(OffloadedWeightsLoader(state_dict={"w": held}))
    assert torch.equal(opened["w"], held)
    assert torch.equal(opened.slice("w", 1), held[1])
    assert list(opened) == ["w"] and len(opened) == 1
    assert opened.files == {}  # nothing opened from disk


def test_snapshot_downloads_only_the_config_and_safetensors(monkeypatch, tmp_path):
    import huggingface_hub

    seen = {}

    def snapshot_download(name, revision, allow_patterns):
        seen.update(name=name, revision=revision, patterns=allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert memory._snapshot("org/model", "abc") == tmp_path
    assert seen == {"name": "org/model", "revision": "abc",
                    "patterns": ["*.json", "*.safetensors"]}  # fmt: skip


def test_a_snapshot_folder_on_disk_is_used_as_it_is(saved_gpt2):
    assert memory._snapshot(str(saved_gpt2), None) == saved_gpt2


def test_a_model_without_decoder_layers_is_not_supported():
    from openinterpretability.core.exceptions import ArchitectureNotSupportedError

    with pytest.raises(ArchitectureNotSupportedError, match="no decoder layers found on lin"):
        memory._layers_path(torch.nn.Linear(2, 2), "lin")


def test_recomputed_layers_run_plainly_without_gradients(saved_gpt2, monkeypatch, tmp_path):
    from openinterpretability.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    memory.request("stream")
    try:
        model = memory.load_causal_lm(str(saved_gpt2), "cpu", torch.float32, None).eval()
    finally:
        memory.request("auto")
    runner = NeuronHookRunner(model, FakeTokenizer())
    before = runner.run_with_interventions(["abcde"])
    calls = []
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", lambda *a, **k: calls.append(1))
    with memory.recompute_below(runner, 1), torch.no_grad():
        during = runner.run_with_interventions(["abcde"])
    assert not calls  # no backward pass to recompute for
    torch.testing.assert_close(during, before)
