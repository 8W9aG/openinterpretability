"""The run environment: the hardware name on each platform, and a missing transformers."""

import subprocess
import sys

import torch

from openinterpretability.harness import environment
from openinterpretability.harness.environment import device_name, run_environment
from openinterpretability.harness.hooks import NeuronHookRunner
from tests.conftest import FakeTokenizer


def test_the_environment_has_no_transformers_version_when_transformers_is_missing(
    tiny_gpt2, monkeypatch
):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", None)  # the import then fails
    env = run_environment(runner)
    assert env.transformers_version is None
    assert env.torch_version == torch.__version__


def test_a_cuda_device_is_named_by_its_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Fake GPU 80GB")
    assert device_name(torch.device("cuda")) == "Fake GPU 80GB"


def test_a_cuda_device_whose_name_fails_falls_back_to_the_cpu_name(monkeypatch, tmp_path):
    def broken(device):
        raise RuntimeError("no driver")

    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Fake CPU @ 3.0GHz\n")
    monkeypatch.setattr(torch.cuda, "get_device_name", broken)
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment, "Path", lambda path: cpuinfo)
    assert device_name(torch.device("cuda")) == "Fake CPU @ 3.0GHz"


def test_a_mac_names_its_chip_from_sysctl(monkeypatch):
    def run(args, **kwargs):
        assert args == ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.CompletedProcess(args, 0, stdout="Apple M9\n", stderr="")

    monkeypatch.setattr(environment.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(environment.subprocess, "run", run)
    assert device_name(torch.device("cpu")) == "Apple M9"


def test_a_linux_cpuinfo_without_a_model_name_falls_back_to_the_processor(monkeypatch, tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nflags\t: fpu\n")
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "processor", lambda: "x86_64")
    monkeypatch.setattr(environment, "Path", lambda path: cpuinfo)
    assert device_name(torch.device("cpu")) == "x86_64"


def test_a_missing_cpuinfo_and_no_processor_gives_no_name(monkeypatch, tmp_path):
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "processor", lambda: "")
    monkeypatch.setattr(environment, "Path", lambda path: tmp_path / "missing")
    assert device_name(torch.device("cpu")) is None
