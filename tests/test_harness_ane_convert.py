"""The Core ML conversion calls, with coremltools absent or replaced by a fake."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openinterpretability.harness.ane import convert, forward
from tests.test_ane import tiny_qwen2

__all__ = ["tiny_qwen2"]  # the fixture, used by name below


def test_without_coremltools_nothing_is_available_and_no_version(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "coremltools", None)
    assert convert.available() is False
    assert convert.version() is None


class FakeCoreMl(types.ModuleType):
    """The parts of coremltools that ``convert.py`` calls, which record their arguments."""

    def __init__(self):
        super().__init__("coremltools")
        self.__version__ = "9.0-fake"
        self.calls = {}
        self.TensorType = lambda **kw: SimpleNamespace(**kw)
        self.precision = SimpleNamespace(FLOAT16="fp16")
        self.target = SimpleNamespace(macOS15="macOS15-target")
        self.ComputeUnit = SimpleNamespace(CPU_AND_NE="cpu-and-ne")
        self.utils = SimpleNamespace(compile_model=self._compile_model)
        self.models = SimpleNamespace(CompiledMLModel=self._compiled)

    def convert(self, traced, **kw):
        self.calls["convert"] = (traced, kw)
        return SimpleNamespace(save=lambda path: Path(path).mkdir())

    def _compile_model(self, package, dest):
        self.calls["compile"] = (Path(package), Path(package).is_dir())
        Path(dest).mkdir()

    def _compiled(self, path, compute_units):
        return SimpleNamespace(path=path, compute_units=compute_units)


@pytest.fixture
def ct(monkeypatch):
    fake = FakeCoreMl()
    monkeypatch.setitem(sys.modules, "coremltools", fake)
    return fake


def test_with_coremltools_it_is_available_and_gives_its_version(ct):
    assert convert.available() is True
    assert convert.version() == "9.0-fake"


def test_a_chunk_is_traced_converted_to_fp16_and_compiled(ct, tiny_qwen2, tmp_path):
    net = forward.ToLayer(tiny_qwen2, 0, 2, 16)
    ids = torch.zeros((2, 16), dtype=torch.long)
    dest = tmp_path / "out.mlmodelc"
    convert.compile_to(net, ids, dest)
    traced, kw = ct.calls["convert"]
    assert isinstance(traced, torch.jit.ScriptModule)
    names = [spec.name for spec in kw["inputs"]]
    assert names == ["ids", *net.extra]
    assert [spec.dtype for spec in kw["inputs"]] == [np.int32, np.float16, np.float16, np.float16]
    assert kw["inputs"][0].shape == (2, 16) and kw["outputs"][0].name == "acts"
    assert kw["compute_precision"] == "fp16" and kw["minimum_deployment_target"] == "macOS15-target"
    package, saved = ct.calls["compile"]
    assert saved and package.name == "to_layer.mlpackage"
    assert dest.is_dir() and not package.parent.exists()  # the work folder is deleted


def test_a_compiled_model_loads_for_the_cpu_and_the_ane(ct, tmp_path):
    model = convert.load(tmp_path / "m.mlmodelc")
    assert model.path == str(tmp_path / "m.mlmodelc") and model.compute_units == "cpu-and-ne"


def test_extra_inputs_are_fp16_or_int32_arrays():
    fed = convert.feed({"mask": torch.zeros(2, 2), "positions": torch.arange(2)})
    assert fed["mask"].dtype == np.float16 and fed["positions"].dtype == np.int32


@pytest.mark.parametrize(
    ("inputs", "name", "dtype"),
    [
        (torch.zeros((2, 3), dtype=torch.long), "ids", np.int32),
        (torch.zeros((2, 3, 4)), "resid", np.float16),
    ],
)
def test_a_prediction_names_its_input_keeps_it_and_returns_fp32(inputs, name, dtype):
    seen = {}

    class Model:
        def predict(self, feed):
            seen.update(feed)
            return {"acts": np.ones((2, 3, 5), dtype=np.float16)}

    model = Model()
    out = convert.predict(model, inputs, {"mask": np.zeros(1)})
    assert set(seen) == {name, "mask"} and seen[name].dtype == dtype
    assert convert._INPUTS[id(model)][-1] is seen[name]  # kept until Core ML is done with it
    assert out.dtype == torch.float32 and out.shape == (2, 3, 5)
    del convert._INPUTS[id(model)]
