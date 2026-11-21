"""The parts of ``harness.layer_major`` that a streamed model leaves out."""

import threading
from types import SimpleNamespace

import torch
from torch import nn

from openinterpretability.harness.layer_major import LayerScheduler, _hooks_per_thread, _locked


def test_installing_and_closing_with_no_pass_loads_nothing_and_restores_forward():
    layer = nn.Linear(2, 2)
    forward = layer.forward
    scheduler = LayerScheduler([layer])
    with scheduler.installed():
        assert layer.forward is not forward
    assert layer.forward == forward
    assert scheduler.loaded is None and scheduler.loads == 0
    scheduler.close()  # closing again with nothing loaded does nothing
    assert scheduler.loaded is None


def test_a_pre_hook_runs_only_in_the_thread_that_registered_it():
    layer = nn.Linear(2, 2)
    calls = []
    register = nn.Module.register_forward_pre_hook
    with _hooks_per_thread():
        handle = layer.register_forward_pre_hook(lambda mod, args: calls.append("hook"))
        worker = threading.Thread(target=layer, args=(torch.zeros(1, 2),))
        worker.start()
        worker.join()
        assert calls == []
        layer(torch.zeros(1, 2))
        assert calls == ["hook"]
    handle.remove()
    assert nn.Module.register_forward_pre_hook is register


def test_a_locked_attribute_set_on_the_instance_is_put_back():
    def encode(text):
        return [len(text)]

    tokenizer = SimpleNamespace(encode=encode)
    with _locked(tokenizer, "encode"):
        assert tokenizer.encode is not encode
        assert tokenizer.encode("abc") == [3]
    assert tokenizer.encode is encode


def test_a_forward_hook_runs_only_in_the_thread_that_registered_it():
    layer = nn.Linear(2, 2)
    calls = []
    register = nn.Module.register_forward_hook
    with _hooks_per_thread():
        handle = layer.register_forward_hook(lambda mod, args, out: calls.append("hook"))
        worker = threading.Thread(target=layer, args=(torch.zeros(1, 2),))
        worker.start()
        worker.join()
        assert calls == []
        layer(torch.zeros(1, 2))
        assert calls == ["hook"]
    handle.remove()
    assert nn.Module.register_forward_hook is register
