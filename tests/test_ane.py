"""Scans on Apple's Neural Engine: the forward pass, the conversion cache and rescoring."""

import json
import subprocess
import sys

import pytest
import torch

from openinterp.harness import devices, scan_backend
from openinterp.harness.ane import backend, cache, forward
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.layer_cache import LayerCache
from openinterp.harness.scan import Piece, TopExamples, cut_pieces, layer_chunks, scan_neuron
from tests.conftest import VOCAB, FakeTokenizer
from tests.test_agents import TEXTS


@pytest.fixture
def tiny_qwen2():
    """A randomly initialised Qwen2 built from config: real HF architecture, no download."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(config).eval()


def _hf_activations(model, layer: int, ids: torch.Tensor) -> torch.Tensor:
    """The MLP activations of ``layer`` from HF's own forward pass."""
    if model.config.model_type == "gpt2":
        write = model.transformer.h[layer].mlp.c_proj
    else:
        write = model.model.layers[layer].mlp.down_proj
    seen = []
    handle = write.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    with torch.no_grad():
        model(ids)
    handle.remove()
    return seen[0]


@pytest.mark.parametrize("name", ["tiny_gpt2", "tiny_qwen2"])
def test_the_rewritten_pass_matches_hf_in_fp32(request, name):
    import copy

    model = request.getfixturevalue(name)
    ids = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(3))
    for layer in (0, 1):
        net = forward.ToLayer(copy.deepcopy(model), layer, 2, 16).eval()
        with torch.no_grad():
            ours = net(ids)
        hf = _hf_activations(model, layer, ids)
        assert torch.allclose(ours, hf, atol=1e-5, rtol=1e-4)


def test_projections_are_scaled_by_powers_of_two(tiny_qwen2):
    net = forward.ToLayer(tiny_qwen2, 1, 2, 16)
    scaled = [m for m in net.modules() if isinstance(m, forward.Rescaled)]
    assert scaled  # every projection up to the layer
    assert all(s.scale == 2.0 ** round(torch.log2(torch.tensor(s.scale)).item()) for s in scaled)


def test_supported_architectures(tiny_gpt2, tiny_qwen2, toy_model):
    assert forward.supported(tiny_gpt2) and forward.supported(tiny_qwen2)
    assert not forward.supported(toy_model)


def test_each_length_gets_the_smallest_shape_that_holds_it():
    assert backend.shape_for(1, 2048) == (16, 128)
    assert backend.shape_for(129, 2048) == (8, 256)
    assert backend.shape_for(2048, 2048) == (1, 2048)
    assert backend.shape_for(600, 512) is None  # longer than the model's context


# -- the conversion cache ----------------------------------------------------


def _fake(error: float = 0.001):
    calls = {"build": 0, "check": 0}

    def build(dest):
        calls["build"] += 1
        dest.mkdir()
        (dest / "weights.bin").write_bytes(b"x" * 100)

    def check(path):
        calls["check"] += 1
        assert path.parent.name and not path.parent.name.startswith(".tmp")  # final path
        return error

    return calls, build, check


def _key(**changes):
    key = cache.fields("org/model", "abc123", 3, 16, 128)
    return {**key, **changes}


def test_an_entry_is_built_and_checked_once(tmp_path):
    store = cache.ConversionCache(tmp_path)
    calls, build, check = _fake()
    first = store.get(_key(), build, check)
    again = store.get(_key(), build, check)
    assert first == again and first.name == cache.MODEL
    assert calls == {"build": 1, "check": 1}
    meta = json.loads((first.parent / cache.META).read_text())
    assert meta["fields"]["code"] == cache.code_digest() and meta["error"] == 0.001


@pytest.mark.parametrize(
    "change",
    [{"revision": "def456"}, {"code": "0" * 16}, {"coremltools": "10.0"}, {"macos": "16.0"}],
)
def test_a_new_revision_code_library_or_os_builds_a_new_entry(tmp_path, change):
    store = cache.ConversionCache(tmp_path)
    calls, build, check = _fake()
    assert store.get(_key(), build, check) != store.get(_key(**change), build, check)
    assert calls["build"] == 2


def test_the_code_digest_covers_the_conversion_code():
    assert len(cache.code_digest()) == 16
    assert set(cache.SOURCES) == {"forward.py", "convert.py"}


def test_a_model_that_disagrees_with_fp32_is_rejected_and_not_kept(tmp_path):
    store = cache.ConversionCache(tmp_path)
    _, build, check = _fake(error=cache.MAX_ERROR * 2)
    with pytest.raises(cache.ConversionRejected):
        store.get(_key(), build, check)
    assert not store.path(_key()).exists()


def test_an_entry_left_without_meta_is_built_again(tmp_path):
    store = cache.ConversionCache(tmp_path)
    store.path(_key()).mkdir(parents=True)  # a build that stopped before its meta
    calls, build, check = _fake()
    store.get(_key(), build, check)
    assert calls["build"] == 1


def test_the_least_recently_used_entries_go_first(tmp_path):
    store = cache.ConversionCache(tmp_path)
    _, build, check = _fake()
    old = store.get(_key(layer=1), build, check).parent
    store.max_bytes = int(store.entries()[0][1] * 2.5)  # room for two entries
    store.get(_key(layer=2), build, check)
    store.get(_key(layer=1), build, check)  # used again: now the newest
    store.get(_key(layer=3), build, check)
    assert old.exists() and not store.path(_key(layer=2)).exists()


def test_the_cache_can_be_turned_off(monkeypatch):
    monkeypatch.setenv(cache.ENV, "off")
    assert cache.ConversionCache.default() is None


# -- rescoring ---------------------------------------------------------------


def _top(peaks):
    top = TopExamples()
    for doc, peak in enumerate(peaks):
        top.add(Piece(doc, 0, 3, f"doc{doc}"), [1, 2, 3], [0.0, peak, 0.0])
    return top


def test_the_top_and_a_margin_below_the_20th_are_chosen():
    peaks = [float(100 - i) for i in range(70)] + [-5.0] * 200
    picked = scan_backend.chosen(_top(peaks).candidates)
    assert set(range(scan_backend.RESCORE_TOP)) <= picked
    assert len(picked) <= scan_backend.RESCORE_CAP + 1  # with and without token 0
    crowded = [10.0] * 20 + [9.9] * 200  # 200 inside the 6% margin
    assert len(scan_backend.chosen(_top(crowded).candidates)) == scan_backend.RESCORE_CAP


def test_rescoring_makes_every_kept_example_exact():
    exact = [float(i % 97) for i in range(300)]
    noisy = [v * (1.03 if i % 2 else 0.97) for i, v in enumerate(exact)]  # ±3%
    top = _top(noisy)
    measured = scan_backend.rescore(
        top, lambda texts: [[0.0, exact[int(t[3:])], 0.0] for t in texts], k=20
    )
    assert scan_backend.RESCORE_TOP <= measured <= 2 * scan_backend.RESCORE_CAP
    for i in scan_backend.kept(top.candidates, 20):
        assert top.candidates[i][2][1] == exact[i]
    best = sorted(exact, reverse=True)[:20]
    got = sorted((v[1] for *_, v in top.candidates), reverse=True)[:20]
    assert got == best


class FakeAne:
    """Torch's pass with 1% noise, as the ANE's fp16: the scan must rescore its top."""

    name = backend.NAME

    def __init__(self, fail_at=None):
        self.used, self.noted, self.broken, self.fail_at = False, set(), None, fail_at

    def prepare(self, lengths, layer, batch_tokens):
        return self.broken

    def chunks(self, runner, pieces, layer, batch_tokens):
        generator = torch.Generator().manual_seed(4)
        for i, chunk in enumerate(layer_chunks(runner, pieces, layer, batch_tokens)):
            if i == self.fail_at:
                raise scan_backend.AneError("RuntimeError: E5 stream failed")
            noise = 1 + 0.01 * torch.randn(chunk.acts.shape, generator=generator)
            chunk.acts = chunk.acts * noise
            yield chunk


def test_a_scan_on_the_ane_records_the_devices_numbers(tiny_gpt2):
    texts = [t * 3 for t in TEXTS] * 4
    plain = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    ane = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    ane.scan_backend = FakeAne()
    exact = scan_neuron(plain, texts, 1, 2, "positive", k=5, opposite_k=3)
    got = scan_neuron(ane, texts, 1, 2, "positive", k=5, opposite_k=3)
    assert ane.scan_backend.used
    assert [e.activations for e in got.examples] == [e.activations for e in exact.examples]
    assert [e.activations for e in got.opposite.examples] == [
        e.activations for e in exact.opposite.examples
    ]
    assert got.values != exact.values  # the histogram keeps the ANE's values


def test_a_scan_on_the_ane_has_its_own_layer_cache_key(tiny_gpt2, tmp_path):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    runner.model_name = "tiny"
    pieces = list(cut_pieces(runner.tokenizer, TEXTS, None, None))
    on_torch = LayerCache.fields(runner, 1, pieces, 64)
    runner.scan_backend = FakeAne()
    on_ane = LayerCache.fields(runner, 1, pieces, 64)
    assert on_ane["device"] == backend.NAME and on_torch["device"] == "cpu"


def test_the_record_names_the_scan_backend_when_the_ane_ran(tiny_gpt2):
    from openinterp.evidence import _scan_environment

    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    assert _scan_environment(runner).scan_backend is None
    runner.scan_backend = FakeAne()
    assert _scan_environment(runner).scan_backend is None  # attached, not used
    runner.scan_backend.used = True
    assert _scan_environment(runner).scan_backend == backend.NAME


# -- the planner's ANE tier --------------------------------------------------


def test_a_busy_gpu_sends_scans_to_a_free_ane():
    plan = devices.choose(devices.Load({"mps": [5], "ane": []}), True, 50.0, ane_ok=True)
    assert (plan.device, plan.scan, plan.fallback) == ("cpu", "ane", True)


def test_a_busy_gpu_and_a_busy_ane_send_everything_to_the_cpu():
    plan = devices.choose(devices.Load({"mps": [5], "ane": [6]}), True, 50.0, ane_ok=True)
    assert (plan.device, plan.scan) == ("cpu", None)
    assert "the ANE (process 6)" in plan.reason


def test_a_free_gpu_never_uses_the_ane():
    plan = devices.choose(devices.Load({"mps": [], "ane": []}, 0.0), True, 50.0, ane_ok=True)
    assert (plan.device, plan.scan) == ("mps", None)


def test_device_ane_forces_scans_on_the_ane(monkeypatch):
    from openinterp.cli import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(devices, "ane_possible", lambda: True)
    assert resolve_device("ane") == "mps"  # the GPU is free in tests
    assert devices.scans_on_ane()


# -- the real ANE ------------------------------------------------------------


@pytest.mark.skipif(not devices.ane_possible(), reason="needs Apple's Neural Engine")
def test_the_ane_scans_a_tiny_model_and_the_cache_keeps_it(tiny_qwen2, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "SEQS", (16, 32, 64))
    monkeypatch.setattr(backend, "SHAPE_TOKENS", 64)
    runner = NeuronHookRunner(tiny_qwen2, FakeTokenizer())
    pieces = list(cut_pieces(runner.tokenizer, TEXTS, None, None))
    store = cache.ConversionCache(tmp_path)
    builds = []
    real = backend.convert.compile_to
    monkeypatch.setattr(backend.convert, "compile_to", lambda *a: builds.append(a) or real(*a))
    ane = backend.AneScan(tiny_qwen2, "tiny-qwen2", None, store)
    got = torch.cat([c.acts for c in ane.chunks(runner, pieces, 1, 64)])
    want = torch.cat([c.acts for c in layer_chunks(runner, pieces, 1, 64)])
    assert float((got - want).norm() / want.norm()) < cache.MAX_ERROR
    assert builds and len(store.entries()) == len(builds)
    again = backend.AneScan(tiny_qwen2, "tiny-qwen2", None, store)
    list(again.chunks(runner, pieces, 1, 64))
    assert len(builds) == len(store.entries())  # the second scan converted nothing


@pytest.mark.skipif(not devices.ane_possible(), reason="needs Apple's Neural Engine")
def test_an_idle_model_does_not_crash_the_process(tiny_qwen2, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "SEQS", (16, 32, 64))
    monkeypatch.setattr(backend, "SHAPE_TOKENS", 64)
    ane = backend.AneScan(tiny_qwen2, "tiny-qwen2", None, cache.ConversionCache(tmp_path))
    ane.compiled(1, 4, 16)
    path = next(tmp_path.rglob(cache.MODEL))
    script = (
        "import sys, time, torch\n"
        "from openinterp.harness.ane import convert\n"
        "model = convert.load(sys.argv[1])\n"
        "convert.predict(model, torch.zeros((4, 16), dtype=torch.long))\n"
        "time.sleep(2)  # Core ML resets an idle stream after about 0.5 s\n"
        "print('ok')\n"
    )
    done = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.strip() == "ok"


# -- preparing a scan: shapes, memory and failures ---------------------------


def test_the_shapes_a_scan_needs_follow_its_batches():
    assert backend.needed_shapes([100, 120, 300], 2048, 2048) == {(4, 512)}  # one batch, 300 wide
    assert backend.needed_shapes([100] * 30, 2048, 2048) == {(16, 128)}
    assert backend.needed_shapes([3000], 4096, 2048) is None


def test_a_conversion_copies_only_the_layers_it_needs(tiny_qwen2):
    copied = backend.copy_to_layer(tiny_qwen2, 0)
    assert len(copied.model.layers) == 1 and len(tiny_qwen2.model.layers) == 2
    assert isinstance(copied.get_output_embeddings(), torch.nn.Identity)
    assert isinstance(tiny_qwen2.get_output_embeddings(), torch.nn.Linear)  # unchanged
    ids = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        small = forward.ToLayer(copied, 0, 2, 16)(ids)
    assert torch.allclose(small, _hf_activations(tiny_qwen2, 0, ids), atol=1e-5, rtol=1e-4)


def _ane(model, monkeypatch, tmp_path, compiled=None):
    ane = backend.AneScan(model, "tiny", None, cache.ConversionCache(tmp_path))
    monkeypatch.setattr(backend, "SEQS", (16, 32, 64))
    monkeypatch.setattr(backend, "SHAPE_TOKENS", 64)
    calls = []
    monkeypatch.setattr(ane, "compiled", compiled or (lambda *a: calls.append(a)))
    return ane, calls


def test_a_conversion_that_wont_fit_in_memory_keeps_the_scan_on_torch(
    tiny_qwen2, monkeypatch, tmp_path
):
    import openinterp.harness.memory as memory

    monkeypatch.setattr(memory, "available_bytes", lambda device=None: 1000)
    ane, calls = _ane(tiny_qwen2, monkeypatch, tmp_path)
    reason = ane.prepare([10, 12], 1, 64)
    assert "GB is free" in reason and calls == []
    monkeypatch.setattr(ane, "_cached", lambda *a: True)  # nothing to convert: memory is moot
    assert ane.prepare([10, 13], 1, 64) is None and calls == [(1, 4, 16)]


def test_a_failed_conversion_keeps_the_scan_on_torch_and_says_why_once(
    tiny_qwen2, monkeypatch, tmp_path, capsys
):
    def fail(*a):
        raise cache.ConversionRejected("relative error 0.5")

    ane, _ = _ane(tiny_qwen2, monkeypatch, tmp_path, compiled=fail)
    runner = NeuronHookRunner(tiny_qwen2, FakeTokenizer())
    runner.scan_backend = ane
    pieces = list(cut_pieces(runner.tokenizer, TEXTS, None, None))
    assert scan_backend.ane_scan(runner, pieces, 1, 64) is None
    assert scan_backend.ane_scan(runner, pieces, 1, 64) is None
    err = capsys.readouterr().err
    assert err.count("not the ANE: its conversion failed") == 1


def test_one_scan_gets_one_answer(tiny_qwen2, monkeypatch, tmp_path):
    ane, calls = _ane(tiny_qwen2, monkeypatch, tmp_path)
    assert ane.prepare([10], 1, 64) is None
    monkeypatch.setattr(ane, "_prepare", lambda *a: "changed its mind")
    assert ane.prepare([10], 1, 64) is None  # the cache key and the pass must agree


def test_only_a_qualified_model_scans_on_the_ane(tiny_qwen2, monkeypatch):
    from openinterp.harness.ane import qualified

    monkeypatch.setattr(backend.convert, "available", lambda: True)
    assert backend.AneScan.for_model(tiny_qwen2, "tiny", None) is None
    monkeypatch.setitem(qualified.QUALIFIED, "tiny", ((1,), "2026-11-12"))
    assert backend.AneScan.for_model(tiny_qwen2, "tiny", None) is not None


def test_an_ane_failure_during_a_scan_redoes_it_on_torch(tiny_gpt2, capsys):
    texts = [t * 3 for t in TEXTS] * 4
    plain = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    ane = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    ane.scan_backend = FakeAne(fail_at=1)
    exact = scan_neuron(plain, texts, 1, 2, "positive", k=5, batch_tokens=64)
    got = scan_neuron(ane, texts, 1, 2, "positive", k=5, batch_tokens=64)
    assert got.values == exact.values  # every value from torch: nothing kept from the ANE
    assert [e.activations for e in got.examples] == [e.activations for e in exact.examples]
    assert "it failed during a scan" in ane.scan_backend.broken
    assert "The ANE failed, so this scan runs again on cpu" in capsys.readouterr().err


def test_a_retry_after_an_mps_failure_scans_on_the_ane(monkeypatch):
    from openinterp.cli import resolve_device

    monkeypatch.setenv(devices.FORCE_ENV, "cpu")
    monkeypatch.setattr(devices, "ane_possible", lambda: True)
    assert resolve_device("auto") == "cpu"
    assert devices.scans_on_ane()  # MPS, then the ANE, then the CPU


def test_a_compile_that_takes_too_long_keeps_the_scan_on_torch(tiny_qwen2, monkeypatch, tmp_path):
    import time

    monkeypatch.setenv(backend.PREPARE_ENV, "0.01")  # 0.6 s
    ane, _ = _ane(tiny_qwen2, monkeypatch, tmp_path, compiled=lambda *a: time.sleep(2))
    reason = ane.prepare([10], 1, 64)
    assert "took more than 0.01 minutes" in reason
    assert ane.prepare([11], 1, 64) == reason  # not again in this process
