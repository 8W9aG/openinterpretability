"""Run independent forward passes layer by layer over a streamed model.

A streamed model (``--memory stream``, :mod:`openinterp.harness.memory`) reads each
decoder layer from disk on every pass. The review makes hundreds of passes, one after
another, so it reads every layer hundreds of times. :func:`map_layer_major` runs the
items of a loop at once, each in its own thread. It runs each layer once for all the
passes that wait for it:

1. Each thread runs the model's own forward, so each pass computes what it computes
   alone. Only the order of the passes changes.
2. Every thread waits at a layer or has finished. Then the scheduler loads the next
   layer that a thread waits at, in the order of the layers, and lets those threads run
   it. It then frees the layer and loads the next one. So the layers are read in sweeps
   from the first to the last, and a sweep serves every waiting pass.
3. While the threads run, a hook is applied only in the thread that registered it.
   The harness's hooks stay where they are, on modules that every thread shares.

A model that isn't streamed runs the loop as before, one item after another.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import torch
from torch import nn

from openinterp.harness.memory import is_streamed, shared_by, stream_rows

T = TypeVar("T")
R = TypeVar("R")

# The most passes that run at once. Each holds its own activations.
MAX_WORKERS = 32

# The prompt length that :func:`workers_for` plans each pass's memory with.
PLANNED_TOKENS = 256

_active = False  # a layer-major loop is running: a loop inside it runs in order


class LayerScheduler:
    """Lets threads through the streamed layers in sweeps, one loaded layer at a time."""

    def __init__(self, layers: Sequence[nn.Module]) -> None:
        self.layers = list(layers)
        self.streamed = [
            [m for m in layer.modules() if getattr(getattr(m, "_hf_hook", None), "offload", False)]
            for layer in self.layers
        ]
        self.cond = threading.Condition()
        self.running = 0  # threads outside the scheduler's waits
        self.inside = 0  # threads inside the loaded layer
        self.waiting: dict[int, int] = {}  # thread id -> the layer it waits at
        self.loaded: int | None = None
        self.loads = 0  # layers read from disk, for tests and reports

    def start(self) -> None:
        """A worker thread starts."""
        with self.cond:
            self.running += 1

    def finish(self) -> None:
        """A worker thread ends, normally or with an exception."""
        with self.cond:
            self.running -= 1
            self._advance_if_idle()
            self.cond.notify_all()

    def enter(self, layer: int) -> None:
        """Block until ``layer`` is loaded for this thread's pass."""
        me = threading.get_ident()
        with self.cond:
            self.running -= 1
            self.waiting[me] = layer
            self._advance_if_idle()
            self.cond.notify_all()
            while self.loaded != layer or me not in self.waiting:
                self.cond.wait()
            del self.waiting[me]
            self.running += 1
            self.inside += 1

    def leave(self, layer: int) -> None:
        """This thread's pass has run ``layer``, or stopped in it."""
        with self.cond:
            self.inside -= 1

    def _advance_if_idle(self) -> None:
        # Called with the lock held. Only when no thread can still reach the loaded layer.
        if self.running or self.inside or not self.waiting:
            return
        targets = set(self.waiting.values())
        if self.loaded in targets:
            return  # its waiters are being let through
        after = [layer for layer in targets if self.loaded is None or layer > self.loaded]
        self._load(min(after) if after else min(targets))

    def _load(self, layer: int) -> None:
        if self.loaded is not None:
            self._release(self.loaded)
        for module in self.streamed[layer]:
            module._hf_hook.pre_forward(module)  # reads the weights, as each pass did
            module._hf_hook.offload = False  # so a pass doesn't read or free them again
        self.loaded = layer
        self.loads += 1

    def _release(self, layer: int) -> None:
        for module in self.streamed[layer]:
            module._hf_hook.offload = True
            module._hf_hook.post_forward(module, None)  # frees the weights

    def close(self) -> None:
        """Free the layer that is still loaded."""
        if self.loaded is not None:
            self._release(self.loaded)
            self.loaded = None

    @contextmanager
    def installed(self) -> Iterator[None]:
        """Route every pass through :meth:`enter` and :meth:`leave` at each layer."""
        originals = [layer.forward for layer in self.layers]

        def through(index: int, forward: Callable) -> Callable:
            def run(*args, **kwargs):
                self.enter(index)
                try:
                    return forward(*args, **kwargs)
                finally:
                    self.leave(index)

            return run

        for index, (layer, forward) in enumerate(zip(self.layers, originals, strict=True)):
            layer.forward = through(index, forward)
        try:
            with _hooks_per_thread():
                yield
        finally:
            for layer, forward in zip(self.layers, originals, strict=True):
                layer.forward = forward
            self.close()


@contextmanager
def _hooks_per_thread() -> Iterator[None]:
    """While it is open, a forward hook registered in a thread is applied only in it."""
    register_hook, register_pre_hook = (
        nn.Module.register_forward_hook,
        nn.Module.register_forward_pre_hook,
    )

    def owned(fn: Callable) -> Callable:
        owner = threading.get_ident()

        def hook(*args, **kwargs):
            return fn(*args, **kwargs) if threading.get_ident() == owner else None

        return hook

    def forward_hook(self, hook, *args, **kwargs):
        return register_hook(self, owned(hook), *args, **kwargs)

    def forward_pre_hook(self, hook, *args, **kwargs):
        return register_pre_hook(self, owned(hook), *args, **kwargs)

    nn.Module.register_forward_hook = forward_hook
    nn.Module.register_forward_pre_hook = forward_pre_hook
    try:
        yield
    finally:
        nn.Module.register_forward_hook = register_hook
        nn.Module.register_forward_pre_hook = register_pre_hook


@contextmanager
def _locked(obj: Any, name: str) -> Iterator[None]:
    """Let one thread at a time call ``obj.name``: a fast tokenizer isn't thread-safe."""
    own = vars(obj).get(name)  # an attribute set on the instance, to put back after
    method, lock = getattr(obj, name), threading.Lock()

    def call(*args, **kwargs):
        with lock:
            return method(*args, **kwargs)

    setattr(obj, name, call)
    try:
        yield
    finally:
        if own is None:
            delattr(obj, name)
        else:
            setattr(obj, name, own)


def workers_for(runner, n_items: int, tokens: int, logits: bool) -> int:
    """How many passes run at once: as many as a quarter of the planned memory holds.

    Each pass is planned at ``tokens`` tokens, with its logits if ``logits``.
    """
    rows = stream_rows(runner, tokens, 1, logits=logits)
    return max(1, min(n_items, MAX_WORKERS, rows))


def prompt_tokens(runner, texts: Sequence[str]) -> int:
    """The token count of the longest of ``texts``: what a pass over one of them holds."""
    return max((runner.encode([t])["input_ids"].shape[1] for t in texts), default=1)


def map_layer_major(
    runner,
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    tokens: int = PLANNED_TOKENS,
    logits: bool = True,
) -> list[R]:
    """``[fn(item) for item in items]``, run layer-major when ``runner``'s model streams.

    ``tokens`` and ``logits`` describe the largest pass an item makes, so that the
    passes that run at once fit in the planned memory (:func:`workers_for`). ``fn`` must
    not change state that another item reads. The results are in the order of
    ``items``. The first exception raised by an item is raised again.
    """
    global _active
    items = list(items)
    if len(items) < 2 or _active or not is_streamed(runner.model):
        return [fn(item) for item in items]  # a loop inside a running one runs in order
    n_workers = workers_for(runner, len(items), tokens, logits)
    scheduler = LayerScheduler(runner.layers)
    _active = True
    try:
        with scheduler.installed(), _locked(runner, "encode"), shared_by(n_workers):
            results = _run_workers(scheduler, fn, items, n_workers)
    finally:
        _active = False
    runner.layer_loads = getattr(runner, "layer_loads", 0) + scheduler.loads
    return results


def _run_workers(scheduler: LayerScheduler, fn: Callable, items: list, n_workers: int) -> list:
    """Run ``fn`` over ``items`` in ``n_workers`` threads; raise the first exception again."""
    results: list[Any] = [None] * len(items)
    errors: list[BaseException] = []
    queue = iter(range(len(items)))
    lock = threading.Lock()
    grad = torch.is_grad_enabled()  # grad mode is per thread: carry the caller's over

    def work() -> None:
        torch.set_grad_enabled(grad)
        try:
            while not errors:
                with lock:
                    index = next(queue, None)
                if index is None:
                    return
                results[index] = fn(items[index])
        except BaseException as exc:  # raised again in the calling thread
            errors.append(exc)
        finally:
            scheduler.finish()

    threads = [threading.Thread(target=work) for _ in range(n_workers)]
    for thread in threads:
        scheduler.start()
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results
