"""The staged sister search over model pairs (harness/sister_passes.py)."""

import copy
import random

import pytest
import torch

from openinterpretability.harness import sister_passes
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.sister_passes import Job, search
from openinterpretability.harness.sisters import find_sisters
from tests.conftest import FakeTokenizer
from tests.test_sisters import _same
from tests.test_wiring import qwen_runner  # noqa: F401  (fixture)

UNITS = [(1, 5, "positive"), (2, 11, "negative"), (0, 3, "positive")]


@pytest.fixture
def texts():
    rng = random.Random(0)
    return ["".join(rng.choice("abcdefghijklmno") for _ in range(30)) for _ in range(8)]


def _pair(qwen_runner):  # noqa: F811
    other = NeuronHookRunner(copy.deepcopy(qwen_runner.model), FakeTokenizer())
    runners = {"q": qwen_runner, "t": other}
    return runners, (lambda name, revision: runners[name])


def test_rotations_of_several_layers_in_one_pass_match_the_search_per_pair(
    qwen_runner,  # noqa: F811
    texts,
):
    runners, load = _pair(qwen_runner)
    job = Job(("q", None), ("t", None), UNITS, other_name="t")
    search([job], load, texts, k=3, batch_tokens=64, rotation_bytes=10**9)
    assert len({m[0].layer for m in job.matches}) > 1  # the rotations span several layers
    found, tokens = find_sisters(runners["q"], runners["t"], UNITS, texts, other_name="t", k=3)
    _same(job.found, found)
    assert job.tokens == tokens


def test_freeing_memory_empties_only_the_caches_of_present_devices(monkeypatch):
    emptied = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: emptied.append("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append("cuda"))
    for mps, cuda in ((False, False), (True, True)):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda m=mps: m)
        monkeypatch.setattr(torch.cuda, "is_available", lambda c=cuda: c)
        sister_passes._free()
    assert emptied == ["mps", "cuda"]
