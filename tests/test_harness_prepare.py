"""Misaligned weights are copied into aligned memory (harness/prepare.py)."""

import torch
from torch import nn

from openinterpretability.harness.prepare import align_parameters


def test_a_misaligned_parameter_is_copied_and_an_aligned_one_is_kept():
    storage = torch.arange(17, dtype=torch.float32)
    model = nn.Module()
    model.odd = nn.Parameter(storage[1:], requires_grad=False)  # 4 bytes past the start
    model.even = nn.Parameter(torch.ones(4), requires_grad=False)
    assert model.odd.data_ptr() % 64 and not model.even.data_ptr() % 64
    even = model.even.data_ptr()
    align_parameters(model)
    assert model.odd.data_ptr() % 64 == 0
    assert torch.equal(model.odd, torch.arange(1, 17, dtype=torch.float32))
    assert model.even.data_ptr() == even
