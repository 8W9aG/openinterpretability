"""The helpers behind scripts/rest_by_position.py (#7202)."""

import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from rest_by_position import (  # noqa: E402
    bucket_names,
    bucket_stats,
    energy_shares,
    long_windows,
    read_rests,
)
from resting_conditions import CONDITIONS  # noqa: E402
from train_resting import build_model  # noqa: E402


def test_read_rests_gives_each_layers_rest_by_bucket_of_its_c_fc_output():
    condition = replace(CONDITIONS["pythia_size"], n_layer=2, n_embd=16, window=32)
    model = build_model(condition, 50, 0).eval()
    windows = torch.randint(50, (5, 32), generator=torch.Generator().manual_seed(0))
    seen = []
    handle = model.transformer.h[1].mlp.c_fc.register_forward_hook(
        lambda _m, args, out: seen.append((args[0], out))
    )
    with torch.no_grad():
        model(windows)
    handle.remove()
    rests = read_rests(model, [b.mlp.c_fc for b in model.transformer.h], windows)
    assert len(rests) == 2
    x, h = seen[0]
    assert rests[1] == {**bucket_stats(x, h), **energy_shares(x[:, 1:])}
    assert list(rests[1]["rest"]) == bucket_names(32) == ["0-0", "1-15", "16-31"]


def test_long_windows_keep_each_documents_tokens_in_order():
    ids = torch.arange(12).view(6, 2)
    docs = torch.tensor([0, 0, 0, 1, 1, 2])
    out = long_windows(ids, docs, -1, 5)
    assert out.flatten().tolist() == [0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, -1, 10, 11, -1]
