"""Wiring rows for archived records (wiring_backfill.py)."""

import torch
from torch import nn

from openinterpretability import wiring_backfill
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.wiring_backfill import WiringRow, backfill, load_rows, sort_rows, wire
from tests.conftest import D_MLP, D_MODEL, FakeTokenizer, ToyModel
from tests.test_circuits import at


class _OddMLP(nn.Module):
    """An MLP whose input projection has a name the wiring does not know."""

    def __init__(self) -> None:
        super().__init__()
        self.w_in = nn.Linear(D_MODEL, D_MLP)
        self.act = nn.SiLU()
        self.down_proj = nn.Linear(D_MLP, D_MODEL)

    def forward(self, x):
        return self.down_proj(self.act(self.w_in(x)))


def test_a_model_the_wiring_cannot_read_gets_an_empty_row(record):
    torch.manual_seed(0)
    model = ToyModel().eval()
    for block in model.model.layers:
        block.mlp = _OddMLP()
    rec = at(record, 1, 5)
    row = wire(NeuronHookRunner(model, FakeTokenizer()), rec)
    assert (row.weights, row.activations) == (None, None)
    assert (row.slug, row.experiment_id) == wiring_backfill.row_key(rec)


def test_a_limited_backfill_loads_one_model_for_records_that_share_it(tmp_path, record,
                                                                       tiny_gpt2):  # fmt: skip
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    records = [at(record, 1, n) for n in (5, 6, 7)]
    out = tmp_path / "wiring.jsonl"
    loads, echoed = [], []
    wrote = backfill(records, lambda r: loads.append(r) or runner, out, limit=2,
                     echo=echoed.append)  # fmt: skip
    assert wrote == 2 and len(loads) == 1  # one model for both records
    assert [line.split()[0] for line in echoed] == ["[1/2]", "[2/2]"]
    assert len(load_rows(out)) == 2


def test_sorting_rows_gives_a_stable_order(tmp_path):
    out = tmp_path / "wiring.jsonl"
    rows = [WiringRow(slug=s, experiment_id=e) for s, e in (("b", "1"), ("a", "2"), ("a", "1"))]
    out.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    sort_rows(out)
    keys = [(r.slug, r.experiment_id) for r in map(WiringRow.model_validate_json,
                                                   out.read_text().splitlines())]  # fmt: skip
    assert keys == [("a", "1"), ("a", "2"), ("b", "1")]


def test_releasing_memory_empties_only_the_caches_of_present_devices(monkeypatch):
    emptied = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: emptied.append("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append("cuda"))
    for mps, cuda in ((False, False), (True, True)):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda m=mps: m)
        monkeypatch.setattr(torch.cuda, "is_available", lambda c=cuda: c)
        wiring_backfill._release_cached_memory()
    assert emptied == ["mps", "cuda"]
