import json

import pytest
import torch
from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.core.probing import ProbeComparison
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.probing import (
    Window,
    auroc,
    balanced_accuracy,
    choose_tokens,
    collect,
    fit_threshold,
    label_tokens,
    mass_mean,
    sparse_curve,
    windows,
)
from tests.conftest import N_LAYERS, FakeTokenizer


class OffsetTokenizer(FakeTokenizer):
    """The toy character tokenizer, with the offsets a fast tokenizer returns."""

    pad_token_id = 0

    def __call__(self, texts, return_tensors=None, padding=True, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            return {
                "input_ids": [self._id(c) for c in texts],
                "offset_mapping": [(i, i + 1) for i in range(len(texts))],
            }
        return super().__call__(texts, return_tensors, padding)


def test_tokens_overlapping_a_span_are_positive():
    ids, labels = label_tokens(OffsetTokenizer(), "abcdef", [(1, 3), (5, 6)])
    assert len(ids) == 6
    assert labels == [False, True, True, False, False, True]


def test_windows_drop_lone_tokens_and_choose_skips_first_positions():
    rows = [("abcde", [(0, 5)]), ("xy", [])]
    wins = windows(OffsetTokenizer(), rows, window_tokens=2)
    assert [(len(w.ids), w.document) for w in wins] == [(2, 0), (2, 0), (2, 1)]  # 'e' dropped
    keep = choose_tokens(wins, max_positive=10, max_negative=10, seed=0)
    assert keep == [[1], [1], [1]]
    assert sum(map(len, choose_tokens(wins, max_positive=1, max_negative=0, seed=0))) == 1


def test_auroc_counts_ties_half_and_threshold_maximises_balanced_accuracy():
    labels = torch.tensor([True, True, False, False])
    assert auroc(torch.tensor([0.0, 0.0, 0.0, 0.0]), labels) == pytest.approx(0.5)
    assert auroc(torch.tensor([2.0, 1.0, 1.0, 0.0]), labels) == pytest.approx(0.875)
    scores = torch.tensor([3.0, 2.0, 1.0, 0.0])
    assert fit_threshold(scores, labels) == 2.0
    assert balanced_accuracy(scores >= 2.0, labels) == 1.0


def planted(n=400, d=20, column=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.rand(n, generator=g) < 0.3
    x = torch.randn(n, d, generator=g)
    x[:, column] -= 4.0 * y  # the concept lives in one coordinate, negatively
    return x, y


def test_a_planted_neuron_ranks_first_and_one_neuron_suffices():
    (x, y), (xt, yt) = planted(), planted(seed=1)
    points, rank = sparse_curve(x, y, xt, yt, neuron=7, l2=1e-3, ks=(1, 2, 64))
    assert rank == 1
    assert [p.k for p in points] == [1, 2]  # k above the width is skipped
    assert points[0].includes_neuron and points[0].score.balanced_accuracy > 0.95
    d = mass_mean(x, y)
    assert d[7] < 0 and d[7].abs() / d.norm() > 0.9


def test_collect_reads_hidden_and_residual_at_kept_positions(toy_model):
    runner = NeuronHookRunner(toy_model, OffsetTokenizer())
    layer = 1
    wins = [Window([1, 2, 3, 4], [False, True, False, True], 0), Window([5, 6], [False, True], 1)]
    got = collect(runner, wins, [[1, 3], [1]], layer)
    _, cache = runner.run_with_cache(torch.tensor([[1, 2, 3, 4]]), [layer])
    assert torch.allclose(got.hidden[:2], cache[layer][0, [1, 3]])
    with torch.no_grad():
        x = toy_model.embed(torch.tensor([[5, 6]]))
        for block in toy_model.model.layers[: layer + 1]:
            x = block(x)
    assert torch.allclose(got.residual[2], x[0, 1])
    assert got.labels.tolist() == [True, True, True]


def test_probe_compare_writes_a_report(tmp_path, monkeypatch, toy_model, submission):
    from openinterp.core.schemas import AuditRecord, NeuronIdentifier

    runner = NeuronHookRunner(toy_model, OffsetTokenizer())
    monkeypatch.setattr("openinterp.cli._load_runner", lambda *_a, **_k: runner)
    ident = NeuronIdentifier(model_name="toy", layer=N_LAYERS - 1, neuron_idx=3, pole="positive")
    record = tmp_path / "record.json"
    sub = submission.model_copy(update={"identifier": ident})
    record.write_text(AuditRecord(submission=sub).model_dump_json())
    texts = ["abcabcxyz" * 4, "zyxcba" * 5, "aaabbbccc" * 3, "xxyyzz" * 6] * 3
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"text": t, "spans": [[i, i + 1] for i, c in enumerate(t) if c == "a"]})
            + "\n"
            for t in texts
        )
    )
    out = tmp_path / "report.json"
    args = ["probe-compare", "--record", str(record), "--train", str(labels)]
    args += ["--concept", "the letter a", "--window-tokens", "8", "--out", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    report = ProbeComparison.model_validate_json(out.read_text())
    assert report.split == "document" and report.slug == ident.slug
    assert [s.site for s in report.sites] == ["mlp_hidden", "residual"]
    assert report.neuron.line == pytest.approx(0.25 * sub.max_activation)
    # "a" is one token id, so the hidden layer separates it perfectly
    assert report.sites[0].logistic.balanced_accuracy == pytest.approx(1.0)
