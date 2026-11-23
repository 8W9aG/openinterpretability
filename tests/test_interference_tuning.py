"""A sequence of narrow fine-tunes, and the sequential-tuning script (#5125)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.interference import tuning
from tests.conftest import FakeTokenizer, ToyModel

PAIRS = [("q" * 3, "ab"), ("r" * 4, "cde"), ("s" * 2, "f")]


def _toy(seed: int = 0) -> ToyModel:
    torch.manual_seed(seed)
    return ToyModel().eval()


def test_pair_formats_each_task():
    sst2, xsum, gsm8k = tuning.TASKS
    assert tuning.pair(sst2, {"sentence": " dull ", "label": 0})[1] == "negative"
    prompt, target = tuning.pair(xsum, {"document": "x" * 5_000, "summary": " Short. "})
    assert prompt.count("x") == tuning.DOC_CHARS and target == "Short."
    prompt, target = tuning.pair(gsm8k, {"question": "1+1?", "answer": "2 #### 2"})
    assert prompt.startswith("### Instruction:\nSolve") and "1+1?" in prompt
    assert target == "2 #### 2"


def test_sample_keeps_every_row_when_asked_for_more():
    assert tuning.sample([3, 1, 2], 5, 0) == [3, 1, 2]
    got = tuning.sample(list(range(100)), 10, 0)
    assert len(got) == 10 and got == sorted(got)


def test_load_rows_reads_the_pinned_parquet(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"sentence": ["a"], "label": [1], "idx": [0]}), path)
    calls = []

    def download(repo, name, repo_type, revision):
        calls.append((repo, name, repo_type, revision))
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    rows = tuning.load_rows(tuning.TASKS[0], "heldout")
    assert rows == [{"sentence": "a", "label": 1}]  # only the task's columns
    assert calls == [("stanfordnlp/sst2", tuning.TASKS[0].heldout, "dataset",
                      tuning.TASKS[0].revision)]  # fmt: skip


def test_encode_cuts_the_prompt_first(monkeypatch):
    monkeypatch.setattr(tuning, "MAX_TOKENS", 6)
    ids, start = tuning.encode(FakeTokenizer(), "abcdef", "gh")
    assert len(ids) == 6 and start == 3  # "def" + "gh" + the end of sequence
    ids, start = tuning.encode(FakeTokenizer(), "ab", "cdefghij")
    assert len(ids) == 6 and start == 1  # the response is cut too, and one prompt token stays


def test_collate_marks_only_response_tokens():
    ids, attention, mask = tuning.collate([([1, 2, 3], 1), ([4, 5], 2)])
    assert ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert attention.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert mask.tolist() == [[False, True, True], [False, False, False]]


def test_response_loss_matches_a_direct_cross_entropy():
    model = _toy()
    ids, attention, mask = tuning.collate([tuning.encode(FakeTokenizer(), "abc", "de")])
    with torch.no_grad():
        loss, n = tuning.response_loss(model, ids, attention, mask, "cpu")
        logits = model(input_ids=ids).logits[0, :-1]
        want = torch.nn.functional.cross_entropy(logits[2:], ids[0, 3:], reduction="sum")
    assert n == 3 and float(loss) == pytest.approx(float(want), rel=1e-5)


def test_train_stage_lowers_the_loss_and_freezes_the_model():
    model = _toy()
    pairs = [("ab", "cd")] * 50
    before = tuning.heldout_loss(model, FakeTokenizer(), pairs[:2], "cpu")
    lines = []
    losses = tuning.train_stage(model, FakeTokenizer(), pairs, device="cpu", epochs=1, lr=1e-2,
                                batch=1, accumulate=1, seed=0, progress=lines.append)  # fmt: skip
    assert len(losses) == 50 and lines == [f"epoch 1, step 50: loss {losses[-1]:.3f}"]
    assert tuning.heldout_loss(model, FakeTokenizer(), pairs[:2], "cpu") < before
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_train_stage_uses_gradient_checkpointing_when_the_model_has_it(tiny_gpt2):
    losses = tuning.train_stage(tiny_gpt2, FakeTokenizer(), PAIRS, device="cpu", epochs=2,
                                lr=1e-3, batch=1, accumulate=2, seed=0)  # fmt: skip
    assert len(losses) == 2  # one step of two pairs in each epoch; the odd pair is left out
    assert not tiny_gpt2.is_gradient_checkpointing and tiny_gpt2.config.use_cache is False


def test_stage_kl_is_zero_for_one_model_and_positive_for_two():
    a, b = _toy(0), _toy(1)
    assert tuning.stage_kl(a, a, FakeTokenizer(), PAIRS, "cpu") == pytest.approx(0, abs=1e-6)
    assert tuning.stage_kl(a, b, FakeTokenizer(), PAIRS, "cpu") > 0


def test_arc_accuracy_scores_each_row_once():
    model = _toy()
    row = {"question": "q", "choices": {"text": ["ab", "cdef"], "label": ["A", "B"]}}
    rows = [{**row, "answerKey": "A"}, {**row, "answerKey": "B"}]
    correct, items = tuning.arc_accuracy(model, FakeTokenizer(), rows, "cpu")
    assert items == 2 and correct == 1  # one choice wins, so exactly one key matches


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sequential_tuning.py"
    spec = importlib.util.spec_from_file_location("sequential_tuning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(arm, stage, slug, keeps, r=0.9):
    return {"arm": arm, "stage": stage, "slug": slug, "keeps": keeps, "pearson": r}


def test_stage_summary_counts_reproduced_poles_only():
    script = _script()
    rows = [_row("base", 1, "a", True, 0.5), _row("base", 1, "b", False, 0.9),
            _row("base", 1, "c", True, None)]  # fmt: skip
    got = script.stage_summary(rows, {"a", "b"})
    assert got["poles"] == 2 and got["keeps"] == 1 and got["keep_share"] == 0.5
    assert got["median_pearson"] == 0.9
    assert script.stage_summary(rows, set())["keep_share"] is None


def test_first_failures_finds_each_poles_first_failing_stage():
    script = _script()
    rows = [_row("base", s, "a", k) for s, k in enumerate([True, True, False, True])]
    rows += [_row("base", s, "b", True) for s in range(4)]
    rows += [_row("instruct", s, "a", k) for s, k in enumerate([False, False, False, False])]
    rows += [_row("base", s, "z", False) for s in range(4)]  # not reproduced: left out
    base, instruct = script.first_failures(rows, {"a", "b"}, 3)
    assert base == {"table": "first_fail", "arm": "base", "counts": [0, 0, 1, 0], "never": 1,
                    "back": 1}  # fmt: skip
    assert instruct["counts"] == [1, 0, 0, 0] and instruct["back"] == 0


def test_task_data_samples_each_task(monkeypatch):
    script = _script()
    rows = {"sst2": {"sentence": "s", "label": 1}, "xsum": {"document": "d", "summary": "t"},
            "gsm8k": {"question": "q", "answer": "a"}}  # fmt: skip
    monkeypatch.setattr(script, "load_rows", lambda task, split: [rows[task.name]] * 5)
    data = script.task_data(3)
    assert len(data["sst2"]["train"]) == 3 and len(data["gsm8k"]["heldout"]) == 5
    assert data["sst2"]["train"][0][1] == "positive"


def _records(*pairs):
    ident = lambda n, pole: SimpleNamespace(layer=1, neuron_idx=n, pole=pole,  # noqa: E731
                                            slug=f"M/L1/N{n}-{pole}")  # fmt: skip
    return [SimpleNamespace(submission=SimpleNamespace(identifier=ident(n, p))) for n, p in pairs]


def test_pole_rows_scale_by_each_poles_ratio(monkeypatch):
    script = _script()
    scales = []

    def read(runner, record, scale):
        scales.append(scale)
        return {"specificity": 1.0, "sensitivity": 0.9, "passes": True}

    monkeypatch.setattr(script, "readings", read)
    records = _records((2, "positive"), (0, "negative"))
    layers = script.by_layer(records)
    base_cols = {1: ([0, 2], torch.tensor([[1.0, 2.0], [-2.0, 0.0]]))}
    now = {1: ([0, 2], torch.tensor([[1.0, 1.0], [-1.0, 0.0]]))}
    assert list(layers) == [1]
    rows = script.pole_rows(None, records * 25, base_cols, now, "base", 1, 0.0)
    assert scales[:2] == [0.5, 0.5] and rows[0]["keeps"] and rows[0]["stage"] == 1
    flat = {1: ([0, 2], torch.zeros(2, 2))}
    script.pole_rows(None, records[:1], flat, flat, "base", 1, 0.0)
    assert scales[-1] == 1.0  # a pole that never fires keeps its scale


def test_tune_and_measure_stage_on_a_toy_model(monkeypatch):
    script = _script()
    monkeypatch.setattr(script, "EPOCHS", 1)
    monkeypatch.setattr(script, "LR", 1e-2)  # one step must move the toy model
    runner = NeuronHookRunner(_toy(), FakeTokenizer())
    pairs = {"train": [("ab", "cd")] * 8, "heldout": PAIRS}
    got = script.tune(runner, pairs, "base stage 1", 1, "cpu", 0.0)
    assert got["steps"] == 1 and got["kl"] > 0
    arc = [{"question": "q", "choices": {"text": ["a", "b"], "label": ["A", "B"]},
            "answerKey": "A"}]  # fmt: skip
    measured = script.measure_stage(runner, {"t": pairs}, arc, "cpu")
    assert measured["arc_items"] == 1 and set(measured["task_loss"]) == {"t"}
    cols = script.columns(runner, [torch.tensor([[1, 2, 3]])], {1: [SimpleNamespace(neuron_idx=2)]})
    assert cols[1][0] == [2] and cols[1][1].shape == (2, 1)
