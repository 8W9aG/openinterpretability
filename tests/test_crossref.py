"""Dictionary latents aligned with neuron poles (`openinterp crossref`), on a random GPT-2
and Qwen2 with planted latents stored locally (no download)."""

import pytest
import torch
from safetensors.torch import save_file

from openinterp.builder.compiler import compile_site
from openinterp.core.schemas import NeuronIdentifier
from openinterp.crossref import (
    AlignmentRow,
    crossref,
    describe,
    format_for_reviewer,
    load_rows,
    neuronpedia_url,
)
from openinterp.harness import dictionaries
from openinterp.harness.crossref import align_layer
from openinterp.harness.dictionaries import DictionarySpec
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.wiring import _matrix
from tests.conftest import FakeTokenizer
from tests.test_wiring import qwen_runner  # noqa: F401  (fixture)

N_FEATURES, PLANTED, NEURON = 64, 9, 7


def _save(tmp_path, name, w_enc, w_dec, layers=(1,)):
    for layer in layers:
        (tmp_path / name / str(layer)).mkdir(parents=True)
        weights = {
            "W_enc": w_enc,
            "b_enc": torch.zeros(w_enc.shape[1]),
            "W_dec": w_dec,
            "b_dec": torch.zeros(w_dec.shape[1]),
        }
        save_file(weights, tmp_path / name / str(layer) / "tc.safetensors")


def _spec(tmp_path, monkeypatch, name, model, site, w_enc, w_dec, **kw):
    _save(tmp_path, name, w_enc.contiguous(), w_dec.contiguous())
    spec = DictionarySpec(
        model_name=model,
        kind="transcoder",
        repo=str(tmp_path / name),
        revision="local",
        path="{layer}/tc.safetensors",
        layers=(1,),
        input_site=site,
        **kw,
    )
    monkeypatch.setitem(dictionaries.DICTIONARIES, name, spec)
    dictionaries._load.cache_clear()
    return spec


@pytest.fixture
def gpt2_runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


@pytest.fixture
def planted(tmp_path, monkeypatch, gpt2_runner):
    """A transcoder on GPT-2's normalised MLP input whose latent PLANTED writes neuron
    NEURON's w_out and reads what it reads."""
    torch.manual_seed(3)
    block = gpt2_runner.layers[1]
    w_enc, w_dec = torch.randn(32, N_FEATURES), torch.randn(N_FEATURES, 32)
    w_dec[PLANTED] = _matrix(block.mlp.c_proj)[:, NEURON] * 2
    # The neuron reads the gained input; the latent the input without the gain.
    w_enc[:, PLANTED] = _matrix(block.mlp.c_fc)[NEURON] * block.ln_2.weight.detach()
    return _spec(
        tmp_path,
        monkeypatch,
        "tiny-tc",
        "tiny-gpt2",
        "mlp_in_normalized",
        w_enc,
        w_dec,
        neuronpedia="tiny/{layer}-tc",
    )


def test_a_planted_latent_is_the_positive_poles_best_match_on_both_sides(gpt2_runner, planted):
    found = align_layer(gpt2_runner, planted, 1, [(NEURON, "positive"), (NEURON, "negative")])
    write, read = found[(NEURON, "positive")]
    assert write.matches[0].feature_idx == PLANTED
    assert write.matches[0].cosine == pytest.approx(1.0, abs=1e-5)
    assert read.matches[0].feature_idx == PLANTED and read.matches[0].channel == "in"
    assert read.matches[0].cosine == pytest.approx(1.0, abs=1e-5)
    assert write.percentile == read.percentile == pytest.approx(1 - 1 / 128)
    assert 0 < write.chance < 1
    # A latent fires non-negatively, so it writes the positive pole's direction only.
    write, read = found[(NEURON, "negative")]
    assert PLANTED not in [m.feature_idx for m in write.matches + read.matches]
    assert all(m.cosine < 0.99 for m in write.matches)


def test_a_gated_neurons_negative_pole_reads_with_an_open_gate_and_negative_up(
    tmp_path,
    monkeypatch,
    qwen_runner,  # noqa: F811
):
    torch.manual_seed(4)
    mlp = qwen_runner.layers[1].mlp
    w_enc, w_dec = torch.randn(32, N_FEATURES), torch.randn(N_FEATURES, 32)
    w_enc[:, PLANTED] = -_matrix(mlp.up_proj)[NEURON]
    w_enc[:, PLANTED + 1] = _matrix(mlp.gate_proj)[NEURON]
    spec = _spec(tmp_path, monkeypatch, "tiny-qwen-tc", "tiny-qwen", "mlp_in", w_enc, w_dec)
    found = align_layer(qwen_runner, spec, 1, [(NEURON, "positive"), (NEURON, "negative")])
    negative = {m.feature_idx: m for m in found[(NEURON, "negative")][1].matches}
    positive = {m.feature_idx: m for m in found[(NEURON, "positive")][1].matches}
    assert negative[PLANTED].channel == "up" and negative[PLANTED].cosine > 0.99
    assert negative[PLANTED + 1].channel == "gate" and negative[PLANTED + 1].cosine > 0.99
    assert PLANTED not in positive and positive[PLANTED + 1].channel == "gate"


def test_crossref_writes_one_row_per_pole_and_resumes(
    tmp_path, gpt2_runner, planted, record, monkeypatch
):
    def pole(idx, sign):
        ident = NeuronIdentifier(model_name="tiny-gpt2", layer=1, neuron_idx=idx, pole=sign)
        return record.model_copy(
            update={"submission": record.submission.model_copy(update={"identifier": ident})}
        )

    records = [pole(NEURON, "positive"), pole(NEURON, "positive"), pole(3, "negative")]
    other = NeuronIdentifier(model_name="tiny-gpt2", layer=0, neuron_idx=1, pole="positive")
    records.append(
        record.model_copy(
            update={"submission": record.submission.model_copy(update={"identifier": other})}
        )
    )  # layer 0: no dictionary covers it
    out = tmp_path / "alignment.jsonl"
    loads = []

    def load(model_name, revision):
        loads.append((model_name, revision))
        return gpt2_runner

    assert crossref(records, load, out, k=3, echo=lambda _: None) == 2
    rows = load_rows(out)
    assert set(rows) == {"tiny-gpt2/L1/N7-positive", "tiny-gpt2/L1/N3-negative"}
    row = rows["tiny-gpt2/L1/N7-positive"][0]
    assert row.dictionary == "tiny-tc" and row.path == "1/tc.safetensors"
    assert row.writes.matches[0].feature_idx == PLANTED and len(row.writes.matches) == 3
    assert loads == [("tiny-gpt2", None)]  # no record pins a revision
    assert crossref(records, load, out, echo=lambda _: None) == 0  # already done

    # The neuron page links each latent to Neuronpedia, where its preset says it is.
    assert neuronpedia_url(row, PLANTED) == f"https://www.neuronpedia.org/tiny/1-tc/{PLANTED}"
    site = tmp_path / "site"
    compile_site(records, site, alignment_rows=rows)
    page = (site / "models/tiny-gpt2/L1/N7.html").read_text()
    assert "Dictionary latents" in page and f"tiny/1-tc/{PLANTED}" in page


def test_the_reviewer_sees_only_above_chance_latents_with_a_description(planted):
    row = AlignmentRow.model_validate(
        {
            "slug": "tiny-gpt2/L1/N7-positive",
            "dictionary": "tiny-tc",
            "repo": "r",
            "revision": "v",
            "path": "1/tc.safetensors",
            "model_revision": None,
            "writes": {
                "matches": [
                    {"feature_idx": PLANTED, "cosine": 0.9},
                    {"feature_idx": 2, "cosine": 0.5},
                    {"feature_idx": 3, "cosine": 0.1},
                ],
                "chance": 0.2,
                "percentile": 0.99,
            },
        }
    )
    registry = {f"tiny-gpt2/L1/tiny-tc/F{PLANTED}": "closing quotes"}
    fetched = []

    def fetch(_row, idx):
        fetched.append(idx)
        return None

    text = format_for_reviewer(describe([row], registry, fetch=fetch))
    assert fetched == [2]  # the registry answers for PLANTED; F3 is below chance
    assert f"tiny-tc latent {PLANTED} writes" in text and "closing quotes" in text
    assert "latent 2" not in text  # no description anywhere
    assert format_for_reviewer(describe([row], {}, fetch=lambda *_: None)) == ""


def test_chance_is_not_the_initial_weights_of_dead_latents(tmp_path, monkeypatch, gpt2_runner):
    # Dead latents keep their initialisation; small seeds would reproduce it as "random".
    init = [torch.randn(256, 32, generator=torch.Generator().manual_seed(s)) for s in range(4)]
    w_dec = torch.cat(init)
    spec = _spec(
        tmp_path, monkeypatch, "init-tc", "tiny-gpt2", "mlp_in", torch.randn(32, 1024), w_dec
    )
    write, read = align_layer(gpt2_runner, spec, 1, [(0, "positive")])[(0, "positive")]
    assert write.chance < 0.95 and read.chance < 0.95
