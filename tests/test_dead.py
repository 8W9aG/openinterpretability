from datetime import UTC, datetime

import pytest
import torch

from openinterp.builder.stopped import explore_focus
from openinterp.builder.tree import build_tree
from openinterp.core.draws import UnauditableDraw, load_draws, write_draw
from openinterp.core.identifiers import NeuronIdentifier
from openinterp.core.schemas import AuditRecord, ExperimentProvenance
from openinterp.evidence import dead_unit
from openinterp.harness.dead import prove_dead
from openinterp.harness.hooks import NeuronHookRunner
from tests.conftest import FakeTokenizer

PROVENANCE = ExperimentProvenance(
    dtype="float32",
    corpus="corpus.jsonl",
    corpus_sha256="ab" * 32,
    n_snippets=1,
    tokens_scanned=100,
    selection="random",
    harness_version="test",
)


def neuron(layer, idx, pole="positive", model="openai-community/gpt2"):
    return NeuronIdentifier(model_name=model, layer=layer, neuron_idx=idx, pole=pole)


@pytest.fixture
def gpt2_runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2, FakeTokenizer())


@pytest.fixture
def qwen_runner():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return NeuronHookRunner(Qwen2ForCausalLM(config).eval(), FakeTokenizer())


def _pre_activations(tiny_gpt2, layer, idx, n=4000):
    """The neuron's pre-activation on random residual streams of many scales."""
    block = tiny_gpt2.transformer.h[layer]
    x = torch.randn(n, 32) * torch.logspace(-3, 3, n).unsqueeze(1)
    with torch.no_grad():
        return block.mlp.c_fc(block.ln_2(x))[:, idx]


def test_gpt2_bound_holds_on_every_input(gpt2_runner, tiny_gpt2):
    """No input reaches past the bound, and live poles are not proven dead."""
    h = _pre_activations(tiny_gpt2, 1, 5)
    pos, neg = (
        prove_dead(gpt2_runner, neuron(1, 5)),
        prove_dead(gpt2_runner, neuron(1, 5, "negative")),
    )
    assert h.max() <= pos.bound and -h.min() <= neg.bound
    assert not pos.proven and not neg.proven


def test_gpt2_bias_far_below_the_bound_proves_the_positive_pole_dead(gpt2_runner, tiny_gpt2):
    with torch.no_grad():
        tiny_gpt2.transformer.h[1].mlp.c_fc.bias[5] = -1000.0
    proof = prove_dead(gpt2_runner, neuron(1, 5))
    assert proof.proven and proof.bound < 0
    assert _pre_activations(tiny_gpt2, 1, 5).max() < 0
    assert not prove_dead(gpt2_runner, neuron(1, 5, "negative")).proven


def test_gpt2_proof_is_tight_at_the_bound(gpt2_runner, tiny_gpt2):
    """Shifting the bias 1% of R either side of the bound flips the proof."""
    pos = prove_dead(gpt2_runner, neuron(1, 5)).bound
    neg = prove_dead(gpt2_runner, neuron(1, 5, "negative")).bound
    c, r = (pos - neg) / 2, (pos + neg) / 2  # bounds are c + R and R − c
    bias = tiny_gpt2.transformer.h[1].mlp.c_fc.bias
    with torch.no_grad():
        bias[5] -= c + 0.99 * r  # now c + R = 0.01 R: some input fires it
    assert not prove_dead(gpt2_runner, neuron(1, 5)).proven
    with torch.no_grad():
        bias[5] -= 0.02 * r  # now c + R = −0.01 R
    assert prove_dead(gpt2_runner, neuron(1, 5)).proven


def test_relu_negative_pole_is_always_dead(gpt2_runner, tiny_gpt2):
    tiny_gpt2.config.activation_function = "relu"
    assert prove_dead(gpt2_runner, neuron(0, 2, "negative")).proven
    assert not prove_dead(gpt2_runner, neuron(0, 2)).proven


def test_unknown_activation_is_not_proven(gpt2_runner, tiny_gpt2):
    tiny_gpt2.config.activation_function = "tanh"
    with torch.no_grad():
        tiny_gpt2.transformer.h[1].mlp.c_fc.bias[5] = -1000.0
    assert not prove_dead(gpt2_runner, neuron(1, 5)).proven


def test_gated_mlp_is_proven_dead_only_by_a_zero_read(qwen_runner):
    mlp = qwen_runner.model.model.layers[1].mlp
    assert not prove_dead(qwen_runner, neuron(1, 7, model="q")).proven
    with torch.no_grad():
        mlp.gate_proj.weight[7] = 0.0
    for pole in ("positive", "negative"):
        assert prove_dead(qwen_runner, neuron(1, 7, pole, model="q")).proven


def test_dead_stop_records_the_proof(gpt2_runner, tiny_gpt2, record):
    stop = dead_unit(gpt2_runner, neuron(1, 5), PROVENANCE)
    assert not stop.draw.proven_dead and "don't prove it dead" in stop.message
    with torch.no_grad():
        tiny_gpt2.transformer.h[1].mlp.c_fc.bias[5] = -1000.0
    stop = dead_unit(gpt2_runner, neuron(1, 5), PROVENANCE)
    assert stop.draw.proven_dead and stop.draw.outcome == "dead" and "is dead" in stop.message


def test_proven_dead_poles_count_as_verified(tmp_path, record, monkeypatch):
    """The planner and the site count a proven-dead pole's neuron as verified."""
    from openinterp import planner_models
    from openinterp.planner import plan_next

    monkeypatch.setattr(planner_models, "model_dims", lambda model, download=True: (2, 10))
    (tmp_path / "pyproject.toml").write_text('[tool.openinterp]\nmodels = ["m/a", "m/b"]\n')
    wiki = tmp_path / "wiki"
    for i, proven in enumerate([True, True, False]):
        draw = UnauditableDraw(
            identifier=neuron(0, i, model="m/a"),
            outcome="dead",
            proven_dead=proven,
            provenance=PROVENANCE,
            timestamp=datetime(2026, 11, 2, 12, i, tzinfo=UTC),
        )
        write_draw(draw, wiki)
    plan = plan_next(wiki, seed=1, epsilon=1.0)
    assert plan.coverage["m/a"].verified == 2 and plan.coverage["m/b"].verified == 0

    ident = neuron(1, 3, model="m/a")
    sub = record.submission.model_copy(update={"identifier": ident, "model_mlp_width": 10})
    tree = build_tree([AuditRecord(submission=sub, verdict=record.verdict)])
    explore = explore_focus(tree, load_draws(wiki))
    assert record.verdict.status == "VERIFIED"
    assert explore["models"] == {"m/a": {"verified": 3, "dead": 2}}  # L1/N3, and 2 dead


def test_proven_dead_chosen_stops_count_as_verified(tmp_path, monkeypatch):
    """A chosen unit's proven-dead stop counts as verified, but stays out of the draws."""
    from openinterp import planner_models
    from openinterp.core.draws import ChosenStop, load_stops, stop_path
    from openinterp.planner import plan_next

    monkeypatch.setattr(planner_models, "model_dims", lambda model, download=True: (2, 10))
    (tmp_path / "pyproject.toml").write_text('[tool.openinterp]\nmodels = ["m/a"]\n')
    wiki = tmp_path / "wiki"
    manual = PROVENANCE.model_copy(update={"selection": "manual", "seed": None})
    for i, proven in enumerate([True, False]):
        stop = ChosenStop(
            identifier=neuron(0, i, model="m/a"),
            outcome="dead",
            proven_dead=proven,
            provenance=manual,
            timestamp=datetime(2026, 11, 2, 12, i, tzinfo=UTC),
        )
        path = stop_path(wiki, stop)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stop.model_dump_json() + "\n")

    assert load_draws(wiki) == []  # random-selection statistics don't read stops
    plan = plan_next(wiki, seed=1, epsilon=1.0)
    assert plan.coverage["m/a"].verified == 1  # the proven stop only
    assert [s.proven_dead for s in load_stops(wiki)] == [True, False]


@pytest.mark.parametrize(
    ("proven", "says"),
    [(True, "the weights prove that no input"), (False, "the weights don't prove it dead")],
)
def test_a_stop_page_on_a_dead_unit_with_no_records(tmp_path, proven, says):
    from openinterp.builder.wiki import write_stop_page
    from openinterp.checks import check_registry

    stop = UnauditableDraw(
        identifier=NeuronIdentifier(model_name="toy", layer=1, neuron_idx=7, pole="positive"),
        outcome="dead",
        proven_dead=proven,
        dead_bound=-1.0 if proven else None,
        provenance=PROVENANCE.model_copy(update={"selection": "manual", "seed": None}),
        timestamp=datetime(2026, 11, 5, tzinfo=UTC),
    )
    page = write_stop_page(stop, tmp_path)
    text = page.read_text()
    assert "scan stopped (dead)" in text and says in text
    assert "has no experiment record yet" in text and "Newest record" not in text
    # A second write in the same second keeps the first (write-once).
    page.write_text("kept")
    write_stop_page(stop, tmp_path)
    assert page.read_text() == "kept"
    assert check_registry(tmp_path).stops == 1


def test_the_wiki_home_links_a_stop_only_on_a_unit_with_no_index(tmp_path, record):
    from openinterp.builder.wiki import write_experiment_page, write_stop_page

    studied = record.model_copy(
        update={"submission": record.submission.model_copy(update={"identifier": neuron(1, 7)})}
    )
    write_experiment_page(studied, tmp_path)
    for unit in (neuron(1, 7), neuron(1, 7, "negative")):
        write_stop_page(
            UnauditableDraw(
                identifier=unit,
                outcome="default_on",
                active_share=0.6,
                provenance=PROVENANCE.model_copy(update={"selection": "manual", "seed": None}),
                timestamp=datetime(2026, 11, 5, tzinfo=UTC),
            ),
            tmp_path,
        )
    home = (tmp_path / "README.md").read_text()
    stem = "stops/openai-community--gpt2/L1/N7"
    assert f"{stem}-negative/20261105T000000Z.md" in home  # no experiment, so no index
    assert f"{stem}-positive/" not in home  # the unit's own index links this one
