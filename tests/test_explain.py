"""`open-interpretability explain` (explain.py), on the toy model and tiny GPT-2.

The issues are #3250 (neuron poles) and #3617 (attention heads and paths).
"""

import json

import pytest
import torch
from click.testing import CliRunner

from openinterpretability import explain as ex
from openinterpretability.core.identifiers import NeuronIdentifier

TEXTS = ["a b c d e f g h", "h g f e d c b a", "a c e g b d f h"] * 20
PROMPT = "a b c d"


@pytest.fixture
def means(toy_runner, tmp_path, monkeypatch):
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    return ex.unit_means(toy_runner, "toy", TEXTS, 200)


def test_means_are_cached_and_leave_out_each_windows_first_token(toy_runner, means, tmp_path):
    assert sorted(means.neurons) == list(range(toy_runner.n_layers)) and means.heads == {}
    assert len(list((tmp_path / "means").glob("*.pt"))) == 1
    ids = [i for t in TEXTS for i in toy_runner.tokenizer.encode(t)][:200]
    windows = torch.tensor([ids[i : i + 4] for i in range(0, len(ids) - 3, 4)])
    _, cache = toy_runner.run_with_cache(windows, [0])
    assert torch.allclose(means.neurons[0], cache[0][:, 1:].reshape(-1, cache[0].shape[-1]).mean(0))


def test_each_contribution_goes_to_the_pole_its_sign_names(toy_runner, means):
    token = int(ex.logits_with(toy_runner, PROMPT, {}, -1).argmax())
    scores, _ = ex.attribute(toy_runner, PROMPT, means, token, -1)
    _, cache = toy_runner.run_with_cache(PROMPT, [0])
    act = cache[0][0, 1:]  # the first token is left out
    for (layer, neuron, pole), _ in scores.items():
        if layer == 0:
            values = act[:, neuron]
            assert (values > 0).any() if pole == "positive" else (values < 0).any()


def test_ablating_a_last_layer_pole_moves_the_logit_by_minus_its_attribution(toy_runner, means):
    # The toy model has no attention, and its last layer reaches the logits linearly, so
    # first-order attribution is exact there.
    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=1000, confirm=1000)
    last = [p for p in result.poles if p.layer == toy_runner.n_layers - 1 and p.ablation_change]
    assert last
    for p in last:
        assert p.ablation_change == pytest.approx(-p.attribution, abs=1e-4)


def test_shares_and_labels(toy_runner, means):
    first = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=3)
    assert first.verified_share == 0.0 and first.joint == {"poles": 0}
    assert all(p.status == "unstudied" and p.explanation is None for p in first.poles)
    top = first.poles[0]
    slug = NeuronIdentifier(
        model_name="toy", layer=top.layer, neuron_idx=top.neuron, pole=top.pole
    ).slug
    labels = {slug: ("VERIFIED", "fires on b", 2.0)}
    result = ex.explain(toy_runner, "toy", PROMPT, means, labels, top=3)
    shown = result.poles[0]
    assert shown.status == "VERIFIED" and shown.explanation == "fires on b"
    assert result.verified_share == pytest.approx(abs(shown.attribution) / result.total)
    assert result.joint["poles"] == 1 and "logit_change" in result.joint
    rejected = ex.explain(toy_runner, "toy", PROMPT, means, {slug: ("REJECTED", "x", 2.0)}, top=3)
    assert rejected.poles[0].explanation is None  # only a VERIFIED explanation is shown


def test_cli(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    base = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200"]
    args = [*base, "--wiki-dir", str(tmp_path / "wiki"), "--top", "3"]
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    assert "hold 0.0% of the total" in out.output and "nothing here is explained" in out.output
    as_json = CliRunner().invoke(cli, [*args, "--json"])
    assert json.loads(as_json.output)["verified_share"] == 0.0


@pytest.fixture
def gpt2(tiny_gpt2):
    """Tiny GPT-2 in float64, so a small step's logit change is above rounding noise."""
    from openinterpretability.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    return NeuronHookRunner(tiny_gpt2.double(), FakeTokenizer())


@pytest.fixture
def gpt2_means(gpt2, tmp_path, monkeypatch):
    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    return ex.unit_means(gpt2, "tiny", TEXTS, 200)


def _shift(gpt2, unit, means, alpha):
    """Hooks that move one unit a share ``alpha`` of the way to its mean, past the first token."""
    layer, index, pole = unit
    if pole == ex.HEAD:
        cols = slice(index * gpt2.head_dim(layer), (index + 1) * gpt2.head_dim(layer))

        def head(z):
            z = z.clone()
            z[..., 1:, cols] -= alpha * (z[..., 1:, cols] - means.heads[layer][cols])
            return z

        return {}, {layer: head}

    def neuron(act):
        act = act.clone()
        a = act[..., 1:, index]
        active = a > 0 if pole == "positive" else a < 0
        act[..., 1:, index] = torch.where(active, a - alpha * (a - means.neurons[layer][index]), a)
        return act

    return {layer: neuron}, {}


def test_heads_have_means_and_an_attribution_that_matches_a_small_step(gpt2, gpt2_means):
    assert sorted(gpt2_means.heads) == [0, 1]
    assert gpt2_means.heads[0].shape == (gpt2.n_heads * gpt2.head_dim(0),)
    token = int(ex.logits_with(gpt2, PROMPT, {}, -1).argmax())
    scores, logit = ex.attribute(gpt2, PROMPT, gpt2_means, token, -1)
    heads = [u for u in scores if u[2] == ex.HEAD]
    assert sorted(heads) == [(0, 0, ex.HEAD), (0, 1, ex.HEAD), (1, 0, ex.HEAD), (1, 1, ex.HEAD)]
    alpha = 1e-4
    for unit in heads:
        neurons, hooked = _shift(gpt2, unit, gpt2_means, alpha)
        with torch.no_grad(), gpt2.hooks(neurons, None, hooked):
            moved = gpt2.forward(gpt2.encode(PROMPT))[0, -1, token]  # float64, unrounded
        assert float(moved - logit) / -alpha == pytest.approx(scores[unit][0], rel=0.01, abs=1e-7)


def test_an_edge_is_the_change_in_its_targets_contribution(gpt2, gpt2_means):
    token = int(ex.logits_with(gpt2, PROMPT, {}, -1).argmax())
    graph = ex.Graph(gpt2, PROMPT, gpt2_means, token, -1)
    scores = graph.scores(graph.grads)

    def strongest(units):
        return max(units, key=lambda u: abs(scores[u][0]))

    target = strongest(u for u in scores if u[0] == 1 and u[2] != ex.HEAD)
    source = strongest(u for u in scores if u[0] == 0 and u[2] == ex.HEAD)
    (edge,) = [e for e in ex.paths(graph, [target, source], 1) if e["target"] != e["source"]]
    alpha = 1e-4
    neurons, heads = _shift(gpt2, source, gpt2_means, alpha)
    with gpt2.hooks(neurons, None, heads):
        moved = ex.Graph(gpt2, PROMPT, gpt2_means, token, -1)
    moved.grads = graph.grads  # the target's contribution, with ∂logit/∂target held fixed
    change = float((moved.contribution(target) - graph.contribution(target)).detach())
    assert change / -alpha == pytest.approx(edge["weight"], rel=0.01, abs=1e-7)


def test_explain_lists_heads_with_their_labels_and_traces_paths(gpt2, gpt2_means):
    from openinterpretability.core.identifiers import HeadIdentifier

    first = ex.explain(gpt2, "tiny", PROMPT, gpt2_means, {}, top=50, edges=5)
    head = next(u for u in first.poles if u.pole == ex.HEAD)
    assert head.label == f"L{head.layer}/H{head.neuron}"
    assert first.edges and all({"source", "target", "weight"} <= set(e) for e in first.edges)
    slug = HeadIdentifier(model_name="tiny", layer=head.layer, head_idx=head.neuron).slug
    labels = {slug: ("VERIFIED", "copies b", 1.0)}
    labelled = ex.explain(gpt2, "tiny", PROMPT, gpt2_means, labels, top=50)
    shown = next(u for u in labelled.poles if u.key == head.key)
    assert shown.status == "VERIFIED" and shown.explanation == "copies b"
    assert labelled.joint["poles"] == 1 and "logit_change" in labelled.joint


def test_a_superposition_attribution_matches_a_small_step_along_its_direction(gpt2, gpt2_means):
    from openinterpretability.explain_groups import GroupDef, superposition_score

    group = GroupDef(
        "superposition", "L1/Sx", "m/L1/Sx", ((1, 3, 0.8), (1, 40, -0.5), (1, 77, 0.3))
    )
    token = int(ex.logits_with(gpt2, PROMPT, {}, -1).argmax())
    graph = ex.Graph(gpt2, PROMPT, gpt2_means, token, -1)
    expected = superposition_score(graph, group)
    index = torch.tensor([3, 40, 77])
    w = torch.tensor([0.8, -0.5, 0.3], dtype=torch.float64)
    u, alpha = w / w.norm(), 1e-4

    def step(act):
        act = act.clone()
        part = act[..., 1:, :][..., index]
        along = ((part - gpt2_means.neurons[1][index].double()) @ u) * (part @ w > 0)
        act[..., 1:, index] = part - alpha * along[..., None] * u
        return act

    with torch.no_grad(), gpt2.hooks({1: step}):
        moved = gpt2.forward(gpt2.encode(PROMPT))[0, -1, token]
    change = float((moved - graph.logit).detach())
    assert change / -alpha == pytest.approx(expected, rel=0.01, abs=1e-7)


def test_circuits_sum_their_members_and_a_verified_one_explains_them(gpt2, gpt2_means, tmp_path):
    from openinterpretability import explain_groups

    folder = tmp_path / "circuits"
    folder.mkdir()
    hrefs = ["models/tiny/L0/N5.html", "models/tiny/L1/H1.html"]
    line = {"date": "2026-11-16", "event": "new", "id": "C1", "members": hrefs}
    (folder / "tiny.jsonl").write_text(json.dumps(line) + "\n")
    (circuit,) = explain_groups.circuits("tiny", folder)
    assert circuit.neurons == ((0, 5, 1.0),) and circuit.heads == ((1, 1),)
    plain = ex.explain(gpt2, "tiny", PROMPT, gpt2_means, {}, top=1000, groups=[circuit])
    (found,) = plain.groups
    token = int(ex.logits_with(gpt2, PROMPT, {}, -1).argmax())
    scores, _ = ex.attribute(gpt2, PROMPT, gpt2_means, token, -1)
    assert found.attribution == pytest.approx(sum(scores.get(u, (0.0,))[0] for u in circuit.units))
    assert found.status == "unstudied" and found.ablation_change is not None
    assert plain.verified_share == 0.0
    labels = {"C1": ("VERIFIED", "copies the name")}
    verified = ex.explain(
        gpt2, "tiny", PROMPT, gpt2_means, {}, groups=[circuit], group_labels=labels
    )
    assert verified.groups[0].explanation == "copies the name"
    covered = sum(abs(scores[u][0]) for u in circuit.units if u in scores)
    assert verified.verified_share == pytest.approx(covered / verified.total)


def test_circuit_labels_read_each_circuits_newest_record(tmp_path):
    import shutil
    from pathlib import Path

    from openinterpretability.explain_groups import circuit_labels

    source = next(Path("wiki/circuits").glob("*/C*/*.json"))
    target = tmp_path / "circuits" / source.parent.parent.name / source.parent.name
    target.mkdir(parents=True)
    shutil.copy(source, target / source.name)
    record = json.loads(source.read_text())
    model = record["hypothesis"]["model_name"]
    labels = circuit_labels(tmp_path, model)
    assert labels == {
        source.parent.name: (record["verdict"]["status"], record["hypothesis"]["explanation"])
    }


def test_means_are_computed_but_not_kept_with_the_cache_off(toy_runner, tmp_path, monkeypatch):
    monkeypatch.setenv(ex.MEANS_ENV, "off")
    monkeypatch.setattr(ex, "WINDOW", 4)
    first = ex.unit_means(toy_runner, "toy", TEXTS, 200)
    again = ex.unit_means(toy_runner, "toy", TEXTS, 200)
    assert sorted(first.neurons) == list(range(toy_runner.n_layers))
    assert all(torch.equal(first.neurons[la], again.neurons[la]) for la in first.neurons)
    assert not list(tmp_path.rglob("*.pt"))


def test_registry_labels_keep_only_the_models_units(tmp_path, record):
    from datetime import UTC, datetime

    from openinterpretability.builder.wiki import write_experiment_page
    from openinterpretability.core.draws import UnauditableDraw, write_draw
    from tests.test_core_draws import PROVENANCE

    wiki = tmp_path / "wiki"
    write_experiment_page(record, wiki)  # a record of another model
    for model in ("toy", "other/model"):
        ident = NeuronIdentifier(model_name=model, layer=0, neuron_idx=2, pole="positive")
        write_draw(UnauditableDraw(identifier=ident, outcome="dead", provenance=PROVENANCE,
                                   timestamp=datetime(2026, 11, 1, tzinfo=UTC)), wiki)  # fmt: skip
    toy = NeuronIdentifier(model_name="toy", layer=0, neuron_idx=2, pole="positive").slug
    assert ex.registry_labels(wiki, "toy") == {toy: ("stopped", None, None)}
    own = ex.registry_labels(wiki, record.submission.identifier.model_name)
    assert own[record.submission.identifier.slug][0] == record.verdict.status


def test_a_verified_superposition_is_labelled_and_checked_by_removing_it(gpt2, gpt2_means):
    from openinterpretability.explain_groups import GroupDef, remove_direction

    group = GroupDef("superposition", "L1/Sx", "tiny/L1/Sx", ((1, 3, 0.8), (1, 40, -0.5)))
    labels = {"tiny/L1/Sx": ("VERIFIED", "reads b", None)}
    result = ex.explain(gpt2, "tiny", PROMPT, gpt2_means, labels, top=5, groups=[group])
    (found,) = result.groups
    assert (found.kind, found.status, found.explanation) == ("superposition", "VERIFIED", "reads b")
    base = ex.logits_with(gpt2, PROMPT, {}, -1)
    removed = ex.logits_with(gpt2, PROMPT, remove_direction(group, gpt2_means), -1)
    token = result.token_id
    assert found.ablation_change == pytest.approx(float(removed[token] - base[token]))
