"""Backfilled wiring and circuit pages."""

from click.testing import CliRunner

from openinterpretability.builder.circuits import circuits, circuits_and_hubs
from openinterpretability.builder.compiler import compile_site
from openinterpretability.builder.tree import build_tree
from openinterpretability.builder.wiki import archive_record
from openinterpretability.cli import cli
from openinterpretability.core.schemas import AdversarialTestResult
from openinterpretability.core.wiring import Wiring, WiringEdge
from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.wiring_backfill import backfill, load_rows, record_wiring, review_prompts
from tests.conftest import FakeTokenizer
from tests.site_pages import unit_pages

MODEL = "Qwen--Qwen2.5-0.5B"


def at(record, layer, idx, **wiring):
    """``record`` moved to ``L<layer>/N<idx>``, with ``wiring`` set on it."""
    ident = record.submission.identifier.model_copy(update={"layer": layer, "neuron_idx": idx})
    sub = record.submission.model_copy(
        update={"identifier": ident, "wiring": wiring.get("weights")}
    )
    verdict = record.verdict.model_copy(update={"wiring": wiring.get("activations")})
    return record.model_copy(update={"submission": sub, "verdict": verdict})


def wiring(method, up=(), down=()):
    def edges(pairs):
        return [WiringEdge(kind=k, layer=n, index=i, weight=w) for k, n, i, w in pairs]

    return Wiring(
        method=method,
        upstream_neurons=edges(p for p in up if p[0] == "mlp_neuron"),
        upstream_heads=edges(p for p in up if p[0] == "attn_head"),
        downstream_neurons=edges(p for p in down if p[0] == "mlp_neuron"),
        downstream_heads=edges(p for p in down if p[0] == "attn_head"),
        prompts=2 if method == "activations" else 0,
    )


def test_circuits_chain_edges_between_studied_units(tmp_path, record):
    records = [
        # L1/N5 -> L3/N117 -> L6/N2, and L3/N117 -> L9/N9 which has no page
        at(record, 3, 117, weights=wiring("weights", up=[("mlp_neuron", 1, 5, 0.4)],
                                          down=[("mlp_neuron", 9, 9, 0.2)])),
        at(record, 1, 5, activations=wiring("activations", down=[("mlp_neuron", 3, 117, 0.7)])),
        at(record, 6, 2, weights=wiring("weights", up=[("mlp_neuron", 3, 117, -0.3)])),
        at(record, 12, 1),  # wired to nothing
    ]  # fmt: skip
    (model,) = build_tree(records, [])
    (circuit,) = circuits(model, {})
    assert [(u.layer, u.idx) for u in circuit.units] == [(1, 5), (3, 117), (6, 2)]
    first, second = circuit.edges
    assert (first.source.idx, first.target.idx, first.methods) == (
        5,
        117,
        ["activations", "weights"],
    )
    assert (second.source.idx, second.target.idx, second.methods) == (117, 2, ["weights"])

    compile_site(records, tmp_path)
    site = tmp_path / "models" / MODEL
    index = (site / "circuits.html").read_text()
    assert 'href="../../models/' + MODEL + '/circuits/L1-N5.html">L1/N5</a>' in index
    page = (site / "circuits" / "L1-N5.html").read_text()  # named after its root
    assert "Circuit from L1/N5" in page and "3 units of" in page and "layers 1–6" in page
    assert "L9/N9" not in page and "weights +0.4" in page and "activations +0.7" in page
    assert 'href="../../../models/' + MODEL + "/L1/N5-positive/" in page  # the experiment's page
    model_page = (site / "index.html").read_text()
    assert "1 circuit</a>" in model_page
    unit_page = unit_pages(site / "L3" / "N117.html")
    assert "circuits/L1-N5.html" in unit_page and "See its circuit" in unit_page
    lone = unit_pages(site / "L12" / "N1.html")
    assert "circuits/" not in lone

    # Every other unit's name redirects to the page it is on now, or to its own page.
    assert (
        "url=../../../models/" + MODEL + '/circuits/L1-N5.html"'
        in (site / "circuits" / "L3-N117.html").read_text()
    )
    assert "circuits/L1-N5.html" in (site / "hubs" / "L6-N2.html").read_text()
    assert (
        "url=../../../models/" + MODEL + '/L12/N1.html"'
        in (site / "circuits" / "L12-N1.html").read_text()
    )


def test_a_circuit_keeps_its_old_url_when_a_lower_unit_joins(tmp_path, record):
    chain = [
        at(record, 3, 117, weights=wiring("weights", down=[("mlp_neuron", 6, 2, 0.4)])),
        at(record, 6, 2),
    ]
    compile_site(chain, tmp_path / "before")
    assert (tmp_path / "before" / "models" / MODEL / "circuits" / "L3-N117.html").exists()
    lower = at(record, 1, 5, weights=wiring("weights", down=[("mlp_neuron", 3, 117, 0.2)]))
    compile_site([*chain, lower], tmp_path / "after")
    old = (tmp_path / "after" / "models" / MODEL / "circuits" / "L3-N117.html").read_text()
    assert 'http-equiv="refresh"' in old and "circuits/L1-N5.html" in old


def test_hubs_dont_join_circuits(tmp_path, record):
    spoke = wiring("activations", down=[("mlp_neuron", 12, 1, 0.1)])
    records = [
        at(record, 1, 5, weights=wiring("weights", down=[("mlp_neuron", 3, 117, 0.4)])),
        at(record, 3, 117, activations=spoke),
        *(at(record, 2, i, activations=spoke) for i in range(9)),
        at(record, 12, 1),  # 10 units are wired into it, of 12
    ]
    (model,) = build_tree(records, [])
    found, (hub,) = circuits_and_hubs(model, {})
    (circuit,) = found
    assert [(u.layer, u.idx) for u in circuit.units] == [(1, 5), (3, 117)]
    assert (hub.unit.layer, hub.unit.idx, len(hub.neighbours)) == (12, 1, 10)

    compile_site(records, tmp_path)
    site = tmp_path / "models" / MODEL
    index = (site / "circuits.html").read_text()
    assert "hubs/L12-N1.html" in index and "10 studied units" in index
    page = (site / "hubs" / "L12-N1.html").read_text()
    assert "Hub" in page and "Wired to 10 studied units" in page and "(10 edges in, 0 out)" in page
    assert "1 circuit, 1 hub</a>" in (site / "index.html").read_text()
    assert "hubs/L12-N1.html" in unit_pages(site / "L2" / "N4.html")
    assert "circuits/L1-N5.html" in unit_pages(site / "L3" / "N117.html")
    # The hub's name under circuits/ redirects to its hub page.
    assert "hubs/L12-N1.html" in (site / "circuits" / "L12-N1.html").read_text()


def test_no_circuits_no_page(tmp_path, record):
    compile_site([record], tmp_path)
    assert not (tmp_path / "models" / MODEL / "circuits.html").exists()
    assert "circuit" not in (tmp_path / "models" / MODEL / "index.html").read_text()


def test_backfill_writes_resumable_rows_the_build_shows(tmp_path, record, tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    holdout = AdversarialTestResult(
        kind="positive_holdout", prompt="abc abc", passed=True, expected_active=True
    )
    verdict = record.verdict.model_copy(
        update={"adversarial_test_results": [*record.verdict.adversarial_test_results, holdout]}
    )
    rec = at(record.model_copy(update={"verdict": verdict}), 1, 5)
    rec = rec.model_copy(
        update={"submission": rec.submission.model_copy(update={"max_activation": 1e-4})}
    )
    own = at(record, 1, 6, weights=wiring("weights"))  # has its own wiring: left alone
    assert review_prompts(rec) == ['"Hello," she said.', "abc abc"]

    out = tmp_path / "wiring.jsonl"
    loads = []
    assert backfill([rec, own], lambda r: loads.append(r) or runner, out, echo=lambda _: None) == 1
    assert backfill([rec, own], lambda r: runner, out, echo=lambda _: None) == 0  # resumes
    rows = load_rows(out)
    weights, activations, backfilled = record_wiring(rec, rows)
    assert backfilled and len(loads) == 1
    assert weights.method == "weights" and activations.prompts == 2
    assert record_wiring(own, rows)[2] is False

    wiki, site = tmp_path / "wiki", tmp_path / "site"
    archive_record(rec, wiki)
    args = ["build", "--wiki-dir", str(wiki), "--output-dir", str(site), "--wiring", str(out)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    page = unit_pages(site / "models" / MODEL / "L1" / "N5.html")
    assert "Backfilled after the review" in page and "From activations" in page
