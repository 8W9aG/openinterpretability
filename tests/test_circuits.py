"""Backfilled wiring and circuit pages."""

from click.testing import CliRunner

from openinterp.builder.circuits import circuits, circuits_and_hubs
from openinterp.builder.compiler import compile_site
from openinterp.builder.tree import build_tree
from openinterp.builder.wiki import archive_record
from openinterp.cli import cli
from openinterp.core.schemas import AdversarialTestResult
from openinterp.core.wiring import Wiring, WiringEdge
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.wiring_backfill import backfill, load_rows, record_wiring, review_prompts
from tests.conftest import FakeTokenizer

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
    (model,) = build_tree(records)
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
    page = (tmp_path / "models" / MODEL / "circuits.html").read_text()
    assert "Circuit 1" in page and "3 units, layers 1–6" in page and "L9/N9" not in page
    assert "weights +0.4" in page and "activations +0.7" in page
    assert 'href="../../models/' + MODEL + "/L1/N5.html#" in page
    model_page = (tmp_path / "models" / MODEL / "index.html").read_text()
    assert "1 circuit</a>" in model_page
    unit_page = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "circuits.html#circuit-1" in unit_page
    lone = (tmp_path / "models" / MODEL / "L12" / "N1.html").read_text()
    assert "circuits.html" not in lone


def test_hubs_dont_join_circuits(tmp_path, record):
    spoke = wiring("activations", down=[("mlp_neuron", 12, 1, 0.1)])
    records = [
        at(record, 1, 5, weights=wiring("weights", down=[("mlp_neuron", 3, 117, 0.4)])),
        at(record, 3, 117, activations=spoke),
        *(at(record, 2, i, activations=spoke) for i in range(9)),
        at(record, 12, 1),  # 10 units are wired into it, of 12
    ]
    (model,) = build_tree(records)
    found, (hub,) = circuits_and_hubs(model, {})
    (circuit,) = found
    assert [(u.layer, u.idx) for u in circuit.units] == [(1, 5), (3, 117)]
    assert (hub.unit.layer, hub.unit.idx, len(hub.neighbours)) == (12, 1, 10)

    compile_site(records, tmp_path)
    page = (tmp_path / "models" / MODEL / "circuits.html").read_text()
    assert 'id="hub-L12-N1"' in page and "wired to 10 studied units (10 edges in, 0 out)" in page
    assert "1 circuit, 1 hub</a>" in (tmp_path / "models" / MODEL / "index.html").read_text()
    spoke_page = (tmp_path / "models" / MODEL / "L2" / "N4.html").read_text()
    assert "circuits.html#hub-L12-N1" in spoke_page
    unit_page = (tmp_path / "models" / MODEL / "L3" / "N117.html").read_text()
    assert "circuits.html#circuit-1" in unit_page


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
    page = (site / "models" / MODEL / "L1" / "N5.html").read_text()
    assert "Backfilled after the review" in page and "From activations" in page
