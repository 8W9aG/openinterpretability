"""The site shows superpositions: their pages, their members and their counts (issue #3685)."""

import pyarrow.parquet as pq

from openinterp.builder.compiler import compile_site
from openinterp.builder.neuronpedia_export import export_row
from openinterp.builder.tree import build_tree, false_discovery
from openinterp.builder.wiki import rebuild_indexes, write_experiment_page
from openinterp.core.schemas import AuditRecord, NeuronIdentifier
from tests.site_pages import unit_pages
from tests.test_superpositions import MEMBERS, _reviewed
from tests.test_wiring import qwen_runner  # noqa: F401  (fixture)


def _neuron_record(record: AuditRecord, idx: int) -> AuditRecord:
    """An ordinary neuron-pole record on neuron ``idx`` of the superposition's layer."""
    ident = record.submission.identifier
    neuron = NeuronIdentifier(
        model_name=ident.model_name, layer=ident.layer, neuron_idx=idx, pole="positive"
    )
    update = {"identifier": neuron, "superposition_profile": None}
    return AuditRecord(
        submission=record.submission.model_copy(update=update), verdict=record.verdict
    )


def test_superposition_pages_members_counts_and_parquet(qwen_runner, tmp_path):  # noqa: F811
    record = _reviewed(qwen_runner)
    ident = record.submission.identifier
    studied, lonely, smallest = (n for n, _ in MEMBERS)  # by |weight|: 3, 7, 20
    neuron = _neuron_record(record, studied)
    report = compile_site([record, neuron], tmp_path / "dist")

    layer = tmp_path / "dist" / "models" / ident.model_slug / f"L{ident.layer}"
    page = unit_pages(layer / f"S{ident.superposition_id}.html")
    assert "Superposition of 3 neurons" in page and "Method: <code>circuit</code>" in page
    rows = page[page.index('id="members"') :]
    order = [rows.index(f'N{n}.html">N{n}</a>') for n in (studied, lonely, smallest)]
    assert order == sorted(order)  # sorted by |weight|
    assert "+1.00" in rows and "-0.50" in rows and "+0.25" in rows  # relative to the largest

    for idx in (studied, lonely):
        member = (layer / f"N{idx}.html").read_text()
        assert f'S{ident.superposition_id}.html">L{ident.layer}/S{ident.superposition_id}' in member
    assert "has no experiment of its own" in (layer / f"N{lonely}.html").read_text()
    assert "has no experiment of its own" not in (layer / f"N{studied}.html").read_text()

    home = (tmp_path / "dist" / "index.html").read_text()
    assert "1 superpositions studied" in home
    assert "3 neurons studied, 2 of them only as part of a superposition" in home
    model = (tmp_path / "dist" / "models" / ident.model_slug / "index.html").read_text()
    assert "2 of them only as part of a superposition" in model

    rows = {r["kind"]: r for r in pq.read_table(report.parquet).to_pylist()}
    assert rows["superposition"]["superposition_id"] == ident.superposition_id
    assert rows["superposition"]["superposition_members"] == [3, 7, 20]
    assert rows["mlp_neuron_pole"]["superposition_id"] is None
    assert rows["mlp_neuron_pole"]["superposition_members"] is None


def test_superpositions_stay_out_of_native_counts_and_neuronpedia(qwen_runner, tmp_path):  # noqa: F811
    record = _reviewed(qwen_runner)
    (model,) = build_tree([record])
    (unit,) = model.layers[0].units
    assert unit.is_superposition and not unit.is_native and model.neurons == 0
    assert model.superpositions == 1 and false_discovery([model])["claims"] == 0
    (experiment,) = unit.experiments
    assert export_row(experiment, "https://example.org/repo") is None

    write_experiment_page(record, tmp_path / "wiki")
    rebuild_indexes(tmp_path / "wiki")
    index = (tmp_path / "wiki" / "README.md").read_text()
    assert (
        f"L{record.submission.identifier.layer}/S{record.submission.identifier.superposition_id}"
        in index
    )
