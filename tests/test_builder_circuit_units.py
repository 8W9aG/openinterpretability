from pathlib import Path
from types import SimpleNamespace

from openinterpretability.builder.circuit_units import _location, load_circuit_data
from openinterpretability.core.circuit_records import CircuitRecord

REAL = Path(__file__).parents[1] / "wiki/circuits/openai-community--gpt2/C4/20261116T232532Z.json"


def test_load_circuit_data_reads_each_circuit_record_with_its_source(tmp_path):
    wiki = tmp_path / "wiki"
    path = wiki / "circuits" / "openai-community--gpt2" / "C4" / REAL.name
    path.parent.mkdir(parents=True)
    path.write_text(REAL.read_text())
    data = load_circuit_data(wiki, tmp_path / "no-lineage", tmp_path / "no-hubs")
    ((source, record),) = data["openai-community--gpt2"].records
    assert source == f"wiki/circuits/openai-community--gpt2/C4/{REAL.name}"
    assert record == CircuitRecord.model_validate_json(REAL.read_text())


def test_members_with_no_layer_have_no_location():
    model = SimpleNamespace(n_layers=12, mlp_width=3072)
    assert _location(model, ["models/m/index.html"]) == (None, None)
    assert _location(model, ["models/m/L11/N1536.html"]) == (1.0, 0.5)
