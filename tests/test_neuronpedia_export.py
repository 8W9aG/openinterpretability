"""VERIFIED explanations in Neuronpedia's export format (`data/neuronpedia/` of the site)."""

import gzip
import json

from openinterp.builder.compiler import compile_site
from openinterp.builder.neuronpedia_export import EXPORT_DIR, NEURON_SET, TYPE_NAME
from openinterp.core.identifiers import FeatureIdentifier
from openinterp.core.licenses import DATA_LICENSE
from openinterp.core.schemas import NeuronIdentifier

GPT2 = "openai-community/gpt2"


def _with(record, ident, status="VERIFIED"):
    sub = record.submission.model_copy(update={"identifier": ident})
    verdict = record.verdict.model_copy(update={"status": status})
    return record.model_copy(update={"submission": sub, "verdict": verdict})


def _rows(site, source):
    path = site / EXPORT_DIR / "gpt2-small" / source / "explanations" / "batch-0.jsonl.gz"
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def test_the_site_exports_verified_units_in_neuronpedias_layout(record, tmp_path):
    neuron = NeuronIdentifier(model_name=GPT2, layer=3, neuron_idx=12, pole="negative")
    latent = FeatureIdentifier(
        model_name=GPT2, layer=8, dictionary="gpt2-small-transcoders", feature_idx=10336
    )
    rejected = NeuronIdentifier(model_name=GPT2, layer=3, neuron_idx=13, pole="positive")
    other_model = NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")
    records = [
        _with(record, neuron),
        _with(record, latent),
        _with(record, rejected, status="REJECTED"),
        _with(record, other_model),  # Neuronpedia doesn't know the model
    ]
    site = tmp_path / "site"
    compile_site(records, site)

    (row,) = _rows(site, f"3-{NEURON_SET}")
    assert (row["modelId"], row["index"], row["pole"]) == ("gpt2-small", "12", "negative")
    assert row["typeName"] == TYPE_NAME and row["license"] == DATA_LICENSE
    assert row["description"] == record.submission.natural_language_explanation
    assert row["explanationModelName"] == record.submission.submitting_agent_model
    assert row["url"].endswith("/blob/main/wiki/experiments/openai-community--gpt2/L3/"
                               f"N12-negative/{row['url'].rsplit('/', 1)[1]}")  # fmt: skip
    (latent_row,) = _rows(site, "8-tres-dc")  # the source Neuronpedia hosts it under
    assert (latent_row["index"], latent_row["pole"]) == ("10336", None)
    exported = list((site / EXPORT_DIR).rglob("*.gz"))
    assert len(exported) == 2  # not the REJECTED pole, not the unknown model

    before = {p: p.read_bytes() for p in exported}
    compile_site(records, site)  # the same records give the same bytes
    assert {p: p.read_bytes() for p in (site / EXPORT_DIR).rglob("*.gz")} == before
