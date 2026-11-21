"""The dictionary latents aligned with a neuron's poles, for its page (builder/aligned.py)."""

from types import SimpleNamespace

from openinterpretability.builder.aligned import aligned_latents
from openinterpretability.core.schemas import NeuronIdentifier
from openinterpretability.crossref import AlignmentRow

GPT2, TC = "openai-community/gpt2", "gpt2-small-transcoders"


def test_a_row_without_reads_shows_only_its_writes():
    ident = NeuronIdentifier(model_name=GPT2, layer=8, neuron_idx=7, pole="positive")
    model = SimpleNamespace(slug=ident.slug.rsplit("/L", 1)[0])
    neuron = SimpleNamespace(is_neuron=True, sides=("positive",), layer=8, idx=7)
    row = AlignmentRow.model_validate({
        "slug": ident.slug, "dictionary": TC, "repo": "r", "revision": "v", "path": "p",
        "model_revision": None,
        "writes": {"matches": [{"feature_idx": 1, "cosine": 0.9}], "chance": 0.2,
                   "percentile": 0.9},
    })  # fmt: skip
    (pole,) = aligned_latents(model, neuron, {ident.slug: [row]}, set())
    (view,) = pole["rows"]
    assert pole["pole"] == "positive" and [s["name"] for s in view["sides"]] == ["writes"]
    (match,) = view["sides"][0]["matches"]
    assert match["above"] and match["href"] is None
