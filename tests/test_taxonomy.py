"""The registry's taxonomy: one definition, in the protocol and on the About page (#3703)."""

import sys
from pathlib import Path

from openinterpretability.core import identifiers as ids
from openinterpretability.core.taxonomy import AUDITED, TAXONOMY

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from taxonomy_table import PROTOCOL, updated  # noqa: E402


def _concept(term):
    return next(c for c in TAXONOMY if c.term == term)


def test_the_protocol_table_matches_the_module():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert updated(text) == text, "run scripts/taxonomy_table.py"


def test_identifier_forms_come_from_the_identifier_classes():
    model = "openai-community/gpt2"
    pole = ids.NeuronIdentifier(model_name=model, layer=3, neuron_idx=7, pole="positive")
    head = ids.HeadIdentifier(model_name=model, layer=5, head_idx=1)
    feature = ids.FeatureIdentifier(model_name=model, layer=4, feature_idx=12, dictionary="d")
    sup = ids.SuperpositionIdentifier(model_name=model, layer=2, superposition_id="0123456789ab")
    hub = ids.HubIdentifier(model_name=model, layer=3, center="neuron", index=7)
    assert _concept("Pole").identifier == pole.label
    assert _concept("Neuron").identifier == pole.label.rstrip("+")
    assert _concept("Head").identifier == head.label
    assert _concept("Dictionary feature").identifier == feature.label
    assert _concept("Superposition").identifier == sup.label
    assert hub.slug.endswith("/" + _concept("Hub").identifier)
    assert _concept("Model").identifier == model


def test_the_audited_units_are_the_component_kinds():
    audited = {c.term for c in TAXONOMY if c.audited == AUDITED}
    assert audited == {"Pole", "Head", "Superposition", "Dictionary feature", "Circuit"}


def test_the_about_page_shows_each_concept_and_three_figures(tmp_path):
    from html import escape

    from openinterpretability.builder.compiler import compile_site

    compile_site([], tmp_path)
    about = (tmp_path / "about.html").read_text()
    for concept in TAXONOMY:
        assert f'id="{concept.anchor}"' in about and escape(concept.identifier) in about
    assert about.count('role="img" aria-labelledby="tax-') == 4
    assert "/issues/4438" in about and "<code>open-interpretability sisters</code>" in about
    assert "docs/PROTOCOL.md#1-unit-of-analysis" in about
