"""Random draws that could not be audited (openinterpretability/core/draws.py)."""

from datetime import UTC, datetime

from openinterpretability.core.draws import UnauditableDraw, load_draws, write_draw
from openinterpretability.core.identifiers import NeuronIdentifier
from openinterpretability.core.schemas import ExperimentProvenance

PROVENANCE = ExperimentProvenance(
    dtype="float32",
    corpus="corpus.jsonl",
    corpus_sha256="ab" * 32,
    n_snippets=1,
    tokens_scanned=100,
    selection="random",
    harness_version="test",
)


def _draw(proven_dead: bool) -> UnauditableDraw:
    return UnauditableDraw(
        identifier=NeuronIdentifier(model_name="m/a", layer=0, neuron_idx=1, pole="positive"),
        outcome="dead",
        proven_dead=proven_dead,
        provenance=PROVENANCE,
        timestamp=datetime(2026, 11, 2, 12, tzinfo=UTC),
    )


def test_a_draw_is_written_once_and_a_second_write_keeps_the_first(tmp_path):
    first = write_draw(_draw(proven_dead=True), tmp_path)
    again = write_draw(_draw(proven_dead=False), tmp_path)  # the same unit and time
    assert again == first
    assert UnauditableDraw.model_validate_json(first.read_text()).proven_dead
    assert [d.proven_dead for d in load_draws(tmp_path)] == [True]
