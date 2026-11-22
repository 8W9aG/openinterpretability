"""One experiment's row of the registry's Parquet export (builder/parquet_export.py)."""

from openinterpretability.builder.parquet_export import experiment_row
from openinterpretability.builder.tree import build_tree
from openinterpretability.core.attribution import InputAttribution


def _row(record, attribution):
    sub = record.submission.model_copy(update={"input_attribution": attribution})
    (model,) = build_tree([record.model_copy(update={"submission": sub})], [])
    (experiment,) = model.layers[0].units[0].experiments
    return experiment_row(experiment, with_json=False)


def test_an_attribution_with_every_example_skipped_has_empty_attribution_columns(record):
    skipped = InputAttribution(
        baseline_token="<|endoftext|>", tolerance=0.1, examples=[], distances=[],
        mean_distance=3.0, ablation_agreement=0.5,
    )  # fmt: skip
    row = _row(record, skipped)
    assert row["current_token_share"] is None
    assert row["context_mean_distance"] is None  # not 3.0: nothing was measured
    assert row["context_ablation_agreement"] is None
