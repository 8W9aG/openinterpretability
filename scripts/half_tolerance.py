"""How far 16-bit numbers drift from float32 when a record is reproduced (issue #2995).

`openinterp reproduce` compares each number within ``atol × scale + rtol × |recorded|``.
The defaults suit float32 differences between devices. A record made in bfloat16 on one
device drifts by a few percent on another. This script measures that drift, to set the
tolerance for 16-bit records (``reproduce.HALF_TOLERANCE``).

It reproduces each record with the model loaded in ``--dtype`` on ``--device``. Every
number that `reproduce` compares is kept, with the scale it is judged against. The exact
checks that differ are kept too: probe pass/fail, gates, balanced accuracy and status.
One JSON row per record goes to ``--out``.

Two directions measure the same rounding:

- A bfloat16 record, reproduced in float32: what `reproduce` does with a 16-bit record.
- A float32 record, reproduced in bfloat16: many more records, models and layers.

    uv run python scripts/half_tolerance.py --dtype float32 --device cpu \\
        --out half.jsonl wiki/experiments/Qwen--Qwen2.5-0.5B/L12/N2655-positive/*.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import click

from openinterp.reproduce import Tolerance, reproduce_record

NUMERIC = ("activations differ", "max_activation", "observed_delta", ": activation", ": Δ")


@dataclass(frozen=True)
class Recording(Tolerance):
    """A tolerance that passes every number and keeps (recorded, reproduced, scale)."""

    seen: list = field(default_factory=list, compare=False, hash=False)
    label: list = field(default_factory=lambda: ["other"], compare=False, hash=False)

    def close(self, recorded, reproduced, scale):
        """Keep the pair with what it measures, and call it close."""
        if recorded is not None and reproduced is not None:
            self.seen.append((float(recorded), float(reproduced), float(scale), self.label[0]))
        return True

    def for_dtype(self, dtype):
        """Keep recording, whatever the record's dtype."""
        return self


def labelled(name: str, label: str):
    """Wrap ``reproduce.<name>`` so that the numbers it compares carry ``label``."""
    import openinterp.reproduce as module

    inner = getattr(module, name)

    def wrapper(*args, **kwargs):
        tolerance = next(a for a in args if isinstance(a, Recording))
        tolerance.label[0] = label
        try:
            return inner(*args, **kwargs)
        finally:
            tolerance.label[0] = "other"

    setattr(module, name, wrapper)


for _name, _label in [
    ("_check_examples", "top_example"),
    ("_check_counterfactuals", "counterfactual"),
    ("_compare_verdicts", "review"),
    ("_compare_explanation_scores", "explanation_window"),
]:
    labelled(_name, _label)


def summary(seen: list[tuple]) -> dict:
    """The largest drift as a share of the scale, and of the recorded value."""
    by_scale = [abs(b - a) / s for a, b, s, _ in seen if s > 0]
    by_value = [abs(b - a) / abs(a) for a, b, _, _ in seen if abs(a) > 0]
    return {
        "numbers": len(seen),
        "max_over_scale": max(by_scale, default=0.0),
        "max_over_value": max(by_value, default=0.0),
    }


@click.command()
@click.option("--dtype", type=click.Choice(["float32", "bfloat16", "float16"]), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.argument("records", nargs=-1, type=click.Path(exists=True, path_type=Path))
def main(dtype, device, out, records):
    """Reproduce each record in ``--dtype`` and write how far its numbers drift."""
    from openinterp.cli import _load_runner
    from openinterp.core.schemas import AuditRecord
    from openinterp.harness.units import attn_implementation
    from openinterp.reproduce import pinned_environment

    runners: dict[tuple, object] = {}
    for path in records:
        record = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
        model = record.submission.identifier.model_name
        revision, pinned = pinned_environment(record)
        key = (model, revision, attn_implementation(record.submission.identifier))
        if key not in runners:
            runners.clear()  # one model in memory at a time
            runners[key] = _load_runner(model, device, dtype, revision, key[2])
        tolerance = Recording()
        report = reproduce_record(record, runners[key], tolerance, path)
        exact = [m for m in report.mismatches if not any(n in m for n in NUMERIC)]
        exact = [m for m in exact if "the record pins" not in m]
        row = {
            "record": path.as_posix(),
            "model": model,
            "pinned": pinned,
            "loaded": dtype,
            "device": device,
            "recorded_device": record.verdict.environment.device
            if record.verdict and record.verdict.environment
            else None,
            **summary(tolerance.seen),
            "exact_mismatches": exact,
            "pairs": [[round(a, 6), round(b, 6), round(s, 6), k] for a, b, s, k in tolerance.seen],
        }
        line = json.dumps(row)
        with open(out, "a") as handle:
            handle.write(line + "\n")
        click.echo(
            f"{path.name} {model} {pinned}->{dtype}: {row['numbers']} numbers, "
            f"max {row['max_over_scale']:.4f} of scale, exact {len(exact)}"
        )


if __name__ == "__main__":
    main()
