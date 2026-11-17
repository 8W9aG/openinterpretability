"""How far float32 moves the activations of a neuron pole (#4187).

For each neuron record of the named models, measure every top example again on this
machine. Measure in float32 with one CPU thread, in float32 with the default threads,
and in float64. Report the largest |difference| over all tokens, in activation units, and the
record's scale. The absolute floor of each model in
``[tool.openinterpretability.reproduce_floor]`` is set from these numbers.

    uv run python scripts/neuron_float32_floor.py \\
        --out docs/analyses/neuron-float32-floor.jsonl [RECORD ...]
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch

from openinterpretability.cli import _load_unit_runner
from openinterpretability.core.paths import DEFAULT_WIKI_DIR, experiments_dir
from openinterpretability.core.schemas import AuditRecord, NeuronIdentifier
from openinterpretability.harness.layer_major import map_layer_major, prompt_tokens
from openinterpretability.harness.units import submission_unit

# Each model's poles: Pythia-70m's every pole, GPT-2's negative poles. A negative pole of
# a GELU model has a scale of at most 0.17, so a small absolute move is a large share.
SELECTION = {
    "EleutherAI--pythia-70m": ("positive", "negative"),
    "openai-community--gpt2": ("negative",),
}


def neuron_records(wiki_dir: Path = DEFAULT_WIKI_DIR) -> list[Path]:
    """The neuron records of :data:`SELECTION`, by path."""
    found = []
    for model, poles in SELECTION.items():
        for pole in poles:
            found += experiments_dir(wiki_dir).glob(f"{model}/L*/N*-{pole}/*.json")
    return sorted(found)


def _strengths(runner, record: AuditRecord, threads: int | None) -> list[torch.Tensor]:
    """Each top example's per-token activations, measured as ``reproduce`` measures them."""
    before = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        unit = submission_unit(record.submission)
        texts = [ex.text for ex in record.submission.top_examples]
        rows = map_layer_major(
            runner, partial(unit.strengths, runner), texts,
            tokens=prompt_tokens(runner, texts), logits=False,
        )  # fmt: skip
        return [row[0].double() for row in rows]
    finally:
        torch.set_num_threads(before)


def _largest(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    """The largest |a − b| over the tokens of all examples."""
    return max((x[: len(y)] - y[: len(x)]).abs().max().item() for x, y in zip(a, b, strict=True))


def _peak_move(record: AuditRecord, measured: list[torch.Tensor]) -> float:
    """The largest |recorded peak − measured peak| over the top examples."""
    skip = int(submission_unit(record.submission).skip_first)
    pairs = zip(record.submission.top_examples, measured, strict=True)
    return max(abs(ex.max_activation - v[skip:].max().item()) for ex, v in pairs)


def measure(path: Path, runners: dict) -> dict:
    """The largest moves of one record's top examples, in activation units."""
    record = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
    sub = record.submission
    ident = sub.identifier
    if not isinstance(ident, NeuronIdentifier):
        raise SystemExit(f"{path} is not a neuron record")
    if ident.model_name not in runners:
        r32 = _load_unit_runner(ident, "cpu", "float32")
        r64 = _load_unit_runner(ident, "cpu", "float32")
        r64.model.double()
        runners[ident.model_name] = (r32, r64)
    r32, r64 = runners[ident.model_name]
    recorded = [torch.tensor(ex.activations, dtype=torch.float64) for ex in sub.top_examples]
    one, default = _strengths(r32, record, 1), _strengths(r32, record, None)
    exact = _strengths(r64, record, None)
    env = record.verdict.environment if record.verdict else None
    return {
        "record": str(path),
        "model": ident.model_name,
        "pole": ident.pole,
        "recorded_on": f"{env.device_name or env.device} ({env.torch_version})" if env else None,
        "scale": round(sub.max_activation, 6),
        "remeasured_vs_recorded": round(
            max(_largest(one, recorded), _largest(default, recorded)), 6
        ),
        "one_vs_default_threads": round(_largest(one, default), 6),
        "recorded_vs_float64": round(_largest(recorded, exact), 6),
        "float32_vs_float64": round(_largest(default, exact), 6),
        "peak_vs_recorded": round(_peak_move(record, default), 6),
    }


def main() -> None:
    """Measure the named records, or every record of :data:`SELECTION`, one row each."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runners: dict = {}
    with args.out.open("w", encoding="utf-8") as out:
        for path in args.records or neuron_records():
            row = measure(path, runners)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(row["record"], row["scale"], row["remeasured_vs_recorded"], flush=True)


if __name__ == "__main__":
    main()
