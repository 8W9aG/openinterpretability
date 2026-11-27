"""Does a VERIFIED answer route keep on the instruct model? (#5164).

The route, its claim and its probes come from one VERIFIED answer record, unchanged: no new
search. ``answer_review.review`` measures them on the base model and on its ``-Instruct``
model, each with its own unit means from the same corpus tokens. Each model runs with the
record's null seed and :data:`SEEDS` more, so a gate that passes by chance shows up. It writes
one ``gate`` row per model and seed, and one ``prompt`` row per model and prompt. It writes
no registry record.

    uv run python scripts/route_keep.py \\
        --record wiki/answers/Qwen--Qwen2.5-0.5B/15385436b27b/20261127T002913Z.json \\
        --device cuda --out papers/route-keep/results.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEEDS = (1, 2, 3, 4)  # the extra seeds of the random sets, beside the record's own
MEAN_TOKENS = 20_000  # the corpus tokens of each model's unit means, as the review uses


def gate_row(record, model: str, seed: int) -> dict:
    """The ``gate`` row of one review: its status, each gate and the measures they read.

    Args:
        record: The answer record that ``review`` returned.
        model: The model it measured.
        seed: The seed of its random sets.

    Examples:
        >>> from types import SimpleNamespace as N
        >>> m = N(clean=-0.1, all_cut=14.0, route_cut=8.4, sufficiency=0.55, patching=0.8,
        ...       null_cuts=[0.1, 0.3], null_patching=[0.01, -0.02], corrupted_gap=5.1,
        ...       reviewer_patching={"a": 0.6}, reviewer_null_patching={"a": [0.0]})
        >>> v = N(status="VERIFIED", gates={"necessity": True}, measures=m)
        >>> row = gate_row(N(verdict=v), "M", 3)
        >>> row["status"], row["necessity"], row["seed"]
        ('VERIFIED', 0.6, 3)
    """
    v, m = record.verdict, record.verdict.measures
    return {
        "table": "gate", "model": model, "seed": seed, "status": str(v.status),
        "gates": dict(v.gates), "clean": round(m.clean, 4), "all_cut": round(m.all_cut, 4),
        "route_cut": round(m.route_cut, 4),
        "necessity": round(m.route_cut / m.all_cut, 4) if m.all_cut else None,
        "null_cuts": [round(c, 4) for c in m.null_cuts], "sufficiency": round(m.sufficiency, 4),
        "corrupted_gap": round(m.corrupted_gap, 4),
        "patching": None if m.patching is None else round(m.patching, 4),
        "null_patching": [None if p is None else round(p, 4) for p in m.null_patching],
        "reviewer_patching": {k: None if p is None else round(p, 4)
                              for k, p in m.reviewer_patching.items()},
        "reviewer_null_patching": {k: [None if p is None else round(p, 4) for p in ps]
                                   for k, ps in m.reviewer_null_patching.items()},
    }  # fmt: skip


def prompt_rows(record, model: str) -> list[dict]:
    """One ``prompt`` row per control and probe: whether it came out as the claim says.

    Args:
        record: The answer record that ``review`` returned.
        model: The model it measured.

    Examples:
        >>> from types import SimpleNamespace as N
        >>> m = N(controls={"x": True}, probes={"y": False}, top_tokens={"x": " a", "y": " b"})
        >>> [r["kind"] for r in prompt_rows(N(verdict=N(measures=m)), "M")]
        ['control', 'probe']
    """
    m = record.verdict.measures
    return [{"table": "prompt", "model": model, "kind": kind, "prompt": prompt,
             "as_claimed": ok, "top": m.top_tokens.get(prompt)}
            for kind, found in (("control", m.controls), ("probe", m.probes))
            for prompt, ok in found.items()]  # fmt: skip


@click.command()
@click.option("--record", "record_file", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path))  # fmt: skip
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--corpus", default="corpus.jsonl", type=click.Path(path_type=Path))
@click.option("--device", default="cpu", show_default=True)
def main(record_file: Path, out: Path, corpus: Path, device: str) -> None:  # pragma: no cover
    """Review the record's claim and probes on the base and the instruct model.

    Args:
        record_file: The VERIFIED answer record.
        out: The JSONL file to write.
        corpus: The corpus of the unit means.
        device: The device of each model.
    """
    from openinterpretability.answer_review import review
    from openinterpretability.cli import _load_runner
    from openinterpretability.core.answer_records import AnswerRecord
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import unit_means

    old = AnswerRecord.model_validate_json(record_file.read_text(encoding="utf-8"))
    base = old.claim.model_name
    seeds = (old.verdict.measures.null_seed, *SEEDS)
    texts = read_corpus(corpus)[0]
    rows = []
    for model in (base, f"{base}-Instruct"):
        runner = _load_runner(model, device, "float32", None)
        means = unit_means(runner, model, texts, MEAN_TOKENS)
        claim = old.claim.model_copy(update={"model_name": model, "model_revision": None})
        for i, seed in enumerate(seeds):
            record = review(runner, means, claim, old.probes, old.verdict.policy, seed=seed,
                            texts=texts)  # fmt: skip
            rows.append(gate_row(record, model, seed))
            rows += prompt_rows(record, model) if i == 0 else []
            click.echo(json.dumps({k: v for k, v in rows[-1].items() if "null" not in k}))
        del runner
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
