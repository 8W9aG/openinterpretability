"""Follow Pythia-70m's VERIFIED poles back through its training checkpoints (issue #3376).

Pythia publishes its training run at many steps (``revision="step<N>"``). At each step in
:data:`STEPS`, this loads the model with the registry's loader. It measures each VERIFIED
pole with the review's own code (``submission_unit`` and ``peak_strength``):

* ``specificity`` and ``sensitivity``: the pass rates of the pole's recorded hard negatives
  and holdouts. A probe is active at or above 0.25 times the pole's scale at this step.
  The scale is the record's ``max_activation``, times this step's maximum over the windows
  divided by the last step's. At the last step it is the review's own line. ``holds`` says
  whether both rates reach the record's own gates (its policy).
* ``r_final``: the Pearson correlation between the pole's peak strength over fixed corpus
  windows at this step and at the last step. It says how far the pole is formed.
* ``max``: the step's maximum peak strength over the windows.

Each step also gets the mean next-token loss on the same windows. The rows go to
``--out``; nothing is written in ``wiki/``::

    uv run python scripts/neuron_development.py --out papers/neuron-development/results.jsonl
"""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

import click
import torch

MODEL = "EleutherAI/pythia-70m"
SLUG = "EleutherAI--pythia-70m"
# Log-spaced early, where Pythia saves densely, then doubling to the end of training.
STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
         1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000)  # fmt: skip
LINE = 0.25  # the review's activation threshold, a share of the pole's maximum
BORN = 0.8  # a pole is formed from the first step after which r_final stays ≥ this and it holds
# The poles of the paper's first run (2026-11-16), reported beside the full set (#4881).
FIRST_RUN = (
    "EleutherAI--pythia-70m/L5/N674-positive",
    "EleutherAI--pythia-70m/L5/N655-positive",
    "EleutherAI--pythia-70m/L4/N566-positive",
    "EleutherAI--pythia-70m/L5/N1256-positive",
    "EleutherAI--pythia-70m/L4/N751-positive",
    "EleutherAI--pythia-70m/L2/N1181-positive",
    "EleutherAI--pythia-70m/L4/N812-positive",
    "EleutherAI--pythia-70m/L4/N216-positive",
    "EleutherAI--pythia-70m/L1/N535-positive",
)


def verified_records(wiki_dir: Path) -> list[dict]:
    """The newest record of each pole of the model, for the poles where it is VERIFIED.

    A pole whose newest record is not VERIFIED is left out, even with an older VERIFIED
    record: a later review didn't confirm it (#4881).
    """
    return [r for r in _newest(wiki_dir).values() if r["verdict"]["status"] == "VERIFIED"]


def left_out(wiki_dir: Path) -> list[str]:
    """The poles with a VERIFIED record whose newest record is not VERIFIED."""
    verified = {
        f"{p.parent.parent.name}/{p.parent.name}"
        for p in wiki_dir.glob(f"experiments/{SLUG}/L*/N*/*.json")
        if json.loads(p.read_text(encoding="utf-8"))["verdict"]["status"] == "VERIFIED"
    }
    newest = _newest(wiki_dir)
    return sorted(p for p in verified if newest[p]["verdict"]["status"] != "VERIFIED")


def _newest(wiki_dir: Path) -> dict[str, dict]:
    """Each pole's newest record, by pole, in the order of their file names."""
    newest: dict[str, tuple[str, dict]] = {}
    for path in sorted(wiki_dir.glob(f"experiments/{SLUG}/L*/N*/*.json")):
        pole = f"{path.parent.parent.name}/{path.parent.name}"
        if pole not in newest or path.name > newest[pole][0]:
            record = json.loads(path.read_text(encoding="utf-8"))
            record["_slug"] = f"{SLUG}/{pole}"  # the pole's slug, from where its record lives
            newest[pole] = (path.name, record)
    ordered = sorted(newest.items(), key=lambda kv: kv[1][0])
    return {pole: record for pole, (_, record) in ordered}


def probes_of(record: dict) -> list[tuple[str, bool]]:
    """The record's probes as (text, expected active)."""
    return [
        (t["prompt"], bool(t["expected_active"]))
        for t in record["verdict"]["adversarial_test_results"]
        if t["kind"] in ("positive_holdout", "hard_negative")
        and t.get("expected_active") is not None
    ]


def pass_rate(
    strengths: list[float | None], expected: list[bool], line: float, want: bool
) -> float:
    """The share of the measured probes that expect ``want`` and pass at ``line``."""
    got = [(s >= line) == want for s, e in zip(strengths, expected, strict=True)
           if e == want and s is not None]  # fmt: skip
    return sum(got) / len(got) if got else math.nan


def pearson(a: list[float], b: list[float]) -> float:
    """The Pearson correlation of two equal-length lists; NaN when one is constant."""
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else math.nan


def born_at(rows: list[dict]) -> int | None:
    """The first step from which ``r_final`` stays at or above BORN and the hypothesis holds."""
    for i, row in enumerate(rows):
        if all(r["r_final"] >= BORN and r["holds"] for r in rows[i:]):
            return row["step"]
    return None


@torch.no_grad()
def window_loss(runner, windows: list[str]) -> float:
    """The mean next-token loss over ``windows``."""
    total, count = 0.0, 0
    for text in windows:
        ids = runner.tokenizer(text, return_tensors="pt").input_ids.to(runner.model.device)
        if ids.shape[1] < 2:
            continue
        out = runner.model(ids, labels=ids)
        total += float(out.loss) * (ids.shape[1] - 1)
        count += ids.shape[1] - 1
    return total / count


def measure_step(step: int, records: list[dict], texts: list[str], device: str) -> tuple:
    """Each pole's strengths over the windows and probes at ``step``, and the step's loss."""
    from openinterpretability.cli import _load_runner
    from openinterpretability.core.schemas import AuditRecord
    from openinterpretability.harness.layer_major import map_layer_major, prompt_tokens
    from openinterpretability.harness.units import peak_strength, submission_unit

    runner = _load_runner(MODEL, device, "float32", f"step{step}")
    windows = texts
    out = {}
    for record in records:
        data = {k: v for k, v in record.items() if k != "_slug"}
        unit = submission_unit(AuditRecord.model_validate(data).submission)
        probe_texts = [t for t, _ in probes_of(record)]
        everything = [*windows, *probe_texts]
        strengths = map_layer_major(
            runner, partial(peak_strength, runner, unit), everything,
            tokens=prompt_tokens(runner, everything), logits=False,
        )  # fmt: skip
        out[record["_slug"]] = strengths
    return out, window_loss(runner, windows)


def measure_pole(record: dict, by_step: dict, steps: tuple, n_windows: int) -> list[dict]:
    """One pole's row at each step, from its strengths over the windows and its probes."""
    slug = record["_slug"]
    expected = [e for _, e in probes_of(record)]
    policy = record["verdict"]["policy"]
    final = [s or 0.0 for s in by_step[steps[-1]][slug][:n_windows]]
    final_top = max(final)
    pole_rows, previous = [], None
    for step in steps:
        strengths = by_step[step][slug]
        win, probe = strengths[:n_windows], strengths[n_windows:]
        here = [s or 0.0 for s in win]
        top = max((s for s in win if s is not None), default=0.0)
        scale = record["submission"]["max_activation"] * top / final_top if final_top else 0
        spec = pass_rate(probe, expected, LINE * scale, want=False)
        sens = pass_rate(probe, expected, LINE * scale, want=True)
        pole_rows.append({
            "table": "pole", "slug": slug, "step": step, "max": round(top, 5),
            "r_final": round(pearson(here, final), 4),
            # with the checkpoint before: drift against formation (#4156)
            "r_prev": None if previous is None else round(pearson(here, previous), 4),
            "specificity": round(spec, 4), "sensitivity": round(sens, 4),
            "holds": spec >= policy["min_specificity"] and sens >= policy["min_sensitivity"],
        })  # fmt: skip
        previous = here
    return pole_rows


@click.command()
@click.option("--wiki-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("wiki"))
@click.option(
    "--corpus", type=click.Path(dir_okay=False, path_type=Path), default=Path("corpus.jsonl")
)
@click.option("--windows", default=300, show_default=True, help="Fixed corpus windows.")
@click.option("--window-tokens", default=64, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--step", "steps", multiple=True, type=int, help="Default: each of STEPS.")
@click.option("--pole", "poles", multiple=True, help="Only these poles, e.g. L5/N655-positive.")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(wiki_dir, corpus, windows, window_tokens, device, steps, poles, out):
    """Measure every VERIFIED pole of Pythia-70m at each step in STEPS."""
    from datetime import UTC, datetime

    from openinterpretability.cli import _load_runner
    from openinterpretability.harness.explanation_scores import cut_windows

    records = verified_records(wiki_dir)
    if poles:
        records = [r for r in records if r["_slug"].removeprefix(f"{SLUG}/") in poles]
    selection = {"table": "selection", "date": f"{datetime.now(UTC):%Y-%m-%d}",
                 "poles": len(records), "left_out": left_out(wiki_dir)}  # fmt: skip
    docs = [json.loads(line)["text"] for line in corpus.read_text().splitlines()[-200:]]
    final = _load_runner(MODEL, "cpu", "float32", f"step{(sorted(steps) or STEPS)[-1]}")
    texts = cut_windows(final, docs, window_tokens)[:windows]
    del final
    click.echo(f"{len(records)} VERIFIED poles, {len(texts)} windows, {len(STEPS)} steps")

    steps = tuple(sorted(steps)) or STEPS
    by_step, losses = {}, {}
    for step in steps:
        by_step[step], losses[step] = measure_step(step, records, texts, device)
        click.echo(f"step {step}: loss {losses[step]:.3f}")

    rows = [selection, {"table": "first", "slugs": list(FIRST_RUN)}]
    for record in records:
        slug = record["_slug"]
        pole_rows = measure_pole(record, by_step, steps, len(texts))
        rows += pole_rows
        explanation = record["submission"]["natural_language_explanation"]
        rows.append({"table": "born", "slug": slug, "step": born_at(pole_rows),
                     "explanation": explanation})  # fmt: skip
    rows += [{"table": "loss", "step": s, "loss": round(losses[s], 5)} for s in steps]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
