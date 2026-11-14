"""Check VERIFIED explanations against fluent dreams of their pole (issue #3228).

``prepare`` takes a seeded sample of VERIFIED GPT-2 small positive poles. It runs EPO on
each (:mod:`openinterp.harness.dreaming`), and keeps each distinct dream that clears the
record's threshold. It adds as many windows of ``corpus.jsonl`` where the pole is silent.
Then it writes one detection prompt per pole, built the way ``score`` builds one, from the
record's own explanation, with the texts shuffled. An ``explanation-judge`` agent answers
each one. ``report`` reads the answers. A dream that fires, but that the judge calls
inactive by the explanation, is a misprediction: a firing case the explanation misses.

    uv run python scripts/dream_audit.py prepare --device mps --dir exchange/dreams
    # one explanation-judge agent per exchange/dreams/*.prompt.md
    uv run python scripts/dream_audit.py report --dir exchange/dreams \\
        --out docs/analyses/dreaming.jsonl
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import click

MODEL = "openai-community/gpt2"
POLES = 20
SEED = 0
CONTROL_TOKENS = 24


def sample_poles(wiki_dir: Path, n: int, seed: int) -> list:
    """A seeded sample of the standing records of VERIFIED positive poles of :data:`MODEL`."""
    from openinterp.builder.tree import build_tree
    from openinterp.builder.wiki import load_archive
    from openinterp.core.errata import load_errata

    found = []
    for node in build_tree(load_archive(wiki_dir), load_errata(wiki_dir)):
        if node.name != MODEL:
            continue
        for layer in node.layers:
            for unit in layer.units:
                for side, standing in unit.standings.items():
                    if unit.is_neuron and standing.verified and side == "positive":
                        found.append(standing.lead.record)
    found.sort(key=lambda r: r.submission.identifier.slug)
    return random.Random(seed).sample(found, min(n, len(found)))


def silent_windows(runner, ident, threshold: float, count: int, seed: int) -> list[str]:
    """``count`` short windows of the corpus where the pole stays below ``threshold``."""
    rng, out = random.Random(seed), []
    with open("corpus.jsonl", encoding="utf-8") as f:
        docs = [json.loads(line)["text"] for _, line in zip(range(400), f, strict=False)]
    rng.shuffle(docs)
    for text in docs:
        ids = runner.tokenizer(text)["input_ids"]
        if len(ids) < CONTROL_TOKENS + 1:
            continue
        start = rng.randrange(len(ids) - CONTROL_TOKENS)
        window = runner.tokenizer.decode(ids[start : start + CONTROL_TOKENS])
        peak = runner.max_pole_strength(window, ident.layer, ident.neuron_idx, ident.pole)
        if peak < threshold:
            out.append(window)
        if len(out) == count:
            return out
    return out


@click.group()
def cli() -> None:
    """Dream each sampled pole, write judge prompts, and report the answers."""


@cli.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--device", default="cpu", show_default=True)
def prepare(wiki_dir: Path, out_dir: Path, device: str) -> None:
    """Run EPO on each pole and write one detection prompt per pole."""
    from openinterp.agents.judge import DetectionAnswer, ExplanationJudge, TaskWindow
    from openinterp.cli import _load_runner
    from openinterp.core.schemas import ReviewPolicy
    from openinterp.harness.dreaming import dream

    runner = _load_runner(MODEL, device, "float32", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in sample_poles(wiki_dir, POLES, SEED):
        sub = record.submission
        ident = sub.identifier
        policy = record.verdict.policy or ReviewPolicy()
        threshold = policy.activation_threshold * sub.max_activation
        dreams = dream(runner, ident.layer, ident.neuron_idx, ident.pole, seed=SEED)
        firing = {d.text: d for d in dreams if d.strength >= threshold}
        controls = silent_windows(runner, ident, threshold, max(len(firing), 1), SEED)
        items = [{"text": t, "kind": "dream", "strength": d.strength, "lam": d.lam,
                  "cross_entropy": d.cross_entropy} for t, d in firing.items()]  # fmt: skip
        items += [{"text": t, "kind": "control"} for t in controls]
        random.Random(ident.slug).shuffle(items)
        name = ident.slug.replace("/", "--")
        windows = [
            TaskWindow(
                text=i["text"],
                tokens=[
                    runner.tokenizer.decode([t]) for t in runner.tokenizer(i["text"])["input_ids"]
                ],
            )  # fmt: skip
            for i in items
        ]
        prompt = ExplanationJudge.detection_prompt(sub.natural_language_explanation, windows)
        answer = out_dir / f"{name}.json"
        text = ExplanationJudge.render_prompt(DetectionAnswer, prompt, answer)
        (out_dir / f"{name}.prompt.md").write_text(text, encoding="utf-8")
        key = {"slug": ident.slug, "threshold": round(threshold, 4), "scale": sub.max_activation,
               "explanation": sub.natural_language_explanation, "dreams_total": len(dreams),
               "items": items}  # fmt: skip
        (out_dir / f"{name}.key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
        click.echo(f"{ident.slug}: {len(firing)} dreams fire of {len(dreams)}", err=True)


def wilson(k: int, n: int) -> list[float]:
    """The 95% Wilson interval of ``k`` successes in ``n``."""
    if n == 0:
        return [0.0, 1.0]
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


@cli.command()
@click.option("--dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def report(out_dir: Path, out: Path) -> None:
    """Read each judge's answer, and write one row per pole and a summary."""
    from openinterp.agents.judge import DetectionAnswer

    rows = []
    for key_path in sorted(out_dir.glob("*.key.json")):
        key = json.loads(key_path.read_text(encoding="utf-8"))
        answer_path = key_path.with_name(key_path.name.removesuffix(".key.json") + ".json")
        listed = set(DetectionAnswer.model_validate_json(answer_path.read_text()).active)
        items = [i | {"judged_active": n in listed} for n, i in enumerate(key["items"], 1)]
        dreams = [i for i in items if i["kind"] == "dream"]
        controls = [i for i in items if i["kind"] == "control"]
        top = max((i["strength"] for i in dreams), default=0)
        rows.append({
            "table": "pole", "slug": key["slug"], "explanation": key["explanation"],
            "scale": key["scale"], "threshold": key["threshold"],
            "dreams_total": key["dreams_total"], "dreams_firing": len(dreams),
            "mispredicted": sum(not i["judged_active"] for i in dreams),
            "controls": len(controls),
            "control_false_positives": sum(i["judged_active"] for i in controls),
            "max_dream_over_scale": round(top / key["scale"], 4),
            "items": items,
        })  # fmt: skip
    dreams = sum(r["dreams_firing"] for r in rows)
    missed = sum(r["mispredicted"] for r in rows)
    controls = sum(r["controls"] for r in rows)
    false_pos = sum(r["control_false_positives"] for r in rows)
    summary = {
        "table": "summary", "poles": len(rows), "dreams_firing": dreams, "mispredicted": missed,
        "mispredicted_share": round(missed / dreams, 4) if dreams else None,
        "mispredicted_ci95": wilson(missed, dreams),
        "poles_with_a_misprediction": sum(r["mispredicted"] > 0 for r in rows),
        "controls": controls, "control_false_positives": false_pos,
        "control_false_positive_share": round(false_pos / controls, 4) if controls else None,
    }  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in [summary, *rows]), encoding="utf-8")
    click.echo(json.dumps(summary))


if __name__ == "__main__":
    cli()
