"""Dreams against VERIFIED explanations: three arms and a review arm (issue #3779).

A follow-up of ``dream_audit.py`` (#3228, docs/analyses/dreaming.md). It takes a seeded
sample of VERIFIED positive poles of each model. EPO
(:mod:`openinterpretability.harness.dreaming`) runs on each at the paper's k = 512 and
300 steps, from two seeds, in three arms:

- ``max``: dreams that fire. A miss is one that the judge, applying the explanation, calls
  inactive.
- ``min``: it starts from up to two of the record's positive holdouts that fired. Each
  is cut to a window around its peak token, which is frozen. Then the pole is driven
  down. A false positive is a silenced prompt that the judge calls active.
- ``hit``: dreams aimed at the record's active line. The judge's calls show how sharp the
  explanation's boundary is.

Silent corpus windows are the controls, as in #3228. One detection prompt per pole holds
every arm's texts, shuffled, for an ``explanation-judge`` agent. ``review`` adds each
pole's firing max-dreams to the record's own probes. Each expects what the judge said
the explanation predicts. Then it measures the verdict again. It is reported, not gated.

    uv run python scripts/dream_arms.py prepare --device mps --dir exchange/dream-arms
    # one explanation-judge agent per exchange/dream-arms/*.prompt.md
    uv run python scripts/dream_arms.py review --dir exchange/dream-arms
    uv run python scripts/dream_arms.py report --dir exchange/dream-arms \\
        --out docs/analyses/dreaming-arms.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))
from dream_audit import silent_windows, wilson  # noqa: E402

MODELS = ("openai-community/gpt2", "EleutherAI/pythia-70m")
POLES = 20
SEEDS = (0, 1)
TOP_K, STEPS = 512, 300  # the paper's defaults
STARTS = 2  # positive holdouts per pole that the min arm starts from
HIT_BAND = 0.10  # a hit dream is within this share of the active line
BEFORE, AFTER = 20, 3  # tokens kept around a holdout's peak for the min arm


def sample_poles(wiki_dir: Path, model: str, n: int, seed: int) -> list:
    """A seeded sample of the lead records of ``model``'s VERIFIED positive poles.

    Args:
        wiki_dir: The wiki folder.
        model: The model's name.
        n: How many poles.
        seed: The sample's seed.

    Returns:
        Up to ``n`` audit records, sorted by slug before sampling.
    """
    from openinterpretability.builder.tree import build_tree
    from openinterpretability.builder.wiki import load_archive
    from openinterpretability.core.errata import load_errata

    found = [
        standing.lead.record
        for node in build_tree(load_archive(wiki_dir), load_errata(wiki_dir))
        if node.name == model
        for layer in node.layers
        for unit in layer.units
        for side, standing in unit.standings.items()
        if unit.is_neuron and standing.verified and side == "positive"
    ]
    found.sort(key=lambda r: r.submission.identifier.slug)
    return random.Random(seed).sample(found, min(n, len(found)))


def holdout_starts(record, threshold: float) -> list[str]:
    """The record's positive holdouts that fired, strongest first, at most :data:`STARTS`.

    Args:
        record: An audit record.
        threshold: The record's active line.

    Returns:
        The holdouts' texts.
    """
    fired = [
        r
        for r in record.verdict.adversarial_test_results
        if r.kind == "positive_holdout" and (r.activation or 0) >= threshold
    ]
    fired.sort(key=lambda r: -(r.activation or 0))
    return [r.prompt for r in fired[:STARTS]]


def arm_dreams(runner, record, threshold: float) -> list[dict]:
    """Every arm's kept dreams for one pole, as items for the judge's prompt.

    Args:
        runner: The model, with hooks.
        record: The pole's lead record.
        threshold: The record's active line.

    Returns:
        One item per distinct kept text, with its arm, strength and λ.
    """
    import torch

    from openinterpretability.harness.dreaming import dream

    ident = record.submission.identifier
    unit = (ident.layer, ident.neuron_idx, ident.pole)
    items: dict[str, dict] = {}

    def keep(arm: str, dreams, test) -> None:
        """Add each dream that passes ``test`` and is new."""
        for d in dreams:
            if test(d.strength) and d.text not in items:
                items[d.text] = {"text": d.text, "kind": arm, "strength": d.strength,
                                 "lam": d.lam}  # fmt: skip

    for seed in SEEDS:
        run = {"top_k": TOP_K, "steps": STEPS, "seed": seed}
        keep("max", dream(runner, *unit, **run), lambda s: s >= threshold)
        hit = dream(runner, *unit, goal="hit", target=threshold, **run)
        keep("hit", hit, lambda s: abs(s - threshold) <= HIT_BAND * threshold)
    for text in holdout_starts(record, threshold):
        ids = runner.tokenizer(text)["input_ids"]
        strengths = runner.neuron_activations(torch.tensor([ids]), *unit)[0]
        peak = 1 + int(strengths[1:].argmax()) if len(ids) > 1 else 0
        # A window around the peak: a long holdout makes a one-hot too large for the GPU.
        lo = max(0, peak - BEFORE)
        window = ids[lo : peak + AFTER + 1]
        silenced = dream(runner, *unit, goal="min", start=window, frozen=(peak - lo,),
                         top_k=TOP_K, steps=STEPS, seed=SEEDS[0])  # fmt: skip
        keep("min", silenced, lambda s: s < threshold)
    return list(items.values())


@click.group()
def cli() -> None:
    """Dream each sampled pole in three arms, write judge prompts, review and report."""


@cli.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--device", default="cpu", show_default=True)
def prepare(wiki_dir: Path, out_dir: Path, device: str) -> None:
    """Run the three arms on each pole, and write one detection prompt per pole."""
    from openinterpretability.agents.judge import DetectionAnswer, ExplanationJudge, TaskWindow
    from openinterpretability.cli import _load_runner
    from openinterpretability.core.schemas import ReviewPolicy

    out_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        runner = _load_runner(model, device, "float32", None)
        for record in sample_poles(wiki_dir, model, POLES, 0):
            sub, ident = record.submission, record.submission.identifier
            name = ident.slug.replace("/", "--")
            if (out_dir / f"{name}.key.json").exists():
                continue  # done in an earlier run
            policy = record.verdict.policy or ReviewPolicy()
            threshold = policy.activation_threshold * sub.max_activation
            items = arm_dreams(runner, record, threshold)
            count = max(sum(i["kind"] == "max" for i in items), 1)
            items += [{"text": t, "kind": "control"}
                      for t in silent_windows(runner, ident, threshold, count, 0)]  # fmt: skip
            random.Random(ident.slug).shuffle(items)
            tok = runner.tokenizer
            windows = [TaskWindow(text=i["text"], tokens=[tok.decode([t]) for t in
                                                          tok(i["text"])["input_ids"]])
                       for i in items]  # fmt: skip
            prompt = ExplanationJudge.detection_prompt(sub.natural_language_explanation, windows)
            text = ExplanationJudge.render_prompt(DetectionAnswer, prompt, out_dir / f"{name}.json")
            (out_dir / f"{name}.prompt.md").write_text(text, encoding="utf-8")
            key = {"slug": ident.slug, "model": model, "threshold": round(threshold, 4),
                   "explanation": sub.natural_language_explanation, "items": items}  # fmt: skip
            (out_dir / f"{name}.key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
            counts = {a: sum(i["kind"] == a for i in items) for a in ("max", "min", "hit")}
            click.echo(f"{ident.slug}: {counts}", err=True)


def judged(key_path: Path) -> list[dict]:
    """The key's items, each with ``judged_active`` from the judge's answer.

    Args:
        key_path: A ``<pole>.key.json`` file whose answer, ``<pole>.json``, exists.

    Returns:
        The items, in the prompt's order.
    """
    from openinterpretability.agents.judge import DetectionAnswer

    key = json.loads(key_path.read_text(encoding="utf-8"))
    answer = key_path.with_name(key_path.name.removesuffix(".key.json") + ".json")
    listed = set(DetectionAnswer.model_validate_json(answer.read_text()).active)
    return [i | {"judged_active": n in listed} for n, i in enumerate(key["items"], 1)]


def rates(items: list[dict]) -> dict:
    """Each arm's count and the share that matters for it, with a 95% Wilson interval.

    Args:
        items: Judged items from :func:`judged`, of any number of poles.

    Returns:
        ``max``: dreams missed. ``min``: silenced prompts judged active. ``hit``: dreams
        judged active. ``control``: silent windows judged active.

    >>> r = rates([{"kind": "max", "judged_active": False}, {"kind": "max", "judged_active": True}])
    >>> r["max"]["n"], r["max"]["share"]
    (2, 0.5)
    """
    out = {}
    for arm, wanted in (("max", False), ("min", True), ("hit", True), ("control", True)):
        hits = [i["judged_active"] == wanted for i in items if i["kind"] == arm]
        k, n = sum(hits), len(hits)
        out[arm] = {"n": n, "k": k, "share": round(k / n, 4) if n else None, "ci95": wilson(k, n)}
    return out


@cli.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--device", default="cpu", show_default=True)
def review(wiki_dir: Path, out_dir: Path, device: str) -> None:
    """Add each GPT-2 pole's firing dreams to its record's probes, and re-measure the verdict."""
    from null_calibration import review_probes

    from openinterpretability.agents.reviewer import AdversarialProbe, probes_from_verdict
    from openinterpretability.cli import _load_runner

    model = MODELS[0]
    runner = _load_runner(model, device, "float32", None)
    for record in sample_poles(wiki_dir, model, POLES, 0):
        sub = record.submission
        name = sub.identifier.slug.replace("/", "--")
        key_path, out = out_dir / f"{name}.key.json", out_dir / f"{name}.review.json"
        if out.exists() or not (key_path.with_name(f"{name}.json")).exists():
            continue
        own = probes_from_verdict(record.verdict)
        dreams = [AdversarialProbe(text=i["text"], expected_active=i["judged_active"],
                                   rationale="dream")
                  for i in judged(key_path) if i["kind"] == "max"]  # fmt: skip
        reviewer = record.verdict.reviewer_agent_model
        before = review_probes(sub, runner, own, reviewer)
        after = review_probes(sub, runner, own + dreams, reviewer)
        result = {"slug": sub.identifier.slug, "dreams_added": len(dreams),
                  "before": before, "after": after}  # fmt: skip
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        click.echo(f"{sub.identifier.slug}: {before['outcome']} -> {after['outcome']}", err=True)


@cli.command()
@click.option("--dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def report(out_dir: Path, out: Path) -> None:
    """Write one row per pole, one per model, and the review arm's verdict moves."""
    rows, by_model = [], {}
    for key_path in sorted(out_dir.glob("*.key.json")):
        key = json.loads(key_path.read_text(encoding="utf-8"))
        if not key_path.with_name(key_path.name.removesuffix(".key.json") + ".json").exists():
            continue
        items = judged(key_path)
        by_model.setdefault(key["model"], []).extend(items)
        rows.append({"table": "pole", "slug": key["slug"], "model": key["model"],
                     "threshold": key["threshold"], "rates": rates(items),
                     "items": items})  # fmt: skip
    summary = [{"table": "model", "model": m, "poles": sum(r["model"] == m for r in rows),
                "rates": rates(items)} for m, items in by_model.items()]  # fmt: skip
    reviews = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.review.json"))]
    moves = {"table": "review", "poles": len(reviews),
             "verified_before": sum(r["before"]["outcome"] == "VERIFIED" for r in reviews),
             "verified_after": sum(r["after"]["outcome"] == "VERIFIED" for r in reviews),
             "rows": [{"slug": r["slug"], "dreams_added": r["dreams_added"],
                       "before": r["before"]["outcome"], "after": r["after"]["outcome"],
                       "ba_before": r["before"]["balanced_accuracy"],
                       "ba_after": r["after"]["balanced_accuracy"]} for r in reviews]}  # fmt: skip
    lines = [*summary, moves, *rows]
    out.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    click.echo(json.dumps([*summary, {k: v for k, v in moves.items() if k != "rows"}], indent=1))


if __name__ == "__main__":
    cli()
