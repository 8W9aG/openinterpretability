"""Ground-truth calibration (issue #3201, ``docs/analyses/ground-truth-calibration.md``).

Does the pipeline's explanation name a neuron's role when the role is known by
construction? Each plant is a GPT-2 small neuron whose input weights are rewritten as a
logistic probe for a label rule over tokens. Its output row is kept, so its vocabulary
lenses look like any neuron's. Every plant lives in one model copy. A plant is audited
only if, on held-out documents, it passes a gate set before any audit.

    uv run python scripts/ground_truth.py plant --out $S/plan.json   # fit, install, gate
    uv run python scripts/ground_truth.py scan --plan $S/plan.json   # -> ev.prompt.md
    # a neuron-proposer answers each ev.hypothesis.json, measuring only with measure_unit.py
    uv run python scripts/ground_truth.py submit --plan $S/plan.json --agent-model ...
    # a neuron-reviewer answers each review.probes.json
    uv run python scripts/ground_truth.py measure --plan $S/plan.json --agent-model ...
    uv run python scripts/ground_truth.py report --plan $S/plan.json > rows.jsonl

``$S`` must be outside the repository, because the plan names each unit's label rule.
"""

from __future__ import annotations

import functools
import json
import random
import sys
from pathlib import Path

import click
import torch

import openinterp.cli  # noqa: F401  (first: the CLI modules import each other in its order)

sys.path.insert(0, str(Path(__file__).parent))

from untrained_null import CORPUS, MAX_TOKENS, MODEL, load_plan, pending  # noqa: E402

PLAN_DIR = Path.home() / ".cache" / "openinterp" / "p3201"
DEFAULT_PLAN = PLAN_DIR / "plan.json"
WEIGHTS = "plants.pt"  # next to the plan: each plant's input column and bias
SEED = 3201
TARGET_LOGIT = 4.0  # the median pre-activation on labelled tokens, well into GELU's range
GATE = {"precision": 0.8, "recall": 0.8}  # on held-out documents, at the protocol's line
LINE = 0.25  # the protocol's activation_threshold

MONTHS = "January February March April May June July August September October November December"
WEEKDAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday"
COLOURS = "red blue green yellow black white orange purple pink brown grey gray"
NUMBERS = "one two three four five six seven eight nine ten eleven twelve"
ANIMALS = "dog dogs cat cats horse horses bird birds fish cow cows pig pigs sheep mouse lion"
DRINKS = "coffee tea beer wine water milk juice"
BODY = "hand hands head eyes eye face arm arms leg legs foot feet heart"
TITLES = ("Mr", "Mrs", "Dr", "Ms")

# (rule name, layer, description the judge sees). Layers spread over GPT-2's 12.
PLANTS = [
    ("months", 0, "the name of a month (January to December)"),
    ("weekdays", 1, "the name of a day of the week"),
    ("colours", 2, "a colour word (red, blue, green and so on)"),
    ("numbers", 3, "a number word from one to twelve"),
    ("animals", 4, "the name of a common animal (dog, cat, horse, bird and so on)"),
    ("brackets", 5, "a closing bracket token: ')' or ']'"),
    ("after_title", 6, "the token right after the title Mr, Mrs, Dr or Ms (the name)"),
    ("in_quotes", 7, "a token inside double quotation marks"),
    ("line_start", 8, "the first token of a line, right after a newline"),
    ("weekdays_or_drinks", 9, "a day of the week, or a drink (coffee, tea, beer, wine ...)"),
    ("colours_or_body", 10, "a colour word, or a body part (hand, head, eyes, face ...)"),
    ("months_deep", 11, "the name of a month (January to December)"),
]


def word_ids(tokenizer, words: str) -> set[int]:
    """The single-token ids of each word, with and without a leading space, either case."""
    ids = set()
    for w in words.split():
        for v in {w, w.lower(), w.capitalize()}:
            for form in (v, " " + v):
                t = tokenizer.encode(form)
                if len(t) == 1:
                    ids.add(t[0])
    return ids


def label(rule: str, tokenizer, ids: list[int]) -> list[bool]:
    """Whether each position of ``ids`` matches ``rule``. This is the plant's ground truth."""
    texts = [tokenizer.decode([i]) for i in ids]
    sets = {
        "months": MONTHS,
        "months_deep": MONTHS,
        "weekdays": WEEKDAYS,
        "colours": COLOURS,
        "numbers": NUMBERS,
        "animals": ANIMALS,
        "weekdays_or_drinks": WEEKDAYS + " " + DRINKS,
        "colours_or_body": COLOURS + " " + BODY,
    }
    if rule in sets:
        members = word_ids(tokenizer, sets[rule])
        return [i in members for i in ids]
    if rule == "brackets":
        return [t.strip() in (")", "]") for t in texts]
    if rule == "after_title":
        return after_title(texts)
    if rule == "in_quotes":
        return in_quotes(texts)
    if rule == "line_start":
        return [i > 0 and "\n" in texts[i - 1] and texts[i].strip() != "" for i in range(len(ids))]
    raise ValueError(rule)


def after_title(texts: list[str]) -> list[bool]:
    """The token right after Mr, Mrs, Dr or Ms, skipping a full stop after the title."""
    out = [False] * len(texts)
    for i in range(1, len(texts)):
        prev = texts[i - 1].strip()
        if (
            prev in TITLES
            and texts[i].strip() != "."
            or prev == "."
            and i > 1
            and texts[i - 2].strip() in TITLES
        ):
            out[i] = True
    return out


def in_quotes(texts: list[str]) -> list[bool]:
    """Tokens between double quotation marks, straight or curly, without the marks."""
    out, inside = [], False
    for t in texts:
        quotes = t.count('"') + t.count("“") + t.count("”")
        out.append(inside and quotes == 0)
        if "“" in t:
            inside = True
        elif "”" in t:
            inside = False
        elif quotes % 2:
            inside = not inside
    return out


def documents(tokenizer) -> list[list[int]]:
    """The corpus's documents as token ids, cut to GPT-2's window of 1,024."""
    from openinterp.evidence import read_corpus

    return [ids[:1024] for t in read_corpus(CORPUS)[0] if len(ids := tokenizer.encode(t)) > 1]


@torch.no_grad()
def mlp_inputs(runner, docs, layer: int) -> torch.Tensor:
    """``[tokens, d_model]``: what ``layer``'s MLP reads, first tokens left out."""
    rows = []

    def record(_module, args):
        rows.append(args[0][0, 1:].cpu())

    handle = runner.mlp(layer).register_forward_pre_hook(record)
    try:
        for ids in docs:
            runner.forward({"input_ids": torch.tensor([ids], device=runner.device)})
    finally:
        handle.remove()
    return torch.cat(rows)


def install(runner, plants: list[dict], weights: dict) -> None:
    """Write each plant's input column and bias into the model, in place."""
    for p in plants:
        fc = runner.mlp(p["layer"]).c_fc  # GPT-2's Conv1D: weight is [d_model, d_mlp]
        w, b = weights[p["rule"]]
        fc.weight.data[:, p["neuron"]] = w.to(fc.weight)
        fc.bias.data[p["neuron"]] = b.to(fc.bias)


@functools.cache
def planted_runner(plan_file: Path = DEFAULT_PLAN, device: str = "auto"):
    """GPT-2 small with every plant of ``plan_file`` installed."""
    from openinterp.cli import _load_runner

    run = _load_runner(MODEL, device, "float32")
    plan = load_plan(plan_file)
    install(run, plan["plants"], torch.load(plan_file.parent / WEIGHTS))
    return run


def gate_scores(act: torch.Tensor, y: torch.Tensor, test: torch.Tensor, scale: float) -> dict:
    """Precision, recall and active share on the ``test`` tokens, at the protocol's line."""
    fired = act >= LINE * scale
    tp = int((fired & y)[test].sum())
    return {
        "precision": tp / max(int(fired[test].sum()), 1),
        "recall": tp / max(int(y[test].sum()), 1),
        "active_share": float(fired[test].float().mean()),
        "positives": int(y.sum()),
    }


def unstudied(layer: int, rng: random.Random) -> int:
    """A random GPT-2 neuron of ``layer`` with no experiment in the registry."""
    studied = Path("wiki/experiments/openai-community--gpt2") / f"L{layer}"
    taken = (
        {int(d.name[1:].split("-")[0]) for d in studied.glob("N*")} if studied.exists() else set()
    )
    while (n := rng.randrange(3072)) in taken:
        pass
    return n


def fit_plants(run, docs, labels: dict, fit: torch.Tensor, rng: random.Random):
    """Fit each plant's probe on the ``fit`` tokens. Return the plants and their weights."""
    from openinterp.harness.probing import fit_logistic

    plants, weights = [], {}
    for rule, layer, description in PLANTS:
        y, x = labels[rule], mlp_inputs(run, docs, layer)
        w, b = fit_logistic(x[fit], y[fit])
        gain = TARGET_LOGIT / float((x[fit][y[fit]] @ w + b).median())
        weights[rule] = (w * gain, b * gain)
        neuron = unstudied(layer, rng)
        plants.append({"rule": rule, "layer": layer, "neuron": neuron, "description": description})
        click.echo(f"fitted {rule} at L{layer}: {int(y.sum())} labelled tokens")
    return plants, weights


def gate_plants(run, docs, labels: dict, test: torch.Tensor, plants: list[dict]) -> None:
    """Score each installed plant on the ``test`` tokens, and mark whether it passes the gate.

    It runs on the model with every plant installed, which is the model that the audit uses.
    """
    for p in plants:
        x = mlp_inputs(run, docs, p["layer"])
        fc = run.mlp(p["layer"]).c_fc
        pre = x @ fc.weight[:, p["neuron"]].cpu() + fc.bias[p["neuron"]].cpu()
        act = torch.nn.functional.gelu(pre, approximate="tanh")
        p["gate"] = gate_scores(act, labels[p["rule"]], test, float(act.max()))
        p["passed"] = all(p["gate"][m] >= v for m, v in GATE.items())
        click.echo(f"{p['rule']}: {p['gate']} -> {'pass' if p['passed'] else 'DROP'}")


@click.group()
def cli() -> None:
    """Ground-truth calibration (docs/analyses/ground-truth-calibration.md)."""


@cli.command()
@click.option("--out", default=DEFAULT_PLAN, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--work-dir", default="exchange/p3201", type=click.Path(path_type=Path))
def plant(out, work_dir):
    """Fit each plant on even documents, install them all, and gate them on odd documents."""
    from openinterp.cli import _load_runner

    rng = random.Random(SEED)
    run = _load_runner(MODEL, "auto", "float32")
    docs = documents(run.tokenizer)
    doc_of = torch.cat([torch.full((len(ids) - 1,), d) for d, ids in enumerate(docs)])
    fit = doc_of % 2 == 0
    labels = {
        r: torch.cat([torch.tensor(label(r, run.tokenizer, ids)[1:]) for ids in docs])
        for r, _, _ in PLANTS
    }
    plants, weights = fit_plants(run, docs, labels, fit, rng)
    install(run, plants, weights)
    gate_plants(run, docs, labels, ~fit, plants)
    audited = [p for p in plants if p["passed"]]
    rng.shuffle(audited)
    units = [
        {"name": f"u{i:02d}", "dir": str(work_dir / f"u{i:02d}")}
        | {k: p[k] for k in ("rule", "layer", "neuron")}
        for i, p in enumerate(audited)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out.parent / WEIGHTS)
    out.write_text(
        json.dumps({"seed": SEED, "gate": GATE, "plants": plants, "units": units}, indent=2)
    )
    click.echo(f"Wrote {len(units)} of {len(plants)} plants to {out}")


@cli.command()
@click.option(
    "--plan", "plan_file", default=DEFAULT_PLAN, type=click.Path(exists=True, path_type=Path)
)
@click.option("--unit", "names", multiple=True, help="Default: every unit not yet done.")
def scan(plan_file, names):
    """Scan units and write their proposer prompts (ev.prompt.md), or outcome.json on a stop."""
    from untrained_null import window

    from openinterp.cli_evidence import write_evidence
    from openinterp.evidence import ScanRequest, Unauditable, scan_evidence

    for unit in pending(load_plan(plan_file), names, "", ("ev.json", "outcome.json")):
        out = Path(unit["dir"])
        out.mkdir(parents=True, exist_ok=True)
        request = ScanRequest(
            model_name=MODEL,
            layer=unit["layer"],
            neuron=unit["neuron"],
            pole="positive",
            corpus=CORPUS,
            max_tokens=MAX_TOKENS,
            dtype="float32",
            window_tokens=window(),
        )
        try:
            evidence = scan_evidence(planted_runner(plan_file), request)
        except Unauditable as exc:
            outcome = exc.draw.outcome if exc.draw else "unauditable"
            (out / "outcome.json").write_text(json.dumps({"outcome": outcome}))
            click.echo(f"{unit['name']}: {outcome}")
            continue
        write_evidence(evidence, out / "ev.json")
        click.echo(f"{unit['name']}: scanned")


@cli.command()
@click.argument("unit_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("texts", nargs=-1, required=True)
@click.option(
    "--plan", "plan_file", default=DEFAULT_PLAN, type=click.Path(exists=True, path_type=Path)
)
def strengths(unit_dir, texts, plan_file):
    """The proposer's measuring tool: each text's per-token activations and its peak."""
    from untrained_null import print_strengths

    print_strengths(unit_dir, texts, planted_runner(plan_file, "cpu"))


@cli.command()
@click.option(
    "--plan", "plan_file", default=DEFAULT_PLAN, type=click.Path(exists=True, path_type=Path)
)
@click.option("--unit", "names", multiple=True, help="Default: every unit not yet done.")
@click.option("--agent-model", required=True, help="The proposer's model, as recorded.")
def submit(plan_file, names, agent_model):
    """Build each submission on the planted model, and write the reviewer's prompt."""
    from untrained_null import build_and_write

    for unit in pending(load_plan(plan_file), names, "ev.hypothesis.json", ("submission.json",)):
        build_and_write(Path(unit["dir"]), planted_runner(plan_file), agent_model)


@cli.command()
@click.option(
    "--plan", "plan_file", default=DEFAULT_PLAN, type=click.Path(exists=True, path_type=Path)
)
@click.option("--unit", "names", multiple=True, help="Default: every unit not yet done.")
@click.option("--agent-model", required=True, help="The reviewer's model, as recorded.")
def measure(plan_file, names, agent_model):
    """Run each review's probes on the planted model and decide (result.json)."""
    from untrained_null import measure_unit_dir

    for unit in pending(load_plan(plan_file), names, "review.probes.json", ("result.json",)):
        outcome = measure_unit_dir(Path(unit["dir"]), planted_runner(plan_file), agent_model)
        click.echo(f"{unit['name']}: {outcome}")


@cli.command()
@click.option(
    "--plan", "plan_file", default=DEFAULT_PLAN, type=click.Path(exists=True, path_type=Path)
)
def report(plan_file):
    """One JSON line per plant: its rule, gate, outcome, explanation and judgement."""
    plan = load_plan(plan_file)
    units = {u["rule"]: u for u in plan["units"]}
    for p in plan["plants"]:
        row = {k: p[k] for k in ("rule", "layer", "neuron", "description", "gate", "passed")}
        if (unit := units.get(p["rule"])) is not None:
            out = Path(unit["dir"])
            row["name"] = unit["name"]
            for file in ("outcome.json", "result.json", "judgement.json"):
                if (out / file).exists():
                    row |= json.loads((out / file).read_text())
            if (out / "result-corrected.json").exists():  # see the page's label check
                corrected = json.loads((out / "result-corrected.json").read_text())
                row["corrected_outcome"] = corrected["outcome"]
                row["corrected_probe"] = corrected["corrected_probe"]
            if (out / "submission.json").exists():
                sub = json.loads((out / "submission.json").read_text())
                row["explanation"] = sub["natural_language_explanation"]
        click.echo(json.dumps(row))


@cli.command()
@click.option(
    "--plan", "plan_file", default=DEFAULT_PLAN, type=click.Path(exists=True, path_type=Path)
)
def summary(plan_file):
    """Counts with Wilson intervals: gate, recovery, VERIFIED, and VERIFIED by recovery."""
    from null_calibration import wilson

    plan = load_plan(plan_file)
    rows = []
    for unit in plan["units"]:
        out = Path(unit["dir"])
        rows.append(
            json.loads((out / "result.json").read_text())
            | json.loads((out / "judgement.json").read_text())
        )
    n = len(rows)
    verified = [r for r in rows if r["outcome"] == "VERIFIED"]
    table = {
        "plants": len(plan["plants"]),
        "passed_gate": n,
        "ratings": {k: sum(r["rating"] == k for r in rows) for k in ("match", "partial", "miss")},
        "match_interval": wilson(sum(r["rating"] == "match" for r in rows), n),
        "verified": len(verified),
        "verified_interval": wilson(len(verified), n),
        "verified_by_rating": {
            k: sum(r["rating"] == k for r in verified) for k in ("match", "partial", "miss")
        },
        "verified_miss_interval": wilson(
            sum(r["rating"] == "miss" for r in verified), len(verified)
        ),
    }
    click.echo(json.dumps(table, indent=1))


if __name__ == "__main__":
    cli()
