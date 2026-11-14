"""The untrained-model null (issue #3202, ``docs/analyses/untrained-null.md``).

Does the protocol verify units of a model that learned nothing? Each pair is one GPT-2
small positive pole, audited twice. One audit uses the trained weights. The other uses a
randomly initialised copy of the same architecture (``from_config``, seed 0). Both arms
run the whole pipeline: scan, proposer, reviewer and the measured verdict. Units are shuffled under
opaque names, and every prompt names GPT-2, so neither agent is told the arm.

    uv run python scripts/untrained_null.py plan --pairs 20 --out $S/plan.json
    uv run python scripts/untrained_null.py scan --plan $S/plan.json        # -> ev.prompt.md
    # a neuron-proposer answers each ev.hypothesis.json, measuring only with `strengths`
    uv run python scripts/untrained_null.py submit --plan $S/plan.json --agent-model ...
    # a neuron-reviewer answers each review.probes.json
    uv run python scripts/untrained_null.py measure --plan $S/plan.json --agent-model ...
    uv run python scripts/untrained_null.py report --plan $S/plan.json > rows.jsonl

``$S`` should be outside the repository, because the plan says which unit is in which arm.
Proposers measure with ``scripts/measure_unit.py``, whose name says nothing of the study.
"""

from __future__ import annotations

import functools
import json
import random
import sys
from pathlib import Path

import click

import openinterp.cli  # noqa: F401  (first: the CLI modules import each other in its order)

sys.path.insert(0, str(Path(__file__).parent))

MODEL = "openai-community/gpt2"
ARMS = ("trained", "untrained")
LAYERS, WIDTH = 12, 3072
INIT_SEED = 0  # the untrained arm's weights
CORPUS = Path("corpus.jsonl")
MAX_TOKENS = 200_000  # the scan sample of protocol 0.7 and later
N_PROBES = 30
# Outside the repo, so a proposer's prompt need not name the file that holds the arms.
DEFAULT_PLAN = Path.home() / ".cache" / "openinterp" / "p3202" / "plan.json"


@functools.cache
def runner(arm: str, device: str = "auto"):
    """The model runner of ``arm``: GPT-2's trained weights, or fresh ones from INIT_SEED."""
    from default_on_poles import random_init_runner

    from openinterp.cli import _load_runner

    trained = _load_runner(MODEL, device, "float32")
    if arm == "trained":
        return trained
    return random_init_runner(MODEL, trained.tokenizer, INIT_SEED, trained.device)


def window() -> int | None:
    """The scan window that pyproject.toml configures for GPT-2, as the registry's audits use."""
    from openinterp.planner import configured_windows

    return configured_windows().get(MODEL)


def load_plan(path: Path) -> dict:
    """The plan file as a dict."""
    return json.loads(path.read_text())


def pending(plan: dict, names, needs: str, makes: tuple[str, ...]) -> list[dict]:
    """The units named (default: all) that have ``needs`` and none of ``makes`` yet."""
    units = [u for u in plan["units"] if not names or u["name"] in names]
    done = [u for u in units if any((Path(u["dir"]) / m).exists() for m in makes)]
    ready = [u for u in units if not needs or (Path(u["dir"]) / needs).exists()]
    return [u for u in ready if names or u not in done]


@click.group()
def cli() -> None:
    """The untrained-model null (docs/analyses/untrained-null.md)."""


@cli.command()
@click.option("--pairs", default=20, show_default=True)
@click.option("--seed", default=3202, show_default=True)
@click.option("--work-dir", default="exchange/p3202", type=click.Path(path_type=Path))
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def plan(pairs, seed, work_dir, out):
    """Draw (layer, neuron) pairs, uniform layer then neuron; shuffle units into opaque names."""
    rng = random.Random(seed)
    draws = sorted({(rng.randrange(LAYERS), rng.randrange(WIDTH)) for _ in range(pairs)})
    units = [{"arm": a, "layer": la, "neuron": n} for la, n in draws for a in ARMS]
    rng.shuffle(units)
    for i, unit in enumerate(units):
        unit |= {"name": f"u{i:02d}", "dir": str(work_dir / f"u{i:02d}")}
    out.write_text(json.dumps({"seed": seed, "init_seed": INIT_SEED, "units": units}, indent=2))
    click.echo(f"Wrote {len(units)} units to {out}")


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--unit", "names", multiple=True, help="Default: every unit not yet done.")
def scan(plan_file, names):
    """Scan units and write their proposer prompts (ev.prompt.md), or outcome.json on a stop."""
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
            evidence = scan_evidence(runner(unit["arm"]), request)
        except Unauditable as exc:
            outcome = exc.draw.outcome if exc.draw else "unauditable"
            (out / "outcome.json").write_text(json.dumps({"outcome": outcome}))
            click.echo(f"{unit['name']}: {outcome}")
            continue
        write_evidence(evidence, out / "ev.json")


@cli.command()
@click.argument("unit_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("texts", nargs=-1, required=True)
@click.option(
    "--plan",
    "plan_file",
    default=DEFAULT_PLAN,
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
)
def strengths(unit_dir, texts, plan_file):
    """The proposer's measuring tool: each text's per-token activations and its peak.

    The peak is a fraction of the unit's own scale, from its ev.json. So it does not
    say which arm the unit is in. It runs on the CPU.
    """
    from openinterp.core.schemas import ScanEvidence
    from openinterp.harness.units import unit_of

    units = load_plan(plan_file)["units"]
    arm = next(u["arm"] for u in units if Path(u["dir"]).resolve() == unit_dir.resolve())
    evidence = ScanEvidence.model_validate_json((unit_dir / "ev.json").read_text())
    skip = bool(evidence.first_token_floor)
    unit = unit_of(evidence.identifier, None, None, skip)
    scale = evidence.top_examples[0].max_activation
    run = runner(arm, "cpu")
    for text in texts:
        values = unit.strengths(run, text)[0].tolist()
        tokens = [run.tokenizer.decode([i]) for i in run.tokenizer(text)["input_ids"]]
        peak = max(values[int(skip) :], default=0.0)
        click.echo(f"{peak / scale:.3f} of scale: {text!r}")
        click.echo(
            "   " + " ".join(f"{t!r}:{v / scale:.2f}" for t, v in zip(tokens, values, strict=True))
        )


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--unit", "names", multiple=True, help="Default: every unit not yet done.")
@click.option("--agent-model", required=True, help="The proposer's model, as recorded.")
def submit(plan_file, names, agent_model):
    """Build each submission on its arm's weights, and write the reviewer's prompt."""
    from openinterp.agents.proposer import HypothesisDraft, ProposerAgent
    from openinterp.cli_review import write_review_prompt
    from openinterp.cli_submission import build_submission
    from openinterp.core.schemas import ScanEvidence

    for unit in pending(load_plan(plan_file), names, "ev.hypothesis.json", ("submission.json",)):
        out = Path(unit["dir"])
        evidence = ScanEvidence.model_validate_json((out / "ev.json").read_text())
        draft = HypothesisDraft.model_validate_json((out / "ev.hypothesis.json").read_text())
        run = runner(unit["arm"])
        submission = build_submission(ProposerAgent(agent_model), draft, run, evidence)
        (out / "submission.json").write_text(submission.model_dump_json(indent=2))
        write_review_prompt(submission, out / "review.prompt.md", N_PROBES, False)


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--unit", "names", multiple=True, help="Default: every unit not yet done.")
@click.option("--agent-model", required=True, help="The reviewer's model, as recorded.")
def measure(plan_file, names, agent_model):
    """Run each review's probes on its arm and decide (result.json), as today's policy does."""
    from null_calibration import review_probes

    from openinterp.agents.reviewer import ProbeSet
    from openinterp.core.schemas import NeuronAuditSubmission

    for unit in pending(load_plan(plan_file), names, "review.probes.json", ("result.json",)):
        out = Path(unit["dir"])
        sub = NeuronAuditSubmission.model_validate_json((out / "submission.json").read_text())
        try:
            probes = ProbeSet.model_validate_json((out / "review.probes.json").read_text()).probes
        except ValueError as exc:  # left for the reviewer to fix; the others go ahead
            click.echo(f"{unit['name']}: invalid probes, skipped: {exc}", err=True)
            continue
        result = review_probes(sub, runner(unit["arm"]), probes, agent_model)
        (out / "result.json").write_text(json.dumps(result, indent=2))
        click.echo(f"{unit['name']}: {result['outcome']}")


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
def report(plan_file):
    """One JSON line per unit: its arm, its outcome and the hypothesis."""
    for unit in load_plan(plan_file)["units"]:
        out = Path(unit["dir"])
        row = {k: unit[k] for k in ("name", "arm", "layer", "neuron")}
        for file in ("outcome.json", "result.json"):
            if (out / file).exists():
                row |= json.loads((out / file).read_text())
        if (out / "submission.json").exists():
            sub = json.loads((out / "submission.json").read_text())
            row["explanation"] = sub["natural_language_explanation"]
        click.echo(json.dumps(row))


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(at least ``a`` successes in row 1) for the 2×2 table [[a, b], [c, d]], margins fixed."""
    from math import comb

    n1, k, n = a + b, a + c, a + b + c + d
    return sum(comb(k, i) * comb(n - k, n1 - i) for i in range(a, min(n1, k) + 1)) / comb(n, n1)


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
def summary(plan_file):
    """Per arm: units, scan stops, VERIFIED and probe-verified counts, and the Fisher test."""
    from null_calibration import wilson

    arms: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for unit in load_plan(plan_file)["units"]:
        out = Path(unit["dir"])
        result = out / "result.json"
        row = json.loads(result.read_text()) if result.exists() else {"outcome": "UNMEASURED"}
        if (out / "outcome.json").exists():
            row = {"outcome": "STOPPED"}
        arms[unit["arm"]].append(row)
    table = {}
    for arm, rows in arms.items():
        measured = [r for r in rows if r["outcome"] not in ("STOPPED", "UNMEASURED")]
        verified = sum(r["outcome"] == "VERIFIED" for r in measured)
        probe = sum(bool(r.get("probe_verified")) for r in measured)
        accuracy = [
            r["balanced_accuracy"] for r in measured if r.get("balanced_accuracy") is not None
        ]
        table[arm] = {
            "units": len(rows),
            "measured": len(measured),
            "verified": verified,
            "verified_interval": wilson(verified, len(measured)),
            "probe_verified": probe,
            "mean_balanced_accuracy": sum(accuracy) / len(accuracy) if accuracy else None,
            "outcomes": {
                o: sum(r["outcome"] == o for r in rows) for o in {r["outcome"] for r in rows}
            },
        }
    t, u = table["trained"], table["untrained"]
    table["fisher_p_trained_more"] = fisher_one_sided(
        t["verified"], t["measured"] - t["verified"], u["verified"], u["measured"] - u["verified"]
    )
    click.echo(json.dumps(table, indent=1))


if __name__ == "__main__":
    cli()
