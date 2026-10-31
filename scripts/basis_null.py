"""The rotated-basis null: do random directions in the MLP's hidden space verify as often
as native neurons? (``docs/analyses/null-calibration.md``)

Both arms are audited as latents of a ``basis`` (``DictionaryProfile``) reading the
MLP's hidden activations (``mlp_hidden``): the *identity* basis, whose latents are the
native positive poles, and a random orthogonal *rotation*. The two share one dictionary
name and one prompt, and their weights sit in directories named by a hash, so the
proposer and the reviewer can't tell which arm a unit is in. Units are drawn in pairs,
one per arm at the same layer and index.

    uv run python scripts/basis_null.py plan --pairs 30 --out $S/plan.json
    uv run python scripts/basis_null.py weights --plan $S/plan.json
    uv run python scripts/basis_null.py scan --plan $S/plan.json --unit u07   # -> ev.prompt.md
    # proposer answers ev.hypothesis.json
    uv run python scripts/basis_null.py submit --plan $S/plan.json --unit u07  # -> review.prompt.md
    # reviewer answers review.probes.json
    uv run python scripts/basis_null.py measure --plan $S/plan.json --unit u07  # -> result.json
    uv run python scripts/basis_null.py report --plan $S/plan.json

``$S`` should be outside the repository (the plan says which unit is in which arm); the
prompts go to ``--work-dir`` (default ``exchange/basis``).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import click
import torch

import openinterp.cli  # noqa: F401  (first: the CLI modules import each other in its order)

MODEL = "Qwen/Qwen2.5-0.5B"
NAME = "qwen2.5-0.5b-mlp-basis"  # one name for both arms
ARMS = ("identity", "rotation")
CORPUS = Path("corpus.jsonl")
WINDOW_TOKENS = 512


def weights_dir(arm: str, seed: int) -> Path:
    digest = hashlib.sha256(f"{arm}/{seed}".encode()).hexdigest()[:12]
    return Path.home() / ".cache" / "openinterp" / "bases" / digest


def spec(arm: str, seed: int, layers):
    from openinterp.harness.dictionaries import DictionarySpec

    return DictionarySpec(
        model_name=MODEL,
        kind="basis",
        repo=str(weights_dir(arm, seed)),
        revision=None,
        path="layer_{layer}.safetensors",
        layers=tuple(layers),
        input_site="mlp_hidden",
    )


def load_plan(path: Path) -> dict:
    return json.loads(path.read_text())


def register(plan: dict, unit: dict) -> None:
    """Point the shared dictionary name at this unit's arm."""
    from openinterp.harness.dictionaries import DICTIONARIES

    layers = sorted({u["layer"] for u in plan["units"]})
    DICTIONARIES[NAME] = spec(unit["arm"], plan["seed"], layers)


@click.group()
def cli() -> None:
    """The rotated-basis null (docs/analyses/null-calibration.md)."""


@cli.command()
@click.option("--pairs", default=30, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--layers", default=24, show_default=True)
@click.option("--width", default=4864, show_default=True)
@click.option("--work-dir", default="exchange/basis", type=click.Path(path_type=Path))
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
def plan(pairs, seed, layers, width, work_dir, out):
    """Draw unit pairs (uniform layer, then index) and shuffle them into opaque names."""
    rng = random.Random(seed)
    draws = [(rng.randrange(layers), rng.randrange(width)) for _ in range(pairs)]
    units = [{"arm": arm, "layer": la, "feature": f} for la, f in draws for arm in ARMS]
    rng.shuffle(units)
    for i, unit in enumerate(units):
        unit["name"] = f"u{i:02d}"
        unit["dir"] = str(work_dir / unit["name"])
    out.write_text(json.dumps({"seed": seed, "units": units}, indent=2))
    click.echo(f"Wrote {len(units)} units to {out}")


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
def weights(plan_file):
    """Write each arm's basis for every planned layer: W_enc = Q, W_dec = (W_down Q)ᵀ."""
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    plan = load_plan(plan_file)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    for layer in sorted({u["layer"] for u in plan["units"]}):
        w_down = model.model.layers[layer].mlp.down_proj.weight.detach()  # [d_model, d_mlp]
        for arm in ARMS:
            path = weights_dir(arm, plan["seed"]) / f"layer_{layer}.safetensors"
            if path.exists():
                continue
            q = basis(arm, w_down.shape[1], plan["seed"], layer)
            path.parent.mkdir(parents=True, exist_ok=True)
            tensors = {
                "W_enc": q.contiguous(),
                "b_enc": torch.zeros(q.shape[1]),
                "W_dec": (w_down @ q).T.contiguous(),
                "b_dec": torch.zeros(w_down.shape[0]),
            }
            save_file(tensors, str(path))
            click.echo(f"Wrote {path} ({arm}, layer {layer})")


def basis(arm: str, width: int, seed: int, layer: int) -> torch.Tensor:
    """The identity, or a Haar-random orthogonal matrix (QR of a Gaussian, signs fixed)."""
    if arm == "identity":
        return torch.eye(width)
    generator = torch.Generator().manual_seed(seed * 1000 + layer)
    q, r = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).float()


def pending(plan: dict, names, needs: str, makes: tuple[str, ...]) -> list[dict]:
    """The units named (default: all) that have ``needs`` and none of ``makes`` yet."""
    units = [u for u in plan["units"] if not names or u["name"] in names]
    done = [u for u in units if any((Path(u["dir"]) / m).exists() for m in makes)]
    ready = [u for u in units if not needs or (Path(u["dir"]) / needs).exists()]
    return [u for u in ready if names or u not in done]


def runner():
    from openinterp.cli import _load_runner

    return _load_runner(MODEL, "auto", "float32")


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--unit", "names", multiple=True, help="Default: every unit not yet done.")
def scan(plan_file, names):
    """Scan units and write their proposer prompts (ev.prompt.md), or outcome.json if
    the scan stops on them (dead or default-on)."""
    from openinterp.cli_evidence import write_evidence
    from openinterp.evidence import ScanRequest, Unauditable, scan_evidence

    plan, run = load_plan(plan_file), runner()
    for unit in pending(plan, names, "", ("ev.json", "outcome.json")):
        name = unit["name"]
        register(plan, unit)
        request = ScanRequest(
            model_name=MODEL,
            unit="feature",
            layer=unit["layer"],
            feature=unit["feature"],
            dictionary=NAME,
            corpus=CORPUS,
            window_tokens=WINDOW_TOKENS,
            max_tokens=200_000,
            dtype="float32",
        )
        out = Path(unit["dir"])
        try:
            evidence = scan_evidence(run, request)
        except Unauditable as exc:
            out.mkdir(parents=True, exist_ok=True)
            outcome = exc.draw.outcome if exc.draw else "unauditable"
            (out / "outcome.json").write_text(json.dumps({"outcome": outcome}))
            click.echo(f"{name}: {outcome}")
            continue
        write_evidence(evidence, out / "ev.json")


@cli.command()
@click.argument("unit_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("texts", nargs=-1, required=True)
def strengths(unit_dir, texts):
    """The proposer's measuring tool: each text's per-token strengths and its peak as a
    fraction of the unit's scale, read from the unit's own ev.json (so it says nothing of
    which arm the unit is in). Runs on the CPU."""
    from openinterp.cli import _load_runner
    from openinterp.core.schemas import ScanEvidence
    from openinterp.harness.units import unit_of

    evidence = ScanEvidence.model_validate_json((unit_dir / "ev.json").read_text())
    skip = bool(evidence.first_token_floor)
    unit = unit_of(evidence.identifier, None, evidence.dictionary_profile, skip)
    scale = evidence.top_examples[0].max_activation
    run = _load_runner(MODEL, "cpu", "float32")
    for text in texts:
        values = unit.strengths(run, text)[0].tolist()
        ids = run.tokenizer(text)["input_ids"]
        tokens = [run.tokenizer.decode([i]) for i in ids]
        peak = max(values[int(skip) :], default=0.0)
        click.echo(f"{peak / scale:.3f} of scale (peak {peak:.4f}): {text!r}")
        click.echo("   " + " ".join(f"{t!r}:{v:.3f}" for t, v in zip(tokens, values, strict=True)))


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--unit", "names", multiple=True, help="Default: every unit not yet done.")
@click.option("--agent-model", required=True, help="The proposer's model, as recorded.")
def submit(plan_file, names, agent_model):
    """Build each submission from its scan and the proposer's draft (measuring the
    counterfactuals) and write the reviewer's prompt (review.prompt.md)."""
    from openinterp.agents.proposer import HypothesisDraft, ProposerAgent
    from openinterp.cli_review import write_review_prompt
    from openinterp.cli_submission import build_submission
    from openinterp.core.schemas import ScanEvidence

    plan, run = load_plan(plan_file), runner()
    for unit in pending(plan, names, "ev.hypothesis.json", ("submission.json",)):
        register(plan, unit)
        out = Path(unit["dir"])
        evidence = ScanEvidence.model_validate_json((out / "ev.json").read_text())
        draft = HypothesisDraft.model_validate_json((out / "ev.hypothesis.json").read_text())
        submission = build_submission(ProposerAgent(agent_model), draft, run, evidence)
        (out / "submission.json").write_text(submission.model_dump_json(indent=2))
        write_review_prompt(submission, out / "review.prompt.md", 20, False)


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--unit", "names", multiple=True, help="Default: every unit not yet done.")
@click.option("--agent-model", required=True, help="The reviewer's model, as recorded.")
def measure(plan_file, names, agent_model):
    """Run each review's probes and decide (result.json), without causal metrics or
    wiring, which aren't gated."""
    from null_calibration import review_probes  # beside this script

    from openinterp.agents.reviewer import ProbeSet
    from openinterp.core.schemas import NeuronAuditSubmission

    plan, run = load_plan(plan_file), runner()
    for unit in pending(plan, names, "review.probes.json", ("result.json",)):
        register(plan, unit)
        out = Path(unit["dir"])
        sub = NeuronAuditSubmission.model_validate_json((out / "submission.json").read_text())
        try:
            probes = ProbeSet.model_validate_json((out / "review.probes.json").read_text()).probes
        except ValueError as exc:  # left for the reviewer to fix; the others go ahead
            click.echo(f"{unit['name']}: invalid probes, skipped: {exc}", err=True)
            continue
        result = review_probes(sub, run, probes, agent_model)
        (out / "result.json").write_text(json.dumps(result, indent=2))
        click.echo(f"{unit['name']}: {result['outcome']}")


STAGES = (  # file -> the stage a unit has reached once it exists, latest first
    ("result.json", "measured"),
    ("outcome.json", "unauditable"),
    ("review.probes.json", "probed"),
    ("review.prompt.md", "submitted"),
    ("ev.hypothesis.json", "proposed"),
    ("ev.prompt.md", "scanned"),
)


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
def status(plan_file):
    """Each unit's stage, grouped (without saying which arm it is in)."""
    groups: dict[str, list[str]] = {}
    for unit in load_plan(plan_file)["units"]:
        out = Path(unit["dir"])
        stage = next((s for f, s in STAGES if (out / f).exists()), "pending")
        groups.setdefault(stage, []).append(unit["name"])
    for stage, names in groups.items():
        click.echo(f"{stage} ({len(names)}): {' '.join(names)}")


@cli.command()
@click.option("--plan", "plan_file", required=True, type=click.Path(exists=True, path_type=Path))
def report(plan_file):
    """One JSON line per unit, with its arm and outcome."""
    for unit in load_plan(plan_file)["units"]:
        out = Path(unit["dir"])
        row = {k: unit[k] for k in ("name", "arm", "layer", "feature")}
        for file in ("outcome.json", "result.json"):
            if (out / file).exists():
                row |= json.loads((out / file).read_text())
        if (out / "submission.json").exists():
            sub = json.loads((out / "submission.json").read_text())
            row["explanation"] = sub["natural_language_explanation"]
            row["density"] = sub["dictionary_profile"]["density"]
        click.echo(json.dumps(row))


if __name__ == "__main__":
    cli()
