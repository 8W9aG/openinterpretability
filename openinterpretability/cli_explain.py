"""Attribute one answer of a model to its labelled neuron poles and attention heads.

The command is ``open-interpretability explain``.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from pathlib import Path

import click

from openinterpretability.cli import _load_runner, cli, wiki_dir_option, with_model_options


@cli.command(name="explain")
@click.argument("prompt")
@click.option("--model", "model_name", default="Qwen/Qwen2.5-0.5B", show_default=True)
@click.option("--token", help="The answer token to explain. Default: the model's top token.")
@click.option("--position", default=-1, show_default=True, help="The position to explain.")
@click.option("--top", default=20, show_default=True, help="Units to list.")
@click.option(
    "--edges", default=8, show_default=True, help="Top units whose incoming paths are traced."
)
@click.option(
    "--corpus", default="corpus.jsonl", show_default=True, type=click.Path(path_type=Path)
)
@click.option(
    "--mean-tokens",
    default=20_000,
    show_default=True,
    help="Tokens of the corpus for each neuron's mean, the attribution's baseline (cached).",
)
@click.option(
    "--target", default=None, type=click.FloatRange(0, 1),
    help="The explained share to plan experiments for. Default: 1, a complete explanation.",
)  # fmt: skip
@click.option(
    "--experiments", "show", default=10, show_default=True, help="Planned experiments to list."
)
@click.option(
    "--level", type=click.Choice(["mechanism", "long", "paragraph"]), default="mechanism",
    show_default=True, help="mechanism: the evidence. long and paragraph: an LLM narrates it.",
)  # fmt: skip
@click.option("--narrator-model", default=None, help="The narrator's model (agents' default).")
@click.option(
    "--run", "run", default=None, type=click.IntRange(0), is_flag=False, flag_value=0,
    help="Run the planned unit and superposition experiments through the experiment loop, by "
    "coverage gain, until the explained share reaches --target or no work is left (#4473). "
    "--run N stops after N experiments too. Without it, only list them.",
)  # fmt: skip
@click.option("--proposer", default="", help="--run: the LiteLLM proposer model.")
@click.option("--reviewer", default="", help="--run: the LiteLLM reviewer model.")
@click.option("--provider", default="", help="--run: a provider profile (docs/providers.md).")
@click.option(
    "--max-attempts", default=5, show_default=True, type=click.IntRange(1),
    help="--run: runs per unit before giving up on it.",
)  # fmt: skip
@click.option(
    "--record", "write_record", is_flag=True,
    help="Write the explanation to wiki/explanations/, for its page on the site (#3744).",
)  # fmt: skip
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
@wiki_dir_option()
@with_model_options
def explain_command(  # long-function: one argument per option, then one line per step
    prompt, model_name, token, position, top, edges, corpus, mean_tokens, target, show, level,
    narrator_model, run, proposer, reviewer, provider, max_attempts, write_record, as_json,
    wiki_dir, device, dtype,
):  # fmt: skip
    """Explain the model's answer to PROMPT as fully as the registry allows.

    It attributes the answer's logit to every MLP neuron pole and attention head, against
    each unit's mean on the corpus. It labels the top units from the registry. It reports
    the share of the attribution that VERIFIED units hold, and checks it by mean-ablating
    them. It lists the superpositions and circuits that carry the most attribution, and
    traces the paths between the top units. Then it lists the experiments that would
    explain the rest: for each pole, head, superposition and circuit that isn't VERIFIED.
    """
    from openinterpretability import explain_groups, explain_plan
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import explain, registry_labels, unit_means
    from openinterpretability.explain_coverage import POOL, add_gains, prompt_coverage
    from openinterpretability.planner import configured_windows, load_states

    runner = _load_runner(model_name, device, dtype)
    means = unit_means(runner, model_name, read_corpus(corpus)[0], mean_tokens)
    states = load_states(Path(wiki_dir))  # read once: labels and the plan both use it
    labels = registry_labels(Path(wiki_dir), model_name, states)
    groups = explain_groups.superpositions(model_name) + explain_groups.circuits(model_name)
    group_labels = explain_groups.circuit_labels(Path(wiki_dir), model_name)
    result = explain(
        runner, model_name, prompt, means, labels, token=_answer(runner, token),
        position=position, top=top, edges=edges, groups=groups, group_labels=group_labels,
    )  # fmt: skip
    mark_hubs(result, model_name, states)  # a top unit that is a hub's center, and its facts
    planned = explain_plan.plan(
        result, states, model_name, target=explain_plan.TARGET if target is None else target,
        window=configured_windows().get(model_name), groups=groups,
    )  # fmt: skip
    args = (runner, prompt, means, result.token_id, position, result.order)
    coverage = prompt_coverage(*args)
    add_gains(*args, planned, pool=max(POOL, show))  # rank the experiments by coverage gained
    told = _narrate(result, coverage, planned, narrator_model) if level != "mechanism" else {}
    if as_json:
        out = _as_json(result, planned, show) | {"coverage": coverage}
        click.echo(json.dumps(out | told, indent=1, default=str))
        return
    if level != "paragraph":
        _print(result)
        _print_coverage(coverage)
        _print_plan(planned, show)
    _print_narrative(told, level)
    if write_record:
        path = _record(model_name, result, coverage, planned, told, Path(wiki_dir))
        click.echo(f"Explanation → {path}")
    if run is not None:
        agents = (proposer, reviewer, provider)
        _run(result, planned, groups, model_name, agents, max_attempts, run, Path(wiki_dir))


def mark_hubs(result, model: str, states: dict) -> None:
    """Set ``hub`` on each top unit that is a hub's center (#4472)."""
    from openinterpretability.explain import _slug
    from openinterpretability.hubs import explain_notes

    notes = explain_notes(model, states)
    for unit in result.poles:
        note = notes.get(_slug(model, unit.key))
        unit.hub = note.model_dump() if note is not None else None


def hub_line(hub: dict) -> str:
    """One line on a unit that is a hub's center: its id, page and artefact facts."""
    facts = []
    if hub.get("active_share") is not None:
        on = "default-on" if hub.get("default_on") else "not default-on"
        facts.append(f"{on}, active on {hub['active_share']:.0%} of its scan")
    if hub.get("norm") is not None:
        facts.append(f"output norm {hub['norm']:.3g} against its layer's median "
                     f"{hub['layer_median']:.3g} (p{100 * hub['percentile']:.0f})")  # fmt: skip
    from openinterpretability.core.project import project_links

    told = "; ".join(facts) or "no facts recorded"
    url = project_links()["site_url"] + hub["href"]
    return f"hub {hub['hub']} ({url}): {told}. A hub can be an artefact."


def _answer(runner, token: str | None) -> int | None:
    """The answer token's id, or None for the model's top token."""
    if token is None:
        return None
    ids = runner.tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise click.BadParameter(f"{token!r} is {len(ids)} tokens, not one", param_hint="--token")
    return ids[0]


def _as_json(result, planned, show: int) -> dict:
    """The result, without the full ranking, and the next ``show`` planned experiments."""
    out = asdict(result) | {"verified_share": result.verified_share}
    out.pop("order")
    return out | {
        "target": planned.target, "planned_share": planned.planned_share,
        "blocked": planned.blocked, "reaches": planned.reaches,
        "planned": len(planned.experiments), "milestones": dict(_milestones(planned)),
        "experiments": [asdict(e) for e in planned.experiments[:show]],
        "group_experiments": [asdict(e) for e in planned.groups],
        "blocked_units": planned.blocked_units,
    }  # fmt: skip


def _print(result) -> None:
    click.echo(f"Answer at position {result.position}: {result.token!r} (logit {result.logit:.2f})")
    click.echo(
        f"{'unit':<14} {'attribution':>11} {'share':>6} {'ablated Δ':>9} {'peak/max':>9}  status"
    )
    for p in result.poles:
        share = abs(p.attribution) / result.total if result.total else 0.0
        ablated = f"{p.ablation_change:+.2f}" if p.ablation_change is not None else ""
        peak = f"{p.strength / p.max_activation:.2f}" if p.max_activation else "—"
        row = f"{p.label:<14} {p.attribution:>+11.3f} {share:>6.1%} {ablated:>9} {peak:>9}"
        click.echo(f"{row}  {p.status}")
        if p.explanation:
            click.echo(f"{'':<14} {p.explanation[:110]}")
        if p.hub:
            click.echo(f"{'':<14} {hub_line(p.hub)}")
    click.echo(
        f"Explained units (VERIFIED, or in a VERIFIED superposition or circuit) hold "
        f"{result.verified_share:.1%} of the total |attribution|; "
        f"{1 - result.verified_share:.1%} is unexplained."
    )
    joint = result.joint
    if joint.get("poles"):
        changed = (
            f"the top token changes to {joint['new_top_token']!r}"
            if joint["top_token_changed"]
            else "the top token stays"
        )
        click.echo(
            f"Mean-ablating the {joint['poles']} VERIFIED units above together moves the answer's "
            f"logit by {joint['logit_change']:+.2f}, and {changed}."
        )
    else:
        click.echo("No unit above is VERIFIED, so nothing here is explained yet.")
    _print_groups(result)
    _print_paths(result)
    click.echo(
        "Attribution is first order and against each unit's mean. The ablated Δ column is "
        "the exact check: mean-ablating a unit moves the logit by about minus its attribution. "
        "Unverified units carry no label."
    )


def _print_groups(result) -> None:
    """The superpositions and circuits, which overlap with the units and aren't in the total."""
    if not result.groups:
        return
    click.echo("Superpositions and circuits (they overlap with the units above):")
    for g in result.groups[:10]:
        ablated = f"{g.ablation_change:+.2f}" if g.ablation_change is not None else ""
        name = f"{g.name} ({g.kind}, {g.members} members)"
        click.echo(f"  {name:<46} {g.attribution:>+9.3f} {ablated:>9}  {g.status}")
        if g.explanation:
            click.echo(f"  {'':<46} {g.explanation[:110]}")


def _print_paths(result, per_target: int = 3) -> None:
    """Each traced unit's strongest incoming edges from the other top units."""
    incoming: dict[str, list[dict]] = {}
    for edge in result.edges:  # sorted by |weight|, strongest first
        incoming.setdefault(edge["target"], []).append(edge)
    if not incoming:
        return
    click.echo("Paths: the top units that drive each unit's contribution (first order).")
    for unit in result.poles:
        edges = incoming.get(unit.label, [])[:per_target]
        if edges:
            sources = ", ".join(f"{e['source']} {e['weight']:+.3f}" for e in edges)
            click.echo(f"  {unit.label:<12} ← {sources}")


def _print_plan(planned, show: int) -> None:
    """The planned experiments: how far they reach, the first ``show``, and what is blocked."""
    from openinterpretability.explain_plan import CAVEAT

    n, reach = len(planned.experiments), planned.reaches
    goal = "reach" if reach >= planned.target - 1e-9 else f"reach only {reach:.1%}, short of"
    click.echo(
        f"Plan: {n} unit experiment{'s' if n != 1 else ''} {goal} the {planned.target:.0%} target "
        f"(explained now: {planned.explained:.1%}), and {len(planned.groups)} for "
        "superpositions and circuits."
    )
    if marks := _milestones(planned):
        click.echo("  " + "; ".join(f"{k} experiments reach {m:.0%}" for m, k in marks) + ".")
    for e in planned.experiments[:show] + planned.groups[:show]:
        held = f"; in {', '.join(e.covered_by)}" if e.covered_by else ""
        gain = f"{e.gain:+.1%} coverage" if e.gain is not None else "gain not measured"
        click.echo(
            f"  {e.target:<20} {gain:<18} {e.share:>6.1%} attr.  {e.status:<14} {e.draw}{held}"
        )
        for line in textwrap.wrap(e.rationale, 96):
            click.echo(f"      {line}")
        for command in e.commands:
            click.echo(f"    $ {command}")
    if n > show:
        click.echo(f"  … and {n - show} more unit experiments (--experiments, or --json).")
    if planned.blocked:
        click.echo(
            f"Blocked: {planned.blocked:.1%} of the attribution, on "
            f"{len(planned.blocked_units)} units that no experiment can raise "
            "(stopped, or no follow-up)."
        )
    click.echo(CAVEAT)


def _milestones(planned, marks=(0.5, 0.8, 0.9, 0.99)) -> list[tuple[float, int]]:
    """How many unit experiments, in rank order, reach each explained share below the target."""
    out, reached, k = [], planned.explained, 0
    for mark in (m for m in marks if planned.explained < m < planned.target):
        while reached < mark and k < len(planned.experiments):
            reached += planned.experiments[k].share
            k += 1
        if reached >= mark:
            out.append((mark, k))
    return out


def _print_coverage(coverage: dict) -> None:
    """Prompt coverage (explain_coverage.py): what the explained units bring back, causally."""
    share, base = coverage["coverage"], coverage["random"]
    if share is None:
        click.echo("Prompt coverage: ablating every unit doesn't change the answer.")
        return
    against = (
        f"; {coverage['units']} random units bring back {base:.1%}" if base is not None else ""
    )
    click.echo(
        f"Prompt coverage: keeping only the {coverage['units']} explained units, and mean-ablating "
        f"the rest, brings back {share:.1%} of the answer's log-probability{against}."
    )


def _narrate(result, coverage, planned, model: str | None) -> dict:
    """The narrator's long form and paragraph, the evidence it read, and its checks (#3743)."""
    from openinterpretability.agents.base import DEFAULT_MODEL
    from openinterpretability.agents.narrator import Narrator
    from openinterpretability.core import costs
    from openinterpretability.explain_narrate import check_citations, evidence

    ev = evidence(result, coverage, planned)
    narrator = Narrator(model or DEFAULT_MODEL)
    costs.reset()  # so that the costs below are the narrator's alone
    narrative = narrator.narrate(ev)
    return {
        "evidence": ev, "narrative": narrative.model_dump(), "narrator_model": narrator.model,
        "citations_not_in_evidence": check_citations(narrative, ev),
        "narrator_costs": costs.measured().model_dump(mode="json"),
    }  # fmt: skip


def _print_narrative(told: dict, level: str) -> None:
    """The long form and the paragraph, or the paragraph alone, with any unsupported citation."""
    if not told:
        return
    narrative = told["narrative"]
    if level == "long":
        click.echo("")
        for section in narrative["sections"]:
            click.echo(f"## {section['title']}\n")
            click.echo(textwrap.fill(section["text"], 100) + "\n")
        click.echo("## In one paragraph\n")
    click.echo(textwrap.fill(narrative["paragraph"], 100))
    click.echo(f"\n(narrated by {told['narrator_model']} from the evidence only)")
    if told["citations_not_in_evidence"]:
        missing = ", ".join(told["citations_not_in_evidence"])
        click.echo(f"Warning: the text cites units that the evidence doesn't hold: {missing}.")


def _run(result, planned, groups, model: str, agents: tuple[str, str, str], max_attempts: int,
         limit: int, wiki_dir: Path) -> None:  # fmt: skip
    """Run the plan through the experiment loop until the explained share reaches its target.

    Units and superpositions run by coverage gain (#4473). Circuits wait for an agent to write
    their task (:func:`~openinterpretability.explanation_targets.circuit_notes`).
    """
    from openinterpretability.cli_loop import _api_model, _models, run_loop
    from openinterpretability.explanation_targets import (
        ShareMeter,
        TargetedDraws,
        by_gain,
        circuit_notes,
        targets,
    )
    from openinterpretability.planner import configured_windows

    for model_name in agents[:2]:
        if model_name:
            _api_model(None, None, model_name)  # an external model has nobody to answer it
    proposer, reviewer = _models(*agents)
    ordered = by_gain([*planned.experiments, *planned.groups])
    found = targets(ordered, model, groups)
    target = _reachable(planned, circuit_notes(ordered))
    click.echo(f"Running {len(found)} experiments, by coverage gain, until the explained share "
               f"reaches {target:.0%}: " + ", ".join(t.label for t in found))  # fmt: skip
    carrying = {(g.kind, g.name) for g in result.groups}  # the groups that carry the answer
    meter = ShareMeter(model, result.order, [g for g in groups if (g.kind, g.name) in carrying],
                       wiki_dir)  # fmt: skip
    draws = TargetedDraws(found, reviewer, max_attempts, meter=meter, target=target, limit=limit,
                          window=configured_windows().get(model))  # fmt: skip
    raise SystemExit(run_loop(proposer, reviewer, draws))


def _reachable(planned, circuits: list[str]) -> float:
    """The target the run aims at, said with what blocks the rest (#4473).

    The run reaches at most the share now plus what the planned unit experiments add.
    Blocked units can't rise, and circuits wait for an agent.
    """
    click.echo(f"Explained now: {planned.explained:.1%}. Blocked: {planned.blocked:.1%} "
               f"(stopped units, or no follow-up). The planned experiments reach "
               f"{planned.reaches:.1%}.")  # fmt: skip
    for note in circuits:
        click.echo(f"Not run: circuit {note}")
    if planned.target <= planned.reaches:
        return planned.target
    click.echo(f"The target {planned.target:.0%} is out of reach, so the run stops at "
               f"{planned.reaches:.1%}, the most it can explain.")  # fmt: skip
    return planned.reaches


def _record(model: str, result, coverage: dict, planned, told: dict, wiki_dir: Path) -> Path:
    """Write the explanation record (core/explanation_records.py) and return its path."""
    from datetime import UTC, datetime

    from openinterpretability import __version__
    from openinterpretability.core.explanation_records import ExplanationRecord, Narration
    from openinterpretability.core.explanation_records import record_path as path_of

    units = _explained_units(model, result)
    narration = None
    if told:
        n = told["narrative"]
        narration = Narration(
            sections=n["sections"], paragraph=n["paragraph"],
            narrator_model=told["narrator_model"],
            citations_not_in_evidence=told["citations_not_in_evidence"],
        )  # fmt: skip
    record = ExplanationRecord(
        model_name=model, prompt=result.prompt, position=result.position, answer=result.token,
        explained_share=result.verified_share, blocked_share=planned.blocked,
        prompt_coverage=coverage.get("coverage"), random_coverage=coverage.get("random"),
        units=units, narration=narration, planned_experiments=len(planned.experiments),
        package_version=__version__, written=datetime.now(UTC),
    )  # fmt: skip
    path = path_of(wiki_dir, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return path


def _explained_units(model: str, result) -> list:
    """The record's mechanism: the top units, then the top superpositions and circuits."""
    from openinterpretability.core.explanation_records import ExplainedUnit
    from openinterpretability.explain import HEAD, _slug

    total = result.total or 1.0
    units = []
    for u in result.poles:
        kind = "head" if u.pole == HEAD else "pole"
        units.append(ExplainedUnit(
            unit=u.label, slug=_slug(model, u.key), kind=kind, share=abs(u.attribution) / total,
            attribution=u.attribution, status=u.status, explanation=u.explanation, hub=u.hub,
        ))  # fmt: skip
    for g in result.groups[:10]:
        units.append(ExplainedUnit(
            unit=g.name, kind=g.kind, share=abs(g.attribution) / total,
            attribution=g.attribution, status=g.status, explanation=g.explanation,
        ))  # fmt: skip
    return units
