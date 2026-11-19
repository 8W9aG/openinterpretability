"""Attribute one answer of a model to its labelled neuron poles and attention heads.

The command is ``open-interpretability explain``.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import click

from openinterpretability import explain_plan
from openinterpretability.cli import _load_runner, cli, wiki_dir_option, with_model_options
from openinterpretability.explain_reuse import look_first
from openinterpretability.explain_run import run_plan
from openinterpretability.explain_stopped import lines as stopped_lines
from openinterpretability.hubs import hub_line, mark_hubs


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
    help="The causal coverage to aim at: the share of the answer's log-probability that the "
    "explained units bring back, the rest mean-ablated (#4476). Default: 1.",
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
@click.option(
    "--land/--no-land", "land", default=True, show_default=True,
    help="--record: land the record as its own PR, merged on green (#4475). --no-land "
    "writes it to --wiki-dir only.",
)  # fmt: skip
@click.option(
    "--complete", is_flag=True,
    help="From the prompt to a published explanation in one command (#4476): --run to "
    "--target, --level long and --record, landed as a PR.",
)  # fmt: skip
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
@wiki_dir_option()
@with_model_options
def explain_command(  # long-function: one argument per option, then one line per step
    prompt, model_name, token, position, top, edges, corpus, mean_tokens, target, show, level,
    narrator_model, run, proposer, reviewer, provider, max_attempts, write_record, land, complete,
    as_json,
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
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import unit_means

    if complete:  # run to the target, narrate in full, and land the record (#4476)
        run, level, write_record, land = 0 if run is None else run, "long", True, True
    reuse = look_first(Path(wiki_dir), model_name, prompt, position, target, level,
                       write_record or run is not None)  # fmt: skip
    if reuse.action == "done":
        return
    if reuse.action == "level":
        run = None  # an earlier explanation already reaches the target: only narrate (#4503)
    runner = _load_runner(model_name, device, dtype)
    texts, _, corpus_sha, corpus_source = read_corpus(corpus)
    means = unit_means(runner, model_name, texts, mean_tokens)
    ask = Ask(model_name, prompt, _answer(runner, token), position, top, edges, target, show)
    found = _analyse(runner, means, ask, Path(wiki_dir))
    home = Path(wiki_dir).resolve()  # the record goes here, though --run moves to its worktree
    ran, code = None, 0
    if run is not None:  # run, then explain the answer as the registry stands after it (#4474)
        agents = (proposer, reviewer, provider)
        ran, code = run_plan(found, ask, agents, max_attempts, run, Path(wiki_dir),
                             measure=(runner, means))  # fmt: skip
        found = _analyse(runner, means, ask, Path(wiki_dir))
        ran = ran.model_copy(update={"share_after": found.coverage.get("coverage") or 0.0})
    result, planned, coverage = found.result, found.planned, found.coverage
    told = {}
    if level != "mechanism":
        told = _narrate(result, coverage, planned, narrator_model, found.answer)
    if as_json:
        out = _as_json(result, planned, show) | {
            "coverage": coverage,
            "coverage_plan": found.reach,
            "answer_experiment": found.answer,
            "stopped": [{k: v for k, v in r.items() if k != "unit"} for r in found.stopped],
        }
        click.echo(
            json.dumps(out | told | {"run": ran and ran.model_dump()}, indent=1, default=str)
        )
    elif level != "paragraph":
        _print(result)
        _print_coverage(coverage)
        _print_plan(planned, show, found.reach)
        click.echo("\n".join(stopped_lines(found.stopped, ask.model)))
        click.echo("\n".join(explain_plan.answer_lines(found.answer)))
    if not as_json:
        _print_narrative(told, level)
    if write_record:
        from openinterpretability.explain_figure_data import figure_data
        from openinterpretability.reproduce_explanation import inputs_of

        figures = figure_data(runner, means, result, coverage, planned, found.groups)
        record = _record(model_name, result, coverage, planned, told, ran, figures,
                         found.reach["target"])  # fmt: skip
        inputs = inputs_of(runner, ask, result, planned, (corpus_sha, corpus_source), mean_tokens)
        record = record.model_copy(update=reuse.fields | {"inputs": inputs})
        _keep(record, home, land)
    if not as_json:
        click.echo(reached(result.verified_share, found.reach["target"], coverage, ran))
    if code:
        raise SystemExit(code)


@dataclass(frozen=True)
class Ask:
    """What to explain: the model, the prompt and its answer, and how much to show."""

    model: str
    prompt: str
    token: int | None  # the answer's token id, or None for the model's top token
    position: int
    top: int
    edges: int
    target: float | None
    show: int


def _analyse(runner, means, ask: Ask, wiki_dir: Path) -> SimpleNamespace:
    """Attribute, label, mark hubs, measure the coverage and plan, from the registry now.

    ``explain --run`` calls it twice: to plan the run, and after it (#4474).
    """
    from openinterpretability import explain_groups
    from openinterpretability.explain import explain, registry_labels
    from openinterpretability.explain_coverage import (
        POOL,
        add_gains,
        coverage_plan,
        prompt_coverage,
    )
    from openinterpretability.explain_stopped import rank as rank_stopped
    from openinterpretability.planner import configured_windows, load_states

    model = ask.model
    states = load_states(wiki_dir)  # read once: labels and the plan both use it
    labels = registry_labels(wiki_dir, model, states)
    groups = explain_groups.superpositions(model) + explain_groups.circuits(model)
    result = explain(
        runner, model, ask.prompt, means, labels, token=ask.token, position=ask.position,
        top=ask.top, edges=ask.edges, groups=groups,
        group_labels=explain_groups.circuit_labels(wiki_dir, model),
    )  # fmt: skip
    mark_hubs(result, model, states)  # a top unit that is a hub's center, and its facts
    planned = explain_plan.plan(  # every unit experiment: the coverage target picks how many
        result, states, model, target=explain_plan.TARGET,
        window=configured_windows().get(model), groups=groups,
    )  # fmt: skip
    args = (runner, ask.prompt, means, result.token_id, ask.position, result.order)
    coverage = prompt_coverage(*args)
    add_gains(*args, planned, pool=max(POOL, ask.show))  # rank the experiments by coverage gained
    stopped = rank_stopped(*args, planned, wiki_dir, model)  # the blocked units, by gain (#4951)
    target = explain_plan.TARGET if ask.target is None else ask.target
    reach = coverage_plan(*args, planned, target)  # the target is causal coverage (#4476)
    answer = explain_plan.answer_step(
        wiki_dir, model, ask.prompt, ask.position, result.token, coverage
    )  # how the units make the answer (#4778)
    return SimpleNamespace(result=result, planned=planned, coverage=coverage, groups=groups,
                           reach=reach, answer=answer, stopped=stopped)  # fmt: skip


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


def _print_plan(planned, show: int, reach: dict) -> None:
    """The planned experiments: how far they reach, the first ``show``, and what is blocked.

    ``reach`` is the coverage estimate (``explain_coverage.coverage_plan``, #4476).
    """
    from openinterpretability.explain_plan import CAVEAT

    k, target = reach["experiments"], reach["target"]
    goal = (f"about {k} unit experiments reach the {target:.0%} coverage target" if k is not None
            else f"the {target:.0%} coverage target is out of reach: every planned unit "
            f"brings back {reach['reach']:.1%}")  # fmt: skip
    if reach.get("overshoot"):
        goal += "; with every planned unit kept, the answer overshoots the clean model"
    click.echo(f"Plan: {goal} (coverage now {reach['now']:.1%}). There are "
               f"{len(planned.experiments)} unit experiments in all, and "
               f"{len(planned.groups)} for superpositions and circuits.")  # fmt: skip
    if reach.get("chosen"):
        click.echo("  Greedy picks, by joint coverage: " + ", ".join(reach["chosen"][:20]))
    if marks := _milestones(planned):
        click.echo(
            "  Of the |attribution|: "
            + "; ".join(f"{k} experiments reach {m:.0%}" for m, k in marks)
            + "."
        )
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
    if (n := len(planned.experiments)) > show:
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


def reached(share: float, target: float, coverage: dict, ran=None) -> str:
    """The last line: the coverage reached against its target, and the share (#4476)."""
    start = f" (from {ran.share_before:.1%} before the run)" if ran is not None else ""
    found = coverage.get("coverage")
    text = (f"Reached: prompt coverage {found:.1%}{start}, target {target:.0%}" if found is not None
            else f"Reached: no prompt coverage, target {target:.0%}")  # fmt: skip
    if found is not None and coverage.get("random") is not None:
        text += f", against {coverage['random']:.1%} for random units matched by kind and layer"
    return text + f"; explained share of |attribution| {share:.1%}."


def _print_coverage(coverage: dict) -> None:
    """Prompt coverage (explain_coverage.py): what the explained units bring back, causally."""
    share, base = coverage["coverage"], coverage["random"]
    if share is None:
        click.echo("Prompt coverage: ablating every unit doesn't change the answer.")
        return
    against = f"; {coverage['units']} matched random units: {base:.1%}" if base is not None else ""
    click.echo(
        f"Prompt coverage: keeping only the {coverage['units']} explained units, and mean-ablating "
        f"the rest, brings back {share:.1%} of the answer's log-probability{against}."
    )
    if coverage.get("overshoot"):
        click.echo(f"Overshoot: the raw coverage is {coverage['raw']:.1%}. The kept units push "
                   "the answer above the clean model, so the credit stops at 100%.")  # fmt: skip


def _narrate(result, coverage, planned, model: str | None, answer: dict | None = None) -> dict:
    """The narrator's long form and paragraph, the evidence it read, and its checks (#3743)."""
    from openinterpretability.agents.base import DEFAULT_MODEL
    from openinterpretability.agents.narrator import Narrator
    from openinterpretability.core import costs
    from openinterpretability.explain_narrate import check_citations, evidence

    ev = evidence(result, coverage, planned, answer=answer)
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


def _keep(record, home: Path, land: bool) -> None:
    """Land ``record`` as its own PR, or with ``--no-land`` write it under ``home`` (#4475)."""
    from openinterpretability.explain_land import land_record, write_local

    if not land:
        click.echo(f"Explanation → {write_local(record, home)}")
        return
    url, merged = land_record(record)
    click.echo(f"Explanation → {url} ({'merged' if merged else 'left open'})")


def _record(model: str, result, coverage: dict, planned, told: dict, run=None, figures=None,
            target: float | None = None):  # fmt: skip
    """The explanation record (core/explanation_records.py), as the registry stands."""
    from datetime import UTC, datetime

    from openinterpretability import __version__
    from openinterpretability.core.explanation_records import ExplanationRecord, Narration

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
        raw_coverage=coverage.get("raw"),
        units=units, narration=narration, planned_experiments=len(planned.experiments),
        package_version=__version__, written=datetime.now(UTC), run=run, figures=figures,
        target=target, coverage_detail=ExplanationRecord.detail(coverage),
    )  # fmt: skip
    return record


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
