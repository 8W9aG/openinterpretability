"""The steps of one drawn experiment as exact commands: claim, read the history, run, post.

``openinterp next --steps`` prints them. A driving agent then runs them in order. It
does not work them out again from the skills and ``docs/PROTOCOL.md`` on every
experiment. The skills stay the reference for *why*. This is only the *what*, filled in
for the draw.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from openinterp.core.paths import DEFAULT_EXCHANGE_DIR
from openinterp.planner import Candidate, Plan, claim_key
from openinterp.planner_models import model_dims, model_heads

SCAN_KINDS = ("explore", "opposite_pole", "revise", "new_hypothesis", "rescan", "sister")
SUPERSEDING_KINDS = ("revise", "new_hypothesis", "rescan")
REVIEW_KINDS = ("independent_review", "replicate", "challenge")


@dataclass(frozen=True)
class Step:
    """One step of a draw: a title, the commands to run and a note."""

    title: str
    commands: list[str] = field(default_factory=list)
    note: str = ""


def random_neuron(model: str, seed: int, dims: tuple[int, int], pole: str | None = None) -> str:
    """The slug ``openinterp audit --model M --random --seed S`` scans, without a scan.

    Mirrors :func:`openinterp.evidence.pick_unit`'s draws (layer, neuron, pole), assuming
    every layer has the same MLP width.
    """
    from openinterp.core.identifiers import NeuronIdentifier

    rng = random.Random(seed)
    layer = rng.randrange(dims[0])
    neuron = rng.randrange(dims[1])
    pole = pole or rng.choice(["positive", "negative"])
    return NeuronIdentifier(model_name=model, layer=layer, neuron_idx=neuron, pole=pole).slug


def random_head(model: str, seed: int, dims: tuple[int, int]) -> str:
    """The slug ``openinterp audit --model M --random --unit head --seed S`` scans.

    Mirrors :func:`openinterp.evidence.pick_unit` for heads: the layer, then the head.
    """
    from openinterp.core.identifiers import HeadIdentifier

    rng = random.Random(seed)
    layer = rng.randrange(dims[0])
    head = rng.randrange(dims[1])
    return HeadIdentifier(model_name=model, layer=layer, head_idx=head).slug


def draw_slug(choice: Candidate) -> str | None:
    """Return the unit that the draw studies.

    For a random pick (an explore draw, or a sister study of a random pole), it is worked
    out from the draw's seed.
    """
    options = (choice.audit or "").split()
    if choice.slug or "--random" not in options:
        return choice.slug
    if "--seed" not in options:
        return None
    seed = int(options[options.index("--seed") + 1])
    if "--unit" in options and options[options.index("--unit") + 1] == "head":
        heads = model_heads(choice.model)
        return random_head(choice.model, seed, heads) if heads else None
    if (dims := model_dims(choice.model)) is None:
        return None
    pole = options[options.index("--pole") + 1] if "--pole" in options else None
    return random_neuron(choice.model, seed, dims, pole)


def steps(
    plan: Plan,
    corpus: str = "corpus.jsonl",
    proposer: str | None = None,
    reviewer: str | None = None,
    fresh_corpus: bool = False,
    ledger: str | None = None,
) -> list[Step]:
    """Return every step of ``plan``'s draw.

    A ``proposer`` or ``reviewer`` of None, or one named ``external/…``, means that a
    Claude Code subagent answers a prompt that the CLI writes. A new scan reads ``corpus``,
    or with ``fresh_corpus`` a corpus that it first builds from the draw's seed. A review of
    a record reads the corpus that the record scanned, whichever these say. ``ledger`` is
    the draw's ledger entry (:mod:`openinterp.core.ledger`), which the claim commits.
    """
    c = plan.choice
    slug = draw_slug(c)
    key = claim_key(slug) if slug else "<KEY>"
    out = f"{DEFAULT_EXCHANGE_DIR}/{key}"
    corpus, build = _corpus(c, out, corpus, plan.seed, fresh_corpus)
    run = _run(c, key, out, corpus, _api(proposer), _api(reviewer))
    return [
        _claim(c, key, plan.seed, known=slug is not None, ledger=ledger),
        *([] if c.kind in ("explore", "opposite_pole") else [_history(c)]),
        *build,
        *run,
        _post(key, c.kind),
    ]


def _corpus(c, out, corpus, seed, fresh: bool) -> tuple[str, list[Step]]:
    """Return the corpus that the draw reads, and the steps that build it, if any."""
    if c.kind in REVIEW_KINDS:
        scanned = f"{out}/scan-corpus.jsonl"
        command = f"uv run openinterp corpus --same-as {c.record} --out {scanned}"
        return scanned, [Step("Rebuild the corpus the record scanned", [command], _NO_REBUILD)]
    if fresh and c.kind in SCAN_KINDS:
        # Not under `out`: an explore draw's key can be unknown until its scan starts.
        built = f"{DEFAULT_EXCHANGE_DIR}/fresh/corpus-{seed}.jsonl"
        command = f"uv run openinterp corpus --seed {seed} --out {built}"
        return built, [Step("Build a fresh scan corpus (about a minute)", [command])]
    return corpus, []


def _api(model: str | None) -> str | None:
    return None if model is None or model.startswith("external/") else model


def _claim(c: Candidate, key: str, seed: int, known: bool, ledger: str | None = None) -> Step:
    branch = (
        f"experiment/{key}"
        if c.kind in ("explore", "opposite_pole")
        else f"experiment/{key}-{c.kind}"
    )
    # The claim commits the draw's ledger entry, so the draw reaches main whatever the run does.
    commit = (
        [f"git add {ledger}", f'git commit -m "Claim {key}"']
        if ledger
        else [f'git commit --allow-empty -m "Claim {key}"']
    )
    commands = [
        f"git switch -c {branch}",
        *commit,
        "git push -u origin HEAD",
        f'gh pr create --draft --title "experiment: {key} ({c.kind})" '
        f'--body "openinterp next seed: {seed}"',
    ]
    note = (
        ""
        if known
        else (
            "The unit isn't known until the scan starts (the model's config isn't cached): start "
            "the scan (a later step), then claim with the slug its `Unit:` line names, '/' → '-'."
        )
    )
    return Step("Claim it before any other work", commands, note)


def _history(c: Candidate) -> Step:
    unit_dir = c.record.rsplit("/", 1)[0] if c.record else f"wiki/experiments/{c.slug}"
    title = (
        "Read the VERIFIED pole's history (its explanation is the hypothesis under test)"
        if c.kind == "sister"
        else "Read the history (revisions must address the failed gates and probes it lists)"
    )
    return Step(title, [f"uv run openinterp show {unit_dir}"])


def _run(c, key, out, corpus, proposer, reviewer) -> list[Step]:
    if c.kind == "reevaluate":
        command = f"uv run openinterp reevaluate --record {c.record}"
        return [Step("Re-evaluate (no LLM calls)", [command])]
    if c.kind in REVIEW_KINDS:
        return _review(c.record, out, corpus, reviewer, independent=True)
    submission = f"submissions/{key}.json"
    scan = _scan(c, out, corpus, proposer, submission)
    return scan + _review(submission, out, corpus, reviewer, independent=False)


def _scan(c, out, corpus, proposer, submission) -> list[Step]:
    supersedes = f" --supersedes {c.record}" if c.kind in SUPERSEDING_KINDS else ""
    if c.kind == "sister":  # the scan's options carry it; the submit step needs it too
        supersedes = f" --sister-of {c.record}"
    stopped = _STOPPED if "--random" in (c.audit or "").split() else _STOPPED_CHOSEN
    audit = f"uv run openinterp audit {c.audit} --corpus {corpus}"
    if proposer:
        once = "" if c.kind == "sister" else supersedes  # already in c.audit
        command = f"{audit} --agent-model {proposer}{once} --out {submission}"
        return [Step(f"Scan and propose ({_SLOW})", [command], stopped)]
    submit = (
        f"uv run openinterp audit --evidence {out}/ev.json "
        f"--hypothesis {out}/ev.hypothesis.json "
        f"--agent-model external/<proposer model ID>{supersedes} --out {submission}"
    )
    return [
        Step(f"Scan ({_SLOW})", [f"{audit} --evidence-out {out}/ev.json"], stopped),
        _agent("neuron-proposer", f"{out}/ev.prompt.md", f"{out}/ev.hypothesis.json"),
        Step("Submit the hypothesis", [submit]),
    ]


_SLOW = "about 15 min on CPU: run it in the background"
_STOPPED_CHOSEN = (
    "If the scan stops because the unit never fires or is default-on, that is a result: "
    "commit the stop record and page it writes under wiki/stops/, note the unit in the PR, "
    "then mark it ready and land it like any other result (it must be merged, not closed "
    "or left as a draft) before drawing again."
)
_STOPPED = (
    "If the scan stops because the unit never fires or is default-on, that is a result: "
    "commit the draw record it writes under wiki/draws/, note the unit and seed in the PR, "
    "then mark it ready and land it like any other result (it must be merged, not left as "
    "a draft) before drawing again."
)
_NO_REBUILD = (
    "If this stops because the record's corpus can't be rebuilt, the draw is infeasible: close "
    "the claim's PR with that reason and draw again. Don't review on another corpus."
)
_INDEPENDENT = (
    "The reviewer's model must differ from the record's proposer; if it doesn't, "
    "say so in the PR instead of claiming independence."
)


def _agent(agent: str, prompt: str, answer: str) -> Step:
    return Step(
        f"Ask `{agent}` to answer {prompt}, then record its cost",
        [f"uv run openinterp usage <agent ID> --answer {answer}"],
        "Give it only that path. Note the model ID it reports.",
    )


def _review(audit_file, out, corpus, reviewer, independent: bool) -> list[Step]:
    note = _INDEPENDENT if independent else ""
    if reviewer:
        command = (
            f"uv run openinterp verify --audit-file {audit_file} --corpus {corpus} "
            f"--agent-model {reviewer}"
        )
        return [Step("Review", [command], note)]
    review = (
        f"uv run openinterp review --audit-file {audit_file} "
        f"--probes {out}/review.probes.json --agent-model external/<reviewer model ID> "
        f"--explanation-scores {out}/score.scores.json"
    )
    return [
        *_probes(audit_file, out),
        *_scores(audit_file, out, corpus),
        Step("Review", [review], note),
    ]


def _probes(audit_file, out) -> list[Step]:
    """The reviewer subagent writes probes, and checks them as tokens when that matters."""
    emit = (
        f"uv run openinterp review --audit-file {audit_file} --emit-prompt {out}/review.prompt.md"
    )
    check = (
        f"uv run openinterp review --audit-file {audit_file} "
        f"--check-probes {out}/review.probes.json"
    )
    return [
        Step("Write the reviewer prompt", [emit]),
        _agent("neuron-reviewer", f"{out}/review.prompt.md", f"{out}/review.probes.json"),
        Step(
            "Show the reviewer its probes as tokens (every review)",
            [check],
            f"Send the same reviewer {out}/review.check.prompt.md with SendMessage; "
            "it rewrites the probes. Don't skip it: reviewers label by words, and a wrong "
            "guess about a token split fails correct hypotheses (issue #3592). If it changed "
            "any probe, run the check again, so new probes are checked too, until it "
            "changes none (at most 3 rounds).",
        ),
    ]


def _scores(audit_file, out, corpus) -> list[Step]:
    """Two judge subagents score the explanation on held-out windows."""
    emit = (
        f"uv run openinterp score --audit-file {audit_file} --corpus {corpus} "
        f"--emit-task {out}/score.task.json"
    )
    score = (
        f"uv run openinterp score --task {out}/score.task.json "
        "--judge-model external/<judge model ID>"
    )
    return [
        Step("Draw held-out windows for the judges", [emit]),
        Step(
            f"Ask one `explanation-judge` to answer {out}/score.detection.prompt.md and a "
            f"second to answer {out}/score.fuzzing.prompt.md (if written), then record costs",
            [
                f"uv run openinterp usage <first agent ID> --answer {out}/score.detection.json",
                f"uv run openinterp usage <second agent ID> --answer {out}/score.fuzzing.json",
            ],
            "Give each only its own path. Note the model ID they report.",
        ),
        Step("Score", [score]),
    ]


def _post(key: str, kind: str) -> Step:
    return Step(
        "Post the result, whatever it is",
        [
            "uv run openinterp wiki",
            "make check lint test",
            f'git add wiki && git commit -m "{VERBS[kind]} {key} (<VERDICT>, <balanced accuracy>)"',
            "git push && gh pr ready",
        ],
        "Commit the record's .sig if verify printed 'Signed as …'. Never abandon a claim "
        "silently: if you stop, say why in the PR and land it with only its ledger entry "
        "(wiki/ledger/), so the draw still counts. Don't close it.",
    )


VERBS = {
    "explore": "Explore",
    "reevaluate": "Reevaluate",
    "independent_review": "Independent review of",
    "revise": "Revise",
    "new_hypothesis": "New hypothesis for",
    "rescan": "Rescan",
    "replicate": "Replicate",
    "challenge": "Challenge",
    "opposite_pole": "Explore",
    "sister": "Sister study of",
}


def render(steps_: list[Step]) -> str:
    """The steps as a numbered list, with each command after a ``$``."""
    lines = []
    for n, step in enumerate(steps_, 1):
        lines.append(f"{n}. {step.title}")
        lines += [f"   $ {command}" for command in step.commands]
        if step.note:
            lines.append(f"   {step.note}")
    return "\n".join(lines)
