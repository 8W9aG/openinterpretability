"""The steps of one drawn experiment as exact commands: claim, read the history, run, post.

``openinterp next --steps`` prints them, so a driving agent runs them in order instead of
re-deriving them from the skills and ``docs/PROTOCOL.md`` on every experiment. The
skills stay the reference for *why*; this is only the *what*, filled in for the draw.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from openinterp.planner import Candidate, Plan, claim_key

SCAN_KINDS = ("explore", "opposite_pole", "revise", "new_hypothesis", "rescan", "sister")
SUPERSEDING_KINDS = ("revise", "new_hypothesis", "rescan")
REVIEW_KINDS = ("independent_review", "replicate", "challenge")


@dataclass(frozen=True)
class Step:
    title: str
    commands: list[str] = field(default_factory=list)
    note: str = ""


def model_dims(model: str) -> tuple[int, int] | None:
    """(layers, MLP width) from the model's ``config.json``, or None if it can't be read.

    The file is in the Hugging Face cache once the model has been used; reading it never
    loads weights."""
    from huggingface_hub import hf_hub_download

    try:
        with open(hf_hub_download(model, "config.json")) as fh:
            config = json.load(fh)
    except Exception:  # offline with nothing cached, or an unknown model
        return None
    layers = config.get("num_hidden_layers") or config.get("n_layer")
    width = config.get("intermediate_size") or config.get("n_inner")
    if width is None and config.get("n_embd"):  # GPT-2: n_inner null means 4 × n_embd
        width = 4 * config["n_embd"]
    return (layers, width) if layers and width else None


def random_neuron(model: str, seed: int, dims: tuple[int, int], pole: str | None = None) -> str:
    """The slug ``openinterp audit --model M --random --seed S`` scans, without a scan.

    Mirrors :func:`openinterp.evidence.pick_unit`'s draws (layer, neuron, pole), assuming
    every layer has the same MLP width."""
    from openinterp.core.identifiers import NeuronIdentifier

    rng = random.Random(seed)
    layer = rng.randrange(dims[0])
    neuron = rng.randrange(dims[1])
    pole = pole or rng.choice(["positive", "negative"])
    return NeuronIdentifier(model_name=model, layer=layer, neuron_idx=neuron, pole=pole).slug


def draw_slug(choice: Candidate) -> str | None:
    """The unit the draw studies; for an explore draw, worked out from its seed."""
    if choice.slug or choice.kind != "explore":
        return choice.slug
    options = (choice.audit or "").split()
    if "--seed" not in options or (dims := model_dims(choice.model)) is None:
        return None
    seed = int(options[options.index("--seed") + 1])
    pole = options[options.index("--pole") + 1] if "--pole" in options else None
    return random_neuron(choice.model, seed, dims, pole)


def steps(
    plan: Plan,
    corpus: str = "corpus.jsonl",
    proposer: str | None = None,
    reviewer: str | None = None,
) -> list[Step]:
    """Every step of ``plan``'s draw. A ``proposer`` or ``reviewer`` of None, or one named
    ``external/…``, means a Claude Code subagent answers a prompt the CLI writes."""
    c = plan.choice
    slug = draw_slug(c)
    key = claim_key(slug) if slug else "<KEY>"
    out = f"exchange/{key}"
    run = _run(c, key, out, corpus, _api(proposer), _api(reviewer))
    return [
        _claim(c, key, plan.seed, known=slug is not None),
        *([] if c.kind in ("explore", "opposite_pole") else [_history(c)]),
        *run,
        _post(key, c.kind),
    ]


def _api(model: str | None) -> str | None:
    return None if model is None or model.startswith("external/") else model


def _claim(c: Candidate, key: str, seed: int, known: bool) -> Step:
    branch = (
        f"experiment/{key}"
        if c.kind in ("explore", "opposite_pole")
        else f"experiment/{key}-{c.kind}"
    )
    commands = [
        f"git switch -c {branch}",
        f'git commit --allow-empty -m "Claim {key}"',
        "git push -u origin HEAD",
        f'gh pr create --draft --title "experiment: {key} ({c.kind})" '
        f'--body "openinterp next seed: {seed}"',
    ]
    note = (
        ""
        if known
        else (
            "The unit isn't known until the scan starts (the model's config isn't cached): start "
            "the scan (next step), then claim with the slug its `Unit:` line names, '/' → '-'."
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
    "If the scan stops because the unit never fires or is default-on, say so in the PR and "
    "close it."
)
_STOPPED = (
    "If the scan stops because the unit never fires or is default-on, that is a result: "
    "commit the draw record it writes under wiki/draws/, note the unit and seed in the PR, "
    "then mark it ready and land it like any other result (it must be merged, not left as "
    "a draft) before drawing again."
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
            "For a head or any claim about tokens: show the reviewer its probes as tokens",
            [check],
            f"Send the same reviewer {out}/review.check.prompt.md with SendMessage; "
            "it rewrites the probes.",
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
            f'git add wiki && git commit -m "{_VERBS[kind]} {key} '
            '(<VERDICT>, <balanced accuracy>)"',
            "git push && gh pr ready",
        ],
        "Commit the record's .sig if verify printed 'Signed as …'. Never abandon a claim "
        "silently: if you stop, close the PR saying why.",
    )


_VERBS = {
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
    lines = []
    for n, step in enumerate(steps_, 1):
        lines.append(f"{n}. {step.title}")
        lines += [f"   $ {command}" for command in step.commands]
        if step.note:
            lines.append(f"   {step.note}")
    return "\n".join(lines)
