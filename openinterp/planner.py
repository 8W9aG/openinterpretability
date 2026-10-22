"""Choose the next experiment: randomised, exploration-balanced, herd-resistant.

Every neuron in the registry generates candidate follow-up experiments (re-evaluate
under the current protocol, independent re-review, revise, replicate, challenge,
audit the opposite pole, try a new hypothesis after a rejection). Each candidate
gets a weight from its expected information value, decayed by how often the neuron
has been studied, by recent activity and by open PRs that already claim it. Weights
are never zero: repeating an experiment is allowed, and a null result is data.

With probability ``epsilon`` the planner explores instead: a uniformly random new
neuron (the only selection that supports unbiased registry statistics), in a model
drawn from the configured list (``[tool.openinterp] models`` in ``pyproject.toml``)
with weight ``1/√(1 + units studied or claimed)``, so models with little coverage
get explored without the well-covered ones being dropped. Otherwise
it samples a candidate in proportion to ``weight ** (1 / temperature)``. Sampling,
rather than taking the best candidate, keeps independent agents from piling onto
the same neuron. The seed is reported so any choice can be reproduced.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from openinterp.builder.wiki import load_archive_paths
from openinterp.core.identifiers import model_slug
from openinterp.core.schemas import PROTOCOL_VERSION, AuditRecord

# Expected information value per experiment kind, before decay.
BASE_WEIGHTS = {
    "reevaluate": 3.0,  # verdict predates the current protocol; cheap, no LLM calls
    "independent_review": 2.5,  # proposer reviewed its own hypothesis
    "revise": 2.0,  # NEEDS_REVISION: failed gates say exactly what to fix
    "replicate": 1.2,  # VERIFIED by one reviewer model, or never by an independent one
    "opposite_pole": 1.0,  # the other half of a studied neuron
    "new_hypothesis": 0.8,  # REJECTED: the null result stands; try another idea
    "challenge": 0.6,  # VERIFIED and replicated: hunt for counterexamples
}
SKILLS = {
    "explore": ["native-neuron-audit", "neuron-family-triage", "vocab-projection-lookahead"],
    "reevaluate": [],
    "independent_review": ["interpretability-illusions-review"],
    "revise": ["native-neuron-audit", "explanation-simulation-scoring"],
    "replicate": ["interpretability-illusions-review"],
    "opposite_pole": ["native-neuron-audit", "vocab-projection-lookahead"],
    "new_hypothesis": ["native-neuron-audit", "neuron-steering-tests"],
    "challenge": ["interpretability-illusions-review", "neuron-steering-tests"],
}
DEFAULT_MODELS = ("Qwen/Qwen2.5-0.5B",)


@dataclass(frozen=True)
class Candidate:
    kind: str
    model: str
    weight: float
    rationale: str
    commands: list[str]
    slug: str | None = None
    skills: list[str] = field(default_factory=list)


@dataclass
class NeuronState:
    slug: str
    record: AuditRecord
    record_path: Path
    experiments: int = 0
    reviewers: set[str] = field(default_factory=set)
    independent_reviews: int = 0  # experiments reviewed by a model other than the proposer's
    last: datetime | None = None


def claim_key(slug: str) -> str:
    """Branch-safe form of a neuron slug; PR branches and titles should contain it."""
    return slug.replace("/", "-")


def open_pr_claims() -> list[str]:
    """Titles and branch names of open PRs (empty if `gh` is unavailable)."""
    if not shutil.which("gh"):
        return []
    try:
        out = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                "200",
                "--json",
                "title,headRefName",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return [f"{pr['title']} {pr['headRefName']}" for pr in json.loads(out)]


def configured_models(pyproject: Path = Path("pyproject.toml")) -> tuple[str, ...]:
    """Models the planner explores: ``[tool.openinterp] models`` (empty if unset)."""
    try:
        with open(pyproject, "rb") as fh:
            config = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return ()
    return tuple(config.get("tool", {}).get("openinterp", {}).get("models", ()))


def explore_weights(
    pool: Sequence[str], states: dict[str, NeuronState], claims: Sequence[str]
) -> dict[str, float]:
    """Explore probability per model, ∝ ``1/√(1 + units studied or claimed)``."""
    studied = Counter(s.record.submission.identifier.model_name for s in states.values())
    weights = {}
    for model in pool:
        key = f"{model_slug(model)}-L"  # how a claim on this model's units starts
        claimed = sum(key in c for c in claims)
        weights[model] = 1.0 / math.sqrt(1 + studied[model] + claimed)
    total = sum(weights.values())
    return {m: w / total for m, w in weights.items()}


def load_states(wiki_dir: Path) -> dict[str, NeuronState]:
    """One state per neuron pole, derived from all of its archived experiments.

    Follow-ups are planned from the most recent experiment; the count, reviewers and
    recency cover every experiment on the neuron.
    """
    states: dict[str, NeuronState] = {}
    for path, record in load_archive_paths(Path(wiki_dir)):  # oldest first
        slug = record.submission.identifier.slug
        state = states.setdefault(slug, NeuronState(slug, record, path))
        state.record, state.record_path = record, path
        state.experiments += 1
        state.reviewers.add(record.verdict.reviewer_agent_model)
        state.independent_reviews += not record.self_reviewed
        state.last = record.verdict.timestamp
    return states


def _decay(state: NeuronState, claims: Sequence[str], now: datetime) -> tuple[float, list[str]]:
    factor, notes = 1.0 / math.sqrt(state.experiments), []
    if state.experiments > 1:
        notes.append(f"{state.experiments} experiments so far")
    age = now - state.last
    if age < timedelta(hours=6):
        factor *= 0.2
        notes.append("studied in the last 6h")
    elif age < timedelta(hours=24):
        factor *= 0.5
        notes.append("studied in the last 24h")
    key = claim_key(state.slug)
    if any(key in c or state.slug in c for c in claims):
        factor *= 0.05
        notes.append("an open PR already claims it")
    return factor, notes


def candidates_for(state: NeuronState, slugs: set[str], claims: Sequence[str], now: datetime):
    record, verdict = state.record, state.record.verdict
    ident = record.submission.identifier
    is_head = ident.kind == "attn_head"
    path = state.record_path.as_posix()
    unit = f"--head {ident.head_idx}" if is_head else f"--neuron {ident.neuron_idx}"
    target = f"--model {ident.model_name} --layer {ident.layer} {unit}"
    other = "negative" if ident.pole == "positive" else "positive"
    gates = verdict.metrics.gates if verdict.metrics else {}
    failed = [g for g, ok in gates.items() if not ok]
    proposer = record.submission.submitting_agent_model
    review = (
        f"openinterp verify --audit-file {path} --corpus <scan corpus> "
        f"--agent-model <model other than {proposer}>"
    )
    pole = "" if is_head else f" --pole {ident.pole}"
    revise = f"openinterp audit {target}{pole} --supersedes {path}"

    kinds: list[tuple[str, str, list[str], str | None]] = []
    stale = verdict.policy is None or verdict.policy.protocol_version != PROTOCOL_VERSION
    if stale:
        kinds.append(
            (
                "reevaluate",
                "verdict predates the current protocol",
                [f"openinterp reevaluate --record {path}"],
                None,
            )
        )
    # Not the gate: it passes self-reviews unless the policy requires independence.
    if record.self_reviewed:
        kinds.append(
            ("independent_review", "the proposer reviewed its own hypothesis", [review], None)
        )
    if verdict.status == "NEEDS_REVISION":
        kinds.append(
            (
                "revise",
                f"failed gates: {', '.join(failed) or 'unknown'}",
                [revise, "openinterp verify --audit-file <new submission> --corpus <scan corpus>"],
                None,
            )
        )
    elif verdict.status == "REJECTED":
        kinds.append(
            (
                "new_hypothesis",
                "rejected; the null result stands, so try another idea",
                [revise, "openinterp verify --audit-file <new submission> --corpus <scan corpus>"],
                None,
            )
        )
    elif len(state.reviewers) < 2:
        kinds.append(("replicate", "verified by only one reviewer model", [review], None))
    elif not state.independent_reviews:
        kinds.append(("replicate", "every review so far was a self-review", [review], None))
    else:
        kinds.append(
            ("challenge", "verified and replicated; look for counterexamples", [review], None)
        )
    other_slug = ident.slug.rsplit("-", 1)[0] + f"-{other}"
    if not is_head and other_slug not in slugs:  # a head has no poles
        kinds.append(
            (
                "opposite_pole",
                f"the {other} pole is unstudied",
                [f"openinterp audit {target} --pole {other}"],
                other_slug,
            )
        )

    factor, notes = _decay(state, claims, now)
    for kind, why, commands, slug in kinds:
        yield Candidate(
            kind=kind,
            model=ident.model_name,
            weight=BASE_WEIGHTS[kind] * factor,
            rationale="; ".join([why, *notes]),
            commands=commands,
            slug=slug or ident.slug,
            skills=SKILLS[kind],
        )


@dataclass
class Plan:
    seed: int
    choice: Candidate
    explored: bool
    candidates: list[Candidate]
    explore_models: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def plan_next(
    wiki_dir: Path = Path("wiki"),
    *,
    seed: int | None = None,
    epsilon: float = 0.7,
    temperature: float = 1.0,
    models: Sequence[str] = (),
    claims: Sequence[str] = (),
    now: datetime | None = None,
) -> Plan:
    """Draw the next experiment.

    Explore draws use ``models`` if given, else the models configured in the
    ``pyproject.toml`` next to ``wiki_dir``, else the registry's, else ``DEFAULT_MODELS``.
    """
    seed = random.SystemRandom().getrandbits(32) if seed is None else seed
    rng = random.Random(seed)
    now = now or datetime.now(UTC)
    states = load_states(wiki_dir)
    slugs = set(states)
    candidates = sorted(
        (c for s in states.values() for c in candidates_for(s, slugs, claims, now)),
        key=lambda c: -c.weight,
    )
    pool = sorted(
        set(models)
        or set(configured_models(Path(wiki_dir).parent / "pyproject.toml"))
        or {s.record.submission.identifier.model_name for s in states.values()}
        or set(DEFAULT_MODELS)
    )
    explore = explore_weights(pool, states, claims)

    if not candidates or rng.random() < epsilon:
        model = rng.choices(pool, weights=[explore[m] for m in pool], k=1)[0]
        audit_seed = rng.getrandbits(32)
        rationale = "uniformly random new neuron (unbiased; counts towards registry statistics)"
        if len(pool) > 1:
            rationale += f"; {model} drawn with probability {explore[model]:.0%} by coverage"
        choice = Candidate(
            kind="explore",
            model=model,
            weight=0.0,
            rationale=rationale,
            commands=[f"openinterp audit --model {model} --random --seed {audit_seed}"],
            skills=SKILLS["explore"],
        )
        return Plan(seed, choice, True, candidates, explore)

    weights = [c.weight ** (1.0 / temperature) for c in candidates]
    choice = rng.choices(candidates, weights=weights, k=1)[0]
    return Plan(seed, choice, False, candidates, explore)
