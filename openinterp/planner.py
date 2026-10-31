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
    layer: int | None = None
    record: str | None = None  # the record a follow-up builds on
    audit: str | None = None  # `openinterp audit` options, for kinds that scan


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


def _config(pyproject: Path) -> dict:
    try:
        with open(pyproject, "rb") as fh:
            return tomllib.load(fh).get("tool", {}).get("openinterp", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def configured_models(pyproject: Path = Path("pyproject.toml")) -> tuple[str, ...]:
    """Models the planner explores: ``[tool.openinterp] models`` (empty if unset)."""
    return tuple(_config(pyproject).get("models", ()))


def configured_windows(pyproject: Path = Path("pyproject.toml")) -> dict[str, int]:
    """``[tool.openinterp.window_tokens]``: the scan window of each model whose context is
    shorter than the corpus's documents (GPT-2 reads 1,024 positions)."""
    return dict(_config(pyproject).get("window_tokens", {}))


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
    commands = _commands(state)
    kinds = _review_kinds(state, commands) + _opposite_pole(state, slugs, commands)
    gates = verdict.metrics.gates if verdict.metrics else {}
    if gates.get("not_default_on") is False:
        # A default-on unit can't be audited as a detector, so no revision or review of it
        # can pass; only its other pole is still open.
        kinds = [k for k in kinds if k[0] in ("reevaluate", "opposite_pole")]

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
            layer=ident.layer,
            record=state.record_path.as_posix(),
            audit=_audit_options(commands),
        )


def _audit_options(commands: list[str]) -> str | None:
    """The options of the ``openinterp audit`` among ``commands``, without --supersedes."""
    for command in commands:
        if command.startswith("openinterp audit "):
            return command.removeprefix("openinterp audit ").split(" --supersedes ")[0]
    return None


@dataclass(frozen=True)
class _Commands:
    """The commands a follow-up on one record runs."""

    path: str
    target: str  # --model, --layer and the unit, as the record was scanned
    review: str
    revise: str


def _commands(state: NeuronState) -> _Commands:
    record = state.record
    ident = record.submission.identifier
    path = state.record_path.as_posix()
    if ident.kind == "attn_head":
        unit = f"--head {ident.head_idx}"
    elif ident.kind == "dictionary_feature":
        unit = f"--dictionary {ident.dictionary} --feature {ident.feature_idx}"
    else:
        unit = f"--neuron {ident.neuron_idx}"
    target = f"--model {ident.model_name} --layer {ident.layer} {unit}"
    provenance = record.submission.provenance
    if provenance and provenance.snippet_tokens:  # rescan as the record was scanned
        target += f" --window-tokens {provenance.snippet_tokens}"
    proposer = record.submission.submitting_agent_model
    review = (
        f"openinterp verify --audit-file {path} --corpus <scan corpus> "
        f"--agent-model <model other than {proposer}>"
    )
    pole = f" --pole {ident.pole}" if ident.pole else ""
    revise = f"openinterp audit {target}{pole} --supersedes {path}"
    return _Commands(path, target, review, revise)


Kind = tuple[str, str, list[str], str | None]  # kind, why, commands, slug (if another unit's)


def _review_kinds(state: NeuronState, c: _Commands) -> list[Kind]:
    """Follow-ups on the record's own verdict: re-evaluate, review, revise or replicate."""
    record, verdict = state.record, state.record.verdict
    kinds: list[Kind] = []
    if verdict.policy is None or verdict.policy.protocol_version != PROTOCOL_VERSION:
        reevaluate = [f"openinterp reevaluate --record {c.path}"]
        kinds.append(("reevaluate", "verdict predates the current protocol", reevaluate, None))
    # Not the gate: it passes self-reviews unless the policy requires independence.
    if record.self_reviewed:
        kinds.append(
            ("independent_review", "the proposer reviewed its own hypothesis", [c.review], None)
        )
    gates = verdict.metrics.gates if verdict.metrics else {}
    failed = [g for g, ok in gates.items() if not ok]
    resubmit = [c.revise, "openinterp verify --audit-file <new submission> --corpus <scan corpus>"]
    if verdict.status == "NEEDS_REVISION":
        kinds.append(("revise", f"failed gates: {', '.join(failed) or 'unknown'}", resubmit, None))
    elif verdict.status == "REJECTED":
        why = "rejected; the null result stands, so try another idea"
        kinds.append(("new_hypothesis", why, resubmit, None))
    elif len(state.reviewers) < 2:
        kinds.append(("replicate", "verified by only one reviewer model", [c.review], None))
    elif not state.independent_reviews:
        kinds.append(("replicate", "every review so far was a self-review", [c.review], None))
    else:
        why = "verified and replicated; look for counterexamples"
        kinds.append(("challenge", why, [c.review], None))
    return kinds


def _opposite_pole(state: NeuronState, slugs: set[str], c: _Commands) -> list[Kind]:
    ident = state.record.submission.identifier
    other = "negative" if ident.pole == "positive" else "positive"
    other_slug = ident.slug.rsplit("-", 1)[0] + f"-{other}"
    if not ident.pole or other_slug in slugs:  # heads and features have no poles
        return []
    audit = f"openinterp audit {c.target} --pole {other}"
    return [("opposite_pole", f"the {other} pole is unstudied", [audit], other_slug)]


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
    pyproject = Path(wiki_dir).parent / "pyproject.toml"
    pool = _model_pool(models, pyproject, states)
    explore = explore_weights(pool, states, claims)
    if not candidates or rng.random() < epsilon:
        choice = _explore(rng, pool, explore, pyproject)
        return Plan(seed, choice, True, candidates, explore)

    weights = [c.weight ** (1.0 / temperature) for c in candidates]
    choice = rng.choices(candidates, weights=weights, k=1)[0]
    return Plan(seed, choice, False, candidates, explore)


def _model_pool(models: Sequence[str], pyproject: Path, states) -> list[str]:
    """``models``, else the configured models, else the registry's, else the defaults."""
    return sorted(
        set(models)
        or set(configured_models(pyproject))
        or {s.record.submission.identifier.model_name for s in states.values()}
        or set(DEFAULT_MODELS)
    )


def _explore(rng: random.Random, pool: list[str], explore: dict, pyproject: Path) -> Candidate:
    """A random new unit of a model drawn by coverage (two draws from ``rng``)."""
    model = rng.choices(pool, weights=[explore[m] for m in pool], k=1)[0]
    audit_seed = rng.getrandbits(32)
    rationale = "uniformly random new neuron (unbiased; counts towards registry statistics)"
    if len(pool) > 1:
        rationale += f"; {model} drawn with probability {explore[model]:.0%} by coverage"
    audit = f"openinterp audit --model {model} --random --seed {audit_seed}"
    if window := configured_windows(pyproject).get(model):
        audit += f" --window-tokens {window}"
    return Candidate(
        kind="explore",
        model=model,
        weight=0.0,
        rationale=rationale,
        commands=[audit],
        skills=SKILLS["explore"],
        audit=_audit_options([audit]),
    )
