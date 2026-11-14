"""Choose the next experiment: randomised, exploration-balanced, herd-resistant.

Every neuron in the registry generates candidate follow-up experiments:

- Re-evaluate under the current protocol, or re-review independently.
- Revise, replicate or challenge.
- Audit the opposite pole, or try a new hypothesis after a rejection.
- Rescan a record that was scanned on the built-in sample.

A VERIFIED pole also offers its unstudied matches in other configured models
(``open-interpretability sisters``), audited with its explanation. Any match can be drawn, and any
pole of another model at random. But the pole's sister is drawn far more often than the
rest.

Each candidate gets a weight from its expected information value. The weight decays
with how often the neuron has been studied, with recent activity, and with open PRs
that already claim it. Weights are never zero: repeating an experiment is allowed, and
a null result is data.

With probability ``epsilon``, the planner explores instead. It picks a uniformly random
new neuron, the only selection that supports unbiased registry statistics. The model
comes from the configured list (``[tool.openinterp] models`` in ``pyproject.toml``).
The smallest model below its coverage target gets most of these draws, and the others
share the rest (``openinterp.planner_models``). Otherwise, the planner
samples a candidate in proportion to ``weight ** (1 / temperature)``. Sampling,
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
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from openinterp.core.agent_names import canonical_agent
from openinterp.core.licenses import ModelLicense
from openinterp.core.paths import DEFAULT_WIKI_DIR
from openinterp.core.schemas import (
    PROTOCOL_VERSION,
    AuditRecord,
    ReviewPolicy,
    scan_active_share,
)
from openinterp.planner_models import (
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_OTHERS_SHARE,
    ModelCoverage,
    explore_weights,
    focus_model,
    model_coverage,
)

# Expected information value per experiment kind, before decay.
# Probability of an explore draw (a random new unit) instead of a follow-up.
DEFAULT_EPSILON = 0.7
BASE_WEIGHTS = {
    "reevaluate": 3.0,  # verdict predates the current protocol; cheap, no LLM calls
    "independent_review": 2.5,  # proposer reviewed its own hypothesis
    "revise": 2.0,  # NEEDS_REVISION: failed gates say exactly what to fix
    "rescan": 2.0,  # scanned on the built-in sample: its max activation means little
    "replicate": 1.2,  # VERIFIED by one reviewer model, or never by an independent one
    "opposite_pole": 1.0,  # the other half of a studied neuron
    "new_hypothesis": 0.8,  # REJECTED: the null result stands; try another idea
    "challenge": 0.6,  # VERIFIED and replicated: hunt for counterexamples
    "sister": 10.0,  # a ready hypothesis on its likeliest carrier (others: LINK_WEIGHT)
}
SKILLS = {
    "explore": ["native-neuron-audit", "neuron-family-triage", "vocab-projection-lookahead"],
    "reevaluate": [],
    "independent_review": ["interpretability-illusions-review"],
    "revise": ["native-neuron-audit", "explanation-simulation-scoring"],
    "rescan": ["native-neuron-audit", "neuron-family-triage"],
    "replicate": ["interpretability-illusions-review"],
    "opposite_pole": ["native-neuron-audit", "vocab-projection-lookahead"],
    "new_hypothesis": ["native-neuron-audit", "neuron-steering-tests"],
    "challenge": ["interpretability-illusions-review", "neuron-steering-tests"],
    "sister": ["native-neuron-audit", "interpretability-illusions-review"],
}
DEFAULT_MODELS = ("Qwen/Qwen2.5-0.5B",)


@dataclass(frozen=True)
class Candidate:
    """One experiment the planner can draw, with its weight and the commands to run it."""

    kind: str
    model: str
    weight: float
    rationale: str
    commands: list[str]
    slug: str | None = None
    skills: list[str] = field(default_factory=list)
    layer: int | None = None
    record: str | None = None  # the record a follow-up builds on
    audit: str | None = None  # `open-interpretability audit` options, for kinds that scan


@dataclass
class NeuronState:
    """What the registry holds on one unit: its latest record and how much it has been studied."""

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
    """Return ``[tool.openinterp.window_tokens]``, the scan window of some models.

    These are the models whose context is shorter than the corpus's documents. GPT-2
    reads 1,024 positions.
    """
    return dict(_config(pyproject).get("window_tokens", {}))


def configured_licenses(pyproject: Path = Path("pyproject.toml")) -> dict[str, ModelLicense]:
    """Return ``[tool.openinterp.licenses]``, the terms of each model (empty if unset).

    A malformed entry raises a validation error, so a typo cannot hide a term.
    """
    licenses = _config(pyproject).get("licenses", {})
    return {model: ModelLicense(**terms) for model, terms in licenses.items()}


def coverage_target(pyproject: Path = Path("pyproject.toml")) -> float:
    """Return ``[tool.openinterp] coverage_target``, or the default.

    A model's verified share must pass it before most explore draws move to the next model.
    """
    return float(_config(pyproject).get("coverage_target", DEFAULT_COVERAGE_TARGET))


def others_share(pyproject: Path = Path("pyproject.toml")) -> float:
    """Return ``[tool.openinterp] others_share``, or the default.

    It is the share of explore draws for the models that are not the focus.
    """
    return float(_config(pyproject).get("others_share", DEFAULT_OTHERS_SHARE))


def load_states(wiki_dir: Path) -> dict[str, NeuronState]:
    """One state per neuron pole, derived from all of its archived experiments.

    Follow-ups are planned from the most recent experiment; the count, reviewers and
    recency cover every experiment on the neuron.
    """
    from openinterp.builder.wiki import load_archive_paths  # Jinja: keep the CLI start light

    states: dict[str, NeuronState] = {}
    for path, record in load_archive_paths(Path(wiki_dir)):  # oldest first
        slug = record.submission.identifier.slug
        state = states.setdefault(slug, NeuronState(slug, record, path))
        state.record, state.record_path = record, path
        state.experiments += 1
        state.reviewers.add(canonical_agent(record.verdict.reviewer_agent_model))
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
    """The follow-up candidates for one studied unit, weighted."""
    record = state.record
    ident = record.submission.identifier
    commands = _commands(state)
    if record.scanned_on_builtin_sample:
        # Every review kind would inherit a 552-token scan, and so would its default-on
        # gate: rescan on the real corpus instead.
        kinds = _rescan(commands)
    else:
        kinds = _review_kinds(state, commands)
        if _default_on(record):
            # A single default-on pole can't be audited as a detector: a new audit stops at the
            # scan, and a review fails the gate. Re-evaluating can still correct an old verdict.
            kinds = [k for k in kinds if k[0] == "reevaluate"]
    kinds += _opposite_pole(state, slugs, commands)

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


def _default_on(record: AuditRecord) -> bool:
    """Return whether the record shows a default-on unit, by its gate or by its histogram.

    Records from before protocol 0.8 have no ``not_default_on`` gate, so the histogram is
    read as ``check_default_on`` reads it, with the current policy.
    """
    metrics = record.verdict.metrics
    if metrics and metrics.gates.get("not_default_on") is False:
        return True
    policy = ReviewPolicy()
    share = scan_active_share(record.submission, policy)
    return share is not None and share >= policy.max_active_share


def _audit_options(commands: list[str]) -> str | None:
    """The options of the audit command among ``commands``, without --supersedes.

    The audit command is the one that starts with ``open-interpretability audit``.
    """
    for command in commands:
        if command.startswith("open-interpretability audit "):
            return command.removeprefix("open-interpretability audit ").split(" --supersedes ")[0]
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
        f"open-interpretability verify --audit-file {path} --corpus <scan corpus> "
        f"--agent-model <model other than {proposer}>"
    )
    pole = f" --pole {ident.pole}" if ident.pole else ""
    revise = f"open-interpretability audit {target}{pole} --supersedes {path}"
    return _Commands(path, target, review, revise)


Kind = tuple[str, str, list[str], str | None]  # kind, why, commands, slug (if another unit's)


def _review_kinds(state: NeuronState, c: _Commands) -> list[Kind]:
    """Follow-ups on the record's own verdict: re-evaluate, review, revise or replicate."""
    record, verdict = state.record, state.record.verdict
    kinds: list[Kind] = []
    if verdict.policy is None or verdict.policy.protocol_version != PROTOCOL_VERSION:
        reevaluate = [f"open-interpretability reevaluate --record {c.path}"]
        kinds.append(("reevaluate", "verdict predates the current protocol", reevaluate, None))
    # Not the gate: it passes self-reviews unless the policy requires independence.
    if record.self_reviewed:
        kinds.append(
            ("independent_review", "the proposer reviewed its own hypothesis", [c.review], None)
        )
    gates = verdict.metrics.gates if verdict.metrics else {}
    failed = [g for g, ok in gates.items() if not ok]
    resubmit = [
        c.revise,
        "open-interpretability verify --audit-file <new submission> --corpus <scan corpus>",
    ]
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


def _rescan(c: _Commands) -> list[Kind]:
    """Return a new audit on the scan corpus, to supersede a built-in-sample record.

    Re-evaluating or re-reviewing the record would reuse its scan.
    """
    why = "scanned on the 552-token built-in sample; rescan on the corpus"
    verify = "open-interpretability verify --audit-file <new submission> --corpus <scan corpus>"
    return [("rescan", why, [c.revise, verify], None)]


def _opposite_pole(state: NeuronState, slugs: set[str], c: _Commands) -> list[Kind]:
    ident = state.record.submission.identifier
    other = "negative" if ident.pole == "positive" else "positive"
    other_slug = ident.slug.rsplit("-", 1)[0] + f"-{other}"
    if not ident.pole or other_slug in slugs:  # heads and features have no poles
        return []
    audit = f"open-interpretability audit {c.target} --pole {other}"
    return [("opposite_pole", f"the {other} pole is unstudied", [audit], other_slug)]


@dataclass
class Plan:
    """The planner's choice, the candidates it drew from, and the seed that replays it."""

    seed: int
    choice: Candidate
    explored: bool
    candidates: list[Candidate]
    explore_models: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, ModelCoverage] = field(default_factory=dict)
    focus: str | None = None  # the model that explore draws favour

    def to_json(self) -> str:
        """The plan as indented JSON."""
        return json.dumps(asdict(self), indent=2)


def plan_next(
    wiki_dir: Path = DEFAULT_WIKI_DIR,
    *,
    seed: int | None = None,
    epsilon: float = DEFAULT_EPSILON,
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
    pyproject = Path(wiki_dir).parent / "pyproject.toml"
    pool = _model_pool(models, pyproject, states)
    # Imported here: planner_sisters builds its candidates from this module's weights.
    from openinterp.planner_sisters import sister_candidates

    sisters = sister_candidates(states, claims, now, pool, Path(wiki_dir).parent, seed)
    candidates = sorted(
        [*(c for s in states.values() for c in candidates_for(s, slugs, claims, now)), *sisters],
        key=lambda c: -c.weight,
    )
    records = (s.record for s in states.values())
    coverage = model_coverage(pool, records, claims, dead(wiki_dir), stopped(wiki_dir))
    target = coverage_target(pyproject)
    explore = explore_weights(coverage, target, others_share(pyproject))
    focus = focus_model(coverage.values(), target)
    if not candidates or rng.random() < epsilon:
        choice = _explore(rng, pool, explore, pyproject, focus, slugs | set(claims))
        return Plan(seed, choice, True, candidates, explore, coverage, focus)

    weights = [c.weight ** (1.0 / temperature) for c in candidates]
    choice = rng.choices(candidates, weights=weights, k=1)[0]
    return Plan(seed, choice, False, candidates, explore, coverage, focus)


def stopped(wiki_dir: Path) -> list:
    """Every unit that a scan stopped on (dead or default-on), chosen or drawn at random.

    Such a unit has no experiment, but it counts as studied.
    """
    from openinterp.core.draws import load_draws, load_stops

    wiki_dir = Path(wiki_dir)
    return [d.identifier for d in (*load_draws(wiki_dir), *load_stops(wiki_dir))]


def dead(wiki_dir: Path) -> list:
    """Every neuron pole that a scan found dead and the weights proved dead.

    The proof does not depend on how the unit was selected. So this reads the stops of
    chosen units (:func:`load_stops`) as well as random draws. Random-selection
    statistics still read the draws only.
    """
    from openinterp.core.draws import load_draws, load_stops

    wiki_dir = Path(wiki_dir)
    return [d.identifier for d in (*load_draws(wiki_dir), *load_stops(wiki_dir)) if d.proven_dead]


def _model_pool(models: Sequence[str], pyproject: Path, states) -> list[str]:
    """``models``, else the configured models, else the registry's, else the defaults."""
    return sorted(
        set(models)
        or set(configured_models(pyproject))
        or {s.record.submission.identifier.model_name for s in states.values()}
        or set(DEFAULT_MODELS)
    )


def _explore(
    rng: random.Random, pool: list[str], explore: dict, pyproject: Path, focus: str | None,
    taken: set[str] = frozenset(),
) -> Candidate:  # fmt: skip
    """A new unit of a model drawn by size and coverage: random, or weighted by effect."""
    from openinterp.planner_effects import effect_draw
    from openinterp.planner_heads import head_draw, neuron_draw

    model = rng.choices(pool, weights=[explore[m] for m in pool], k=1)[0]
    audit_seed = rng.getrandbits(32)
    window = configured_windows(pyproject).get(model)
    if head := head_draw(model, audit_seed, pyproject, window):  # the unit kind, its own draw
        return head
    if weighted := effect_draw(rng, model, pyproject, taken, window):
        return weighted
    why = "the smallest model below its verified target" if model == focus else "coverage"
    return neuron_draw(model, audit_seed, window, explore[model] if len(pool) > 1 else None, why)
