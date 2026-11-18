"""Answer experiments: a claim about how a model produces one answer (issue #4778).

An explanation can keep VERIFIED units and still not say how the model gets its answer.
An answer experiment tests one **route**: the source tokens, the units in order, and how
they write the answer. The proposer also gives counterfactual prompts. Each one predicts if
the answer stays the model's top token. The reviewer writes more of them, blind to the
results. :func:`decide` judges the measures against :class:`AnswerPolicy`:

- *Necessity.* Mean-ablating the route cuts the answer's log-probability by at least half
  the cut from ablating every unit. The cut is also 3× the mean cut of matched random sets.
- *Sufficiency.* With only the route live, the answer comes back by at least half.
- *Path patching.* The route's activations from the corrupted prompt move the answer by at
  least half the clean-to-corrupted gap. Matched random sets move it by at most 0.1.
- *Prompts.* Every proposer prompt, and 80% of the reviewer's prompts, come out as
  predicted.

One answer can take several attempts. Each attempt is its own record and sees the earlier
ones. The attempts stop at the first VERIFIED claim, or after :data:`MAX_ATTEMPTS`.

Answer verdicts have their own rule version, :data:`ANSWER_PROTOCOL_VERSION`, as circuits
have. ``PROTOCOL_VERSION``, the rule for units, is unchanged (``docs/PROTOCOL.md`` §4).
Records live in ``wiki/answers/<model slug>/<prompt id>/<UTC stamp>.json``.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from openinterpretability.core.explanation_records import prompt_id
from openinterpretability.core.identifiers import model_slug
from openinterpretability.core.strict import StrictModel

ANSWER_PROTOCOL_VERSION = "1.0"
ANSWERS_WIKI = "answers"  # records live in wiki/answers/
MAX_ATTEMPTS = 5  # answer claims on one prompt before the plan stops proposing more
AnswerStatus = Literal["VERIFIED", "REJECTED", "NEEDS_REVISION"]
# L3/N7+ or L3/N7- (or −) for a neuron pole, L3/H2 for a head.
UNIT = re.compile(r"^L(?P<layer>\d+)/(?:H(?P<head>\d+)|N(?P<neuron>\d+)(?P<sign>[+\-−]))$")
CAUSAL_GATES = ("necessity", "beats_null", "sufficiency", "patching", "patching_null")


def parse_unit(label: str) -> tuple[int, int, str]:
    """``(layer, index, pole or "head")`` from a label such as ``L3/N7+`` or ``L2/H6``."""
    found = UNIT.match(label)
    if found is None:
        raise ValueError(f"not a unit label: {label!r} (write L3/N7+, L3/N7- or L2/H6)")
    layer = int(found["layer"])
    if found["head"] is not None:
        return layer, int(found["head"]), "head"
    return layer, int(found["neuron"]), "positive" if found["sign"] == "+" else "negative"


class Counterfactual(StrictModel):
    """A changed prompt, and whether the claim predicts that the answer stays the top token."""

    prompt: str = Field(min_length=1)
    expect: Literal["kept", "changed"]
    why: str = Field(default="", description="What the route predicts here, and why.")


class AnswerClaim(StrictModel):
    """How the model produces ``answer`` at ``position`` of ``prompt``: the route."""

    model_name: str
    model_revision: str | None = None
    prompt: str = Field(min_length=1)
    position: int = -1
    answer: str = Field(min_length=1, description="The answer token, as text, e.g. ' sky'.")
    sources: list[str] = Field(min_length=1, description="The prompt's tokens the route reads.")
    route: list[str] = Field(min_length=1, description="The units in order: L2/H6, L3/N1920+.")
    roles: dict[str, str] = Field(default_factory=dict, description="Unit → what it does here.")
    explanation: str = Field(min_length=1, description="The route in a few sentences.")
    corrupted: str = Field(
        min_length=1,
        description="The prompt with the sources changed, of the same token length.",
    )
    counterfactuals: list[Counterfactual] = Field(min_length=4, max_length=8)
    attempt: int = Field(default=1, ge=1)
    supersedes: str | None = Field(default=None, description="The earlier attempt's id.")
    submitting_agent_model: str

    @property
    def prompt_id(self) -> str:
        """The prompt's id, as its explanations have it."""
        return prompt_id(self.model_name, self.position, self.prompt)


class AnswerProbes(StrictModel):
    """The reviewer's prompts, written blind to the measures, to break the claim."""

    prompts: list[Counterfactual] = Field(min_length=10, max_length=30)
    reviewer_agent_model: str


class AnswerPolicy(StrictModel):
    """The gates of an answer verdict (module docstring)."""

    min_necessity: float = 0.5  # of the cut from ablating every unit
    null_factor: float = 3.0  # the route's cut against the mean of matched random sets
    null_sets: int = 8
    min_sufficiency: float = 0.5  # the answer that the route alone brings back
    min_patching: float = 0.5  # of the clean-to-corrupted gap
    max_null_patching: float = 0.1  # matched random sets, on the mean
    min_controls: float = 1.0  # the proposer's counterfactuals that come out as predicted
    min_probes: float = 0.8  # the reviewer's prompts that come out as predicted
    protocol_version: str = ANSWER_PROTOCOL_VERSION


class LayerView(StrictModel):
    """The last token through the layers: a lens row and a spectrum row per layer (#4917)."""

    lens: list[dict[str, float]] = Field(default_factory=list)
    spectrum: list[list[list[float]]] = Field(default_factory=list)  # [layer][token][direction]
    space: list[list[list[float]]] = Field(default_factory=list)  # [layer][token][x, y, z]


class RouteReading(StrictModel):
    """The route on one prompt: what each unit does there, for the page's figures (#4917).

    Reported, never gated: the maintainer decided on 2026-11-18 that these don't change a
    verdict. Each unit is keyed by its label in the claim's route.
    """

    tokens: list[str]
    top: str  # the model's top token after the prompt
    answer_log_prob: float  # the claim's answer, at the prompt's last position
    route_share: float | None  # the route's cut over the cut from ablating every unit
    strength: dict[str, list[float]]  # unit → its strength at each token, 0 at the first
    attribution: dict[str, float]  # unit → its first-order attribution to the answer's logit
    edges: list[dict[str, Any]] = Field(default_factory=list)  # {source, target, weight}
    # The answer's log-probability with the original prompt's route values patched in. A
    # share of the gap would divide by almost 0 where the answer stays the same.
    patched_log_prob: float | None = None
    # Through the layers, the embedding first (answer_lens): the logit lens over the shared
    # candidate tokens, and the residual stream on the shared principal directions.
    lens: list[dict[str, float]] = Field(default_factory=list)
    spectrum: list[list[list[float]]] = Field(default_factory=list)  # [layer][token][direction]
    # Each token in the model's 3D embedding space, at each layer (embedding_space, #4929).
    space: list[list[list[float]]] = Field(default_factory=list)  # [layer][token][x, y, z]
    # The same with units switched off (mean-ablated). Every prompt has the whole route off.
    # The original prompt also has each route unit off on its own, keyed by its label.
    route_off: LayerView | None = None
    unit_off: dict[str, LayerView] = Field(default_factory=dict)


class AnswerMeasures(StrictModel):
    """What the review measured, in the answer's log-probability at its position."""

    clean: float
    all_cut: float  # clean minus every unit mean-ablated
    route_cut: float  # clean minus the route mean-ablated
    null_cuts: list[float]  # matched random sets
    sufficiency: float | None  # (route kept − all ablated) / (clean − all ablated), at most 1
    corrupted_gap: float  # clean minus the corrupted prompt
    patching: float | None  # the route patched from the corrupted run, over the gap
    null_patching: list[float | None]
    controls: dict[str, bool]  # proposer prompt → came out as predicted
    probes: dict[str, bool]  # reviewer prompt → came out as predicted
    top_tokens: dict[str, str] = Field(default_factory=dict)  # each prompt's top token
    # The route on each prompt: the original, the controls and the probes (#4917).
    readings: dict[str, RouteReading] = Field(default_factory=dict)

    @property
    def necessity(self) -> float:
        """The route's cut as a share of the cut from ablating every unit, or 0."""
        return self.route_cut / self.all_cut if self.all_cut > 0 else 0.0


class AnswerVerdict(StrictModel):
    """The decision, with the gates it rests on."""

    status: AnswerStatus
    gates: dict[str, bool]
    policy: AnswerPolicy
    measures: AnswerMeasures
    reviewer_agent_model: str
    timestamp: datetime

    @property
    def failed(self) -> list[str]:
        """The gates that failed, in the order :func:`decide` checks them."""
        return [gate for gate, ok in self.gates.items() if not ok]


class AnswerRecord(StrictModel):
    """One answer experiment, at ``wiki/answers/<model>/<prompt id>/<stamp>.json``."""

    claim: AnswerClaim
    probes: AnswerProbes
    verdict: AnswerVerdict
    remeasures: str | None = Field(  # #4917: the same claim and probes, measured again
        default=None, description="The id of the record this one measures again."
    )

    @property
    def id(self) -> str:
        """Its UTC stamp, as in its file name."""
        return f"{self.verdict.timestamp.astimezone(UTC):%Y%m%dT%H%M%SZ}"


def _rate(results: dict[str, bool]) -> float:
    return sum(results.values()) / len(results) if results else 0.0


def decide(measures: AnswerMeasures, policy: AnswerPolicy) -> tuple[AnswerStatus, dict[str, bool]]:
    """The status and the gates from the measures.

    VERIFIED when every gate passes. NEEDS_REVISION when the route carries the answer but
    the prompts don't come out as the claim says. Then the claim is too broad or too
    narrow. REJECTED otherwise.
    """
    m, p = measures, policy
    nulls = [n for n in m.null_patching if n is not None]
    null_cut = sum(m.null_cuts) / len(m.null_cuts) if m.null_cuts else 0.0
    gates = {
        "necessity": m.necessity >= p.min_necessity,
        "beats_null": m.route_cut > 0 and m.route_cut >= p.null_factor * max(null_cut, 0.0),
        "sufficiency": m.sufficiency is not None and m.sufficiency >= p.min_sufficiency,
        "patching": m.patching is not None and m.patching >= p.min_patching,
        "patching_null": bool(nulls) and sum(nulls) / len(nulls) <= p.max_null_patching,
        "controls": _rate(m.controls) >= p.min_controls,
        "probes": _rate(m.probes) >= p.min_probes,
    }
    if all(gates.values()):
        return "VERIFIED", gates
    if all(gates[g] for g in CAUSAL_GATES):
        return "NEEDS_REVISION", gates
    return "REJECTED", gates


def working_dir(prompt_hash: str) -> Path:
    """One prompt's prompts, claims and probes: ``answer-<prompt id>`` in the exchange folder.

    It is a working folder, not a cited input, so it is never published.
    """
    from openinterpretability.core.paths import DEFAULT_EXCHANGE_DIR

    return DEFAULT_EXCHANGE_DIR / f"answer-{prompt_hash}"


def folder(wiki_dir: Path, model: str, prompt_hash: str) -> Path:
    """``wiki/answers/<model slug>/<prompt id>/``."""
    return wiki_dir / ANSWERS_WIKI / model_slug(model) / prompt_hash


def record_path(wiki_dir: Path, record: AnswerRecord) -> Path:
    """``wiki/answers/<model slug>/<prompt id>/<UTC stamp>.json``."""
    c = record.claim
    return folder(wiki_dir, c.model_name, c.prompt_id) / f"{record.id}.json"


def load(wiki_dir: Path, model: str, prompt_hash: str) -> list[AnswerRecord]:
    """The answer records of one prompt, oldest first. Invalid files are left out."""
    return [record for _, record in load_paths(wiki_dir, model, prompt_hash)]


def load_paths(wiki_dir: Path, model: str, prompt_hash: str) -> list[tuple[Path, AnswerRecord]]:
    """The answer records of one prompt with their files, oldest first (#4909)."""
    from pydantic import ValidationError

    out = []
    for path in sorted(folder(wiki_dir, model, prompt_hash).glob("*.json")):
        try:
            out.append((path, AnswerRecord.model_validate_json(path.read_text(encoding="utf-8"))))
        except ValidationError:
            continue  # `open-interpretability check` rejects it (check_answers)
    return out


def summary(record: AnswerRecord) -> dict[str, Any]:
    """The claim and its verdict, for the plan and the narrator's evidence (#4780)."""
    c, v = record.claim, record.verdict
    m = v.measures
    return {
        "status": v.status, "attempt": c.attempt, "route": c.route, "sources": c.sources,
        "roles": c.roles, "claim": c.explanation, "failed_gates": v.failed,
        "necessity": round(m.necessity, 4), "sufficiency": m.sufficiency,
        "patching": m.patching, "reviewer_prompts_as_predicted": f"{sum(m.probes.values())}"
        f" of {len(m.probes)}",
    }  # fmt: skip


def next_attempt(records: list[AnswerRecord]) -> int | None:
    """The next attempt's number, or None when one is VERIFIED or the cap is reached."""
    claims = {r.claim.attempt for r in records}  # a re-measure is not an attempt (#4917)
    if any(r.verdict.status == "VERIFIED" for r in records) or len(claims) >= MAX_ATTEMPTS:
        return None
    return len(claims) + 1


def check_answers(wiki_dir: Path, report: Any, scope: set[Path] | None = None) -> None:
    """Add a problem to ``report`` for each answer record that fails a check.

    Each record must validate and sit at its own path. Under the current rule version, its
    status and gates must be what :func:`decide` gives from its stored measures.
    """
    from pydantic import ValidationError

    root = wiki_dir / ANSWERS_WIKI
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_dir() or (scope is not None and path not in scope):
            continue
        try:
            record = AnswerRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, UnicodeDecodeError) as error:
            msg = error.errors()[0]["msg"] if isinstance(error, ValidationError) else str(error)
            report.add(path, f"not a valid answer record: {msg}")
            continue
        if record_path(wiki_dir, record) != path:
            report.add(path, f"should be at {record_path(wiki_dir, record)}")
        verdict = record.verdict
        if verdict.policy.protocol_version == ANSWER_PROTOCOL_VERSION:
            status, gates = decide(verdict.measures, verdict.policy)
            if (status, gates) != (verdict.status, verdict.gates):
                report.add(path, f"the measures give {status}, not {verdict.status}")
