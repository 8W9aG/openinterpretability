"""Explanation records: why a model gives one answer, as the registry stood (issue #3744).

``open-interpretability explain --record`` writes one. It holds the prompt and the answer,
the shares and prompt coverage. It holds the mechanism: each unit and group, with its share
and status. It also holds the narrated long form and paragraph (#3743), with the narrator.
Each record lives at ``wiki/explanations/<model slug>/<prompt id>/<UTC>.json``, so a prompt
keeps its history as the registry grows.

An explanation is a snapshot. Each unit's status is the one at the time of writing. When a
cited unit's status has changed since, the site marks the explanation stale
(:mod:`openinterpretability.builder.explanation_pages`). A newer explanation of the prompt
is listed beside it on the prompt's page (#4503).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import Field

from openinterpretability.core.corpora import CorpusSource
from openinterpretability.core.identifiers import model_slug
from openinterpretability.core.run_environment import RunEnvironment
from openinterpretability.core.strict import StrictModel

EXPLANATIONS_WIKI = "explanations"  # records live in wiki/explanations/
# The values that reach file paths and generated commands. A model name is `org/name`, or a
# bare name in tests. A unit slug is a pole's or a head's, matched in full.
MODEL_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$"
STAMP = r"^\d{8}T\d{6}Z$"  # a record's id: the UTC time it was written
FUTURE_SLACK = timedelta(hours=1)  # clock skew allowed for a record's written stamp (#5173)
LEVELS = ("mechanism", "paragraph", "long")  # each level holds the ones before it
UNIT_SLUG = r"^[A-Za-z0-9][A-Za-z0-9._-]*/L\d+/(N\d+-(positive|negative)|H\d+)$"


class HubNote(StrictModel):
    """A top unit that is a hub's center, and the facts that say whether it is an artefact.

    A hub is wired to many studied units, and it can be an artefact. A unit that is on most
    of the time, or that writes with a large norm, touches many units anyway. These facts
    show which, and they are reported, not gated (``docs/circuits-and-hubs.md``, #4472).
    """

    hub: str = Field(description="The hub's id: <model>/L5/H1/HUB.")
    href: str = Field(description="Its page on the site.")
    norm: float | None = None  # the center's output-weight norm
    layer_median: float | None = None  # its layer's median norm
    percentile: float | None = None  # the share of its layer with a smaller norm
    active_share: float | None = None  # of the scan's tokens, from its newest experiment
    default_on: bool | None = None  # whether that experiment shows it default-on


class ExplainedUnit(StrictModel):
    """One unit or group of the mechanism, as it stood when the explanation was written."""

    unit: str = Field(description="L3/N7+, L2/H10, L20/S0123456789ab or C4.")
    slug: str | None = Field(
        default=None, pattern=UNIT_SLUG, description="Its registry slug, for a pole or head."
    )
    kind: str  # pole, head, superposition or circuit
    share: float
    attribution: float
    status: str
    explanation: str | None = None  # only when VERIFIED
    hub: HubNote | None = None  # when the unit is a hub's center (#4472)


class Narration(StrictModel):
    """The narrated levels (#3743)."""

    sections: list[dict[str, str]] = Field(default_factory=list)  # {title, text}
    paragraph: str
    narrator_model: str
    citations_not_in_evidence: list[str] = Field(default_factory=list)


class FigureUnit(StrictModel):
    """One top unit, as the explanation page's figures draw it (#4479)."""

    label: str  # L3/N7+ or L5/H1
    layer: int
    kind: str  # pole or head
    attribution: float  # signed
    status: str
    explained: bool  # VERIFIED, or in a VERIFIED superposition or circuit
    blocked: bool = False  # no experiment can raise it


class ExplanationFigures(StrictModel):
    """The measured data of the explanation page's figures, so the build needs no model (#4479)."""

    tokens: list[str]  # the prompt's tokens
    logit: float  # the answer's logit, where the waterfall ends
    units: list[FigureUnit]  # the top units, by |attribution|
    net: float  # the signed attribution of every unit, summed
    edges: list[dict[str, Any]] = Field(default_factory=list)  # {source, target, weight}, by label
    coverage: float | None = None  # what the explained units bring back
    random_draws: list[float] = Field(default_factory=list)  # random units of the same count
    reading: dict[str, list[float]] = Field(default_factory=dict)  # label → strength per token
    ablation: list[dict[str, Any]] = Field(default_factory=list)  # {token, before, after}
    groups: list[dict[str, Any]] = Field(default_factory=list)  # {name, kind, status, members}


# A page cautions that coverage shows little in two cases (#4777). Coverage beats the random
# baseline by less than CAUTION_MARGIN. Or verified units hold less than CAUTION_SHARE.
CAUTION_MARGIN = 0.2
CAUTION_SHARE = 0.25


class CoverageDetail(StrictModel):
    """What prompt coverage rests on (#4777): a matched baseline, each kind, a minimal set."""

    random_matched: bool = True  # the baseline drew each layer's heads and poles alike
    random_spread: float | None = None  # its draws' standard deviation
    heads_only: float | None = None  # coverage with only the explained heads live
    poles_only: float | None = None  # and with only the explained neuron poles live
    heads_kept: dict[int, int] = Field(default_factory=dict)  # explained heads, by layer
    n_heads: int | None = None  # the model's heads per layer
    minimal_units: list[str] = Field(default_factory=list)  # the pruned set's labels
    minimal_coverage: float | None = None  # its coverage

    @classmethod
    def of(cls, coverage: dict[str, Any]) -> CoverageDetail | None:
        """The detail of ``explain_coverage.prompt_coverage``'s result, or None without it."""
        from openinterpretability.explain import PoleAttribution

        if "random_matched" not in coverage:
            return None
        return cls(
            random_spread=coverage.get("random_spread"),
            heads_only=coverage.get("heads_only"),
            poles_only=coverage.get("poles_only"),
            heads_kept=coverage.get("heads_kept") or {},
            n_heads=coverage.get("n_heads"),
            minimal_units=[
                PoleAttribution(layer, index, pole, 0.0, 0.0).label
                for layer, index, pole in coverage.get("minimal_units", [])
            ],
            minimal_coverage=coverage.get("minimal_coverage"),
        )


class ExplainRun(StrictModel):
    """What ``explain --run`` did before this explanation was written (#4474)."""

    share_before: float  # the prompt coverage when the run started (#4476)
    share_after: float  # and when it stopped
    landed: list[str] = Field(default_factory=list, description="<slug> <verdict> <PR URL>")
    left_open: list[str] = Field(default_factory=list, description="PRs that didn't merge")
    # Experiments whose PRs were open when the run ended. The shares count their records,
    # though main doesn't hold them yet (#5826). Older runs waited for CI and have none.
    pending: list[str] = Field(default_factory=list, description="<slug> <verdict> <PR URL>")
    # Experiments of an `explain --local` run: only in the user's wiki/, not in the registry,
    # until `explain-share` opens their PR (#5827). Older runs have none.
    local: list[str] = Field(
        default_factory=list, description="<slug> <verdict> <record path, from the checkout>"
    )
    stopped: str  # why the run stopped
    shares: list[float] = Field(default_factory=list)  # the share before each draw (#4479)


class ExplanationInputs(StrictModel):
    """What ``reproduce`` needs to measure the explanation again, with no registry (#4793).

    The registry changes after a record is written. So the record keeps the unit sets as
    they stood, and the means' corpus and budget, which the attribution is measured against.
    """

    environment: RunEnvironment  # the device, dtype and model revision it was measured with
    token_id: int  # the answer's token id
    means_corpus_sha256: str  # the corpus of the means (explain.unit_means)
    means_corpus_source: CorpusSource | None = None  # how to build that corpus again
    means_tokens: int  # the corpus tokens that the means read
    means_window: int  # explain.WINDOW when it was written
    means_version: int  # explain.MEANS_VERSION when it was written
    top: int  # the top units listed (explain --top)
    edges: int  # the top units whose incoming paths are traced (explain --edges)
    explained: list[str]  # every explained unit's label, in attribution order
    blocked: list[str] = Field(default_factory=list)  # every blocked unit's label
    random_seed: int = 0  # the random baseline's seed (explain_coverage.prompt_coverage)
    random_draws: int = 4  # and its number of draws


def prompt_id(model: str, position: int, prompt: str) -> str:
    """A prompt's id: the start of the SHA-256 of the model, position and prompt."""
    return hashlib.sha256(f"{model}\n{position}\n{prompt}".encode()).hexdigest()[:12]


class ExplanationRecord(StrictModel):
    """One explanation of one answer, as the registry stood (module docstring)."""

    model_name: str = Field(pattern=MODEL_NAME)
    prompt: str
    position: int
    answer: str
    explained_share: float
    blocked_share: float
    prompt_coverage: float | None  # credited: at most 1 (#4507)
    raw_coverage: float | None = None  # uncapped; over 1 is an overshoot (#4507)
    random_coverage: float | None
    # None in records before #4777, whose random baseline was not matched by kind and layer.
    coverage_detail: CoverageDetail | None = None
    units: list[ExplainedUnit]
    narration: Narration | None = None
    planned_experiments: int = 0  # unit experiments that a complete explanation needs
    run: ExplainRun | None = None  # the experiments that --run landed first (#4474)
    figures: ExplanationFigures | None = None  # the page's figures' data (#4479)
    target: float | None = None  # the causal coverage it aimed at (#4476)
    previous: str | None = Field(  # the explanation of this prompt it builds on (#4503)
        default=None, pattern=STAMP, description="That record's id, its UTC stamp."
    )
    reused: bool = False  # no experiments ran: ``previous`` reached the target (#4503)
    # None in records before #4793, which can't be reproduced.
    inputs: ExplanationInputs | None = None
    package_version: str
    written: datetime

    @property
    def id(self) -> str:
        """The record's id among its prompt's explanations: its UTC stamp, as in its file name."""
        return f"{self.written:%Y%m%dT%H%M%SZ}"

    @staticmethod
    def detail(coverage: dict[str, Any]) -> CoverageDetail | None:
        """:meth:`CoverageDetail.of`, for a caller that imports only this class."""
        return CoverageDetail.of(coverage)

    @property
    def coverage_caution(self) -> bool:
        """True when coverage here doesn't show that the answer is understood (#4777).

        That is when coverage is within :data:`CAUTION_MARGIN` of the random baseline, or
        verified units hold less than :data:`CAUTION_SHARE` of the first-order attribution.
        """
        if self.prompt_coverage is None:
            return False
        near = (
            self.random_coverage is not None
            and self.prompt_coverage - self.random_coverage < CAUTION_MARGIN
        )
        return near or self.explained_share < CAUTION_SHARE

    @property
    def overshoot(self) -> bool:
        """True when keeping the explained units raised the answer above the clean model."""
        return self.raw_coverage is not None and self.raw_coverage > 1.0

    @property
    def level(self) -> str:
        """The highest level it has: ``long``, ``paragraph`` or ``mechanism``."""
        if self.narration is None:
            return "mechanism"
        return "long" if self.narration.sections else "paragraph"

    @property
    def prompt_id(self) -> str:
        """The prompt's id (:func:`prompt_id`)."""
        return prompt_id(self.model_name, self.position, self.prompt)


def record_path(wiki_dir: Path, record: ExplanationRecord) -> Path:
    """``wiki/explanations/<model slug>/<prompt id>/<UTC timestamp>.json``."""
    folder = wiki_dir / EXPLANATIONS_WIKI / model_slug(record.model_name) / record.prompt_id
    return folder / f"{record.id}.json"


def load_all(wiki_dir: Path) -> dict[tuple[str, str], list[tuple[Path, ExplanationRecord]]]:
    """Every valid record, by ``(model slug, prompt id)``, oldest first; none without the folder.

    A prompt is a page, and each of its records is an explanation of it (#4503). A record
    that doesn't validate is left out with a warning. So one bad file can't stop the planner or the
    site build. ``open-interpretability check`` rejects it (:func:`check_explanations`).
    """
    import sys

    from pydantic import ValidationError

    root = wiki_dir / EXPLANATIONS_WIKI
    out: dict[tuple[str, str], list[tuple[Path, ExplanationRecord]]] = {}
    for path in sorted(root.glob("*/*/*.json")) if root.is_dir() else []:
        try:
            record = ExplanationRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            print(f"warning: skipping an invalid explanation record: {path}", file=sys.stderr)
            continue
        out.setdefault((path.parent.parent.name, path.parent.name), []).append((path, record))
    return out


def best(records: list[ExplanationRecord]) -> ExplanationRecord | None:
    """The explanation with the highest prompt coverage it measured; a tie goes to the newest."""
    measured = [r for r in records if r.prompt_coverage is not None]
    return max(measured, key=lambda r: (r.prompt_coverage, r.written), default=None)


def prompt_records(
    wiki_dir: Path, model: str, prompt: str, position: int
) -> list[ExplanationRecord]:
    """The explanations of one prompt of one model, oldest first."""
    found = load_all(wiki_dir).get((model_slug(model), prompt_id(model, position, prompt)), [])
    return [r for _, r in found]


def check_explanations(wiki_dir: Path, report: Any, scope: set[Path] | None = None) -> None:
    """Add a problem to ``report`` for each explanation record that fails a check.

    Each record must validate, sit at its own path, and name a configured model.
    """
    from pydantic import ValidationError

    from openinterpretability.planner import configured_models, signed_models

    root = wiki_dir / EXPLANATIONS_WIKI
    if not root.exists():
        return
    models = {*configured_models(), *signed_models()}
    for path in sorted(root.rglob("*")):
        if path.is_dir() or (scope is not None and path not in scope):
            continue
        if path.suffix != ".json":
            report.add(path, "only explanation records (.json) belong in wiki/explanations/")
            continue
        try:
            record = ExplanationRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            report.add(path, f"not a valid explanation record: {error.errors()[0]['msg']}")
            continue
        if record_path(wiki_dir, record) != path:
            report.add(path, f"should be at {record_path(wiki_dir, record)}")
        if record.model_name not in models:
            report.add(path, f"{record.model_name} is not a configured model")
        if record.written > datetime.now(UTC) + FUTURE_SLACK:  # #5173: it would stay newest
            report.add(path, f"it is stamped {record.id}, which is in the future")
