"""The agents page: per agent model and role, its experiments, VERIFIED rate and costs.

A model takes part in an experiment as its proposer (``submitting_agent_model``), its
reviewer (``reviewer_agent_model``) or the judge that scored its explanation. Self-reviewed
experiments (the proposer's model also wrote the probes) are counted apart from the rate,
since a model passing its own probes inflates it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from openinterp.builder.tree import LABEL_PREFIX, Experiment, unit_href
from openinterp.core.costs import ROLE_ORDER, by_agent
from openinterp.core.exceptions import RegistryError

RECENT_PER_AGENT = 20  # rows in an agent page's "Most recent experiments"


def agent_slug(model: str) -> str:
    """The agent page's name: ``external/deepseek-flash[1m]`` → ``external-deepseek-flash-1m``."""
    return re.sub(r"[^a-z0-9._]+", "-", model.lower()).strip("-") or "unknown"


def agent_href(model: str) -> str:
    return f"agents/{agent_slug(model)}.html"


@dataclass
class RoleStats:
    role: str
    experiments: int = 0
    verified: int = 0
    self_reviewed: int = 0  # proposer and reviewer only
    self_verified: int = 0
    costs: dict | None = None  # a :func:`openinterp.core.costs.by_agent` row

    @property
    def independent(self) -> int:
        return self.experiments - self.self_reviewed

    @property
    def independent_verified(self) -> int:
        return self.verified - self.self_verified

    @property
    def rate(self) -> float | None:
        """VERIFIED / experiments, over independent reviews only. A judge scores
        explanations after the verdict, so its experiments' verdicts aren't its rate."""
        if self.role == "judge" or not self.independent:
            return None
        return self.independent_verified / self.independent


@dataclass
class AgentView:
    model: str
    href: str
    roles: dict[str, RoleStats] = field(default_factory=dict)
    experiments: int = 0  # experiments it took any part in, in any role
    recent: list[dict] = field(default_factory=list)  # newest first, any role

    @property
    def role_rows(self) -> list[RoleStats]:
        return [self.roles[r] for r in ROLE_ORDER if r in self.roles]


def _roles_of(e: Experiment) -> dict[str, str]:
    """``role → model`` for the agents behind one (reviewed) experiment."""
    sub, verdict = e.record.submission, e.record.verdict
    roles = {"proposer": sub.submitting_agent_model, "reviewer": verdict.reviewer_agent_model}
    if verdict.explanation_scores and verdict.explanation_scores.judge_model:
        roles["judge"] = verdict.explanation_scores.judge_model
    return roles


def agents_view(experiments: Sequence[Experiment]) -> list[AgentView]:
    """One view per agent model, most experiments first."""
    agents: dict[str, AgentView] = {}
    involved: dict[str, list[tuple[Experiment, list[str]]]] = {}

    def agent(model: str) -> AgentView:
        return agents.setdefault(model, AgentView(model, agent_href(model)))

    for e in experiments:
        verified = e.record.is_verified
        own = e.record.self_reviewed
        by_model: dict[str, list[str]] = {}
        for role, model in _roles_of(e).items():
            stats = agent(model).roles.setdefault(role, RoleStats(role))
            stats.experiments += 1
            stats.verified += verified
            if role != "judge" and own:
                stats.self_reviewed += 1
                stats.self_verified += verified
            by_model.setdefault(model, []).append(role)
        for model, roles in by_model.items():
            involved.setdefault(model, []).append((e, roles))

    costs = by_agent(
        (e.record.verdict.status, e.record.submission.provenance.costs)
        for e in experiments
        if e.record.submission.provenance
    )
    for row in costs:
        if row["agent"] in ROLE_ORDER:
            agent(row["model"]).roles.setdefault(row["agent"], RoleStats(row["agent"])).costs = row

    for model, view in agents.items():
        rows = involved.get(model, [])
        view.experiments = len(rows)
        newest = sorted(rows, key=lambda r: (r[0].record.verdict.timestamp, r[0].id), reverse=True)
        view.recent = [_recent_row(e, roles) for e, roles in newest[:RECENT_PER_AGENT]]

    _check_slugs_unique(agents)
    return sorted(agents.values(), key=lambda a: (-a.experiments, a.model))


def _recent_row(e: Experiment, roles: list[str]) -> dict:
    ident = e.record.submission.identifier
    return {
        "e": e,
        "roles": roles,
        "href": f"{unit_href(ident)}#{e.anchor}",
        "model_name": ident.model_name,
        "unit": f"L{ident.layer}/{LABEL_PREFIX[ident.kind]}{ident.index}",
        "pole": ident.pole,
    }


def _check_slugs_unique(agents: dict[str, AgentView]) -> None:
    seen: dict[str, str] = {}
    for model, view in agents.items():
        other = seen.setdefault(view.href, model)
        if other != model:
            raise RegistryError(f"agent models {other!r} and {model!r} share the page {view.href}")
