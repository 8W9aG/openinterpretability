"""Explanation pages: a page per prompt, and a page per explanation of it (#3744, #4503).

Explanations are sub-units of a model. A prompt is a page,
``models/<model>/explanations/<prompt id>.html``. Each explanation of it is an experiment,
with its own page at ``…/<prompt id>/<UTC stamp>.html``. The prompt's page lists them in a
table, newest first. ``open-interpretability build`` loads every explanation record
(:mod:`openinterpretability.core.explanation_records`) and the registry's current statuses.
An explanation is stale when a unit it cites has a different status now. For example, a
unit that was unstudied when it was written can have a VERIFIED record now. Its page says
which units changed, so a reader knows that a newer explanation would say more.

Each page leads with the newest answer claim's verdict (#4778,
:mod:`openinterpretability.core.answer_records`). The answer is understood only when a claim
about how the model produces it is VERIFIED. Then the page shows the paragraph, the long
form, and the mechanism with a link to each unit's page. ``explanations/index.html`` lists
the prompts by model. The pages that were at ``explanations/<model>/<prompt id>.html``
before #4503 redirect to the prompt's page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from openinterpretability.core.explanation_records import EXPLANATIONS_WIKI

# A pole or head slug, matched in full (the same grammar as explanation_records.UNIT_SLUG).
UNIT_PAGE = re.compile(
    r"^(?P<model>[A-Za-z0-9][A-Za-z0-9._-]*)/L(?P<layer>\d+)/"
    r"(?P<unit>[NH]\d+)(?:-(?P<pole>positive|negative))?$"
)


@dataclass
class ExplanationView:
    """One explanation, with its links and what has changed since it was written."""

    record: object  # ExplanationRecord
    source: str  # its path in the repository, for the source link
    changed: list[dict] = field(default_factory=list)  # {unit, then, now}
    previous: ExplanationView | None = None  # the explanation it builds on (#4503)
    broad: set[str] = field(default_factory=set)  # cited unit slugs that are broad now (#4545)

    @property
    def prompt_href(self) -> str:
        """The prompt's page: ``models/<model slug>/explanations/<prompt id>.html`` (#4503)."""
        from openinterpretability.builder.hrefs import prompt_href
        from openinterpretability.core.identifiers import model_slug

        return prompt_href(model_slug(self.record.model_name), self.record.prompt_id)

    @property
    def href(self) -> str:
        """Its own page: ``models/<model slug>/explanations/<prompt id>/<id>.html`` (#4503)."""
        from openinterpretability.builder.hrefs import explanation_href
        from openinterpretability.core.identifiers import model_slug

        r = self.record
        return explanation_href(model_slug(r.model_name), r.prompt_id, r.id)

    @property
    def stale(self) -> bool:
        """True when a cited unit's status has changed since it was written."""
        return bool(self.changed)

    @property
    def landed(self) -> int:
        """The experiments that its run landed."""
        run = getattr(self.record, "run", None)
        return len(run.landed) if run is not None else 0


@dataclass
class AnswerView:
    """One answer experiment, with its page and its record's file (#4909)."""

    record: object  # AnswerRecord
    source: str  # its path in the repository, for the source link

    @property
    def href(self) -> str:
        """Its page: ``models/<model>/explanations/<prompt id>/answers/<stamp>.html``."""
        from openinterpretability.builder.hrefs import answer_href
        from openinterpretability.core.identifiers import model_slug

        c = self.record.claim
        return answer_href(model_slug(c.model_name), c.prompt_id, self.record.id)

    @property
    def status(self) -> str:
        """Its verdict."""
        return self.record.verdict.status

    @property
    def route(self) -> list[dict]:
        """The route in order: each unit's label, role and unit page."""
        from openinterpretability.core.answer_records import parse_unit
        from openinterpretability.core.identifiers import model_slug

        c, out = self.record.claim, []
        for label in c.route:
            layer, index, pole = parse_unit(label)
            unit = f"H{index}" if pole == "head" else f"N{index}"
            href = f"models/{model_slug(c.model_name)}/L{layer}/{unit}.html"
            out.append({"unit": label, "role": c.roles.get(label, ""), "href": href})
        return out


@dataclass
class PromptView:
    """One prompt of one model: its page lists every explanation of it, newest first (#4503).

    ``record``, ``changed`` and ``stale`` are the newest explanation's, so the planner and
    the lists read a prompt as its newest explanation.
    """

    explanations: list[ExplanationView]  # newest first
    answers: list[AnswerView] = field(default_factory=list)  # oldest first (#4778, #4909)

    @property
    def answer(self) -> AnswerView | None:
        """The newest answer experiment, or None (#4778)."""
        return self.answers[-1] if self.answers else None

    @property
    def standing_answer(self) -> AnswerView | None:
        """The answer experiment the verdict rests on: a VERIFIED one, else the newest."""
        verified = [a for a in self.answers if a.status == "VERIFIED"]
        return verified[-1] if verified else self.answer

    @property
    def understood(self) -> bool:
        """True when a claim about how the model produces the answer is VERIFIED (#4778)."""
        return any(a.status == "VERIFIED" for a in self.answers)

    @property
    def newest(self) -> ExplanationView:
        """The newest explanation of the prompt."""
        return self.explanations[0]

    @property
    def record(self):
        """The newest explanation's record."""
        return self.newest.record

    @property
    def changed(self) -> list[dict]:
        """The newest explanation's changed units."""
        return self.newest.changed

    @property
    def stale(self) -> bool:
        """True when the newest explanation is stale."""
        return self.newest.stale

    @property
    def href(self) -> str:
        """The prompt's page."""
        return self.newest.prompt_href

    @property
    def best_coverage(self) -> float | None:
        """The highest prompt coverage that an explanation of it measured."""
        from openinterpretability.core.explanation_records import best

        found = best([v.record for v in self.explanations])
        return found.prompt_coverage if found is not None else None

    @property
    def model_href(self) -> str:
        """The model's page."""
        from openinterpretability.builder.hrefs import model_href
        from openinterpretability.core.identifiers import model_slug

        return model_href(model_slug(self.record.model_name))

    @property
    def legacy_href(self) -> str:
        """Where the prompt's page was before #4503: ``explanations/<model slug>/<id>.html``."""
        from openinterpretability.core.identifiers import model_slug

        slug = model_slug(self.record.model_name)
        return f"{EXPLANATIONS_WIKI}/{slug}/{self.record.prompt_id}.html"


def unit_href(slug: str | None) -> str | None:
    """The unit page of a pole or head slug: ``models/<model>/L3/N7.html``."""
    found = UNIT_PAGE.match(slug or "")
    return f"models/{found['model']}/L{found['layer']}/{found['unit']}.html" if found else None


def load_views(wiki_dir: Path, states: dict | None = None) -> list[PromptView]:
    """Each prompt with all its explanations, by its newest explanation, newest first.

    Each explanation carries the units whose status has changed since it was written. Its
    text is scrubbed of personal data, as the site publishes it.
    """
    from openinterpretability.builder.pii import scrub
    from openinterpretability.core.answer_records import load_paths as load_answers
    from openinterpretability.core.explanation_records import load_all
    from openinterpretability.planner import load_states

    found = load_all(wiki_dir)
    if not found:
        return []
    states = load_states(wiki_dir) if states is None else states
    prompts = []
    for records in found.values():
        # What the site publishes is scrubbed of personal data and takedowns, as for
        # experiments (builder/pii.py, #4750).
        views = [_view(path, scrub(record), states, wiki_dir) for path, record in records]
        by_id = {v.record.id: v for v in views}
        for v in views:
            v.previous = by_id.get(v.record.previous or "")
        views.sort(key=lambda v: v.record.written, reverse=True)
        first = views[0].record
        answers = load_answers(wiki_dir, first.model_name, first.prompt_id)
        prompts.append(PromptView(views, [AnswerView(scrub(r), _source(path, wiki_dir))
                                          for path, r in answers]))  # fmt: skip
    return sorted(prompts, key=lambda p: p.record.written, reverse=True)


def _view(path: Path, record, states: dict, wiki_dir: Path) -> ExplanationView:
    """The explanation at ``path``, with the cited units whose status has changed since."""
    from openinterpretability.core.explanation import is_broad

    changed, broad = [], set()
    for u in record.units:
        if u.slug is None:
            continue  # a group: its status isn't in the unit states
        state = states.get(u.slug)
        now = state.standing[0] if state is not None else "unstudied"  # the standing (#4900)
        if now != u.status and u.status != "stopped":
            changed.append({"unit": u.unit, "then": u.status, "now": now})
        if state is not None and is_broad(state.record.verdict):
            broad.add(u.slug)
    return ExplanationView(record, _source(path, wiki_dir), changed, broad=broad)


def _source(path: Path, wiki_dir: Path) -> str:
    """A record's path in the repository, for its page's source link."""
    return path.relative_to(wiki_dir.parent).as_posix() if wiki_dir.parent else str(path)


def by_model(views) -> dict[str, list]:
    """The views of each model, keyed by model name, in the order given (#4206)."""
    out: dict[str, list] = {}
    for view in views:
        out.setdefault(view.record.model_name, []).append(view)
    return out


def render_explanations(render, prompts: list[PromptView]) -> None:
    """Write the index, each prompt's page, each explanation's page and the old pages' redirects.

    A record with figure data (#4479) gets its figures as SVG files in a folder beside its page.
    """
    from openinterpretability.builder.compiler import _write_redirect

    if not prompts:
        return
    render(f"{EXPLANATIONS_WIKI}/index.html", "explanations.html", groups=by_model(prompts))
    for prompt in prompts:
        render(prompt.href, "explanation_prompt.html", prompt=prompt)
        old = render.output_dir / prompt.legacy_href
        _write_redirect(old, f"../../{prompt.href}", render.report)
        for answer in prompt.answers:  # one page per answer experiment (#4909)
            from openinterpretability.builder.route_figures import route3d

            routes = write_routes(render, answer)  # the route on each prompt (#4905, #4917)
            render(answer.href, "answer.html", answer=answer, prompt=prompt, routes=routes,
                   route3d=route3d(answer.record))  # fmt: skip
        for view in prompt.explanations:
            drawn = write_figures(render, view)
            render(view.href, "explanation.html", view=view, prompt=prompt, unit_href=unit_href,
                   figures=drawn)  # fmt: skip


def write_routes(render, answer: AnswerView) -> dict:
    """Draw the route on each prompt into ``<page>/fig-route-<n>.svg``; prompt → figure."""
    from openinterpretability.builder.route_figures import answer_figures

    drawn = answer_figures(answer.record)
    if not drawn:
        return {}
    folder = render.output_dir / answer.href.removesuffix(".html")
    folder.mkdir(parents=True, exist_ok=True)
    for _, d in drawn:
        (folder / f"fig-{d.name}.svg").write_text(d.svg, encoding="utf-8")
        (folder / f"fig-{d.name}-static.svg").write_text(d.static or d.svg, encoding="utf-8")
    return dict(drawn)


def write_figures(render, view: ExplanationView) -> list:
    """Draw ``view``'s figures into ``<page>/fig-<name>.svg``, and return them for the page."""
    from openinterpretability.builder.explanation_figures import with_run

    data = getattr(view.record, "figures", None)
    if data is None:
        return []
    folder = render.output_dir / view.href.removesuffix(".html")
    folder.mkdir(parents=True, exist_ok=True)
    drawn = with_run(data, getattr(view.record, "run", None), view.record.answer)
    for d in drawn:
        (folder / f"fig-{d.name}.svg").write_text(d.svg, encoding="utf-8")
        if d.static:
            (folder / f"fig-{d.name}-static.svg").write_text(d.static, encoding="utf-8")
    return drawn
