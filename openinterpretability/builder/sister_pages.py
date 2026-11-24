"""A page for each sister match, and an index of each model's matches (#6639).

A sister is a mutual best match above the rotated baseline (:attr:`sisters.SisterRow.sister`).
One match can be found from both of its poles' models, so it can have two rows. It gets
one page, named by its two poles in sorted order (:func:`hrefs.sister_match_href`). The
page shows both poles, each row's measures and every *sister study* between the two. A
sister study tests one pole with the other pole's explanation (``submission.sister_of``).
A best match that isn't a sister has no page, and stays in the neuron page's table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openinterpretability.builder.hrefs import sister_index_href, sister_match_href
from openinterpretability.builder.sister_graph import sister_graphs
from openinterpretability.sisters import SisterRow, pole_href

__all__ = ["render_sister_pages", "sister_graphs", "sister_matches"]


@dataclass
class PoleView:
    """One pole of a match: where it is, and what the registry says of it."""

    slug: str
    href: str | None  # its unit page, when the registry has one
    status: str | None = None
    explanation: str | None = None

    @property
    def model(self) -> str:
        """The model's name, such as ``openai-community/gpt2``.

        Examples:
            >>> PoleView("openai-community--gpt2/L0/N5-positive", None).model
            'openai-community/gpt2'
        """
        return self.slug.split("/", 1)[0].replace("--", "/", 1)

    @property
    def label(self) -> str:
        """The pole's label in its model: ``L0/N1101+`` or ``L0/N1101−``.

        Examples:
            >>> PoleView("m--a/L0/N1101-negative", None).label
            'L0/N1101−'
        """
        _, layer, unit = self.slug.split("/")
        neuron, pole = unit.split("-")
        return f"{layer}/{neuron}{'+' if pole == 'positive' else '−'}"


@dataclass
class StudyView:
    """One sister study: the pole it tested, the pole whose explanation it used."""

    tested: str
    source: str
    status: str
    when: object  # the verdict's datetime
    href: str  # the experiment's page


@dataclass
class MatchView:
    """One sister match: its two poles, its rows and its studies."""

    a: PoleView
    b: PoleView
    rows: list[SisterRow] = field(default_factory=list)
    studies: list[StudyView] = field(default_factory=list)

    @property
    def href(self) -> str:
        """The match's page.

        Examples:
            >>> a, b = PoleView("m--a/L0/N7-negative", None), PoleView("m--b/L1/N2-positive", None)
            >>> MatchView(a, b).href
            'sisters/m--a-L0-N7-negative~m--b-L1-N2-positive.html'
        """
        return sister_match_href(self.a.slug, self.b.slug)

    @property
    def pearson(self) -> float:
        """The strongest correlation that a row of the match reports.

        Examples:
            >>> from types import SimpleNamespace as NS
            >>> rows = [NS(sister=NS(pearson=p), rotated=0.1) for p in (0.8, 0.9)]
            >>> MatchView(PoleView("m--a/L0/N7-negative", None), None, rows).pearson
            0.9
        """
        return max(r.sister.pearson for r in self.rows)

    @property
    def rotated(self) -> float:
        """The highest rotated baseline that a row of the match reports.

        Examples:
            >>> from types import SimpleNamespace as NS
            >>> rows = [NS(sister=None, rotated=r) for r in (0.2, 0.1)]
            >>> MatchView(PoleView("m--a/L0/N7-negative", None), None, rows).rotated
            0.2
        """
        return max(r.rotated for r in self.rows)


def pole_info(models) -> dict[str, tuple[str, str, str]]:
    """Each studied neuron pole's unit page, standing and explanation, by its slug.

    Args:
        models: The build's model tree (``tree.build_tree``).

    Returns:
        ``{slug: (href, status, explanation)}``.

    Examples:
        >>> pole_info([])
        {}
    """
    out = {}
    for model in models:
        for layer in model.layers:
            for node in layer.units:
                if not node.is_neuron:
                    continue
                for standing in node.standings.values():
                    sub = standing.lead.record.submission
                    out[sub.identifier.slug] = (node.href, standing.status,
                                                sub.natural_language_explanation)  # fmt: skip
    return out


def _pole(slug: str, info: dict, studied: set) -> PoleView:
    """A pole's view: its page when it has one, and its standing when it is studied.

    Args:
        slug: The pole's slug.
        info: Each studied pole's page, standing and explanation (:func:`pole_info`).
        studied: The unit pages that the site has.

    Returns:
        The view.
    """
    href = pole_href(slug)
    found = info.get(slug)
    status, explanation = (found[1], found[2]) if found else (None, None)
    return PoleView(slug, href if href in studied else None, status, explanation)


def sister_matches(rows: dict, experiments, models, studied: set) -> list[MatchView]:
    """Every sister match in ``rows``, each with its studies, strongest first.

    Args:
        rows: The sister rows by source slug (``sisters.load_rows``).
        experiments: The registry's experiments, which hold the sister studies.
        models: The build's model tree.
        studied: The unit pages that the site has.

    Returns:
        The matches.

    Examples:
        >>> sister_matches({}, [], [], set())
        []
    """
    info, found = pole_info(models), {}
    # One match can be found from both poles' models: key it by the sorted pair.
    for row in (r for rs in rows.values() for r in rs):
        if row.sister is None:
            continue
        a, b = sorted((row.slug, row.sister.slug(row.other)))
        match = found.setdefault(
            (a, b), MatchView(_pole(a, info, studied), _pole(b, info, studied))
        )
        match.rows.append(row)
    # A sister study tests one pole with the other's explanation, from either side.
    for e in experiments:
        ref = e.record.submission.sister_of
        if not ref:
            continue
        tested, source = e.record.submission.identifier.slug, ref.rsplit("/", 1)[0]
        match = found.get(tuple(sorted((tested, source))))
        if match is not None:
            v = e.record.verdict
            match.studies.append(StudyView(tested, source, v.status, v.timestamp, e.href))
    for match in found.values():
        match.studies.sort(key=lambda s: s.when, reverse=True)
    return sorted(found.values(), key=lambda m: (-m.pearson, m.href))


def by_model(matches: list[MatchView]) -> dict[str, dict[str, list[MatchView]]]:
    """Each model's matches, grouped by the other model, as ``{model: {other: matches}}``.

    Args:
        matches: The sister matches.

    Returns:
        The groups. Each match is under both of its models.

    Examples:
        >>> a, b = PoleView("m--a/L0/N7-negative", None), PoleView("m--b/L1/N2-positive", None)
        >>> by_model([MatchView(a, b)])
        {'m/a': {'m/b': [MatchView(...)]}, 'm/b': {'m/a': [MatchView(...)]}}
    """
    out: dict[str, dict[str, list[MatchView]]] = {}
    for m in matches:
        for here, there in ((m.a, m.b), (m.b, m.a)):
            out.setdefault(here.model, {}).setdefault(there.model, []).append(m)
    return out


def render_sister_pages(render, models, rows: dict, experiments, studied: set) -> None:
    """Render each sister match's page, and the index of each model's matches.

    Args:
        render: The site's renderer.
        models: The build's model tree.
        rows: The sister rows by source slug.
        experiments: The registry's experiments.
        studied: The unit pages that the site has.

    Examples:
        >>> pages = []
        >>> render_sister_pages(lambda *args, **kw: pages.append(args), [], {}, [], set())
        >>> pages
        []
    """
    matches = sister_matches(rows, experiments, models, studied)
    for m in matches:
        render(m.href, "sister_match.html", m=m)
    names = {model.name: model for model in models}
    for name, groups in by_model(matches).items():
        model = names.get(name)
        if model is not None:  # a sibling the registry doesn't audit has no model page
            render(sister_index_href(model.slug), "sisters.html", model=model, groups=groups)
