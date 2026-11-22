"""The shared marks of the explain post's figures (scripts/figures_explain_post.py, #5851).

The cycle, the row height, the style, and the small marks that each figure draws: ticks,
crosses, arrows, units and travelling rings.
"""

import textwrap

from openinterpretability.builder.explanation_figures import DEFS, EXTRA
from openinterpretability.figures.motion import animate, reveal, translate
from openinterpretability.figures.svg import Figure

CYCLE = "10s"
END = 80.0  # percent of the cycle at which the last mark shows; the rest holds the final frame
ROW = 17  # the height of one row of a list figure
STYLE = (
    ".mark{fill:none;stroke:var(--ink);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}"
    ".thin{fill:none;stroke:var(--ink2);stroke-width:1.2}"
    ".rule{stroke:var(--ink);stroke-width:1.4;stroke-dasharray:5 3}"
    ".arrow{fill:none;stroke:var(--ink2);stroke-width:1.6;marker-end:url(#tip)}"
    ".tipfill{fill:var(--ink2)}"
    ".box{fill:var(--surface);stroke:var(--ink2);stroke-width:1.3}"
    ".bigtext{fill:var(--ink);font-size:15px;font-weight:600}"
)
TIP = (
    '<defs><marker id="tip" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse"><path class="tipfill" d="M0 0L8 4L0 8Z"/>'
    "</marker></defs>"
)


def _figure(title: str, subtitle: str, alt: str, drawn: bool = False) -> Figure:
    """A figure with the explanation figures' patterns and this script's styles.

    With ``drawn``, its groups appear in turn, so a series draws itself over the cycle.
    Without it, the figure is still, apart from any mark that :func:`_travel` moves.

    Args:
        title: The title of the figure.
        subtitle: The text under the title. It wraps at 106 characters.
        alt: The text that describes the figure for a screen reader.
        drawn: If true, the groups of the figure appear in turn.
    """
    fig = Figure([title], textwrap.wrap(subtitle, 106), alt)
    fig.marks += [DEFS, TIP]
    fig.motion = EXTRA + STYLE
    fig.drawn = drawn
    return fig


def _at(k: int, count: int) -> float:
    """The share of the cycle, in percent, at which step ``k`` of ``count`` shows.

    Args:
        k: The index of the step.
        count: The number of steps.
    """
    return END * k / max(1, count - 1) if count > 1 else 0.0


def _group(fig: Figure, css: str, at: float, cycle: str = CYCLE) -> None:
    """Open a group that shows from ``at`` percent of the cycle. Close it with ``_end``.

    In a still figure (``_figure`` without ``drawn``) the group always shows.

    Args:
        fig: The figure to draw in.
        css: The CSS class of the group.
        at: The percent of the cycle at which the group shows.
        cycle: The duration of the cycle, as a CSS time.
    """
    if fig.drawn:
        fig.motion += reveal(css, at, cycle)
    fig.marks.append(f'<g class="{css}">')


def _end(fig: Figure) -> None:
    """Close the group that ``_group`` opened.

    Args:
        fig: The figure to draw in.
    """
    fig.marks.append("</g>")


def _travel(fig: Figure, name: str, points: list[tuple[float, tuple[float, float]]]) -> None:
    """A ring that travels through ``points``, each ``(percent of the cycle, (x, y))``.

    It moves in straight lines between the points and rests on the last one.

    Args:
        fig: The figure to draw in.
        name: The CSS class and the animation name of the ring.
        points: The stops of the ring, each a percent of the cycle and a point.
    """
    stops = [(f"{at:.1f}%", translate(x, y)) for at, (x, y) in points]
    last = translate(*points[-1][1])
    fig.motion += animate(name, stops, CYCLE, timing="ease-in-out", hold=last, rest=last)
    fig.marks.append(f'<circle class="mark {name}" cx="0" cy="0" r="9"/>')


def _tick(fig: Figure, x: float, y: float) -> None:
    """A pass: a check mark.

    Args:
        fig: The figure to draw in.
        x: The x position of the mark.
        y: The y position of the mark.
    """
    fig.marks.append(f'<path class="mark" d="M{x - 5:.1f} {y:.1f}l3.5 4 7-9"/>')


def _cross(fig: Figure, x: float, y: float) -> None:
    """A fail: a cross.

    Args:
        fig: The figure to draw in.
        x: The x position of the centre of the cross.
        y: The y position of the centre of the cross.
    """
    fig.marks.append(
        f'<path class="mark" d="M{x - 4.5:.1f} {y - 4.5:.1f}l9 9'
        f'M{x + 4.5:.1f} {y - 4.5:.1f}l-9 9"/>'
    )


def _arrow(fig: Figure, a: tuple[float, float], b: tuple[float, float], bend: float = 0) -> None:
    """An arrow from ``a`` to ``b``, bent sideways by ``bend`` pixels.

    Args:
        fig: The figure to draw in.
        a: The start point of the arrow.
        b: The end point of the arrow.
        bend: The sideways offset of the curve in pixels. Zero gives a straight arrow.
    """
    if bend:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - bend
        d = f"M{a[0]:.1f} {a[1]:.1f}Q{mx:.1f} {my:.1f} {b[0]:.1f} {b[1]:.1f}"
    else:
        d = f"M{a[0]:.1f} {a[1]:.1f}L{b[0]:.1f} {b[1]:.1f}"
    fig.marks.append(f'<path class="arrow" d="{d}"/>')


def _unit(fig: Figure, x: float, y: float, label: str) -> None:
    """A unit of the route: a head is a square, a neuron pole a circle, with its label.

    Args:
        fig: The figure to draw in.
        x: The x position of the centre of the unit.
        y: The y position of the centre of the unit.
        label: The label of the unit, for example L0/H1 or L0/N513+.
    """
    if "/H" in label:
        fig.marks.append(
            f'<rect class="fc1" x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="2"/>'
        )
    else:
        fig.dot(x, y, 6, "fhatch")
        fig.dot(x, y, 6, "thin")
    fig.text(x + 10, y + 4, label, "small")


def _token(text: str) -> str:
    """A token as the figures print it: quoted, with its leading space shown.

    Args:
        text: The text of the token.
    """
    return "“" + text.replace("\n", "⏎") + "”"


# The baseline ---------------------------------------------------------------------------
