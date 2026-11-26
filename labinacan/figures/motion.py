"""Motion for paper figures: CSS keyframes inside the SVG, from measured frames (#4210).

Every animated figure builds its keyframes here, so they share one form. The SVG has no
JavaScript and no SMIL, so it moves on GitHub, on the site and in a plain ``<img>``. Under
``prefers-reduced-motion`` the base CSS of ``figure_svg.py`` stops every animation. Each
animated mark then shows its resting state, which :func:`animate` sets to the final
frame. See the ``paper-animation`` skill for when and how to animate.
"""

from __future__ import annotations

from collections.abc import Sequence

Stop = tuple[str, str]  # (keyframe selector such as "12.5%" or "0%,25%", its CSS)


def translate(x: float, y: float) -> str:
    """CSS that moves a mark drawn at the origin to (x, y)."""
    return f"transform:translate({x:.1f}px,{y:.1f}px)"


def animate(
    name: str,
    stops: Sequence[Stop],
    cycle: str,
    *,
    timing: str = "linear",
    hold: str | None = None,
    rest: str = "",
    alternate: bool = False,
) -> str:
    """A ``@keyframes`` rule called ``name`` and the class that runs it.

    ``hold`` is the CSS of a final ``100%`` stop, so the mark waits on its last frame
    before the loop starts again. ``rest`` is the class's own CSS, which shows when
    motion is off: give it the final frame. Use ``linear`` timing for measured series
    and ``ease-in-out`` for a morph between two states.
    """
    body = "".join(f"{at}{{{css}}}" for at, css in stops)
    if hold is not None:
        body += f"100%{{{hold}}}"
    loop = " alternate" if alternate else ""
    run = f"animation:{name} {cycle} {timing} infinite{loop};{rest}"
    return f"@keyframes {name}{{{body}}}.{name}{{{run}}}"


def even_stops(frames: Sequence[str], until: float) -> list[Stop]:
    """``frames`` spread evenly from 0% to ``until``%, one stop each."""
    last = len(frames) - 1
    return [(f"{until * i / last if last else 0:.1f}%", css) for i, css in enumerate(frames)]


def reveal(name: str, at: float, cycle: str) -> str:
    """Hide a mark until ``at``% of the cycle, then show it to the end.

    Its resting state is shown, so a reader with motion off sees every mark.
    """
    return animate(
        name, [("0%", "opacity:0"), (f"{at:.1f}%", "opacity:1")], cycle,
        timing="step-end", hold="opacity:1", rest="opacity:1",
    )  # fmt: skip


def counter(name: str, labels_at: Sequence[float], cycle: str) -> list[tuple[str, str]]:
    """One class per label, each shown only while the animation is at its frame.

    ``labels_at`` gives the share of the cycle, in percent, at which each label starts.
    The last label stays to the end of the cycle and is the resting one. Returns
    ``(class, css)`` for each label, in order.
    """
    out = []
    for i, start in enumerate(labels_at):
        end = labels_at[i + 1] if i + 1 < len(labels_at) else 100.0
        cls = f"{name}{i}"
        if i + 1 < len(labels_at):
            stops = [
                ("0%", "opacity:0"), (f"{start:.1f}%", "opacity:1"), (f"{end:.1f}%", "opacity:0")
            ]  # fmt: skip
            css = animate(cls, stops, cycle, timing="step-end", hold="opacity:0", rest="opacity:0")
        else:
            css = animate(cls, [("0%", "opacity:0"), (f"{start:.1f}%", "opacity:1")], cycle,
                          timing="step-end", hold="opacity:1", rest="opacity:1")  # fmt: skip
        out.append((cls, css))
    return out
