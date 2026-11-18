"""The animated route of an answer experiment, on one prompt (issues #4905 and #4917).

Each answer experiment's page draws its claim's route on the original prompt and on every
counterfactual prompt. It draws them from the record's readings, so the build needs no model
(:class:`~openinterpretability.core.answer_records.RouteReading`).

- *Tokens* on the left. A token's bar is the route's total strength at that token.
- *Units* in columns by layer. A head is a square and a neuron pole is a circle. A unit's
  size is its peak strength on this prompt against the original prompt, so a route that
  goes quiet shrinks.
- *Edges.* A token's edge goes to each first-layer unit, from the token where it is
  strongest. The edges between units are traced (``explain.paths``). Each unit's edge to
  the answer is its attribution. Width is weight, and a dashed edge pushes the answer down.
  Edges under :data:`MIN_EDGE` of the strongest are left out, so the route stays legible.
- *The answer* on the right: the model's top token and the claim's answer's log-probability.

The signal builds layer by layer, then the answer shows. The static copy shows the last
frame, for reduced motion and the page's button. Nothing rests on colour: kind is shape,
strength is size and width, and sign is a line style.
"""

from __future__ import annotations

from openinterpretability.builder.explanation_figures import (
    CYCLE,
    DEFS,
    EXTRA,
    Drawn,
    _edge,
    _finish,
)
from openinterpretability.core.answer_records import parse_unit
from openinterpretability.figures.motion import reveal
from openinterpretability.figures.svg import WIDTH, Figure, number

LEFT, RIGHT = 130, WIDTH - 130  # the units' columns lie between the tokens and the answer
MAX_SCALE = 2.0  # a unit's mark grows to at most twice its size on the original prompt
MIN_EDGE = 0.1  # edges under this share of the strongest one are left out, so the route reads
ROW = 34  # the least height per unit in a column, so neighbours' labels don't collide


def _peak(values: list[float]) -> float:
    return max(values, default=0.0)


def route_figure(name: str, claim, reading, original) -> Drawn:
    """The route on one prompt; ``original`` is the original prompt's reading, for scale."""
    kept = reading.top.strip() == claim.answer.strip()
    told = "the answer stays" if kept else f"the model answers {reading.top!r}"
    fig = Figure(
        [f"The route on “{_short(''.join(reading.tokens))}”: {told}"],
        [
            "Squares are heads, circles are neuron poles. Size is strength against the original.",
            "Edge width is weight, and a dashed edge pushes the answer down.",
        ],
        f"The claim's route on the prompt {''.join(reading.tokens)!r}. The tokens are on the "
        f"left, the route's {len(claim.route)} units in columns by layer, and the model's "
        f"answer, {reading.top!r}, on the right.",
    )
    fig.marks.append(DEFS)
    fig.motion = EXTRA
    top, layers = fig.body_top + 14, sorted({parse_unit(u)[0] for u in claim.route})
    height = max(200.0, ROW * max(len(reading.tokens), *_column_sizes(claim, layers)))
    tokens = _tokens(fig, reading, original, top, height)
    pos = _units(fig, claim, reading, original, layers, top, height)
    answer = (WIDTH - 110, top + height / 2)
    _edges(fig, claim, reading, layers, tokens, pos, answer)
    fig.motion += reveal("ranswer", 88, CYCLE)
    fig.marks.append('<g class="ranswer">')
    fig.text(answer[0] + 8, answer[1] - 4, repr(reading.top)[1:-1], "head")
    fig.text(answer[0] + 8, answer[1] + 12,
             f"log p({claim.answer.strip()}) {reading.answer_log_prob:.2f}", "small")  # fmt: skip
    fig.marks.append("</g>")
    fig.bottom = top + height + 6
    return _finish(name, fig, _table(claim, reading, original), moves=True)


def _edges(fig, claim, reading, layers, tokens, pos, answer) -> None:
    """The edges, revealed a layer at a time: from the tokens, between units, to the answer."""
    peak = max([abs(e["weight"]) for e in reading.edges]
               + [abs(a) for a in reading.attribution.values()] + [1e-9])  # fmt: skip
    for k, layer in enumerate(layers):
        cls = f"rl{k}"
        fig.motion += reveal(cls, 10 + 70 * k / len(layers), CYCLE)
        for unit in (u for u in claim.route if parse_unit(u)[0] == layer):
            values = reading.strength.get(unit, [])
            at = max(range(len(values)), key=values.__getitem__) if _peak(values) > 0 else None
            if k == 0 and at is not None and at < len(tokens):  # the route reads its sources
                _edge(fig, tokens[at], pos[unit], 0.4, cls)
            for e in reading.edges:
                if (
                    e["target"] == unit
                    and e["source"] in pos
                    and abs(e["weight"]) >= MIN_EDGE * peak
                ):
                    _edge(fig, pos[e["source"]], pos[unit], e["weight"] / peak, cls)
            weight = reading.attribution.get(unit, 0.0)
            if abs(weight) >= MIN_EDGE * peak:
                _edge(fig, pos[unit], answer, weight / peak, cls)


def _short(text: str, limit: int = 48) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _scale(reading, original, unit: str) -> float:
    """The unit's peak strength here against the original prompt, at most :data:`MAX_SCALE`."""
    there = _peak(original.strength.get(unit, [])) or 1.0
    return min(MAX_SCALE, _peak(reading.strength.get(unit, [])) / there)


def _column_sizes(claim, layers) -> list[int]:
    return [sum(parse_unit(u)[0] == la for u in claim.route) for la in layers]


def _tokens(fig, reading, original, top: float, height: float) -> list[tuple[float, float]]:
    """Each token, with a bar for the route's total strength at it; their anchor points."""
    totals = [sum(v[i] for v in reading.strength.values() if i < len(v))
              for i in range(len(reading.tokens))]  # fmt: skip
    base = _peak([sum(col) for col in zip(*original.strength.values(), strict=False)]) or 1.0
    out = []
    for i, tok in enumerate(reading.tokens):
        y = top + (i + 1) * height / (len(reading.tokens) + 1)
        fig.text(16, y + 4, repr(tok)[1:-1][:14], "small")
        width = 40 * min(MAX_SCALE, totals[i] / base)
        fig.marks.append(f'<rect class="fhatch outline" x="84" y="{y - 5:.1f}" '
                         f'width="{width:.1f}" height="10"/>')  # fmt: skip
        out.append((84 + width + 2, y))
    return out


def _units(fig, claim, reading, original, layers, top: float, height: float) -> dict:
    """Each unit's mark, sized by its strength here against the original; their places."""
    pos = {}
    for k, layer in enumerate(layers):
        column = [u for u in claim.route if parse_unit(u)[0] == layer]
        x = LEFT + (k + 1) * (RIGHT - LEFT) / (len(layers) + 1)
        fig.text(x, top + height + 14, f"L{layer}", "small", "middle")
        for j, unit in enumerate(column):
            y = top + (j + 1) * height / (len(column) + 1)
            pos[unit] = (x, y)
            size = 2 + 5 * _scale(reading, original, unit)
            fig.marks.append(f'<g class="rl{k}">')
            if parse_unit(unit)[2] == "head":
                fig.marks.append(f'<rect class="outline" x="{x - size:.1f}" y="{y - size:.1f}" '
                                 f'width="{2 * size:.1f}" height="{2 * size:.1f}"/>')  # fmt: skip
            else:
                fig.dot(x, y, size, "fc1")
            # Above and to the right, clear of the edges that run through the mark.
            fig.text(x + size + 2, y - size - 2, unit.split("/")[1], "small")
            fig.marks.append("</g>")
    return pos


def _table(claim, reading, original) -> list[list[str]]:
    rows = [["unit", "role", "peak strength", "on the original", "attribution"]]
    for unit in claim.route:
        rows.append([
            unit, claim.roles.get(unit, ""),
            number(_peak(reading.strength.get(unit, [])), 3),
            number(_peak(original.strength.get(unit, [])), 3),
            number(reading.attribution.get(unit, 0.0), 3),
        ])  # fmt: skip
    return rows


def answer_figures(record) -> list[tuple[str, Drawn]]:
    """``(prompt, figure)`` for every prompt with a reading, the original first."""
    readings, claim = record.verdict.measures.readings, record.claim
    original = readings.get(claim.prompt)
    if original is None:
        return []
    return [(prompt, route_figure(f"route-{i}", claim, reading, original))
            for i, (prompt, reading) in enumerate(_ordered(readings, claim))]  # fmt: skip


def _ordered(readings: dict, claim) -> list:
    """The readings' ``(prompt, reading)`` pairs, the original prompt's first."""
    return sorted(readings.items(), key=lambda kv: kv[0] != claim.prompt)


def route3d(record) -> dict | None:
    """The data of the 3D route explorer on an answer experiment's page (#4905, #4917).

    Each unit has its layer and kind. Each prompt has its tokens, the model's answer, each
    unit's strength against the original and its strength at each token, and the edges.
    The page's script draws it with three.js. The SVG figures stay as the static form.
    """
    readings, claim = record.verdict.measures.readings, record.claim
    original = readings.get(claim.prompt)
    if original is None:
        return None
    m = record.verdict.measures
    expect = {c.prompt: (c.expect, "proposer") for c in claim.counterfactuals}
    expect |= {c.prompt: (c.expect, "reviewer") for c in record.probes.prompts}
    came_out = m.controls | m.probes
    units = [{"unit": u, "layer": parse_unit(u)[0], "head": parse_unit(u)[2] == "head",
              "role": claim.roles.get(u, "")} for u in claim.route]  # fmt: skip
    prompts = []
    for prompt, r in _ordered(readings, claim):
        scale = {u: _scale(r, original, u) for u in claim.route}
        what, who = expect.get(prompt, ("original", "original"))
        prompts.append({
            "prompt": prompt, "tokens": r.tokens, "top": r.top, "lp": r.answer_log_prob,
            "share": r.route_share, "expect": what, "by": who,
            "ok": came_out.get(prompt), "scale": scale, "strength": r.strength,
            "attribution": r.attribution, "edges": r.edges,
            "lens": r.lens, "spectrum": r.spectrum, "space": r.space,
            "route_off": r.route_off.model_dump() if r.route_off else None,
            "unit_off": {u: v.model_dump() for u, v in r.unit_off.items()},
        })  # fmt: skip
    candidates = list(original.lens[0]) if original.lens else []
    return {"answer": claim.answer, "units": units, "prompts": prompts,
            "candidates": candidates}  # fmt: skip
