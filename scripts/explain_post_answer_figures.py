"""The explain post's figures of the answer experiments (scripts/figures_explain_post.py).

The route claim, its cuts, the patching and the probes for " sky", then the " days"
mistake. Each figure draws from an answer record or an explanation record only.
"""

from pathlib import Path

from explain_post_marks import (
    CYCLE,
    END,
    ROW,
    _arrow,
    _cross,
    _end,
    _figure,
    _group,
    _tick,
    _token,
    _travel,
    _unit,
)

from openinterpretability.figures.motion import animate
from openinterpretability.figures.svg import WIDTH, Panel, number

ROOT = Path(__file__).parents[1]

STAGES = (
    ("Find the sun", ("L0/N513+", "L0/N1335+", "L0/N353+", "L0/N1418+")),
    ("Set the frame", ("L0/N1669+", "L0/H0", "L0/H1", "L0/H2")),
    ("Copy “sun” forward", ("L1/H6", "L2/H3", "L3/H3")),
    ("Turn it into “sky”", ("L5/H7", "L5/N1677+")),
)


def route(answer: dict) -> str:  # long-function: one figure, drawn mark by mark
    """The proposer's route, stage by stage, on the prompt's positions and the layers.

    Args:
        answer: The answer record of the sun prompt.
    """
    claim = answer["claim"]
    tokens = ["The", " sun", " rises", " in", " the"]
    stated = {u for _, units in STAGES for u in units}
    if stated != set(claim["route"]):
        raise ValueError("the figure's stages don't hold the claim's route")
    fig = _figure(
        "A claim about how: the route from “sun” to “sky”",
        "The proposer's claim, as it states it. Columns: the prompt's tokens. Rows: layers. "
        "Square: a head · hatched circle: a neuron pole. The ring carries “sun” along the route.",
        "A grid of the prompt's tokens across and the model's six layers up. A ring travels from "
        "“sun” up the route to “sky”. The route has four stages: four neurons in layer 0 detect "
        "“sun”; a neuron and three heads in layer 0 "
        "mark the “in the” frame; heads in layers 1 to 3 copy “sun” to the last position; a head "
        "and a neuron in layer 5 turn it into “sky”.",
    )
    top = fig.body_top + 34
    left, col = 80, 116
    row = 44

    def at(token: int, layer: int, dx: float = 0, dy: float = 0) -> tuple[float, float]:
        """The point of a token's column and a layer's row, moved by an offset.

        Args:
            token: The index of the token in the prompt.
            layer: The layer. Layer 0 is the bottom row.
            dx: The horizontal offset in pixels.
            dy: The vertical offset in pixels.
        """
        return left + col * token + dx, top + row * (5 - layer) + dy

    for la in range(6):
        fig.text(28, at(0, la)[1] + 4, f"L{la}", "small")
        fig.line(left - 20, at(0, la)[1], left + col * 5 + 20, at(0, la)[1], "grid")
    base = at(0, 0)[1] + 66
    for i, t in enumerate(tokens):
        fig.text(at(i, 0)[0], base, _token(t), "head", "middle")
    fig.text(at(5, 0)[0] + 10, at(5, 5)[1] + 4, "→ “sky”", "bigtext")
    places = {
        "L0/N513+": at(1, 0, -62, 4), "L0/N1335+": at(1, 0, -62, 26),
        "L0/N353+": at(1, 0, 14, 4), "L0/N1418+": at(1, 0, 14, 26),
        "L0/N1669+": at(4, 0, -62, 4), "L0/H0": at(4, 0, 14, 4),
        "L0/H1": at(4, 0, -62, 26), "L0/H2": at(4, 0, 14, 26),
        "L1/H6": at(4, 1), "L2/H3": at(4, 2), "L3/H3": at(4, 3),
        "L5/H7": at(4, 5, -30), "L5/N1677+": at(4, 5, 40),
    }  # fmt: skip
    for k, (_, units) in enumerate(STAGES):
        _group(fig, f"s{k}", 0.0)
        for u in units:
            _unit(fig, *places[u], u)
        if k == 0:
            _arrow(fig, (at(1, 0)[0], base - 14), (at(1, 0)[0], at(1, 0)[1] + 34))
        if k == 1:
            _arrow(fig, (at(3, 0)[0], base - 14), (at(4, 0)[0] - 70, at(4, 0)[1] + 30), bend=-8)
        if k == 2:
            for la in (1, 2):
                _arrow(
                    fig,
                    (at(1, 0)[0] + 10, at(1, la)[1]),
                    (at(4, la)[0] - 10, at(4, la)[1]),
                    bend=14,
                )
            fig.text(at(2, 1)[0] + 50, at(2, 1)[1] - 14, "attends to “sun”", "small", "middle")
        if k == 3:
            _arrow(fig, (at(4, 3)[0], at(4, 3)[1] - 10), (at(4, 5)[0], at(4, 5)[1] + 12))
        _end(fig)
    _travel(fig, "carry", [
        (0.0, (at(1, 0)[0], base - 5)), (10.0, (at(1, 0)[0], base - 5)),
        (28.0, (at(1, 0)[0], at(1, 0)[1] + 15)), (42.0, (at(1, 1)[0], at(1, 1)[1])),
        (60.0, (at(4, 1)[0], at(4, 1)[1])), (80.0, (at(5, 5)[0] + 40, at(5, 5)[1])),
    ])  # fmt: skip
    fig.bottom = base + 6
    return fig.render()


# Take the route away, or keep only it ---------------------------------------------------


def cuts(answer: dict) -> str:  # long-function: one figure, drawn mark by mark
    """How far each cut lowers the answer, and what the route alone brings back.

    Args:
        answer: The answer record, with the measures and the policy of its verdict.
    """
    m, p = answer["verdict"]["measures"], answer["verdict"]["policy"]
    null = sum(m["null_cuts"]) / len(m["null_cuts"])
    necessity = m["route_cut"] / m["all_cut"]
    fig = _figure(
        "Take the route away, or keep only the route",
        "How far “sky” falls (log-probability) when units are set to their baseline.",
        f"Switching off every unit lowers “sky” by {m['all_cut']:.2f}. "
        f"Switching off the 13-unit route lowers it by {m['route_cut']:.2f}, {necessity:.0%} of "
        f"that. Eight random sets of the same shape lower it by {null:.2f} on average, so the "
        f"route's cut is {m['route_cut'] / null:.1f} times theirs. With only the route kept, "
        f"{m['sufficiency']:.0%} of the answer comes back.",
    )
    top = fig.body_top + 16
    panel = Panel(
        fig, (250, top, WIDTH - 310, 2 * 30 + 8 * 9 + 14), (0, m["all_cut"] * 1.05), (0, 1)
    )
    panel.grid_x([0, 2, 4, 6], 0)
    panel.x_title("how far “sky” falls")
    rows = [
        ("every unit off", m["all_cut"], "fc1", 0.0),
        ("only the route off", m["route_cut"], "fhatch", 18.0),
    ]
    y = top
    for k, (label, value, cls, at) in enumerate(rows):
        _group(fig, f"c{k}", at)
        fig.bar(panel.x(0), y + 4, panel.x(value) - panel.x(0), 18, cls)
        if cls == "fhatch":
            fig.bar(panel.x(0), y + 4, panel.x(value) - panel.x(0), 18, "thin")
        fig.text(panel.left - 10, y + 17, label, "small", "end")
        fig.text(panel.x(value) + 6, y + 17, number(value, 2), "small")
        _end(fig)
        y += 30
    _group(fig, "nulls", 36.0)
    fig.text(panel.left - 10, y + 12, "8 random sets off", "small", "end")
    for i, v in enumerate(m["null_cuts"]):
        fig.bar(panel.x(0), y + 2 + 9 * i, panel.x(max(v, 0)) - panel.x(0), 6, "fneutral")
    _end(fig)
    y += 8 * 9 + 14 + 54
    lines = [
        (f"The route's cut is {necessity:.0%} of the whole cut",
         f"rule ≥ {p['min_necessity']:.0%}", 50.0),
        (f"and {m['route_cut'] / null:.1f}× the random sets' mean cut ({null:.2f})",
         f"rule ≥ {p['null_factor']:.0f}×", 60.0),
        (f"With only the route on, {m['sufficiency']:.0%} of “sky” comes back",
         f"rule ≥ {p['min_sufficiency']:.0%}", 70.0),
    ]  # fmt: skip
    for k, (text, rule, at) in enumerate(lines):
        _group(fig, f"l{k}", at)
        _tick(fig, 30, y + 2)
        fig.text(44, y + 6, text, "head")
        fig.text(WIDTH - 24, y + 6, rule, "small", "end")
        _end(fig)
        y += 22
    fig.bottom = y - 10
    return fig.render()


# Swap the subject -----------------------------------------------------------------------


def patching(answer: dict) -> str:  # long-function: one figure, drawn mark by mark
    """The route patched from the corrupted prompt, against random sets patched the same way.

    Args:
        answer: The answer record, with its claim and its verdict.
    """
    claim, m, p = answer["claim"], answer["verdict"]["measures"], answer["verdict"]["policy"]
    null = sum(m["null_patching"]) / len(m["null_patching"])
    fig = _figure(
        "Swap the subject, and move only the route",
        "The route's values are copied from the second prompt into the first. How far does “sky” "
        "fall, as a share of how far it falls on the second prompt?",
        f"Two prompts: “{claim['prompt']}” and “{claim['corrupted']}”. Copying the route's "
        f"values from the second into the first makes “sky” fall {m['patching']:.0%} as far as the "
        f"second prompt does. "
        f"Random sets of the same shape, copied the same way, move it {null:.1%} on average.",
    )
    top = fig.body_top + 18
    fig.text(20, top + 4, "clean prompt", "small")
    fig.text(130, top + 4, claim["prompt"].replace(" sun ", " SUN "), "head")
    fig.text(20, top + 26, "corrupted", "small")
    fig.text(130, top + 26, claim["corrupted"].replace(" price ", " PRICE "), "head")
    track = Panel(fig, (130, top + 70, WIDTH - 200, 70), (-0.1, 1.0), (0, 1))
    track.grid_x([0, p["max_null_patching"], p["min_patching"], 1.0], 1)
    track.x_title("share of the fall in “sky” that the second prompt causes")
    y = track.top + 20
    fig.text(track.left - 10, y + 4, "the route", "small", "end")
    fig.motion += animate(
        "slide",
        [("0%,8%", f"transform:translateX({track.x(0) - track.x(m['patching']):.1f}px)"),
         ("40%", "transform:translateX(0px)")],
        CYCLE, timing="ease-in-out",
        hold="transform:translateX(0px)", rest="transform:translateX(0px)",
    )  # fmt: skip
    fig.marks.append('<g class="slide">')
    fig.dot(track.x(m["patching"]), y, 7, "fc1")
    _end(fig)
    _group(fig, "routepct", 40.0)
    fig.text(track.x(m["patching"]), y - 12, f"{m['patching']:.1%}", "head", "middle")
    _end(fig)
    y2 = track.top + 52
    _group(fig, "nulls", 55.0)
    fig.text(track.left - 10, y2 + 4, "8 random sets", "small", "end")
    for v in m["null_patching"]:
        fig.dot(track.x(v), y2, 4, "open sneutral")
    fig.line(track.x(null), y2 - 9, track.x(null), y2 + 9, "mark")
    _end(fig)
    y = track.top + track.height + 62
    lines = [
        (f"The route moves the answer {m['patching']:.0%} of the way",
         f"rule ≥ {p['min_patching']:.0%}", 50.0),
        (f"Random sets move it {null:.1%} on average",
         f"rule ≤ {p['max_null_patching']:.0%}", 68.0),
    ]  # fmt: skip
    for k, (text, rule, at) in enumerate(lines):
        _group(fig, f"l{k}", at)
        _tick(fig, 30, y + 2)
        fig.text(44, y + 6, text, "head")
        fig.text(WIDTH - 24, y + 6, rule, "small", "end")
        _end(fig)
        y += 22
    fig.bottom = y - 10
    return fig.render()


# The reviewer's probes ------------------------------------------------------------------


def probes(  # long-function: one figure, drawn mark by mark
    answer: dict, title: str = "Try to break it: a second agent writes the test prompts"
) -> str:
    """Each of the reviewer's prompts, what the claim predicts and what the model said.

    Args:
        answer: The answer record, with the reviewer's prompts and their measures.
        title: The title of the figure.
    """
    m = answer["verdict"]["measures"]
    rows = answer["probes"]["prompts"]
    word = answer["claim"]["answer"].strip()
    cycle = "14s"
    fig = _figure(
        title,
        f"The reviewer sees only the claim, never a measurement. “keeps” or “leaves”: what the "
        f"claim predicts for “{word}”. ✓ as predicted · ✗ not.",
        f"A list of the reviewer's {len(rows)} prompts, each with what the claim "
        f"predicts, the model's top token and a check mark when it came out as predicted. "
        f"{sum(m['probes'].values())} of {len(m['probes'])} did. The proposer's own "
        f"{len(m['controls'])} control prompts all did too.",
    )
    top = fig.body_top + 14
    fig.text(48, top, "prompt", "small")
    fig.text(520, top, "claim", "small")
    fig.text(600, top, "model says", "small")
    top += 8
    for i, row in enumerate(rows):
        at = END * i / len(rows)
        y = top + ROW * i
        _group(fig, f"q{i}", at, cycle)
        ok = m["probes"][row["prompt"]]
        (_tick if ok else _cross)(fig, 28, y + ROW / 2)
        text = row["prompt"] if len(row["prompt"]) < 68 else row["prompt"][:65] + "…"
        fig.text(48, y + ROW - 4, text, "small")
        fig.text(520, y + ROW - 4, "keeps" if row["expect"] == "kept" else "leaves", "small")
        fig.text(600, y + ROW - 4, _token(m["top_tokens"].get(row["prompt"], "?")), "small")
        _end(fig)
    y = top + ROW * len(rows) + 22
    passed = sum(m["probes"].values())
    fig.text(20, y, f"{passed} of {len(rows)} as predicted", "bigtext")
    controls = sum(m["controls"].values())
    _group(fig, "ctl", END, cycle)
    fig.text(WIDTH - 24, y, f"and the proposer's controls: {controls} of {len(m['controls'])}",
             "head", "end")  # fmt: skip
    _end(fig)
    fig.bottom = y - 4
    return fig.render()


# The " days" mistake --------------------------------------------------------------------

DAYS_EXPLANATION = (
    ROOT / "wiki/explanations/EleutherAI--pythia-70m/970a91e11d8c/20261118T235000Z.json"
)
COUNT_WORD = ("L0/N1877+", "L0/N1277+", "L0/N1108+", "L0/N141+", "L0/N1654+", "L0/N883+")
TIME_UNIT = ("L0/N1101+", "L0/N912+", "L0/N2008+", "L0/N780+")
RIGHT = " months"  # the right answer, which the records don't name


def days_mistake(explanation: dict) -> str:  # long-function: one figure, drawn mark by mark
    """The model's top next tokens for the " days" prompt, with the right answer marked.

    Args:
        explanation: The explanation record of the " days" prompt.
    """
    rows = explanation["figures"]["ablation"]
    fig = _figure(
        "The mistake: the model is sure, and wrong",
        f"pythia-70m's most likely next tokens after “{explanation['prompt']}”.",
        "Bar chart of the five most likely next tokens. “days” has "
        f"{rows[0]['before']:.0%}. The right answer, “months”, has "
        f"{next(r['before'] for r in rows if r['token'] == RIGHT):.0%}.",
    )
    top = fig.body_top + 16
    panel = Panel(fig, (150, top, WIDTH - 260, 26 * len(rows)), (0, 1), (0, 1))
    panel.grid_x([0, 0.25, 0.5, 0.75, 1.0], 2)
    panel.x_title("probability")
    for i, r in enumerate(rows):
        y = top + 26 * i
        right = r["token"] == RIGHT
        fig.bar(panel.x(0), y + 4, panel.x(r["before"]) - panel.x(0), 18,
                "fhatch" if right else "fc1")  # fmt: skip
        if right:
            fig.bar(panel.x(0), y + 4, panel.x(r["before"]) - panel.x(0), 18, "thin")
        fig.text(panel.left - 10, y + 17, _token(r["token"]), "head" if i == 0 else "", "end")
        note = f"{r['before']:.1%}" + ("  ← the right answer" if right else "")
        fig.text(panel.x(r["before"]) + 8, y + 17, note, "head" if right else "small")
    fig.bottom = top + 26 * len(rows) + 4
    return fig.render()


def days_copy(answer: dict) -> str:  # long-function: one figure, drawn mark by mark
    """Where the count-word and time-unit neurons fire, and the copy the claim describes.

    Args:
        answer: The VERIFIED answer record of the " days" prompt.
    """
    claim, m = answer["claim"], answer["verdict"]["measures"]
    reading = m["readings"][claim["prompt"]]
    tokens, strength = reading["tokens"], reading["strength"]
    fig = _figure(
        "How it goes wrong: the model copies the pattern",
        "Bars: how strongly the claim's layer-0 neurons fire at each token (measured). The ring: "
        "the copy step, as the claim states it.",
        "The prompt's ten tokens. Count-word neurons fire on “seven” and “twelve”; time-unit "
        "neurons fire on “week”, “days” and “year”. A ring starts at “twelve”, jumps back to the "
        "earlier count word “seven”, steps to the word after it, “days”, and carries “days” to "
        "the end, where the model says it.",
    )
    top = fig.body_top + 30
    left, col = 40, (WIDTH - 140) / len(tokens)

    def x(i: int) -> float:
        """The x position of the middle of a token's column.

        Args:
            i: The index of the token in the prompt.
        """
        return left + col * (i + 0.5)

    lanes = [("count-word neurons", COUNT_WORD, "fc1"), ("time-unit neurons", TIME_UNIT, "fhatch")]
    peak = max(
        sum(strength[u][i] for u in units) for _, units, _ in lanes for i in range(len(tokens))
    )
    lane_h, copy_y = 46, top + 20
    base = copy_y + 56 + len(lanes) * (lane_h + 18)
    fig.text(20, copy_y - 22, "copy heads L3/H3, L4/H0 and L4/H7, as the claim states it", "small")
    fig.line(x(0) - 20, copy_y, x(len(tokens) - 1) + 20, copy_y, "grid")
    for k, (name, units, cls) in enumerate(lanes):
        floor = copy_y + 56 + k * (lane_h + 18) + lane_h
        fig.text(20, floor - lane_h - 4, name, "small")
        fig.line(x(0) - 20, floor, x(len(tokens) - 1) + 20, floor, "axis")
        for i in range(len(tokens)):
            value = sum(strength[u][i] for u in units)
            if value > 0.05:
                h = lane_h * value / peak
                fig.bar(x(i) - 11, floor, 22, -h, cls)
                if cls == "fhatch":
                    fig.bar(x(i) - 11, floor, 22, -h, "thin")
    for i, t in enumerate(tokens):
        fig.text(x(i), base, _token(t), "head", "middle")
    end = x(len(tokens) - 1) + 54
    fig.text(end - 8, copy_y + 5, "→ “days”", "bigtext")
    seven, days, twelve = (tokens.index(" seven"), tokens.index(" days"),
                           len(tokens) - 1)  # fmt: skip
    _arrow(fig, (x(twelve), copy_y - 4), (x(seven) + 6, copy_y - 4), bend=34)
    fig.text((x(twelve) + x(seven)) / 2, copy_y - 32, "look back for the last count word",
             "small", "middle")  # fmt: skip
    _travel(fig, "copy", [
        (0.0, (x(twelve), copy_y)), (10.0, (x(twelve), copy_y)),
        (32.0, (x(seven), copy_y)), (42.0, (x(seven), copy_y)),
        (52.0, (x(days), copy_y)), (60.0, (x(days), copy_y)),
        (80.0, (end - 20, copy_y)),
    ])  # fmt: skip
    fig.bottom = base + 4
    return fig.render()
