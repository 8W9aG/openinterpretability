"""Top-activating examples as the site's token heatmaps show them."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from markupsafe import Markup, escape

from openinterp.core.examples import ActivatingExample
from openinterp.sensitive import is_offensive


@dataclass(frozen=True, slots=True)
class HeatToken:
    """One token of a heatmap, shaded by its strength."""

    text: str
    shade: float  # 0..1 of the pole's max activation; the template caps the tint
    value: float  # the stored pole strength
    peak: bool
    marked: bool = False  # fuzzing: a token the judge was told is active


@dataclass(frozen=True)
class Attended:
    """One token that a head's peak attends to."""

    text: str
    weight: float
    offset: int  # tokens from the peak (≤ 0)


@dataclass(frozen=True)
class HeatExample:
    """One top-activating example, ready for the token heatmap."""

    max_activation: float
    window: list[HeatToken]  # tokens around the peak
    tokens: list[HeatToken]  # the whole snippet
    clipped_start: bool
    clipped_end: bool
    attended: list[Attended] = field(default_factory=list)  # a head's peak attention
    offensive: bool = False  # the snippet holds a word of the site's warning list


def token_heat(
    example: ActivatingExample, scale: float, before: int = 30, after: int = 10
) -> HeatExample:
    """Shade every token by its pole strength relative to ``scale`` (the neuron's max).

    The window keeps ``before`` tokens before the peak and ``after`` after it, so long
    snippets stay readable. The full snippet is kept for an expandable view.
    """
    scale = scale or example.max_activation or 1.0
    tokens = [
        HeatToken(
            tok.replace("\n", "↵"),  # a newline the neuron fires on must stay visible
            round(min(1.0, max(0.0, act / scale)), 3),
            act or 0.0,  # strengths of inactive tokens can be stored as -0.0
            i == example.max_token_index,
        )
        for i, (tok, act) in enumerate(zip(example.tokens, example.activations, strict=True))
    ]
    start = max(0, example.max_token_index - before)
    end = min(len(tokens), example.max_token_index + after + 1)
    return HeatExample(
        max_activation=example.max_activation,
        window=tokens[start:end],
        tokens=tokens,
        clipped_start=start > 0,
        clipped_end=end < len(tokens),
        offensive=is_offensive(example.text),
        attended=[
            Attended(tokens[i].text, w, i - example.max_token_index)
            for i, w in zip(example.attends_to, example.attention_weights, strict=True)
        ],
    )


def _short(shade: float) -> str:
    """Return ``shade`` with no leading ``0`` and no trailing ``.0``: ``.25``, ``1``."""
    text = str(shade)
    text = text[1:] if text.startswith("0.") else text
    return text[:-2] if text.endswith(".0") else text


def heat_html(tokens: Iterable[HeatToken]) -> Markup:
    """Return the tokens as one ``<span class="hm">`` of ``<i>`` elements, one per token.

    Heatmaps are most of the site's bytes, so the markup is short. ``base.html`` styles a
    token as a ``.tok.heat`` span. A token with no ``style`` has ``--a: 0``, and a token
    with no ``data-v`` has the value 0.000. The classes ``p`` and ``m`` mark the peak and a
    token that the fuzzing judge was told is active. Hover (or tap) a token for its value.
    """
    parts = ['<span class="hm">']
    for t in tokens:
        classes = " ".join(c for c, on in (("p", t.peak), ("m", t.marked)) if on)
        value = f"{t.value:.3f}" + (" · peak" if t.peak else "") + (" · marked" if t.marked else "")
        attrs = f' class="{classes}"' if classes else ""
        attrs += f" style=--a:{_short(t.shade)}" if t.shade else ""
        attrs += f' data-v="{value}"' if value != "0.000" else ""
        parts.append(f"<i{attrs}>{escape(t.text)}</i>")
    parts.append("</span>")
    return Markup("".join(parts))  # noqa: S704  (the token text is escaped above)


PEAK_ROW = "font-semibold text-slate-900 dark:text-slate-100"  # the peak's row in a values table


def values_rows(tokens: Iterable[HeatToken]) -> Markup:
    """Return one table row per token: the token and its value, as a heatmap's values table.

    The rows leave out their closing tags, as HTML allows, and ``base.html`` styles the
    cells of a ``.vt`` table. So a row is short: tables of values are a large part of a page.
    """
    rows = []
    for t in tokens:
        start = f'<tr class="{PEAK_ROW}">' if t.peak else "<tr>"
        rows.append(f'{start}<td><code class="tok">{escape(t.text)}</code><td>{t.value:.3f}')
    return Markup("".join(rows))  # noqa: S704  (the token text is escaped above)
