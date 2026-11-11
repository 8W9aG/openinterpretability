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


def heat_html(tokens: Iterable[HeatToken]) -> Markup:
    """Return the tokens as shaded ``<span>`` elements, as the ``heat_tokens`` macro shows them.

    Pages show many thousands of tokens. So this builds the HTML in Python, which is much
    faster than a loop in the template. Hover (or tap) a token for its exact value.
    """
    parts = []
    for t in tokens:
        flags = (" peak" if t.peak else "") + (" marked" if t.marked else "")
        notes = (" · peak" if t.peak else "") + (" · marked" if t.marked else "")
        parts.append(
            f'<span class="tok heat{flags}" style="--a: {t.shade}" '  # a float: nothing to escape
            f'data-v="{t.value:.3f}{notes}">{escape(t.text)}</span>'
        )
    return Markup("".join(parts))  # noqa: S704  (every value in it is escaped above)
