"""Papers' LaTeX is typeset as MathML at build time (builder/maths.py, #4211)."""

import pytest

from openinterpretability.builder.maths import MathsError, to_mathml
from openinterpretability.builder.research import _read_paper

PAPER = """# A paper

The abstract. A pole is $\\mathrm{silu}(g) \\cdot u$, and a KL of $1.3 \\times 10^{-5}$.

## Method

$$
\\mathrm{KL}(p \\Vert q) = \\sum_i p_i \\log \\frac{p_i}{q_i}
$$

It costs $5 and $6, and `$OUT` and `$P` are code.
"""


def test_a_paper_renders_inline_and_display_maths(tmp_path):
    (tmp_path / "README.md").write_text(PAPER, encoding="utf-8")
    body = _read_paper(tmp_path, "https://github.com/x/y").body
    assert body.count("<math ") == 3
    assert 'display="block"' in body and "<mfrac>" in body and "<msup>" in body
    assert "\\mathrm" not in body and "$$" not in body
    assert "costs $5 and $6" in body  # not maths
    assert "<code>$OUT</code> and <code>$P</code>" in body


@pytest.mark.parametrize(
    ("latex", "why"),
    [
        ("x^{", "braces"),
        ("\\frac{a", "braces"),
        ("a}{b", "braces"),  # a brace closes before one opens
        ("\\nosuch{x}", "unknown"),
        ("\\left( x", "can.t typeset"),
    ],
)
def test_bad_latex_fails_the_build(latex, why):
    with pytest.raises(MathsError, match=why):
        to_mathml(latex, display=False)


def test_the_error_names_the_paper(tmp_path):
    (tmp_path / "README.md").write_text("# T\n\nText $x^{$ here.\n", encoding="utf-8")
    with pytest.raises(MathsError, match="README.md: can't typeset"):
        _read_paper(tmp_path, "https://github.com/x/y")
