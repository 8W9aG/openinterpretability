"""Long table cells wrap on prose pages, and short or numeric cells stay on one line (#5570)."""

from openinterpretability.builder.research import _MARKDOWN, LONG_CELL


def test_only_long_cells_are_marked_to_wrap():
    reason = "a reason written as a full sentence, longer than one line in a narrow column"
    assert len(reason) > LONG_CELL
    html = _MARKDOWN.render(
        f"| Page | Status | Why |\n| --- | --- | --- |\n| A | used | {reason} |\n"
    )
    assert f'<td class="wrap">{reason}</td>' in html
    assert "<td>A</td>" in html and "<td>used</td>" in html and "<th>Why</th>" in html


def test_a_numeric_table_has_no_wrapping_cell():
    html = _MARKDOWN.render(
        "| Model | AUC | Interval |\n| --- | --- | --- |\n| GPT-2 | 0.51 | 0.45–0.57 |\n"
    )
    assert 'class="wrap"' not in html


def test_the_prose_styles_let_marked_cells_wrap():
    from pathlib import Path

    prose = (
        Path(__file__).parents[1] / "openinterpretability/builder/templates/_prose.html"
    ).read_text()
    assert ".paper td.wrap, .paper th.wrap { white-space: normal; }" in prose
