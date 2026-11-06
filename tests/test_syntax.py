import random

import pytest
import torch

from openinterp import syntax
from openinterp.core.confounds import (
    BRACKET_DEPTH,
    INDENT_DEPTH,
    ROUND_DEPTH,
    SYNTAX_CONSTRUCT,
    SYNTAX_NODE,
)
from openinterp.evidence import _Texts
from openinterp.harness.annotations import UNLABELLED, Span, TokenLabels, align
from openinterp.harness.scan import Piece
from openinterp.syntax import (
    BRACKET,
    COMMENT,
    GRAMMARS,
    IDENTIFIER,
    KEYWORD,
    NUMBER,
    STRING,
    TOP_LEVEL,
    WHITESPACE,
    bracket_depths,
    indent_depths,
    parse_labels,
    syntax_taggers,
)
from tests.test_annotations import CharTokenizer

PYTHON = 'import os\n\ndef f(a):\n    return g(a, "x(", 1)  # return (\n'


def label_at(text, labels, word, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(word, start + 1)
    return labels[start]


def test_node_kinds_keep_the_role_of_one_word():
    text = 'return "return"  # return\n'
    wrapped = f"def f():\n    {text}"
    nodes, _ = parse_labels(wrapped, "Python")
    assert label_at(wrapped, nodes, "return", 0) == KEYWORD
    assert label_at(wrapped, nodes, "return", 1) == STRING
    assert label_at(wrapped, nodes, "return", 2) == COMMENT


def test_node_kinds_of_a_python_line():
    nodes, _ = parse_labels(PYTHON, "Python")
    assert label_at(PYTHON, nodes, "g(") == IDENTIFIER
    assert label_at(PYTHON, nodes, "(a,") == BRACKET
    assert label_at(PYTHON, nodes, "1)") == NUMBER
    assert label_at(PYTHON, nodes, "\n\n") == WHITESPACE


def test_constructs_take_the_innermost_and_the_root_is_top_level():
    _, constructs = parse_labels(PYTHON, "Python")
    assert label_at(PYTHON, constructs, "os") == "import"
    assert label_at(PYTHON, constructs, "\n\n", 0) == TOP_LEVEL
    assert label_at(PYTHON, constructs, "a)") == "parameters"
    assert label_at(PYTHON, constructs, "return") == "function"
    assert label_at(PYTHON, constructs, "1)") == "call arguments"


def test_brackets_in_strings_and_comments_do_not_count():
    nodes, _ = parse_labels(PYTHON, "Python")
    depths = bracket_depths(PYTHON, nodes)
    assert label_at(PYTHON, depths, "def") == "0"
    assert label_at(PYTHON, depths, "(a)") == "1"  # a bracket has the depth of what it holds
    assert label_at(PYTHON, depths, "1)") == "1"
    assert depths[PYTHON.index("# return") :][0] == "0"  # the string's "(" didn't open one


def test_round_depth_counts_only_round_brackets():
    text = "if (a[0]) { b(c); }\n"
    nodes, _ = parse_labels(text, "C")
    round_only = bracket_depths(text, nodes, "(")
    assert label_at(text, round_only, "a") == "1"
    assert label_at(text, round_only, "b") == "0"
    assert label_at(text, round_only, "c") == "1"
    assert label_at(text, bracket_depths(text, nodes), "c") == "2"


def test_the_proposer_sees_syntax_labels():
    from types import SimpleNamespace

    from openinterp.agents.proposer import label_note

    labels = {SYNTAX_NODE: NUMBER, ROUND_DEPTH: "2"}
    assert label_note(SimpleNamespace(peak_labels=labels)) == "syntax node number, round depth 2"


def test_bracket_depth_is_capped_and_a_stray_closer_stays_at_zero():
    text = ")" + "(" * (syntax.MAX_DEPTH + 2)
    depths = bracket_depths(text, [BRACKET] * len(text))
    assert depths[0] == "0"
    assert depths[-1] == f"{syntax.MAX_DEPTH}+"


@pytest.mark.parametrize(
    ("indent", "unit"), [("  ", "two spaces"), ("    ", "four spaces"), ("\t", "tabs")]
)
def test_indent_levels_do_not_depend_on_the_width(indent, unit):
    text = f"a\n{indent}b\n\n{indent}{indent}c\nd\n"
    depths = indent_depths(text)
    assert len(depths) == len(text)
    assert [text[i] + depths[i] for i in map(text.index, "abcd")] == ["a0", "b1", "c2", "d0"]
    assert depths[text.index("\n\n") + 1] == "1"  # a blank line keeps the level before it


def test_a_comment_star_does_not_set_the_indent_unit():
    text = "/**\n * doc\n */\nclass A {\n    int a;\n    void f() {\n        g();\n    }\n}\n"
    depths = indent_depths(text)
    assert depths[text.index("g()")] == "2"
    assert depths[text.index("* doc")] == "0"


@pytest.mark.parametrize("language", GRAMMARS)
def test_every_grammar_loads_and_labels_every_character(language):
    text = "x = (1)\n"
    nodes, constructs = parse_labels(text, language)
    assert len(nodes) == len(constructs) == len(text)


def test_non_ascii_offsets_are_characters_not_bytes():
    text = 'é = "ü€𝒳"  # 𝒳\nx = (1)\n'
    nodes, _ = parse_labels(text, "Python")
    assert label_at(text, nodes, "x =") == IDENTIFIER
    assert label_at(text, nodes, "1") == NUMBER
    assert label_at(text, nodes, "€") == STRING


def test_truncated_code_still_labels_what_it_can():
    text = 'def f(a):\n    return "unfinished'
    nodes, constructs = parse_labels(text, "Python")
    assert len(nodes) == len(text)
    assert label_at(text, nodes, "def") == KEYWORD
    assert label_at(text, constructs, "a)") == "parameters"


def test_a_cut_inside_a_class_keeps_the_enclosing_constructs():
    text = "class A {\n    int x;\n    void f(int a) {\n        g(a, 1"
    nodes, constructs = parse_labels(text, "Java")
    assert len(nodes) == len(constructs) == len(text)
    assert label_at(text, constructs, "x;") == "type definition"
    assert label_at(text, constructs, "g(") == "function"
    assert label_at(text, constructs, "1") == "call arguments"
    assert label_at(text, nodes, "1") == NUMBER


def test_taggers_cover_a_document_and_skip_those_without_a_grammar():
    taggers = syntax_taggers(["Python", "Markdown", None])
    names = [SYNTAX_NODE, BRACKET_DEPTH, ROUND_DEPTH, INDENT_DEPTH, SYNTAX_CONSTRUCT]
    assert [t.name for t in taggers] == names
    for tagger in taggers:
        spans = tagger.spans(0, PYTHON)
        assert spans[0].start == 0
        assert spans[-1].end == len(PYTHON)
        assert all(a.end == b.start for a, b in zip(spans, spans[1:], strict=False))
        assert tagger.spans(1, "# title") == []
        assert tagger.spans(2, "plain text") == []


def _align_by_reading_every_span(offsets, spans):
    out = []
    for start, end in offsets:
        best, most = None, 0
        for span in spans:
            overlap = min(end, span.end) - max(start, span.start)
            if overlap > most:
                best, most = span.label, overlap
        out.append(best)
    return out


def test_align_agrees_with_reading_every_span():
    rng = random.Random(0)
    for _ in range(200):
        spans = []
        for i in range(rng.randrange(12)):
            start = rng.randrange(30)
            spans.append(Span(start, start + rng.randrange(1, 12), f"s{i}"))
        offsets = []
        for _ in range(rng.randrange(12)):
            start = rng.randrange(40)
            offsets.append((start, start + rng.randrange(0, 6)))
        assert align(offsets, spans) == _align_by_reading_every_span(offsets, spans)


def _texts(labels):
    return _Texts(["x"], "c", "0" * 64, None, None, None, labels)


def test_only_a_corpus_with_programming_languages_gets_syntax_taggers():
    from openinterp.core.confounds import LANGUAGE, PROGRAMMING_LANGUAGE

    plain = {t.name for t in _texts({LANGUAGE: ["en"]}).taggers()}
    assert not plain & {SYNTAX_NODE, BRACKET_DEPTH, INDENT_DEPTH, SYNTAX_CONSTRUCT}
    coded = [t.name for t in _texts({PROGRAMMING_LANGUAGE: ["Python"]}).taggers()]
    assert coded[:6] == [
        PROGRAMMING_LANGUAGE,
        SYNTAX_NODE,
        BRACKET_DEPTH,
        ROUND_DEPTH,
        INDENT_DEPTH,
        SYNTAX_CONSTRUCT,
    ]


def test_detector_pins_every_grammar_package():
    found = syntax.detector()
    assert "tree-sitter " in found
    for package, _ in GRAMMARS.values():
        assert f"{package} " in found


def test_syntax_labels_reach_a_scan_through_character_offsets():
    text = "x = (1)  # (\n"
    labels = TokenLabels(CharTokenizer(), [text, "plain"], syntax_taggers(["Python", None]))
    piece = Piece(0, 0, 1, text)
    row = labels.at(piece, [0, *map(ord, text)], text.index("1") + 1)
    assert row == {
        SYNTAX_NODE: NUMBER,
        BRACKET_DEPTH: "1",
        ROUND_DEPTH: "1",
        INDENT_DEPTH: "0",
        SYNTAX_CONSTRUCT: TOP_LEVEL,
    }
    other = labels.of(Piece(1, 0, 1, "plain"), torch.tensor([*map(ord, "plain")]))
    assert all((v == UNLABELLED).all() for v in other.values())
