"""Label the tokens of source code from its syntax tree, for the scan's syntax label sets.

A document with a known programming language (a corpus's ``code_language`` field) goes
through tree-sitter. Tree-sitter tolerates the broken code that a cut at ``max_chars``
leaves. Every character of the document then has one label in each of these label sets.

* ``syntax node``: identifier, keyword, string, comment, number, operator, bracket,
  punctuation, whitespace or other. A string or a comment counts as one node, so its
  contents are not read as code.
* ``bracket depth``: how many round, square and curly brackets are open. A bracket has
  the depth of what it holds. Angle brackets don't count, because most grammars do not
  tell them from comparison operators. Depths of ``MAX_DEPTH`` or more share one label.
* ``round depth``: the same count for round brackets only. GPT-2's L2 N676+ follows this
  count, and the count of all brackets hides it (papers/syntax-positive-controls/README.md).
* ``indent depth``: the indentation of the character's line, in levels. A blank line has
  the level of the line before it.
* ``syntax construct``: the innermost function, type definition, parameter list, call's
  arguments, import, loop or conditional around the character, or ``top level``.

The grammars name their nodes differently. So this module sorts node names into shared
labels by pattern. The sort is approximate and is not a language's own specification.
A language with no grammar in :data:`GRAMMARS` has no spans, so its tokens are unlabelled.
Tree-sitter gives offsets in UTF-8 bytes, and this module converts them to characters.
"""

from __future__ import annotations

import importlib
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.metadata import version
from itertools import accumulate, groupby

from openinterp.core.confounds import (
    BRACKET_DEPTH,
    INDENT_DEPTH,
    ROUND_DEPTH,
    SYNTAX_CONSTRUCT,
    SYNTAX_NODE,
)
from openinterp.harness.annotations import Span

# The github-code preset's `language` values and their grammars: (package, module function).
# These are its most frequent programming languages. Markup and build files have no entry.
GRAMMARS = {
    "C": ("tree-sitter-c", "language"),
    "C#": ("tree-sitter-c-sharp", "language"),
    "C++": ("tree-sitter-cpp", "language"),
    "GO": ("tree-sitter-go", "language"),
    "Haskell": ("tree-sitter-haskell", "language"),
    "Java": ("tree-sitter-java", "language"),
    "JavaScript": ("tree-sitter-javascript", "language"),
    "Lua": ("tree-sitter-lua", "language"),
    "PHP": ("tree-sitter-php", "language_php"),
    "Python": ("tree-sitter-python", "language"),
    "Ruby": ("tree-sitter-ruby", "language"),
    "Rust": ("tree-sitter-rust", "language"),
    "Scala": ("tree-sitter-scala", "language"),
    "Shell": ("tree-sitter-bash", "language"),
    "TypeScript": ("tree-sitter-typescript", "language_typescript"),
}

IDENTIFIER = "identifier"
KEYWORD = "keyword"
STRING = "string"
COMMENT = "comment"
NUMBER = "number"
OPERATOR = "operator"
BRACKET = "bracket"
PUNCTUATION = "punctuation"
WHITESPACE = "whitespace"
OTHER = "other"

TOP_LEVEL = "top level"
MAX_DEPTH = 8  # depths from here share a label: few tokens are deeper
DEFAULT_INDENT = 4  # the columns of one level, in a document whose lines don't show it

CLOSER = {"(": ")", "[": "]", "{": "}"}
OPENING, CLOSING = set(CLOSER), set(CLOSER.values())
_SEPARATORS = {",", ";", ":", "::", "."}

_ATOMS = (
    (COMMENT, re.compile(r"comment")),
    (STRING, re.compile(r"(^|_)(string|char|character|heredoc|regex)(_|$)")),
)
_NUMBER = re.compile(r"(^|_)(number|integer|int|float|floating|decimal|hex|octal|binary|real)(_|$)")
_NAMED_KEYWORD = re.compile(
    r"^(true|false|null|nil|none|this|self|super|primitive_type|predefined_type|void_type"
    r"|boolean_type|boolean_literal|null_literal|integral_type|floating_point_type)$"
)
_NAME = re.compile(r"identifier|name|word|variable|constant|symbol|field|type")
# Innermost wins, because the walk paints a child's characters after its parent's.
_CONSTRUCTS = (
    ("import", re.compile(r"import|include|using_directive|use_declaration|namespace_use")),
    ("call arguments", re.compile(r"^(argument_list|arguments|actual_parameters)$")),
    ("parameters", re.compile(r"parameter")),
    (
        "function",
        re.compile(
            r"^(function|method|lambda|closure_expression|func_literal|arrow_function"
            r"|generator_function|anonymous_function)"
            r"(_definition|_declaration|_item|_expression|_creation_expression)?$"
            r"|^constructor_declaration$"
        ),
    ),
    (
        "type definition",
        re.compile(
            r"^(class|struct|interface|enum|trait|union|impl|object|module|namespace|type)"
            r"_(definition|declaration|specifier|item)$|^(class|module)$"
        ),
    ),
    (
        "loop",
        re.compile(
            r"^(for|while|do|foreach|enhanced_for|for_in|for_range|repeat|loop|until)"
            r"(_statement|_expression)?$"
        ),
    ),
    ("conditional", re.compile(r"^(if|unless|match|case)(_statement|_expression)?$|switch_")),
)


def detector() -> str:
    """Name the parser and grammar versions, which fix what the syntax label sets say."""
    packages = ("tree-sitter", *sorted(package for package, _ in GRAMMARS.values()))
    return f"openinterp.syntax ({', '.join(f'{p} {version(p)}' for p in packages)})"


@cache
def _parser(language: str):
    """Return the parser of ``language``, a key of :data:`GRAMMARS`."""
    import tree_sitter

    package, function = GRAMMARS[language]
    module = importlib.import_module(package.replace("-", "_"))
    return tree_sitter.Parser(tree_sitter.Language(getattr(module, function)()))


def _char_offsets(text: str) -> list[int]:
    """Return the map from each UTF-8 byte offset of ``text`` to its character offset."""
    if text.isascii():
        return list(range(len(text) + 1))
    widths = (1 if c < "\x80" else 2 if c < "ࠀ" else 3 if c < "\U00010000" else 4 for c in text)
    out = []
    for i, width in enumerate(widths):
        out += [i] * width
    return [*out, len(text)]


def _encode(text: str) -> bytes:
    """UTF-8 of ``text``. A corpus's JSON can hold a lone surrogate, so this keeps it."""
    return text.encode("utf-8", "surrogatepass")


def _node_kind(node_type: str, named: bool, text: str) -> str:
    """Return the ``syntax node`` label of a tree leaf (or of a string or comment)."""
    if not named:
        if text in OPENING or text in CLOSING:
            return BRACKET
        if text in _SEPARATORS:
            return PUNCTUATION
        return KEYWORD if text[:1].isalpha() or text[:1] == "_" else OPERATOR
    for label, pattern in _ATOMS:
        if pattern.search(node_type):
            return label
    if _NUMBER.search(node_type):
        return NUMBER
    if _NAMED_KEYWORD.match(node_type):
        return KEYWORD
    return IDENTIFIER if _NAME.search(node_type) else OTHER


def _atom(node) -> bool:
    """Is this node a string or a comment, whose contents don't count as code?"""
    return node.is_named and any(pattern.search(node.type) for _, pattern in _ATOMS)


def _construct(node) -> str | None:
    """Return the ``syntax construct`` label of a node, or ``None`` if it opens none."""
    if node.is_named:
        for label, pattern in _CONSTRUCTS:
            if pattern.search(node.type):
                return label
    return None


def _paint(text: str, language: str) -> tuple[list[str], list[str]]:
    """Return each character's ``syntax node`` and ``syntax construct`` labels, from one parse."""
    chars = _char_offsets(text)
    nodes: list[str | None] = [None] * len(text)
    constructs = [TOP_LEVEL] * len(text)
    root = _parser(language).parse(_encode(text)).root_node
    todo = [root]
    while todo:  # not recursive, because deep code would pass Python's recursion limit
        node = todo.pop()
        start, end = chars[node.start_byte], chars[node.end_byte]
        # The root is the file. Python calls it `module`, which is not a Ruby `module`.
        if node != root and (construct := _construct(node)) is not None:
            constructs[start:end] = [construct] * (end - start)
        if _atom(node) or not node.children:
            kind = _node_kind(node.type, node.is_named, text[start:end])
            nodes[start:end] = [kind] * (end - start)
        else:
            todo.extend(reversed(node.children))
    return [_gap_kind(c) if n is None else n for n, c in zip(nodes, text, strict=True)], constructs


def _missing_closers(text: str, nodes: Sequence[str]) -> str:
    """Return what closes the brackets still open at the end of ``text``."""
    stack: list[str] = []
    for char, node in zip(text, nodes, strict=True):
        if node == BRACKET and char in OPENING:
            stack.append(CLOSER[char])
        elif node == BRACKET and char in CLOSING and stack:
            stack.pop()
    # A cut also leaves a statement unfinished, and a block needs its statements to end.
    return "".join(reversed(stack)).replace("}", ";}")


def parse_labels(text: str, language: str) -> tuple[list[str], list[str]]:
    """Return each character's ``syntax node`` and ``syntax construct`` labels.

    The corpus cuts a document at ``max_chars``, usually inside a class or a function. The
    parser then has no node for that class or function, so its contents read as top level.
    So this parses again with the missing closing brackets added, and labels only ``text``.
    """
    nodes, constructs = _paint(text, language)
    closers = _missing_closers(text, nodes)
    if not closers:
        return nodes, constructs
    nodes, constructs = _paint(text + closers, language)
    return nodes[: len(text)], constructs[: len(text)]


def _gap_kind(char: str) -> str:
    """Return the ``syntax node`` label of a character that no tree leaf covers."""
    return WHITESPACE if char.isspace() else OTHER


def _depth_label(depth: int) -> str:
    return f"{MAX_DEPTH}+" if depth >= MAX_DEPTH else str(depth)


def bracket_depths(text: str, nodes: Sequence[str], kinds: str = "([{") -> list[str]:
    """Return each character's depth of the brackets that ``kinds`` opens.

    ``nodes`` holds the characters' ``syntax node`` labels. So a bracket in a string or
    a comment doesn't count. A closing bracket with no opening bracket leaves the depth at 0.
    """
    openers, closers = set(kinds), {CLOSER[k] for k in kinds}
    out, depth = [], 0
    for char, node in zip(text, nodes, strict=True):
        if node == BRACKET and char in openers:
            depth += 1
        out.append(_depth_label(depth))
        if node == BRACKET and char in closers:
            depth = max(0, depth - 1)
    return out


def _indent_unit(widths: Sequence[int]) -> int:
    """Return the columns of one level: the most frequent step between indented lines.

    Lines' first indents can mislead. A `` * `` comment line has one space, and is not a
    level. So this takes the step that the lines repeat most. ``DEFAULT_INDENT`` if there
    is no step.
    """
    steps = Counter(b - a for a, b in zip(widths, widths[1:], strict=False) if b > a)
    return steps.most_common(1)[0][0] if steps else DEFAULT_INDENT


def indent_depths(text: str) -> list[str]:
    """Return each character's ``indent depth`` label, from its line's leading whitespace."""
    lines = text.splitlines(keepends=True)
    tabbed = [len(line) - len(line.lstrip(" \t")) for line in lines]
    blank = [not line.strip() for line in lines]
    tabs = [line[:n].count("\t") for line, n in zip(lines, tabbed, strict=True)]
    spaces = [n - t for n, t in zip(tabbed, tabs, strict=True)]
    solid = [s for s, b, t in zip(spaces, blank, tabs, strict=True) if not b and not t]
    unit = _indent_unit(solid)
    out, level = [], 0
    for line, is_blank, tab, space in zip(lines, blank, tabs, spaces, strict=True):
        if not is_blank:
            level = tab + space // unit
        out += [_depth_label(level)] * len(line)
    return out


def _runs(labels: Sequence[str]) -> list[Span]:
    """Join neighbouring characters of one label into spans."""
    lengths = [(label, len(list(run))) for label, run in groupby(labels)]
    ends = accumulate(n for _, n in lengths)
    return [Span(end - n, end, label) for (label, n), end in zip(lengths, ends, strict=True)]


@dataclass(frozen=True)
class SyntaxTagger:
    """Label a document's characters with ``read``, for documents that have a grammar."""

    name: str
    read: Callable[[int, str], list[Span]]

    def spans(self, doc: int, text: str) -> list[Span]:
        """The document's spans in this label set. A document without a grammar has none."""
        return self.read(doc, text)


def syntax_taggers(languages: Sequence[str | None]) -> list[SyntaxTagger]:
    """Return one tagger per syntax label set, over documents of these ``languages``.

    The taggers share one parse of each document. ``languages`` holds one value per
    document, or ``None``. A document with a language outside :data:`GRAMMARS` has no spans.
    """

    @cache
    def parsed(doc: int, text: str) -> tuple[list[str], list[str]] | None:
        language = languages[doc]
        return parse_labels(text, language) if language in GRAMMARS else None

    def tagger(name: str, labels: Callable[[str, list[str], list[str]], Sequence[str]]):
        def read(doc: int, text: str) -> list[Span]:
            found = parsed(doc, text)
            return [] if found is None else _runs(labels(text, *found))

        return SyntaxTagger(name, read)

    return [
        tagger(SYNTAX_NODE, lambda _, nodes, __: nodes),
        tagger(BRACKET_DEPTH, lambda text, nodes, _: bracket_depths(text, nodes)),
        tagger(ROUND_DEPTH, lambda text, nodes, _: bracket_depths(text, nodes, "(")),
        tagger(INDENT_DEPTH, lambda text, _, __: indent_depths(text)),
        tagger(SYNTAX_CONSTRUCT, lambda _, __, constructs: constructs),
    ]
