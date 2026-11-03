---
name: commenting
description: Write and review comments and docstrings so that what the code does, and why, is never ambiguous, in ASD-STE100 Simplified Technical English. Use when writing or changing a comment or docstring, when adding a public function, class or module, and when reviewing a diff.
---

# Commenting

A comment tells the reader what the code cannot: the reason, the constraint or the source. Write it in ASD-STE100 Simplified Technical English (STE), so that a reader whose first language is not English, or an agent, reads one meaning only.

Linters check the form. This skill is for what they cannot judge. To rewrite a sentence into STE, load the `asd-ste100` skill (`.claude/skills/asd-ste100/`). Its linter, `scripts/ste-lint.py`, checks prose outside Vale's paths, such as skills and docs.

| Check | Tool | Scope |
| --- | --- | --- |
| Missing or malformed docstrings | ruff `D` rules, Google convention (`pyproject.toml`) | everything; `tests/` needs no docstrings (`D1`) |
| Commented-out code | ruff `ERA` | everything |
| Docstring coverage | `interrogate` (`[tool.interrogate] fail-under`, never lowered) | `openinterp/` |
| Sentence length, banned words, passive voice | Vale, `make vale` (`.vale/styles/STE`) | `VALE_PATHS` in the `Makefile`: `openinterp`, `scripts`, `tests` |

## When to comment

Comment when a reader cannot get the reason from the code. These need a comment:

- **A constraint from outside the code.** A protocol rule, an API limit, a file format, a bug in a library. Cite it.
- **A choice that looks wrong.** A sort that seems unnecessary, a broad `except`, a magic number. Say why it is correct.
- **A value with a source.** A threshold, a price, a default. Name the section, paper or page it comes from.
- **Write-once and signed data.** Say that records are signed or write-once when this is why the code does not change them.

Do not comment when the code can say it:

- **A comment that restates the code.** Delete it, or replace it with the reason.
- **A comment that names a variable's meaning.** Rename the variable.
- **A comment that explains a long block.** Extract a function with a name that explains it.
- **A step number** (`# 1. Scan evidence.`). Keep the text, remove the number. The code already gives the order.
- **Commented-out code.** Delete it. Git keeps the history.

## Docstrings

Every public module, class, function and method has one (ruff `D1`). `__init__` and magic methods are described by their class. Tests need none: a test's name says what it checks.

One exception: a Pydantic model that an LLM agent answers in (`HypothesisDraft`, `ProbeSet`, `DetectionAnswer` and the models they contain). Pydantic puts a class docstring into the JSON schema, so a docstring there changes what the agent is told. Mark the class `# noqa: D101` with that reason, and describe the fields with `Field(description=...)`.

- **The first line is one sentence that stands alone** (`D205`), at most 100 characters with its indent. Then a blank line, then the details.
- **A function's first line says what it returns or does.** "Return whether…", "Record…", or a noun phrase for a value, as in `"""Every recorded draw, oldest first."""`.
- **A property or schema field says what the value is**, with its format when that helps: `"""``L3/H7``: the unit within its model."""`.
- **Say the edge cases**: what `None` means, what happens with empty input.

## STE rules we apply

- **Short sentences.** At most 20 words in an instruction and 25 in a description. Vale checks 25.
- **One topic in each sentence.** Split at a semicolon or a colon that starts a new idea.
- **Active voice.** Name who or what does it: "The scan writes…", not "…is written". Schemas often have no actor; there, the passive is acceptable.
- **One meaning for each word, one word for each meaning.** Do not use "unit" and "component" for the same thing in one docstring.
- **Approved words.** Use simple words: "use", not "utilize"; "get", not "obtain"; "through", not "via". `.vale/styles/STE/Words.yml` is the banned list, with the word to use. Add to it when you find a new one. Our list is not the ASD dictionary. ASD's terms do not let us copy from it.
- **Technical names are approved.** Use them exactly, with the meaning that `docs/PROTOCOL.md` gives: pole, unit, neuron, head, feature, dictionary, latent, probe, hard negative, holdout, verdict, VERIFIED, gate, scan, record, draw, proposer, reviewer, judge, protocol version, strength, activation line.

## Before and after, from this codebase

A summary that is two sentences joined by a colon:

```python
# Before
"""Where a measurement ran. Reproducing a record pins the model revision and dtype;
the device and library versions explain drift that a different setup can cause.
"""

# After
"""Where a measurement ran.

Reproducing a record pins the model revision and dtype. The device and library
versions explain drift that a different setup can cause.
"""
```

A docstring that does not say what the function does:

```python
# Before
def find_transcript(agent: str) -> Path:
    """``agent`` is a transcript path, or the agent ID Claude Code reports when a subagent
    finishes (``a1b2c3…``, with or without the ``agent-`` prefix).
    """


# After
def find_transcript(agent: str) -> Path:
    """Return the transcript file of ``agent``.

    ``agent`` is a transcript path, or the agent ID that Claude Code reports when a
    subagent finishes (``a1b2c3…``, with or without the ``agent-`` prefix).
    """
```

A docstring that says what, but not why (`openinterp/core/strict.py`):

```python
# Before: no docstring
class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


# After
class StrictModel(BaseModel):
    """The base of every schema.

    Strict types and ``extra="forbid"`` reject a record that does not match its schema.
    ``frozen`` stops code from changing a model after validation, because records are signed.
    """
```

A list wrapped so that a line reads as code (ruff `ERA` flagged `# (Latin), Russian (Cyrillic), Greek, …` in `openinterp/corpus.py`). Restructure the sentence; do not add `# noqa`.

## Citing a source

- **The protocol:** the file and section, as in `(docs/PROTOCOL.md §4)`, or `§4` when the file is clear.
- **A paper:** author and year, as in `(Paulo et al., 2025)`. Put the full reference in the skill or doc that uses it.
- **A protocol version:** `v0.8:` at the start of the comment, for a field or rule that a version added.
- **Another place in the code:** the qualified name, as in ``:func:`openinterp.core.costs.stage` `` or `openinterp.harness.confounds`.

## Reviewing a diff

Flag:

- a docstring that only restates the name, written to satisfy ruff `D1`;
- a comment that restates the code, or that a rename or an extracted function could replace;
- a value, threshold or rule with no source;
- a sentence of more than 25 words, a banned word, or two meanings for one word;
- a comment that the change made wrong. Comments go stale when the code changes and nothing fails.

## How Vale counts a sentence

`.vale/styles/STE/SentenceLength.yml` splits each comment or docstring into sentences itself, because Vale does not do this for source code. A sentence ends at `.`, `!` or `?` before a space (a closing quote or bracket may come between), at a colon that ends a line, and where a list item (`-` or `*`) starts. So `0.5` and `e.g.` do not end a sentence. End each list item with a full stop.

Some lines do not count, because they are not prose: lines indented 4 or more spaces in a docstring (a formula, a file layout, a literal block) and indented command examples (`openinterp …`, `uv …`, `make …`, `git …`, `python …`). Vale's `vale off` comments do not work in Python files, so restructure the text instead.
