"""A document's programming language, for corpora whose dataset doesn't say it.

Used by ``open-interpretability corpus --detect-code``. The detector is Google's Magika, a file-type
model that reads content only (a corpus has no file names). We compared it with go-enry
(GitHub Linguist), a CodeBERTa language classifier and Pygments
(``docs/analyses/code-detection.md``). The tests were labelled GitHub files, prose, prose
followed by code, and the registry's pile-10k corpus. Magika was the most accurate, and
it alone never labelled prose as code.

A document gets a label only when Magika's confidence reaches ``MIN_CONFIDENCE`` and its
label is a language of the github-code preset (``corpus_presets.GITHUB_CODE_LANGUAGES``).
So prose, data files (JSON, CSV, XML) and file types outside that list stay unlabelled. The label
is that preset's name for the language, so exact and detected labels agree. Detection is
per document, never per window: at 200 characters Magika is right on 70% of files, and
on whole files on 91.5%.

Magika's model ships inside its package, so :func:`detector` records the package version
and the model name.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from importlib.metadata import version

# Below this the document is left unlabelled. On 1,000 labelled github-code files it keeps
# 89% at 95.5% accuracy (0.5 keeps 98% at 92.7%). No line labelled any of 300 prose documents.
MIN_CONFIDENCE = 0.8

# Magika's labels of the github-code languages, and those languages' names there
# (tests/test_codeid.py checks that every name in GITHUB_CODE_LANGUAGES has one label).
MAGIKA_LABELS = {
    "asm": "Assembly", "batch": "Batchfile", "c": "C", "cs": "C#", "cpp": "C++",
    "cmake": "CMake", "css": "CSS", "dockerfile": "Dockerfile", "fortran": "FORTRAN",
    "go": "GO", "html": "HTML", "haskell": "Haskell", "java": "Java",
    "javascript": "JavaScript", "julia": "Julia", "lua": "Lua", "makefile": "Makefile",
    "markdown": "Markdown", "php": "PHP", "perl": "Perl", "powershell": "PowerShell",
    "python": "Python", "ruby": "Ruby", "rust": "Rust", "sql": "SQL", "scala": "Scala",
    "shell": "Shell", "latex": "TeX", "typescript": "TypeScript", "vba": "Visual Basic",
}  # fmt: skip


@cache
def _magika():
    from magika import Magika

    return Magika()


@cache
def detector() -> str:
    """The detector and its settings, as a corpus's source sidecar records them."""
    model = _magika().get_model_name()
    return f"magika {version('magika')} (model {model}), confidence >= {MIN_CONFIDENCE}"


def predict(text: str) -> tuple[str | None, float]:
    """Return the github-code language that Magika gives ``text``, and its confidence.

    The language is ``None`` when Magika's label is not one of those languages.
    """
    prediction = _magika().identify_bytes(text.encode("utf-8", "surrogatepass")).prediction
    return MAGIKA_LABELS.get(prediction.dl.label), float(prediction.score)


def detect(texts: Sequence[str]) -> list[str | None]:
    """Each document's programming language, or ``None`` (see the module docstring)."""
    out: list[str | None] = []
    for text in texts:
        language, p = predict(text)
        out.append(language if p >= MIN_CONFIDENCE else None)
    return out
