"""Run the STE linter on prose only, with a looser profile for papers (issues #3137, #4202).

APA 7 lists every author, so a reference can run past the linter's 25 words. This copies
the given Markdown files into a temporary folder, at the same relative paths, with each
line of a ``## References`` section blanked. Line numbers and paths stay the same. Then it
runs ``ste-lint.py`` there, with the arguments given, and returns its exit code.

A paper (``papers/<slug>/README.md``) gets the paper profile. Academic prose relates its
claims with subordinate clauses, so a sentence may have up to :data:`PAPER_MAX_WORDS`
words, not 25. To keep the prose plain, each paper's mean sentence length must be at most
:data:`PAPER_MEAN_WORDS`. Every other STE rule applies to papers as to the other files.

    python3 scripts/ste_prose.py [ste-lint options] FILE...
"""

from __future__ import annotations

import contextlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LINTER = Path(".claude/skills/asd-ste100/scripts/ste-lint.py").resolve()
PAPER_MAX_WORDS = 40  # the paper profile's cap on one sentence
PAPER_MEAN_WORDS = 25  # the paper profile's cap on a paper's mean sentence length


def without_references(text: str) -> str:
    """``text`` with each line of its ``## References`` section blank."""
    out, inside = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            inside = line.strip() == "## References"
        out.append("\n" if inside and not line.startswith("## ") else line)
    return "".join(out)


def is_paper(path: str) -> bool:
    """True for a paper's text, ``papers/<slug>/README.md``; not for the papers' index."""
    parts = Path(path).parts
    return len(parts) == 3 and parts[0] == "papers" and parts[2] == "README.md"


def load_linter():
    """The vendored linter as a module, so the paper profile can change its cap."""
    spec = importlib.util.spec_from_file_location("ste_lint", LINTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lint_papers(options: list[str], papers: list[str]) -> int:
    """Lint ``papers`` with the paper profile, in the current folder; 1 on any failure.

    The linter's cap becomes :data:`PAPER_MAX_WORDS`. A wrapper around its sentence check
    records every sentence's length, so that the mean of each paper can be checked too.
    """
    linter = load_linter()
    linter.MAX_WORDS = PAPER_MAX_WORDS
    lengths: dict[str, list[int]] = {}
    check = linter._long_sentence_findings

    def recording(pieces, filename):
        joined = " ".join(text for _, _, text in pieces)
        starts = linter._sentence_starts(joined)
        for start, end in zip(starts, [*starts[1:], len(joined)], strict=True):
            if n := linter._count_words(joined[start:end]):
                lengths.setdefault(filename, []).append(n)
        return check(pieces, filename)

    linter._long_sentence_findings = recording
    code = linter.main([*options, *papers])
    for paper in papers:
        found = lengths.get(paper, [])
        mean = sum(found) / len(found) if found else 0.0
        if mean > PAPER_MEAN_WORDS:
            print(f"{paper}: mean sentence length {mean:.1f} words (cap {PAPER_MEAN_WORDS})")
            code = 1
    return code


def main(args: list[str]) -> int:
    """Lint ``args``' files, as copies with their references blanked."""
    options = [arg for arg in args if arg.startswith("-")]
    files = [arg for arg in args if not arg.startswith("-") and Path(arg).is_file()]
    papers = [f for f in files if is_paper(f)]
    others = [f for f in files if not is_paper(f)]
    with tempfile.TemporaryDirectory() as tmp:
        for arg in files:
            path, copy = Path(arg), Path(tmp) / arg
            copy.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".md":
                copy.write_text(without_references(path.read_text(encoding="utf-8")))
            else:
                shutil.copy(path, copy)
        code = 0
        if others or not papers:
            command = [sys.executable, str(LINTER), *options, *others]
            code = subprocess.run(command, cwd=tmp).returncode
        if papers:
            with contextlib.chdir(tmp):
                code = lint_papers(options, papers) or code
        return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
