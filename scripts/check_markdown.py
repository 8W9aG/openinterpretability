"""Fail on a Markdown style problem or a broken relative link.

This checks hand-written Markdown only. The pages under `wiki/` are generated, and their
text comes from models and corpora. tests/test_markdown_templates.py lints their templates
through a rendered sample. The style rules are in `[tool.pymarkdown]` in `pyproject.toml`.

Run with `make lint`. CI runs it in the lint step.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parents[1]

EXEMPT = (
    "wiki/*",  # generated pages
    "openinterp/builder/templates/*",  # Jinja, not Markdown, until rendered
    ".claude/skills/asd-ste100/*",  # vendored as is from danyuchn/asd-ste100-skill
)

FENCE = re.compile(r"\s*(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(`+).+?\1")
HEADING = re.compile(r"#{1,6}\s+(.+?)\s*#*\s*$")
INLINE_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE_DEFINITION = re.compile(r"\s{0,3}\[[^\]]+\]:\s*<?([^\s>]+)>?")
HEADING_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def markdown_files() -> list[Path]:
    """Every tracked or new Markdown file that is not exempt."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout
    names = [f for f in out.splitlines() if not any(fnmatch(f, p) for p in EXEMPT)]
    return [REPO / f for f in names if (REPO / f).exists()]


def style_problems(files: list[Path]) -> list[str]:
    """One line for each PyMarkdown finding in ``files``. PyMarkdown errors raise.

    The working directory is the repository root because PyMarkdown reads `pyproject.toml` there.
    """
    if not files:
        return []
    result = subprocess.run(
        [sys.executable, "-m", "pymarkdown", "scan", *map(str, files)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise RuntimeError(f"pymarkdown failed: {result.stderr or result.stdout}")
    return result.stdout.splitlines()


def outside_fences(text: str) -> list[str]:
    """The lines of ``text`` that are not part of a fenced code block."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = FENCE.match(line)
        if fence is None and match:
            fence = match[1]
        elif fence is not None:
            # A closing fence uses the same character and is at least as long.
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = None
        else:
            lines.append(line)
    return lines


def heading_anchors(text: str) -> set[str]:
    """The `#anchor` of each heading in ``text``, with the GitHub rules for duplicates."""
    anchors = set()
    seen: Counter[str] = Counter()
    for line in outside_fences(text):
        match = HEADING.fullmatch(line)
        if not match:
            continue
        slug = github_slug(match[1])
        anchors.add(f"{slug}-{seen[slug]}" if seen[slug] else slug)
        seen[slug] += 1
    return anchors


def github_slug(heading: str) -> str:
    """The anchor GitHub gives ``heading``, before it numbers duplicates.

    GitHub lowercases the text and removes punctuation. Each space becomes a hyphen.
    """
    text = HEADING_LINK.sub(r"\1", heading).replace("`", "").replace("*", "")
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def link_targets(text: str) -> list[str]:
    """Each inline link, image or reference definition target in the prose of ``text``."""
    targets = []
    for line in outside_fences(text):
        line = CODE_SPAN.sub("", line)  # `[a](b)` in a code span is not a link
        targets += INLINE_LINK.findall(line)
        definition = REFERENCE_DEFINITION.match(line)
        if definition:
            targets.append(definition[1])
    return targets


def link_problems(files: list[Path]) -> list[str]:
    """One line for each relative link in ``files`` to a missing file or heading.

    External links are not checked: a host that is down must not fail lint.
    """
    problems = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for target in link_targets(text):
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                continue
            dest = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not dest.exists():
                problems.append(f"{path}: broken link {target}: no such file")
            elif parts.fragment and dest.suffix == ".md":
                anchors = heading_anchors(dest.read_text(encoding="utf-8"))
                if unquote(parts.fragment).lower() not in anchors:
                    problems.append(f"{path}: broken link {target}: no such heading")
    return problems


def main() -> int:
    """Print every problem; return 1 if there is one."""
    files = markdown_files()
    problems = style_problems(files) + link_problems(files)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
