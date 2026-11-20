"""Every work that a paper cites has a review page, or waits for one (issue #5346).

A reference in a paper's *References* that carries an arXiv ID, a DOI or an ACL Anthology
ID names a work. That work needs a page in ``docs/literature/`` whose Citation gives the
same identifier. A work with no page yet must be listed in :data:`AWAITING`, one line each.
The line gives the open ``area: literature`` issue that will write its page::

    2304.01373 #5331

``tests/test_papers.py`` fails on a cited work that has neither. It also fails on a line of
:data:`AWAITING` whose work now has a page, or that no paper cites any more. So the list
only shrinks. The ``literature-review`` skill deletes the line when it writes the page.

    uv run python scripts/cited_works.py   # the cited works with no page, and their issues
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"
LITERATURE = ROOT / "docs" / "literature"
AWAITING = PAPERS / "awaiting_review.txt"
CITATION_CHARS = 4000  # how far into a review page its Citation reaches
IDENTIFIER = re.compile(
    r"arXiv:(\d{4}\.\d{4,5})"
    r"|arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})"
    r"|doi\.org/(10\.[^\s)\]]+)"
    r"|aclanthology\.org/([\w.-]+?)/?[\s)\]]",
    re.IGNORECASE,
)


def identifiers(text: str) -> list[str]:
    """The arXiv IDs, DOIs and ACL Anthology IDs in ``text``, lower case, in order.

    Args:
        text: Any Markdown.
    """
    found = []
    for match in IDENTIFIER.finditer(text + " "):
        value = next(group for group in match.groups() if group).lower().rstrip(".")
        if value not in found:
            found.append(value)
    return found


def cited_works(papers: Path = PAPERS) -> dict[str, list[str]]:
    """For each paper, the first identifier of each reference that has one.

    Args:
        papers: The papers folder.
    """
    out = {}
    for readme in sorted(papers.glob("*/README.md")):
        parts = readme.read_text(encoding="utf-8").split("\n## References\n", 1)
        if len(parts) < 2:
            continue
        section = parts[1].split("\n## ", 1)[0]
        works = [identifiers(line) for line in section.splitlines() if line.startswith("- ")]
        out[readme.parent.name] = [ids[0] for ids in works if ids]
    return out


def reviewed(literature: Path = LITERATURE) -> set[str]:
    """The identifiers that the review pages' Citations give.

    Args:
        literature: The literature folder.
    """
    out: set[str] = set()
    for page in literature.glob("*.md"):
        out.update(identifiers(page.read_text(encoding="utf-8")[:CITATION_CHARS]))
    return out


def awaiting(path: Path = AWAITING) -> dict[str, int]:
    """The listed works with no page yet, and each one's literature issue.

    Args:
        path: The list.
    """
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if line.strip() and not line.startswith("#"):
            work, issue = line.split()[:2]
            out[work.lower()] = int(issue.lstrip("#"))
    return out


def problems(papers: Path = PAPERS, literature: Path = LITERATURE,
             path: Path = AWAITING) -> list[str]:  # fmt: skip
    """Each cited work with no page and no line, and each line that should go.

    Args:
        papers: The papers folder.
        literature: The literature folder.
        path: The list of works that await review.
    """
    pages, listed = reviewed(literature), awaiting(path)
    cited = {work: paper for paper, works in cited_works(papers).items() for work in works}
    out = [f"{paper}: {work} has no page in docs/literature/ and no line in {path.name}"
           for work, paper in sorted(cited.items()) if work not in pages | set(listed)]  # fmt: skip
    for work in sorted(listed):
        if work in pages:
            out.append(f"{path.name}: {work} has a page now; delete its line")
        elif work not in cited:
            out.append(f"{path.name}: {work} is cited by no paper; delete its line")
    return out


def main() -> int:
    """Print each cited work that has no page, with its issue, and any problem."""
    for work, issue in sorted(awaiting().items()):
        print(f"{work} #{issue}")
    found = problems()
    print("\n".join(found) or "Every cited work has a page or a line.", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
