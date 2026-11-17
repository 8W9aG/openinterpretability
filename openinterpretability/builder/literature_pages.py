"""The literature review on the site: a subsection of Research, one page per review (#3293).

The reviews are the Markdown files in ``docs/literature/``. ``README.md`` is their index,
and each other file is one paper or post. Each becomes a page under
``research/literature/``, rendered as the papers are
(:mod:`openinterpretability.builder.research`), with the same search for personal data.
Links between the reviews become links between the pages. Links to other repository files
point at GitHub. An install has no ``docs/`` folder, so it writes no pages.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from openinterpretability.builder.pii import redact_text
from openinterpretability.builder.research import _MARKDOWN, _plain

LITERATURE = Path(__file__).resolve().parents[2] / "docs" / "literature"  # absent in an install
SECTION = "research/literature"
MAP = "map.json"  # scripts/literature_map.py writes it (#3294)
STATUSES = ("used", "planned", "not applicable")  # the index table's status column
_ROW = re.compile(r"^\| \[[^\]]*\]\(([\w-]+)\.md\)[^|]*\| ([a-z ]+) \|", re.M)


@dataclass(frozen=True)
class Review:
    """One page of the literature review."""

    name: str  # the file's name without .md; README is the index
    title: str
    body: str  # rendered HTML without the title

    @property
    def href(self) -> str:
        """The page, relative to the site root."""
        page = "index" if self.name == "README" else self.name
        return f"{SECTION}/{page}.html"


def _rewrite(tokens, repo_url: str) -> None:
    """Point links at the review's pages, and links to other repository files at GitHub."""
    for token in tokens:
        for child in token.children or []:
            href = child.attrGet("href") if child.type == "link_open" else None
            if not href or re.match(r"[a-z]+:|#", href):
                continue
            target, _, fragment = href.partition("#")
            anchor = f"#{fragment}" if fragment else ""
            if "/" not in target and target.endswith(".md"):
                page = "index" if target == "README.md" else target[: -len(".md")]
                child.attrSet("href", f"{page}.html{anchor}")
            else:
                path = posixpath.normpath(posixpath.join("docs/literature", target))
                child.attrSet("href", f"{repo_url}/blob/main/{path}{anchor}")


def _review(path: Path, repo_url: str) -> Review:
    """One file, rendered: its level 1 title, and the rest as HTML."""
    tokens = _MARKDOWN.parse(redact_text(path.read_text(encoding="utf-8")))
    _rewrite(tokens, repo_url)
    start, title = 0, path.stem
    if tokens and tokens[0].type == "heading_open" and tokens[0].tag == "h1":
        title, start = _plain(tokens[1]), 3
    body = _MARKDOWN.renderer.render(tokens[start:], _MARKDOWN.options, {})
    return Review(path.stem, title, body)


def load_reviews(repo_url: str, folder: Path = LITERATURE) -> list[Review]:
    """The index first, then each review in file-name order; none without the folder."""
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*.md"), key=lambda p: (p.name != "README.md", p.name))
    return [_review(path, repo_url) for path in files]


def map_text(path: Path) -> str | None:
    """What the map embeds for one review: its title and *Summary* section, or None.

    The index and the queue have no *Summary* section, so they are left off the map.
    """
    text = path.read_text(encoding="utf-8")
    title = next((line[2:] for line in text.splitlines() if line.startswith("# ")), path.stem)
    found = re.search(r"^## Summary\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return f"{title}\n\n{found.group(1).strip()}" if found else None


def statuses(folder: Path) -> dict[str, str]:
    """Each review's status from the index table: used, planned or not applicable."""
    index = folder / "README.md"
    rows = _ROW.findall(index.read_text(encoding="utf-8")) if index.exists() else []
    return {name: status.strip() for name, status in rows if status.strip() in STATUSES}


WIDTH, HEIGHT, PAD = 720, 420, 40  # the map's viewBox and margin, in SVG units
LABEL = 34  # the longest label, in characters, before it is cut with an ellipsis
CHAR, LINE = 6.2, 13  # a label's width per character and its height, at font size 11


@dataclass(frozen=True)
class Point:
    """One review on the map, with its marker at (cx, cy) and its label at (tx, ty)."""

    name: str
    title: str
    status: str
    cx: float
    cy: float
    tx: float = 0.0
    ty: float = 0.0
    anchor: str = "start"  # "end" puts the label to the left of the marker

    @property
    def moved(self) -> bool:
        """True when the label sits above or below its marker, so it needs a leader line."""
        return self.ty != self.cy + 4

    @property
    def label(self) -> str:
        """The title, cut to :data:`LABEL` characters."""
        return self.title if len(self.title) <= LABEL else self.title[: LABEL - 1] + "…"


def _box(p: Point, tx: float, ty: float, anchor: str) -> tuple[float, float, float, float]:
    """The label's box (left, top, right, bottom) at (tx, ty) with ``anchor``."""
    width = len(p.label) * CHAR
    left = tx if anchor == "start" else tx - width
    return left, ty - LINE + 3, left + width, ty + 3


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


_SPOTS = [(dx, anchor, dy * LINE) for dy in (0, 1, -1, 2, -2)
          for dx, anchor in ((10, "start"), (-10, "end"))] + [(10, "start", 0)]  # fmt: skip


def place_labels(points: list[Point]) -> list[Point]:
    """Give each label the first spot that overlaps no label or marker placed before it.

    The spots are right of the marker, then left of it, then each of those moved up or
    down by one or two lines. When every spot overlaps, the label goes right anyway.
    """
    from dataclasses import replace

    markers = [(p.cx - 7, p.cy - 7, p.cx + 7, p.cy + 7) for p in points]
    taken: list[tuple[float, float, float, float]] = []
    out = []
    for p in sorted(points, key=lambda q: (q.cy, q.cx)):
        for dx, anchor, dy in _SPOTS:
            box = _box(p, p.cx + dx, p.cy + 4 + dy, anchor)
            others = taken + [m for m, q in zip(markers, points, strict=True) if q is not p]
            if box[0] >= 0 and box[2] <= WIDTH and not any(_overlaps(box, o) for o in others):
                break
        taken.append(box)
        out.append(replace(p, tx=p.cx + dx, ty=p.cy + 4 + dy, anchor=anchor))
    return out


def literature_map(folder: Path, reviews: list[Review]) -> tuple[list[Point], list[Review]]:
    """The placed reviews, and the ones not placed yet (new, or changed since the map)."""
    import hashlib
    import json

    path = folder / MAP
    placed = json.loads(path.read_text(encoding="utf-8"))["reviews"] if path.exists() else []
    known = {row["name"]: row for row in placed}
    status = statuses(folder)
    points, missing = [], []
    for review in reviews:
        text = map_text(folder / f"{review.name}.md")
        if text is None:
            continue  # the index and the queue
        row = known.get(review.name)
        if row is None or row["sha256"] != hashlib.sha256(text.encode()).hexdigest():
            missing.append(review)
            continue
        cx = PAD + row["x"] * (WIDTH - 2 * PAD - LABEL * CHAR / 2)
        cy = PAD + (1 - row["y"]) * (HEIGHT - 2 * PAD)
        points.append(Point(review.name, review.title, status.get(review.name, "unknown"),
                            round(cx, 1), round(cy, 1)))  # fmt: skip
    return place_labels(points), missing


def render_reviews(render, reviews: list[Review], folder: Path = LITERATURE) -> None:
    """Write one page per review, and the index with the map of the reviews."""
    points, missing = literature_map(folder, reviews) if reviews else ([], [])
    for review in reviews:
        extra = {"points": points, "missing": missing, "map_size": (WIDTH, HEIGHT)}
        extra = extra if review.name == "README" else {}
        render(review.href, "literature.html", review=review, **extra)
