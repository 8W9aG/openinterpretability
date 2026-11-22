"""The drawn head figure of a paper or a post, for X and for link cards (#5534).

X takes no SVG, and every figure here is SVG. So ``scripts/social_images.py`` draws each
head figure, the first image in the ``README.md``, into the folder's ``media/``:

- ``social.png``: the figure at :data:`SIZE`, with reduced motion, so an animated figure
  shows its final frame, as the site does.
- ``social.mp4``: one full loop of the figure, only when it animates.
- ``social.json``: the SVG's path and SHA-256, and the files drawn. A test compares the hash
  with the SVG, so a figure changed without a redraw fails. Git keeps no file times, so a
  hash is the check.

The files are committed, so the site build and the deploy need no browser.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SIZE = (1280, 720)  # 16:9, which X shows whole; even sides, as H.264 needs
IMAGE, VIDEO, MANIFEST = "social.png", "social.mp4", "social.json"
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_DURATION = re.compile(r"animation:\s*[\w-]+\s+([\d.]+)s")


def head_figure(folder: Path) -> tuple[str, str] | None:
    """The first image of ``folder/README.md``: its path in the folder and its alt text.

    Args:
        folder: A paper's or a post's folder.

    Returns:
        ``(path, alt)``, or None when the README has no image.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "README.md").write_text("# T\\n\\nText.\\n\\n![A curve](media/c.svg)\\n")
        >>> head_figure(folder)
        ('media/c.svg', 'A curve')
    """
    readme = folder / "README.md"
    found = _IMAGE.search(readme.read_text(encoding="utf-8")) if readme.exists() else None
    return (found.group(2), found.group(1)) if found else None


def loop_seconds(svg: str) -> float:
    """How long one loop of an animated SVG lasts: its longest CSS animation, or 0.

    Args:
        svg: The SVG's text.

    Returns:
        The duration in seconds, or 0 when the SVG doesn't animate.

    Examples:
        >>> loop_seconds("<style>.a{animation:a 8s step-end infinite}.b{animation:b 9s}</style>")
        9.0
        >>> loop_seconds("<style>@media (prefers-reduced-motion){*{animation:none}}</style>")
        0.0
    """
    return max((float(s) for s in _DURATION.findall(svg)), default=0.0)


def digest(path: Path) -> str:
    """The SHA-256 of a file, in hex.

    Args:
        path: The file.

    Returns:
        Its SHA-256.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "a.svg"
        >>> _ = path.write_text("<svg/>")
        >>> digest(path)[:12]
        'd4dc56669143'
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stale(folder: Path) -> str | None:
    """Why ``folder``'s drawn files are missing or out of date, or None when they are current.

    A head figure that isn't SVG needs no drawn file, since X takes it as it is.

    Args:
        folder: A paper's or a post's folder.

    Returns:
        A short reason, or None.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "README.md").write_text("# T\\n\\nNo figure.\\n")
        >>> stale(folder) is None
        True
        >>> (folder / "media").mkdir()
        >>> _ = (folder / "media" / "c.svg").write_text("<svg/>")
        >>> _ = (folder / "README.md").write_text("# T\\n\\n![C](media/c.svg)\\n")
        >>> stale(folder)
        'no media/social.json: run scripts/social_images.py'
    """
    head = head_figure(folder)
    if head is None or not head[0].endswith(".svg"):
        return None
    manifest = folder / "media" / MANIFEST
    if not manifest.exists():
        return f"no media/{MANIFEST}: run scripts/social_images.py"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("source") != head[0] or data.get("sha256") != digest(folder / head[0]):
        return f"{head[0]} changed since it was drawn: run scripts/social_images.py"
    missing = [name for name in data.get("files", []) if not (folder / "media" / name).exists()]
    return f"missing media/{missing[0]}" if missing else None


def drawn(folder: Path) -> tuple[str | None, str | None]:
    """The drawn image and video of ``folder``, as paths in the folder, or None for each.

    Args:
        folder: A paper's or a post's folder.

    Returns:
        ``(image, video)``: ``media/social.png`` and ``media/social.mp4`` where they exist.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> (folder / "media").mkdir()
        >>> _ = (folder / "media" / "social.png").write_bytes(b"png")
        >>> drawn(folder)
        ('media/social.png', None)
    """
    media = folder / "media"
    image = f"media/{IMAGE}" if (media / IMAGE).exists() else None
    video = f"media/{VIDEO}" if (media / VIDEO).exists() else None
    return image, video


def write_manifest(folder: Path, source: str, files: list[str]) -> None:
    """Record what was drawn from ``source``, so :func:`stale` can check it.

    Args:
        folder: A paper's or a post's folder.
        source: The head figure's path in the folder.
        files: The names of the drawn files in ``media/``.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> (folder / "media").mkdir()
        >>> _ = (folder / "media" / "c.svg").write_text("<svg/>")
        >>> write_manifest(folder, "media/c.svg", ["social.png"])
        >>> json.loads((folder / "media" / "social.json").read_text())["files"]
        ['social.png']
    """
    data = {**_manifest(folder), "source": source, "sha256": digest(folder / source),
            "files": sorted(files)}  # fmt: skip
    (folder / "media" / MANIFEST).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def figure_names(fig_id: str) -> tuple[str, str]:
    """The drawn image and video of one thread figure (#5833), in ``media/``.

    Args:
        fig_id: The figure's id: its SVG's stem.

    Returns:
        ``(image, video)`` names.

    Examples:
        >>> figure_names("trace")
        ('social-trace.png', 'social-trace.mp4')
    """
    return f"social-{fig_id}.png", f"social-{fig_id}.mp4"


def _manifest(folder: Path) -> dict:
    """``media/social.json`` of ``folder``, or an empty dict when there is none.

    Args:
        folder: A paper's or a post's folder.

    Returns:
        The manifest's data.
    """
    path = folder / "media" / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def figure_files(folder: Path) -> dict[str, list[str]]:
    """The files drawn for each thread figure that are current and present, by figure id.

    A figure whose SVG changed since it was drawn has none, so the thread's check fails.

    Args:
        folder: A paper's or a post's folder.

    Returns:
        The names in ``media/`` of each figure's drawn files.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> (folder / "media").mkdir()
        >>> _ = (folder / "media" / "c.svg").write_text("<svg/>")
        >>> _ = (folder / "media" / "social-c.png").write_bytes(b"png")
        >>> write_figure(folder, "c", ["social-c.png"])
        >>> figure_files(folder)
        {'c': ['social-c.png']}
        >>> _ = (folder / "media" / "c.svg").write_text("<svg>changed</svg>")
        >>> figure_files(folder)
        {'c': []}
    """
    out = {}
    for fig, entry in _manifest(folder).get("figures", {}).items():
        source = folder / "media" / f"{fig}.svg"
        current = source.exists() and entry.get("sha256") == digest(source)
        out[fig] = [
            f for f in entry.get("files", []) if current and (folder / "media" / f).exists()
        ]
    return out


def write_figure(folder: Path, fig_id: str, files: list[str]) -> None:
    """Record what was drawn from a thread figure, beside the head figure's record.

    Args:
        folder: A paper's or a post's folder.
        fig_id: The figure's id.
        files: The names of its drawn files in ``media/``.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> (folder / "media").mkdir()
        >>> _ = (folder / "media" / "c.svg").write_text("<svg/>")
        >>> write_figure(folder, "c", ["social-c.png"])
        >>> json.loads((folder / "media" / MANIFEST).read_text())["figures"]["c"]["files"]
        ['social-c.png']
    """
    data = _manifest(folder)
    source = folder / "media" / f"{fig_id}.svg"
    data.setdefault("figures", {})[fig_id] = {"sha256": digest(source), "files": sorted(files)}
    data["figures"] = dict(sorted(data["figures"].items()))
    (folder / "media" / MANIFEST).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def x_url(folder: Path) -> str | None:
    """The URL of the paper's or post's X post, from ``media/social.json`` (#5840).

    Args:
        folder: A paper's or a post's folder.

    Returns:
        The URL, or None before the paper or post is posted.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> x_url(folder) is None
        True
        >>> (folder / "media").mkdir()
        >>> _ = (folder / "media" / MANIFEST).write_text('{"x_url": "https://x.com/i/status/9"}')
        >>> x_url(folder)
        'https://x.com/i/status/9'
    """
    found = _manifest(folder).get("x_url")
    return found if isinstance(found, str) and found.startswith("https://x.com/") else None
