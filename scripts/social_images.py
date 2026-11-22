"""Draw each paper's and post's head figure for X and for link cards (#5534).

This script reads each folder in ``papers/`` and ``posts/`` whose head figure is an SVG. It
writes three files in the folder's ``media/``:

- ``social.png``: the final frame, drawn with reduced motion.
- ``social.mp4``: one loop, only when the figure animates.
- ``social.json``: the hash that :func:`openinterpretability.builder.social_media.stale`
  checks.

It needs Playwright's Chromium and ``ffmpeg``. The files are committed, so the deploy needs
neither::

    uv run --with playwright==1.65.2 playwright install chromium
    uv run --with playwright==1.65.2 python scripts/social_images.py

Each video frame is drawn at an exact time. The script pauses the page's CSS animations and
sets them to that time. So the video doesn't depend on how fast the browser runs.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openinterpretability.builder.social_media import (  # noqa: E402
    IMAGE,
    SIZE,
    VIDEO,
    figure_files,
    figure_names,
    head_figure,
    loop_seconds,
    stale,
    write_figure,
    write_manifest,
)
from openinterpretability.social.thread import load_thread  # noqa: E402

FPS = 10
PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html, body {{ margin: 0; width: {w}px; height: {h}px; background: #ffffff; }}
body {{ display: flex; align-items: center; justify-content: center; }}
svg {{ max-width: {iw}px; max-height: {ih}px; width: auto; height: auto; }}
</style></head><body>{svg}</body></html>"""


def page(svg: str) -> str:
    """The SVG inlined in a white page of :data:`SIZE`, with a margin.

    Args:
        svg: The SVG's text.

    Returns:
        The HTML page.
    """
    w, h = SIZE
    return PAGE.format(w=w, h=h, iw=w - 64, ih=h - 64, svg=svg)


def draw(browser, folder: Path, source: str, names: tuple[str, str] = (IMAGE, VIDEO)) -> list[str]:
    """Draw the image, and the video when the figure animates; return the files written.

    Args:
        browser: A Playwright browser.
        folder: The paper's or post's folder.
        source: The figure's path in the folder.
        names: The image's and the video's names in ``media/``.

    Returns:
        The names written in ``media/``.
    """
    image, video = names
    svg = (folder / source).read_text(encoding="utf-8")
    media, (w, h) = folder / "media", SIZE
    still = browser.new_page(viewport={"width": w, "height": h}, reduced_motion="reduce")
    still.set_content(page(svg))
    still.screenshot(path=str(media / image))
    still.close()
    files, seconds = [image], loop_seconds(svg)
    if not seconds:
        (media / video).unlink(missing_ok=True)
        return files
    moving = browser.new_page(viewport={"width": w, "height": h}, reduced_motion="no-preference")
    moving.set_content(page(svg))
    moving.evaluate("document.getAnimations().forEach(a => a.pause())")
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(int(seconds * FPS)):
            at = i * 1000 / FPS  # milliseconds into the loop
            moving.evaluate(f"document.getAnimations().forEach(a => a.currentTime = {at})")
            moving.screenshot(path=f"{tmp}/f{i:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", f"{tmp}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(media / video)], check=True)  # fmt: skip
    moving.close()
    return [*files, video]


def draw_thread(browser, folder: Path) -> None:
    """Draw each figure that the folder's thread names and that isn't drawn or is stale (#5833).

    Args:
        browser: A Playwright browser.
        folder: The paper's or post's folder.
    """
    current = figure_files(folder)
    for fig in sorted({m for tweet in load_thread(folder) or [] for m in tweet.media}):
        if current.get(fig) or not (folder / "media" / f"{fig}.svg").exists():
            continue
        files = draw(browser, folder, f"media/{fig}.svg", figure_names(fig))
        write_figure(folder, fig, files)
        print(f"{folder.relative_to(ROOT)}: {', '.join(files)}")


def main() -> int:
    """Redraw every head figure whose drawn files are missing or out of date.

    Returns:
        The exit code: 0.
    """
    from playwright.sync_api import sync_playwright

    folders = sorted(p.parent for p in ROOT.glob("papers/*/README.md"))
    folders += sorted(p.parent for p in ROOT.glob("posts/*/README.md"))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for folder in folders:
            head = head_figure(folder)
            if head is not None and head[0].endswith(".svg") and stale(folder) is not None:
                files = draw(browser, folder, head[0])
                write_manifest(folder, head[0], files)
                print(f"{folder.relative_to(ROOT)}: {', '.join(files)}")
            draw_thread(browser, folder)
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
