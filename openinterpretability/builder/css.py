"""Build the site's stylesheet once, with Tailwind's standalone CLI (issue #3282).

The pages used Tailwind's Play CDN: 451 KB of JavaScript on every page. It generated the
CSS in the browser from the whole page. This module generates the same CSS at build time
instead. It writes one minified file, :data:`STYLESHEET`, under the site, and
``base.html`` links it.

The built pages are about 1 GB, too much for the CLI to scan. So :func:`candidates`
reads each page's class attributes and scripts, where class names are. The CLI scans only
the unique words found there.

The CLI is a single binary at :data:`VERSION`, the same version that the CDN served. No
Node is needed. It is downloaded once to ``~/.cache/openinterpretability/tailwind`` and checked
against the release's SHA-256.

    python -m openinterpretability.builder.css --site dist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from openinterpretability.builder import palette
from openinterpretability.core.legacy_names import cache_base as _cache_base

VERSION = "3.4.16"
STYLESHEET = "assets/site.css"
COLUMN_REM = 60
"""The width of the site's one content column, in rem (#5726). Prose, captions, figures,
tables and maps all fit in it, so a section has one right edge. Tailwind's ``max-w-column``
is this width, and ``max-w-page`` adds the page's 1rem gutter on each side (``px-4``)."""
"""The stylesheet's path under the site, which ``base.html`` links."""

# From the release's sha256sums.txt.
BINARIES = {
    ("Linux", "x86_64"): (
        "tailwindcss-linux-x64",
        "33f254b54c8754f16efbe2be1de38ca25192630dc36f164595a770d4bbf4d893",
    ),
    ("Linux", "aarch64"): (
        "tailwindcss-linux-arm64",
        "1e6746bba6f3d34d7550889a1a009ab90ee3794a5ebce60ed10688ad10680a87",
    ),
    ("Darwin", "arm64"): (
        "tailwindcss-macos-arm64",
        "01751c6019c1b4bf787d2e0b1f221bef1bcc010cef55313fc0691f3b6a3b676f",
    ),
    ("Darwin", "x86_64"): (
        "tailwindcss-macos-x64",
        "220962a6f371fc31605f89569ad647309cbd83471cd8c29b83f235a501c39dce",
    ),
}
FONTS = {
    "sans": ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
    "mono": ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
}


# Any case, and anything between "</script" and ">": a missed script would miss its classes.
CLASSES = re.compile(r'class="([^"]*)"|<script\b[^>]*>(.*?)</script\b[^>]*>', re.S | re.I)
WORD = re.compile(r"[^\s\"'`<>=]+")


def candidates(site: Path) -> set[str]:
    """The words in the class attributes and scripts of every page and script under ``site``.

    These hold every class name a page uses, including those that a script adds.
    """
    found: set[str] = set()
    for path in site.rglob("*.html"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for attr, script in CLASSES.findall(text):
            found.update(WORD.findall(attr or script))
    for path in site.rglob("*.js"):
        found.update(WORD.findall(path.read_text(encoding="utf-8", errors="replace")))
    return found


def config(content: list[str]) -> str:
    """The Tailwind config: the site's colours and fonts, and the files to scan."""
    widths = {"column": f"{COLUMN_REM}rem", "page": f"{COLUMN_REM + 2}rem"}
    extend = {"colors": palette.tailwind_colors(), "fontFamily": FONTS, "maxWidth": widths}
    theme = {"extend": extend}
    return "module.exports = " + json.dumps({"content": content, "theme": theme}, indent=1) + ";\n"


def binary(cache: Path, system: tuple[str, str] | None = None) -> Path:
    """The Tailwind CLI for this machine: from ``cache``, else downloaded and checked."""
    system = system or (platform.system(), platform.machine())
    if system not in BINARIES:
        raise SystemExit(f"no Tailwind CLI for {system[0]} {system[1]}")
    name, sha = BINARIES[system]
    path = cache / f"{name}-{VERSION}"
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/tailwindlabs/tailwindcss/releases/download/v{VERSION}/{name}"
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        path.unlink()
        raise SystemExit(f"{name} {VERSION} has SHA-256 {digest}, not {sha}")
    path.chmod(0o755)
    return path


def build(site: Path, cache: Path) -> Path:
    """Write :data:`STYLESHEET` under ``site`` from the classes its pages use."""
    site = site.resolve()
    out = site / STYLESHEET
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        conf, source = Path(tmp) / "tailwind.config.js", Path(tmp) / "input.css"
        words = Path(tmp) / "classes.txt"
        words.write_text(" ".join(sorted(candidates(site))), encoding="utf-8")
        conf.write_text(config([str(words)]), encoding="utf-8")
        source.write_text("@tailwind base;\n@tailwind components;\n@tailwind utilities;\n")
        args = ["-c", str(conf), "-i", str(source), "-o", str(out), "--minify"]
        subprocess.run([str(binary(cache)), *args], check=True, capture_output=True)
    return out


def main() -> None:
    """Build the stylesheet of the site at ``--site``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", type=Path, default=Path("dist"))
    parser.add_argument("--cache", type=Path, default=_cache_base() / "tailwind")
    args = parser.parse_args()
    out = build(args.site, args.cache)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
