"""Accessibility check of the built site with axe-core (issue #3280).

It opens five pages of a built site in Chromium: home, a model, a layer, a neuron and
an experiment. It opens each one in the light and the dark colour scheme. It runs
axe-core's WCAG 2.0 A and AA and WCAG 2.1 AA rules on each one. It fails on any
violation of impact "serious" or "critical", and prints each one with the element that
fails.

axe-core is one JavaScript file. The check downloads it at ``AXE_VERSION``, checks it
against ``AXE_SHA256`` and keeps it in ``~/.cache/openinterpretability``. Playwright is not a
project dependency, so run the check with it added:

    make sample-run
    uv run --with playwright==1.65.2 playwright install chromium
    uv run --with playwright==1.65.2 python scripts/a11y_check.py --site dist
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from openinterpretability.core.legacy_names import cache_base

AXE_VERSION = "4.10.2"
AXE_SHA256 = "b511cd9dec01c76f4b2ad1723b66b6db37d4c2eb4ed199076e1829d9ee7b75e3"
AXE_URL = f"https://cdn.jsdelivr.net/npm/axe-core@{AXE_VERSION}/axe.min.js"
AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21aa"]
FAIL_IMPACTS = {"serious", "critical"}
SCHEMES = ("light", "dark")


def sample_pages(site: Path) -> list[str]:
    """The pages to check, relative to ``site``: home, then the first of each kind.

    "First" is in sorted path order, so the same site always gives the same pages. A
    kind that the site has no page for is left out.
    """
    kinds = [
        "models/*/index.html",
        "models/*/L*/index.html",
        "models/*/L*/N*.html",
        "models/*/L*/N*-*/*.html",
        "research/index.html",
        "research/*/index.html",  # tables that scroll, and the index's figure links (#5649)
    ]
    pages = ["index.html"] if (site / "index.html").is_file() else []
    for pattern in kinds:
        found = sorted(p for p in site.glob(pattern) if p.is_file())
        if found:
            pages.append(found[0].relative_to(site).as_posix())
    return pages


def failures(violations: list[dict]) -> list[str]:
    """One line for each failing element in axe's ``violations`` list.

    A violation fails when its impact is in ``FAIL_IMPACTS``. Each line gives the rule,
    the impact and the element's CSS selector.
    """
    out = []
    for v in violations:
        if v.get("impact") not in FAIL_IMPACTS:
            continue
        for node in v.get("nodes", []):
            target = " ".join(map(str, node.get("target", [])))
            out.append(f"{v['id']} ({v['impact']}): {target}")
    return out


def axe_source(cache: Path) -> str:
    """axe-core's script text, from the cache or downloaded, checked against its hash."""
    path = cache / f"axe-{AXE_VERSION}.min.js"
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(AXE_URL, timeout=60) as response:  # noqa: S310
            path.write_bytes(response.read())
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != AXE_SHA256:
        path.unlink()
        raise SystemExit(f"axe-core {AXE_VERSION} has SHA-256 {digest}, not {AXE_SHA256}")
    return data.decode("utf-8")


def main() -> int:
    """Check the sample pages in each scheme; print each failure and return 1 if any."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", type=Path, default=Path("dist"), help="the built site")
    parser.add_argument("--channel", help="a browser channel, e.g. chrome (default: Chromium)")
    parser.add_argument("--cache", type=Path, default=cache_base())
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    site = args.site.resolve()
    pages = sample_pages(site)
    if len(pages) < 5:
        print(f"a11y: expected 5 sample pages in {site}, found {pages}")
        return 1
    axe = axe_source(args.cache)
    found = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel) if args.channel else p.chromium.launch()
        for scheme in SCHEMES:
            page = browser.new_context(color_scheme=scheme).new_page()
            for rel in pages:
                page.goto((site / rel).as_uri(), wait_until="load")
                page.add_script_tag(content=axe)
                violations = page.evaluate(
                    "tags => axe.run(document, {runOnly: {type: 'tag', values: tags}})"
                    ".then(r => r.violations)",
                    AXE_TAGS,
                )
                lines = failures(violations)
                found += len(lines)
                print(f"{scheme} {rel}: {len(lines)} serious or critical")
                for line in lines:
                    print(f"  {line}")
        browser.close()
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
