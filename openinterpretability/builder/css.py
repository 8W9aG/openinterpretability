"""Build this site's stylesheet with labinacan's Tailwind build, in this repo's colours (#6628).

python -m openinterpretability.builder.css --site dist
"""

from __future__ import annotations

from labinacan.site import css
from openinterpretability.builder import palette
from openinterpretability.core.legacy_names import cache_base


def main() -> None:
    """Build the stylesheet of the site at ``--site``, with :func:`palette.tailwind_colors`."""
    css.main(palette.tailwind_colors(), cache_base() / "tailwind")


if __name__ == "__main__":
    main()
