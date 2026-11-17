"""Lay the literature reviews out on a map by what they are about (issue #3294).

Each review in docs/literature/ is embedded from its title and its *Summary* section.
The model is all-MiniLM-L6-v2, which did best in the concept-map study (#3290). The
reviews are laid out by the first two principal components of the normalised embeddings.
That is deterministic, and enough for a few dozen papers. Close points are papers about
similar things. The axes themselves have no meaning.

The heavy library stays out of the project. Run it after adding or changing a review::

    uv run --with sentence-transformers python scripts/literature_map.py

It writes docs/literature/map.json, which the site's literature index draws.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.builder.literature_pages import (  # noqa: E402
    LITERATURE,
    MAP,
    map_text,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    """Embed each review, lay them out, and write the map."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    reviews = [p for p in sorted(LITERATURE.glob("*.md")) if map_text(p) is not None]
    texts = [map_text(p) for p in reviews]
    vectors = SentenceTransformer(MODEL, device="cpu").encode(texts, normalize_embeddings=True)
    centred = vectors - vectors.mean(0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    points = centred @ axes[:2].T
    points = (points - points.min(0)) / (points.max(0) - points.min(0))  # into [0, 1]
    rows = [
        {"name": p.stem, "x": round(float(x), 4), "y": round(float(y), 4),
         "sha256": hashlib.sha256(t.encode()).hexdigest()}
        for p, t, (x, y) in zip(reviews, texts, points, strict=True)
    ]  # fmt: skip
    out = LITERATURE / MAP
    out.write_text(json.dumps({"model": MODEL, "reviews": rows}, indent=1) + "\n")
    print(f"{len(rows)} reviews → {out}")


if __name__ == "__main__":
    main()
