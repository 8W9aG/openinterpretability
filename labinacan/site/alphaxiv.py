"""A paper's alphaXiv page, from its ``CITATION.cff`` (#5259).

``scripts/papers.py`` records the URL and checks it. The site links it only when it has
the alphaXiv form, whatever the file says.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml  # a dependency of transformers and huggingface_hub

# A published paper's page on the alphaXiv AI Track (#5259): a ``url`` identifier of its
# CITATION.cff, with this description. The field is outside the paper's frozen hash.
ALPHAXIV = "alphaXiv AI Track"
ALPHAXIV_URL = re.compile(r"https://www\.alphaxiv\.org/abs/[\w.\-]+(?:v\d+)?")

__all__ = ["ALPHAXIV", "ALPHAXIV_URL", "alphaxiv_url", "cited_alphaxiv"]


def cited_alphaxiv(folder: Path) -> str | None:
    r"""The ``url`` identifier that the paper's ``CITATION.cff`` gives for alphaXiv, unchecked.

    ``scripts/papers.py check`` reads it to find a bad value. The site uses
    :func:`alphaxiv_url`, which links only a value of the alphaXiv form.

    Args:
        folder: The paper's folder.

    Returns:
        The value, or None when the paper has no citation file or no such identifier.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> cited_alphaxiv(folder) is None
        True
        >>> _ = (folder / "CITATION.cff").write_text(
        ...     "identifiers:\n- type: url\n  value: javascript:alert(1)\n"
        ...     "  description: alphaXiv AI Track\n")
        >>> cited_alphaxiv(folder)
        'javascript:alert(1)'
    """
    path = folder / "CITATION.cff"
    if not path.is_file():
        return None
    found = yaml.safe_load(path.read_text(encoding="utf-8")).get("identifiers") or []
    return next((str(i["value"]) for i in found
                 if i.get("type") == "url" and i.get("description") == ALPHAXIV), None)  # fmt: skip


def alphaxiv_url(folder: Path) -> str | None:
    r"""The paper's alphaXiv URL, from its ``CITATION.cff``, when it has the alphaXiv form.

    Args:
        folder: The paper's folder.

    Returns:
        The URL. None when the paper gives none, or gives a value that isn't an alphaXiv URL
        (:data:`ALPHAXIV_URL`), so the page never links it.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "CITATION.cff").write_text(
        ...     "identifiers:\n- type: url\n  value: https://www.alphaxiv.org/abs/2611.00001\n"
        ...     "  description: alphaXiv AI Track\n")
        >>> alphaxiv_url(folder)
        'https://www.alphaxiv.org/abs/2611.00001'
        >>> _ = (folder / "CITATION.cff").write_text(
        ...     "identifiers:\n- type: url\n  value: javascript:alert(1)\n"
        ...     "  description: alphaXiv AI Track\n")
        >>> alphaxiv_url(folder) is None
        True
    """
    url = cited_alphaxiv(folder)
    return url if url and ALPHAXIV_URL.fullmatch(url) else None
