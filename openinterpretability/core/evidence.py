"""Bulky record evidence in a content-addressed store, outside the repository (issue #3254).

Two fields make up about 59% of a record: ``submission.top_examples`` and
``verdict.explanation_scores``. :func:`split` takes them out of a record's JSON. Each one
becomes a reference, ``{"evidence_sha256": "<hash>"}``, to a blob that holds the field's
exact JSON. :func:`resolve` puts each blob back, and refuses a blob whose hash is wrong.

A record's signature covers its file's bytes, so it covers the hashes. A hash covers its
blob. So a signed slim record still vouches for all of its evidence. Records are
write-once, so the records that are already whole stay whole.

Blobs live in a local store, ``~/.cache/openinterpretability/evidence/`` by default
(``$OPENINTERPRETABILITY_EVIDENCE_DIR``). A blob that isn't there is fetched, read-only, from the
public dataset :data:`EVIDENCE_REPO`. :func:`openinterpretability.builder.wiki.archive_record`
writes slim records unless ``OPENINTERPRETABILITY_OFFLOAD_EVIDENCE=0``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIELDS = (("submission", "top_examples"), ("verdict", "explanation_scores"))
"""The record fields that :func:`split` moves to the store: (parent key, field)."""
REF_KEY = "evidence_sha256"
# LINT.IfChange
EVIDENCE_REPO = "sackfield/open-interpretability-evidence"
"""The public Hugging Face dataset that holds the blobs, by path ``<sha[:2]>/<sha>.json``."""
# LINT.ThenChange(.github/workflows/evidence-cleanup.yml)
DIR_ENV = "OPENINTERPRETABILITY_EVIDENCE_DIR"
OFFLOAD_ENV = (
    "OPENINTERPRETABILITY_OFFLOAD_EVIDENCE"  # "0" makes archive_record write whole records
)


class EvidenceMissing(ValueError):
    """A record refers to a blob that neither the local store nor the dataset holds."""


def offload_enabled() -> bool:
    """Whether new archived records are written slim: yes, unless the variable is ``0``.

    Phase 3 of issue #3254 turned this on. After the merge, the deploy uploads the blobs from
    ``evidence/<branch>``.
    """
    return os.environ.get(OFFLOAD_ENV, "1") != "0"


def store_dir() -> Path:
    """The local blob store: ``$OPENINTERPRETABILITY_EVIDENCE_DIR``, else ``<cache>/evidence``."""
    from openinterpretability.core.legacy_names import cache_base

    return Path(os.environ.get(DIR_ENV) or cache_base() / "evidence")


def blob_name(sha: str) -> str:
    """A blob's path in the store and in the dataset."""
    return f"{sha[:2]}/{sha}.json"


def canonical(value: Any) -> bytes:
    """The bytes that a blob holds for ``value``: sorted keys, no spaces, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(data: bytes) -> str:
    """The hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def is_ref(value: Any) -> bool:
    """Whether ``value`` is a reference that :func:`split` wrote."""
    return isinstance(value, dict) and set(value) == {REF_KEY}


def put(data: bytes, root: Path | None = None) -> str:
    """Store ``data`` in the local store, and return its hash."""
    sha = sha256(data)
    path = (root or store_dir()) / blob_name(sha)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic: a reader sees no blob or a whole one
    return sha


# Gets a blob by its hash, or gives None when it cannot.
Fetch = Callable[[str], bytes | None]


def _fetch(sha: str) -> bytes | None:
    """The blob from the public dataset, or ``None`` if it can't be had."""
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(EVIDENCE_REPO, blob_name(sha), repo_type="dataset")
    except Exception:  # noqa: BLE001 - offline, no dataset yet, or no such blob
        return None
    return Path(local).read_bytes()


def get(sha: str, root: Path | None = None, fetch: Fetch | None = None) -> bytes:
    """The blob with hash ``sha``: from the local store, else fetched and stored.

    ``fetch`` defaults to the read-only fetch from :data:`EVIDENCE_REPO`. Raises
    :class:`EvidenceMissing` when the blob can't be had or its hash is wrong.
    """
    path = (root or store_dir()) / blob_name(sha)
    data = path.read_bytes() if path.exists() else (fetch or _fetch)(sha)
    if data is None:
        raise EvidenceMissing(
            f"evidence blob {sha} is not in {root or store_dir()} or {EVIDENCE_REPO}"
        )
    if sha256(data) != sha:
        raise EvidenceMissing(f"evidence blob {sha} has the wrong hash; the store holds a bad copy")
    if not path.exists():
        put(data, root)
    return data


def split(data: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """A copy of a record's JSON with each field in :data:`FIELDS` moved to the store.

    A field that is missing, ``None`` or already a reference stays as it is.
    """
    out: dict[str, Any] = json.loads(json.dumps(data))
    for parent, field in FIELDS:
        node = out.get(parent)
        if not isinstance(node, dict) or node.get(field) is None or is_ref(node[field]):
            continue
        node[field] = {REF_KEY: put(canonical(node[field]), root)}
    return out


def resolve(data: Any, root: Path | None = None, fetch: Fetch | None = None) -> Any:
    """``data`` with each reference in :data:`FIELDS` replaced by its blob's value.

    Data with no reference comes back as it is, so whole records cost nothing.
    """
    if not isinstance(data, dict):
        return data
    refs = [
        (parent, field)
        for parent, field in FIELDS
        if isinstance(data.get(parent), dict) and is_ref(data[parent].get(field))
    ]
    if not refs:
        return data
    out = dict(data)
    for parent, field in refs:
        sha = data[parent][field][REF_KEY]
        out[parent] = {**out[parent], field: json.loads(get(sha, root, fetch))}
    return out


def resolved_text(json_data: str | bytes | bytearray) -> str | bytes | bytearray:
    """A record's JSON text with its references resolved; whole records come back as given."""
    text = json_data.decode() if isinstance(json_data, bytes | bytearray) else json_data
    if REF_KEY not in text:
        return json_data
    return json.dumps(resolve(json.loads(text)))


def resolved_dict_text(obj: Any) -> str | None:
    """A record dict's JSON text with its references resolved, or ``None`` if it has none."""
    resolved = resolve(obj)
    return None if resolved is obj else json.dumps(resolved)
