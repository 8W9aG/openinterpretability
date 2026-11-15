"""Check 5 of ``open-interpretability check``: the errata file (:mod:`openinterp.core.errata`).

The file parses, has unique ids, and names records that exist. A key in the allowed
signers (from ``base_ref`` if given) signed it. The signature is checked at the time it
landed on ``main``, as a record's is. With ``base_ref`` the file is append-only. If the
branch changes the file, each erratum must cover a record. Its edits must apply to each
record it covers, and its fix commit must exist.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from openinterp.core.errata import Erratum, apply_errata, errata_path, parse_errata, record_ref
from openinterp.core.paths import experiments_dir
from openinterp.core.schemas import AuditRecord
from openinterp.signing import (
    SignatureError,
    allowed_signers_text,
    landing_times,
    signature_path,
    verify_record,
)

if TYPE_CHECKING:
    from openinterp.checks import CheckReport


def check_errata(
    wiki_dir: Path,
    report: CheckReport,
    base_ref: str | None,
    changes: list[tuple[str, Path]],
    allowed_signers: Path,
) -> None:
    """Add a problem to ``report`` for each way the errata file is not valid."""
    path = errata_path(wiki_dir)
    if any(p == path and status == "D" for status, p in changes):
        report.add(path, "the errata file is append-only and must not be deleted")
    if not path.exists():
        if signature_path(path).exists():
            report.add(signature_path(path), "signature of an errata file that does not exist")
        return
    text = path.read_text(encoding="utf-8")
    try:
        errata = parse_errata(text)
    except ValueError as exc:
        report.add(path, f"invalid erratum: {exc}")
        return
    _check_entries(errata, experiments_dir(wiki_dir), report, path)
    changed = changed_file(path, base_ref, changes)
    if base_ref and changed:
        shown = subprocess.run(
            ["git", "show", f"{base_ref}:{path.as_posix()}"], capture_output=True, text=True
        )
        if shown.returncode == 0 and not text.startswith(shown.stdout):
            report.add(path, "the errata file is append-only: add lines at the end, change none")
        _check_new_errata(errata, experiments_dir(wiki_dir), report, path)
    check_signature(path, report, allowed_signers_text(allowed_signers, base_ref), changed)


def changed_file(path: Path, base_ref: str | None, changes: list[tuple[str, Path]]) -> bool:
    """Whether the branch, or an uncommitted edit, changed ``path`` or its signature."""
    files = {path, signature_path(path)}
    if any(status != "D" and p in files for status, p in changes):
        return True
    if base_ref:
        return False
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *map(str, files)], capture_output=True, text=True
    )
    return status.returncode == 0 and bool(status.stdout.strip())


def _check_entries(errata: list[Erratum], root: Path, report: CheckReport, path: Path) -> None:
    """Ids are unique, and every listed record exists."""
    ids = [e.id for e in errata]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        report.add(path, f"erratum id {dup!r} is used more than once")
    for erratum in errata:
        for ref in erratum.records:
            if not (root / ref).is_file():
                report.add(path, f"erratum {erratum.id}: no record {ref} in {root}")


def _check_new_errata(errata: list[Erratum], root: Path, report: CheckReport, path: Path) -> None:
    """Each erratum covers a record, its edits apply to each, and its fix commit exists."""
    records = [r for r in map(_load, sorted(root.rglob("*.json"))) if r is not None]
    for erratum in errata:
        covered = [r for r in records if erratum.covers(r)]
        if not covered:
            report.add(path, f"erratum {erratum.id} covers no record")
        for record in covered:
            try:
                apply_errata(record, [erratum])
            except ValueError as exc:
                report.add(path, f"erratum {erratum.id} on {record_ref(record)}: {exc}")
                break  # one example is enough
        found = subprocess.run(
            ["git", "cat-file", "-e", f"{erratum.fixed_in}^{{commit}}"], capture_output=True
        )
        if found.returncode != 0:
            report.add(path, f"erratum {erratum.id}: no commit {erratum.fixed_in} in this clone")


def _load(path: Path) -> AuditRecord | None:
    """The record at ``path``, or ``None`` if it is invalid (check 1 reports that)."""
    try:
        return AuditRecord.model_validate_json(path.read_bytes())
    except ValueError:
        return None


def check_signature(
    path: Path, report: CheckReport, signers: str, changed: bool, what: str = "errata file"
) -> None:
    """A trusted key signed ``path``: at the time it landed, or now if it changed since."""
    if not signature_path(path).exists():
        report.add(
            path, f"the {what} is unsigned; a maintainer runs `open-interpretability sign {path}`"
        )
        return
    at = None if changed else landing_times().get(path.resolve())
    try:
        verify_record(path, signers, at)
    except SignatureError as exc:
        report.add(signature_path(path), f"the {what}'s signature is not trusted: {exc}")
