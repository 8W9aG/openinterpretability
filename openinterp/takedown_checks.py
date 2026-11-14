"""Part of check 5 of ``openinterp check``: the takedown file (:mod:`openinterp.core.takedowns`).

The file parses and has unique ids. A key in the allowed signers (from ``base_ref`` if
given) signed it, so a branch can't add a takedown of its own. With ``base_ref`` the file
is append-only, and must not be deleted.

:func:`page_redaction_ok` is the one exception to write-once experiment pages. A page may
change outside its Discussion section when the change only removes what the takedowns
cover.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from openinterp.core.takedowns import all_hashes, load_takedowns, parse_takedowns, redact
from openinterp.core.takedowns import takedowns_path as path_of
from openinterp.errata_checks import changed_file, check_signature
from openinterp.signing import allowed_signers_text, signature_path

if TYPE_CHECKING:
    from openinterp.checks import CheckReport

WHAT = "takedown file"


def check_takedowns(
    wiki_dir: Path,
    report: CheckReport,
    base_ref: str | None,
    changes: list[tuple[str, Path]],
    allowed_signers: Path,
) -> None:
    """Add a problem to ``report`` for each way the takedown file is not valid."""
    path = path_of(wiki_dir)
    if any(p == path and status == "D" for status, p in changes):
        report.add(path, f"the {WHAT} is append-only and must not be deleted")
    if not path.exists():
        if signature_path(path).exists():
            report.add(signature_path(path), f"signature of a {WHAT} that does not exist")
        return
    text = path.read_text(encoding="utf-8")
    try:
        takedowns = parse_takedowns(text)
    except ValueError as exc:
        report.add(path, f"invalid takedown: {exc}")
        return
    ids = [t.id for t in takedowns]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        report.add(path, f"takedown id {dup!r} is used more than once")
    changed = changed_file(path, base_ref, changes)
    if base_ref and changed:
        shown = subprocess.run(
            ["git", "show", f"{base_ref}:{path.as_posix()}"], capture_output=True, text=True
        )
        if shown.returncode == 0 and not text.startswith(shown.stdout):
            report.add(path, f"the {WHAT} is append-only: add lines at the end, change none")
    signers = allowed_signers_text(allowed_signers, base_ref)
    check_signature(path, report, signers, changed, WHAT)


def page_redaction_ok(before: str, after: str, wiki_dir: Path) -> bool:
    """Whether ``after`` is ``before`` with exactly the text that the takedowns cover removed.

    Both are a page's text before its Discussion section. The check that the takedown
    file is signed is :func:`check_takedowns`, which runs in the same ``openinterp check``.
    """
    from openinterp.builder.pii import REMOVED_TEXT

    hashes = all_hashes(load_takedowns(wiki_dir))
    return before != after and redact(before, hashes, REMOVED_TEXT) == after
