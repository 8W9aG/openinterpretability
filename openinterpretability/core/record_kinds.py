"""Every kind of record under ``wiki/``, and how CI can check it (issue #4792).

This is the one list that decides which records CI reproduces and which records a
contributor's PR may add and auto-merge. ``ci.yml`` (its *Record changes* job),
``auto-merge.yml``, ``checks.changed_records`` and ``experiment_loop.merge_blockers`` all
read it. ``docs/PROTOCOL.md`` §5 gives the same table in words.

It imports only the standard library. Run as a script, it prints the reproducible prefixes,
one a line. So a workflow can read it without installing the package. And ``auto-merge.yml``
can read it from the base branch, without running a pull request's code.
"""

from __future__ import annotations

from dataclasses import dataclass

REMEASURED = "re-measured by reproduce"  # CI runs the model again and compares
DERIVED = "re-derived from its own stored measures"  # a consistency check, not a re-run
SIGNED = "signed by a key in .github/allowed_signers"
UNCHECKED = "not checked"


@dataclass(frozen=True)
class RecordKind:
    """One kind of wiki record: where it lives, and how CI checks it."""

    name: str
    prefix: str  # its path under the repository root, with a trailing slash
    check: str  # one of the constants above
    note: str = ""


RECORD_KINDS = (
    RecordKind("experiment", "wiki/experiments/", REMEASURED),
    RecordKind("circuit", "wiki/circuits/", DERIVED, "the stored measures give the status"),
    RecordKind("answer", "wiki/answers/", DERIVED, "the stored measures give the status"),
    RecordKind("explanation", "wiki/explanations/", REMEASURED, "records since #4793"),
    RecordKind("draw", "wiki/draws/", UNCHECKED),
    RecordKind("stop", "wiki/stops/", UNCHECKED),
    RecordKind("ledger entry", "wiki/ledger/", UNCHECKED),
    RecordKind("erratum", "wiki/errata.jsonl", SIGNED),
    RecordKind("patch", "wiki/patches/", UNCHECKED, "a fix to a base model (#5229)"),
)

# The prefixes whose records CI re-runs. A contributor's PR auto-merges only when every file
# it adds is under one of them, so that CI has reproduced each record.
REPRODUCIBLE = tuple(k.prefix for k in RECORD_KINDS if k.check == REMEASURED)


def reproducible(path: str) -> bool:
    """True when ``path`` is a record that CI re-runs."""
    return path.startswith(REPRODUCIBLE)


if __name__ == "__main__":
    print("\n".join(REPRODUCIBLE))
