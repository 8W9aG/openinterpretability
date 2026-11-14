"""Registry consistency checks, run in CI on every pull request (`open-interpretability check`).

An experiment is posted as an archived record ``<id>.json`` plus its page ``<id>.md``
in ``wiki/experiments/<slug>/``. A neuron may have any number of experiments. The
checks enforce that:

1. Every archived record validates, has a verdict, and sits at the path that its
   neuron slug and experiment id give.
2. Every record has its page, and every page has its record. The page's front matter
   matches the record. A record that ``supersedes`` another names an earlier experiment
   on the same unit.
3. The retired ``audits/`` directory holds no records. Import them with
   ``open-interpretability wiki --import-dir audits``.
4. With ``base_ref``: existing experiment pages were not deleted, or edited outside
   their Discussion section, except to remove what a takedown covers. Archived records
   were not touched (write-once). No generated index (``README.md``) was committed.
   Records added since then name their agents in canonical spelling
   (``openinterp.core.agent_names``) and have one shape of model fields
   (``_check_model_fields``).
5. The errata file (``wiki/errata.jsonl``) and the takedown file (``wiki/takedowns.jsonl``)
   are valid and signed (:mod:`openinterp.errata_checks`, :mod:`openinterp.takedown_checks`).
6. The current code recomputes each record's verdict, gates and balanced accuracy from
   its stored results (:mod:`openinterp.verdict_regression`). For each record judged
   under the current protocol, they must match. So a refactor of the gates can't silently
   change what the registry claims.

With ``base_ref``, checks 1 and 2 read only the files that the branch adds or changes,
committed or not. So their cost doesn't grow with the registry. If the branch also
changes a tracked file outside the wiki, the code can read old records differently, so
they read every file. Without ``base_ref``, as on ``main``, they always read every file.

The indexes are rebuilt from the records when they are needed, and are gitignored.
Every experiment PR used to edit the same index rows, so parallel PRs conflicted.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from openinterp.builder.wiki import _read_front_matter, archive_path
from openinterp.core.agent_names import spelling_problem
from openinterp.core.paths import DEFAULT_WIKI_DIR, experiments_dir
from openinterp.core.schemas import AuditRecord
from openinterp.errata_checks import check_errata
from openinterp.signing import ALLOWED_SIGNERS
from openinterp.takedown_checks import check_takedowns, page_redaction_ok
from openinterp.verdict_regression import verdict_drift

DISCUSSION_HEADING = "## Discussion"
LEGACY_AUDITS_DIR = Path("audits")


@dataclass
class Problem:
    """One failed check: the file and what is wrong with it."""

    path: Path
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class CheckReport:
    """What ``open-interpretability check`` found: the problems and how many files it read."""

    problems: list[Problem] = field(default_factory=list)
    records: int = 0
    pages: int = 0
    draws: int = 0
    stops: int = 0
    ledger: int = 0
    changed_only: bool = False  # the record and page checks read only the branch's changes

    @property
    def extra_counts(self) -> dict[str, int]:
        """How many records of each optional kind the check read."""
        return {"draw": self.draws, "stop": self.stops, "ledger entry": self.ledger}

    @property
    def ok(self) -> bool:
        """True when no check failed."""
        return not self.problems

    def add(self, path: Path, message: str) -> None:
        """Record a problem with ``path``."""
        self.problems.append(Problem(Path(path), message))


def check_registry(
    wiki_dir: Path = DEFAULT_WIKI_DIR,
    *,
    base_ref: str | None = None,
    legacy_audits_dir: Path = LEGACY_AUDITS_DIR,
    allowed_signers: Path = ALLOWED_SIGNERS,
) -> CheckReport:
    """Run every registry check on ``wiki_dir`` (see the module docstring)."""
    report = CheckReport()
    wiki_dir = Path(wiki_dir)
    _check_legacy(Path(legacy_audits_dir), report)
    changes = _branch_changes(wiki_dir, base_ref) if base_ref else []
    scope = _scope(changes, wiki_dir) if base_ref else None
    report.changed_only = scope is not None
    _check_records(wiki_dir, report, scope)
    _check_pages(wiki_dir, report, scope)
    _check_draws(wiki_dir, report, scope)
    _check_stops(wiki_dir, report, scope)
    _check_ledger(wiki_dir, report, scope)
    check_errata(wiki_dir, report, base_ref, changes, allowed_signers)
    check_takedowns(wiki_dir, report, base_ref, changes, allowed_signers)
    if base_ref:
        _check_write_once(wiki_dir, base_ref, changes, report)
        root = experiments_dir(wiki_dir)
        changed = [
            path
            for status, path in changes
            if status != "D" and path.suffix == ".json" and path.is_relative_to(root)
        ]
        _check_agent_names(changed, report)
        _check_model_fields(changed, report)
    return report


def _branch_changes(wiki_dir: Path, base_ref: str) -> list[tuple[str, Path]]:
    """Return the status and path of each file changed since the branch left ``base_ref``.

    The list includes uncommitted changes and untracked wiki files, because agents run the
    check before they commit. Untracked files elsewhere are scratch files, so it omits them.
    """
    base = _git("merge-base", base_ref, "HEAD").strip()
    changes = []
    for line in _git("diff", "--name-status", "--no-renames", base).splitlines():
        status, _, name = line.partition("\t")
        changes.append((status, Path(name)))
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", str(wiki_dir))
    return changes + [("A", Path(name)) for name in untracked.splitlines()]


def _scope(changes: list[tuple[str, Path]], wiki_dir: Path) -> set[Path] | None:
    """Return the wiki files to check, or ``None`` for all of them.

    All of them means a change outside the wiki, which can change how any old record reads.
    A changed page brings in its record, because their front matter and values must agree.
    """
    if not all(path.is_relative_to(wiki_dir) for _, path in changes):
        return None
    scope = set()
    for status, path in changes:
        if status != "D":
            scope.add(path)
            if path.suffix == ".md":
                scope.add(path.with_suffix(".json"))
    return scope


def _chosen(paths: Iterable[Path], scope: set[Path] | None) -> Iterator[Path]:
    """Yield the ``paths`` in ``scope``, or every path if ``scope`` is ``None``."""
    return (path for path in paths if scope is None or path in scope)


def _check_legacy(audits_dir: Path, report: CheckReport) -> None:
    if audits_dir.is_dir():
        for path in sorted(audits_dir.rglob("*.json")):
            report.add(
                path,
                "audits/ is retired; experiments live in wiki/experiments/. Run "
                "`open-interpretability wiki --import-dir audits`, then delete audits/",
            )


def _check_records(wiki_dir: Path, report: CheckReport, scope: set[Path] | None = None) -> None:
    root = experiments_dir(wiki_dir)
    if not root.is_dir():
        return
    for path in _chosen(sorted(root.rglob("*.json")), scope):
        report.records += 1
        record = _load(path)
        if record is None:
            report.add(path, "invalid experiment record")
            continue
        if record.verdict is None:
            report.add(
                path, "experiment record has no verdict; run `open-interpretability verify` first"
            )
            continue
        expected = archive_path(wiki_dir, record)
        if path != expected:
            report.add(path, f"experiment record belongs at {expected}")
        _check_supersedes(path, record, report)
        drift = verdict_drift(record)
        if drift:
            report.add(path, drift)
        page = path.with_suffix(".md")
        if not page.exists():
            report.add(path, "experiment record has no page; run `open-interpretability wiki`")
            continue
        meta = _read_front_matter(page)
        values = {
            "slug": record.submission.identifier.slug,
            "status": record.verdict.status,
            "balanced_accuracy": record.verdict.confidence_score,
            "timestamp": record.verdict.timestamp.isoformat(),
            "supersedes": record.submission.supersedes,
        }
        for key, value in values.items():
            if meta.get(key) != value:
                report.add(page, f"front matter {key}={meta.get(key)!r} but record has {value!r}")


def _check_draws(wiki_dir: Path, report: CheckReport, scope: set[Path] | None = None) -> None:
    """Recorded unauditable draws (wiki/draws/) parse and sit where they belong."""
    from pydantic import ValidationError

    from openinterp.core.draws import UnauditableDraw, draw_files, draw_path

    for path in _chosen(draw_files(wiki_dir), scope):
        report.draws += 1
        try:
            draw = UnauditableDraw.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            report.add(path, "invalid draw record")
            continue
        if draw.provenance.selection != "random":
            report.add(path, "only random draws are recorded")
        if path != draw_path(wiki_dir, draw):
            report.add(path, f"draw record belongs at {draw_path(wiki_dir, draw)}")


def _check_ledger(wiki_dir: Path, report: CheckReport, scope: set[Path] | None = None) -> None:
    """Draw-ledger entries (wiki/ledger/) parse and sit where they belong."""
    from pydantic import ValidationError

    from openinterp.core.ledger import DrawEntry, ledger_files, ledger_path

    for path in _chosen(ledger_files(wiki_dir), scope):
        report.ledger += 1
        try:
            entry = DrawEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            report.add(path, "invalid ledger entry")
            continue
        if path != ledger_path(wiki_dir, entry):
            report.add(path, f"ledger entry belongs at {ledger_path(wiki_dir, entry)}")


def _check_stops(wiki_dir: Path, report: CheckReport, scope: set[Path] | None = None) -> None:
    """Recorded stops of chosen units (wiki/stops/) parse, sit where they belong and have a page."""
    from pydantic import ValidationError

    from openinterp.core.draws import ChosenStop, stop_files, stop_path

    for path in _chosen(stop_files(wiki_dir), scope):
        report.stops += 1
        try:
            stop = ChosenStop.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            report.add(path, "invalid stop record")
            continue
        if stop.provenance.selection == "random":
            report.add(path, "a random draw belongs in wiki/draws/")
        if path != stop_path(wiki_dir, stop):
            report.add(path, f"stop record belongs at {stop_path(wiki_dir, stop)}")
        if not path.with_suffix(".md").exists():
            report.add(path, "stop record has no page")


def _check_supersedes(path: Path, record: AuditRecord, report: CheckReport) -> None:
    revises = record.submission.supersedes
    if revises is None:
        return
    if revises >= path.stem:  # ids are UTC timestamps, so they sort chronologically
        report.add(path, f"supersedes {revises}, which is not an earlier experiment")
    elif not path.with_name(f"{revises}.json").exists():
        report.add(path, f"supersedes {revises}, but this unit has no such experiment")


def _check_pages(wiki_dir: Path, report: CheckReport, scope: set[Path] | None = None) -> None:
    root = experiments_dir(wiki_dir)
    if not root.is_dir():
        return
    for page in _chosen(sorted(root.rglob("*.md")), scope):
        if page.name == "README.md":
            continue
        report.pages += 1
        try:
            meta = _read_front_matter(page)
        except ValueError:
            meta = {}
        if not meta:
            report.add(page, "experiment page has no valid front matter")
        if not page.with_suffix(".json").exists():
            report.add(page, f"missing experiment record {page.with_suffix('.json').name}")
    for sig in _chosen(sorted(root.rglob("*.json.sig")), scope):
        if not sig.with_suffix("").exists():
            report.add(sig, f"signature for a missing record {sig.with_suffix('').name}")


def _agent_names(record: AuditRecord) -> dict[str, str | None]:
    """The agent model names in ``record``, by the name of their field."""
    scores = record.verdict.explanation_scores if record.verdict else None
    return {
        "submitting_agent_model": record.submission.submitting_agent_model,
        "reviewer_agent_model": record.verdict.reviewer_agent_model if record.verdict else None,
        "judge_model": scores.judge_model if scores else None,
    }


def _check_agent_names(paths: list[Path], report: CheckReport) -> None:
    """Require new records to spell their agents one way.

    Then the site shows one model as one agent. Older records keep their spellings,
    because they are signed. The build maps them. A follow-up such as ``reevaluate`` copies
    names from an older record of its unit, so a name that the unit already has passes.
    """
    new = set(paths)
    for path in paths:
        record = _load(path)
        if record is None or record.verdict is None:
            continue  # _check_records reports it
        older = [
            _agent_names(loaded)
            for other in path.parent.glob("*.json")
            if other not in new and (loaded := _load(other))
        ]
        for field_name, model in _agent_names(record).items():
            inherited = any(names[field_name] == model for names in older)
            if model and not inherited and (problem := spelling_problem(model)):
                report.add(path, f"{field_name}={model!r}: {problem}")


def _check_model_fields(paths: list[Path], report: CheckReport) -> None:
    """Require new records to have one shape of model fields.

    Each field must be in the file. Layers and width must have a value. Heads per layer
    must have a value on a head record and be null on other records.

    Older records keep their shape, because they are signed. The build fills in their
    widths. A re-review copies the submission of an older record, so a null value that
    the unit already has passes.
    """
    new = set(paths)
    for path in paths:
        record = _load(path)
        if record is None or record.verdict is None:
            continue  # _check_records reports it
        written = json.loads(path.read_text(encoding="utf-8"))["submission"]
        older = [
            loaded.submission
            for other in path.parent.glob("*.json")
            if other not in new and (loaded := _load(other))
        ]
        is_head = record.submission.identifier.kind == "attn_head"
        for name, required in (
            ("model_num_layers", True),
            ("model_mlp_width", True),
            ("model_n_heads", is_head),
        ):
            if name not in written:
                report.add(path, f"{name} is missing; record it with `open-interpretability audit`")
            elif not required and written[name] is not None:
                report.add(path, f"{name} must be null on a {record.submission.identifier.kind}")
            elif (
                required
                and written[name] is None
                and not any(getattr(sub, name) is None for sub in older)
            ):
                report.add(path, f"{name} is null; record it with `open-interpretability audit`")


def _load(path: Path) -> AuditRecord | None:
    try:
        return AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        return None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def _check_write_once(
    wiki_dir: Path, base_ref: str, changes: list[tuple[str, Path]], report: CheckReport
) -> None:
    for status, path in changes:
        if not path.is_relative_to(wiki_dir):
            continue
        if path.suffix == ".json" and status in {"D", "M"}:
            report.add(path, "archived experiment records are write-once")
            continue
        if path.name == "README.md":
            if status == "A":
                report.add(path, "generated wiki indexes are not committed; `git rm --cached` it")
            continue
        if path.suffix != ".md":
            continue
        if status == "D":
            report.add(path, "experiment pages are write-once and must not be deleted")
        elif status == "M":
            before = _git("show", f"{base_ref}:{path.as_posix()}")
            after = path.read_text(encoding="utf-8")
            head, new_head = before.split(DISCUSSION_HEADING)[0], after.split(DISCUSSION_HEADING)[0]
            if head != new_head and not page_redaction_ok(head, new_head, wiki_dir):
                report.add(
                    path,
                    "experiment pages are write-once; only the Discussion section may be edited",
                )


def changed_records(wiki_dir: Path, base_ref: str) -> list[Path]:
    """Experiment records added or modified since ``base_ref``."""
    diff = _git(
        "diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=AM",
        f"{base_ref}...HEAD",
        "--",
        str(experiments_dir(wiki_dir)),
    )
    return [Path(name) for name in diff.splitlines() if name.endswith(".json")]
