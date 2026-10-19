"""Registry consistency checks, run in CI on every pull request (`openinterp check`).

An experiment is posted as an audit record in ``audits/`` plus a wiki page in
``wiki/experiments/``. The checks enforce that:

1. every audit record validates, has a verdict, and is the only record for its neuron;
2. every audit record has the wiki page for its current verdict, and the page's front
   matter matches the record;
3. every wiki experiment page belongs to a neuron that has an audit record, and has
   an archived record (``<id>.json``) that validates and matches it; the archived
   copy of each audit record's current verdict is identical to the record;
4. the generated wiki indexes are up to date;
5. with ``base_ref``, existing experiment pages were not deleted or edited outside
   their Discussion section, and archived records were not touched (write-once).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from openinterp.builder.wiki import (
    _read_front_matter,
    experiment_id,
    neuron_dir,
    rebuild_indexes,
)
from openinterp.core.schemas import AuditRecord

DISCUSSION_HEADING = "## Discussion"


@dataclass
class Problem:
    path: Path
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class CheckReport:
    problems: list[Problem] = field(default_factory=list)
    records: int = 0
    pages: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, path: Path, message: str) -> None:
        self.problems.append(Problem(Path(path), message))


def check_registry(
    audits_dir: Path = Path("audits"),
    wiki_dir: Path = Path("wiki"),
    *,
    base_ref: str | None = None,
) -> CheckReport:
    report = CheckReport()
    audits_dir, wiki_dir = Path(audits_dir), Path(wiki_dir)
    slugs = _check_records(audits_dir, wiki_dir, report)
    _check_pages(wiki_dir, slugs, report)
    _check_indexes(wiki_dir, report)
    if base_ref:
        _check_write_once(wiki_dir, base_ref, report)
    return report


def _check_records(audits_dir: Path, wiki_dir: Path, report: CheckReport) -> dict[str, Path]:
    slugs: dict[str, Path] = {}
    if not audits_dir.is_dir():
        return slugs
    for path in sorted(audits_dir.rglob("*.json")):
        report.records += 1
        try:
            record = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            report.add(path, f"invalid audit record ({exc.error_count()} validation errors)")
            continue
        slug = record.submission.identifier.slug
        if slug in slugs:
            report.add(path, f"duplicate record for {slug} (also {slugs[slug]})")
            continue
        slugs[slug] = path
        if record.verdict is None:
            report.add(path, "audit record has no verdict; run `openinterp verify` first")
            continue

        page = neuron_dir(wiki_dir, record) / f"{experiment_id(record)}.md"
        if not page.exists():
            report.add(path, f"missing wiki page {page}; run `openinterp wiki`")
            continue
        archived = page.with_suffix(".json")
        if archived.exists() and _load(archived) != record:
            report.add(archived, f"archived record differs from {path}; archives are write-once")
        meta = _read_front_matter(page)
        expected = {
            "slug": slug,
            "status": record.verdict.status,
            "balanced_accuracy": record.verdict.confidence_score,
            "timestamp": record.verdict.timestamp.isoformat(),
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                report.add(page, f"front matter {key}={meta.get(key)!r} but record has {value!r}")
    return slugs


def _check_pages(wiki_dir: Path, slugs: dict[str, Path], report: CheckReport) -> None:
    root = wiki_dir / "experiments"
    if not root.is_dir():
        return
    for page in sorted(root.rglob("*.md")):
        if page.name == "README.md":
            continue
        report.pages += 1
        try:
            meta = _read_front_matter(page)
        except ValueError:
            meta = {}
        if not meta:
            report.add(page, "experiment page has no valid front matter")
            continue
        if meta.get("slug") not in slugs:
            report.add(page, f"wiki page for {meta.get('slug')} has no audit record in audits/")
        archived = page.with_suffix(".json")
        if not archived.exists():
            report.add(page, f"missing archived record {archived.name}; run `openinterp wiki`")
            continue
        record = _load(archived)
        if record is None or record.verdict is None:
            report.add(archived, "archived record is invalid or has no verdict")
        elif (record.submission.identifier.slug, record.verdict.timestamp.isoformat()) != (
            meta.get("slug"),
            meta.get("timestamp"),
        ):
            report.add(archived, "archived record does not match its experiment page")
    for archived in sorted(root.rglob("*.json")):
        if not archived.with_suffix(".md").exists():
            report.add(archived, "archived record has no experiment page")


def _load(path: Path) -> AuditRecord | None:
    try:
        return AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        return None


def _check_indexes(wiki_dir: Path, report: CheckReport) -> None:
    if not (wiki_dir / "experiments").is_dir():
        return
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "wiki"
        shutil.copytree(wiki_dir, fresh)
        rebuild_indexes(fresh)
        for generated in sorted(fresh.rglob("README.md")):
            committed = wiki_dir / generated.relative_to(fresh)
            if not committed.exists() or committed.read_text(
                encoding="utf-8"
            ) != generated.read_text(encoding="utf-8"):
                report.add(committed, "wiki index is out of date; run `openinterp wiki`")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def _check_write_once(wiki_dir: Path, base_ref: str, report: CheckReport) -> None:
    diff = _git(
        "diff",
        "--name-status",
        "--no-renames",
        f"{base_ref}...HEAD",
        "--",
        str(wiki_dir / "experiments"),
    )
    for line in diff.splitlines():
        status, _, name = line.partition("\t")
        path = Path(name)
        if path.suffix == ".json" and status in {"D", "M"}:
            report.add(path, "archived experiment records are write-once")
            continue
        if path.name == "README.md" or path.suffix != ".md":
            continue
        if status == "D":
            report.add(path, "experiment pages are write-once and must not be deleted")
        elif status == "M":
            before = _git("show", f"{base_ref}:{name}")
            after = path.read_text(encoding="utf-8")
            if before.split(DISCUSSION_HEADING)[0] != after.split(DISCUSSION_HEADING)[0]:
                report.add(
                    path,
                    "experiment pages are write-once; only the Discussion section may be edited",
                )
