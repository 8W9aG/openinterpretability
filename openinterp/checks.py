"""Registry consistency checks, run in CI on every pull request (`openinterp check`).

An experiment is posted as an archived record ``<id>.json`` plus its page ``<id>.md``
in ``wiki/experiments/<slug>/``. A neuron may have any number of experiments. The
checks enforce that:

1. every archived record validates, has a verdict, and sits at the path its neuron
   slug and experiment id dictate;
2. every record has its page and every page its record, and the page's front matter
   matches the record;
   a record that ``supersedes`` another names an earlier experiment on the same unit;
3. the retired ``audits/`` directory holds no records (import them with
   ``openinterp wiki --import-dir audits``);
4. with ``base_ref``, existing experiment pages were not deleted or edited outside
   their Discussion section, archived records were not touched (write-once), and no
   generated index (``README.md``) was committed, and records added since name their
   agents in canonical spelling (``openinterp.core.agent_names``).

The indexes are rebuilt from the records whenever they are needed and are gitignored:
every experiment PR used to edit the same index rows, so parallel PRs conflicted.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from openinterp.builder.wiki import _read_front_matter, archive_path
from openinterp.core.agent_names import spelling_problem
from openinterp.core.paths import DEFAULT_WIKI_DIR, experiments_dir
from openinterp.core.schemas import AuditRecord

DISCUSSION_HEADING = "## Discussion"
LEGACY_AUDITS_DIR = Path("audits")


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
    draws: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, path: Path, message: str) -> None:
        self.problems.append(Problem(Path(path), message))


def check_registry(
    wiki_dir: Path = DEFAULT_WIKI_DIR,
    *,
    base_ref: str | None = None,
    legacy_audits_dir: Path = LEGACY_AUDITS_DIR,
) -> CheckReport:
    report = CheckReport()
    wiki_dir = Path(wiki_dir)
    _check_legacy(Path(legacy_audits_dir), report)
    _check_records(wiki_dir, report)
    _check_pages(wiki_dir, report)
    _check_draws(wiki_dir, report)
    if base_ref:
        _check_write_once(wiki_dir, base_ref, report)
        _check_agent_names(changed_records(wiki_dir, base_ref), report)
    return report


def _check_legacy(audits_dir: Path, report: CheckReport) -> None:
    if audits_dir.is_dir():
        for path in sorted(audits_dir.rglob("*.json")):
            report.add(
                path,
                "audits/ is retired; experiments live in wiki/experiments/. Run "
                "`openinterp wiki --import-dir audits`, then delete audits/",
            )


def _check_records(wiki_dir: Path, report: CheckReport) -> None:
    root = experiments_dir(wiki_dir)
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.json")):
        report.records += 1
        record = _load(path)
        if record is None:
            report.add(path, "invalid experiment record")
            continue
        if record.verdict is None:
            report.add(path, "experiment record has no verdict; run `openinterp verify` first")
            continue
        expected = archive_path(wiki_dir, record)
        if path != expected:
            report.add(path, f"experiment record belongs at {expected}")
        _check_supersedes(path, record, report)
        page = path.with_suffix(".md")
        if not page.exists():
            report.add(path, "experiment record has no page; run `openinterp wiki`")
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


def _check_draws(wiki_dir: Path, report: CheckReport) -> None:
    """Recorded unauditable draws (wiki/draws/) parse and sit where they belong."""
    from pydantic import ValidationError

    from openinterp.core.draws import UnauditableDraw, draw_files, draw_path

    for path in draw_files(wiki_dir):
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


def _check_supersedes(path: Path, record: AuditRecord, report: CheckReport) -> None:
    revises = record.submission.supersedes
    if revises is None:
        return
    if revises >= path.stem:  # ids are UTC timestamps, so they sort chronologically
        report.add(path, f"supersedes {revises}, which is not an earlier experiment")
    elif not path.with_name(f"{revises}.json").exists():
        report.add(path, f"supersedes {revises}, but this unit has no such experiment")


def _check_pages(wiki_dir: Path, report: CheckReport) -> None:
    root = experiments_dir(wiki_dir)
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
        if not page.with_suffix(".json").exists():
            report.add(page, f"missing experiment record {page.with_suffix('.json').name}")
    for sig in sorted(root.rglob("*.json.sig")):
        if not sig.with_suffix("").exists():
            report.add(sig, f"signature for a missing record {sig.with_suffix('').name}")


def _check_agent_names(paths: list[Path], report: CheckReport) -> None:
    """New records spell their agents one way, so the site shows one model as one agent.
    Older records keep their spellings (they're signed); the build maps them."""
    for path in paths:
        record = _load(path)
        if record is None or record.verdict is None:
            continue  # _check_records reports it
        scores = record.verdict.explanation_scores
        names = {
            "submitting_agent_model": record.submission.submitting_agent_model,
            "reviewer_agent_model": record.verdict.reviewer_agent_model,
            "judge_model": scores.judge_model if scores else None,
        }
        for field_name, model in names.items():
            if model and (problem := spelling_problem(model)):
                report.add(path, f"{field_name}={model!r}: {problem}")


def _load(path: Path) -> AuditRecord | None:
    try:
        return AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        return None


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
        str(wiki_dir),
    )
    for line in diff.splitlines():
        status, _, name = line.partition("\t")
        path = Path(name)
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
            before = _git("show", f"{base_ref}:{name}")
            after = path.read_text(encoding="utf-8")
            if before.split(DISCUSSION_HEADING)[0] != after.split(DISCUSSION_HEADING)[0]:
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
