"""Tests for scripts/check_markdown.py, and a lint of the rendered wiki templates."""

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from openinterp.builder.wiki import write_experiment_page, write_stop_page
from openinterp.core.draws import UnauditableDraw
from openinterp.core.schemas import AuditRecord, ExperimentProvenance

_SPEC = importlib.util.spec_from_file_location(
    "check_markdown", Path(__file__).parents[1] / "scripts" / "check_markdown.py"
)
check_markdown = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_markdown)


@pytest.mark.parametrize(
    ("heading", "anchor"),
    [
        ("Set up (once)", "set-up-once"),
        ("The `audit` command", "the-audit-command"),
        ("§3. Step 9: **decide**", "3-step-9-decide"),
        ("See [the docs](docs/x.md)", "see-the-docs"),
        ("snake_case-name", "snake_case-name"),
    ],
)
def test_github_slug(heading, anchor):
    assert check_markdown.github_slug(heading) == anchor


def test_anchors_number_duplicates_and_skip_fences():
    text = "# Added\n\n## Added\n\n```bash\n# not a heading\n```\n\n~~~\n```\n# nor this\n~~~\n"
    assert check_markdown.heading_anchors(text) == {"added", "added-1"}


def test_link_targets_skip_code():
    text = (
        '[a](a.md) ![b](b.png) [c](c.md "title") [d](<d.md>)\n'
        "`[e](e.md)`\n\n```text\n[f](f.md)\n```\n\n[g]: g.md\n"
    )
    assert check_markdown.link_targets(text) == ["a.md", "b.png", "c.md", "d.md", "g.md"]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_link_problems_reports_each_kind(tmp_path):
    _write(tmp_path / "target.md", "# Real heading\n")
    _write(tmp_path / "data.json", "{}")
    page = _write(
        tmp_path / "page.md",
        "# Here\n\n[ok](target.md) [ok](target.md#real-heading) [ok](#here) [ok](data.json#x)\n"
        "[ok](https://example.com/missing.md) [ok](mailto:a@example.com)\n"
        "[bad](gone.md) [bad](target.md#gone) [bad](#gone)\n",
    )
    problems = check_markdown.link_problems([page])
    assert problems == [
        f"{page}: broken link gone.md: no such file",
        f"{page}: broken link target.md#gone: no such heading",
        f"{page}: broken link #gone: no such heading",
    ]


def test_style_problems_uses_the_repository_config(tmp_path):
    long_line = "word " * 100
    clean = _write(tmp_path / "clean.md", f"---\nname: x\n---\n\n# Title\n\n{long_line.strip()}\n")
    assert check_markdown.style_problems([clean]) == []
    bad = _write(tmp_path / "bad.md", "# Title\n\nText\n- item\n")
    problems = check_markdown.style_problems([bad])
    assert len(problems) == 1 and "MD032" in problems[0]
    assert check_markdown.style_problems([]) == []


def test_style_problems_raises_when_pymarkdown_fails(tmp_path):
    with pytest.raises(RuntimeError, match="pymarkdown failed"):
        check_markdown.style_problems([tmp_path / "missing.md"])


def test_main_returns_one_on_a_problem(tmp_path, monkeypatch, capsys):
    page = _write(tmp_path / "page.md", "# Title\n\n[bad](gone.md)\n")
    monkeypatch.setattr(check_markdown, "markdown_files", lambda: [page])
    assert check_markdown.main() == 1
    assert "no such file" in capsys.readouterr().out
    monkeypatch.setattr(check_markdown, "markdown_files", list)
    assert check_markdown.main() == 0


def test_markdown_files_skips_generated_and_vendored():
    names = {p.relative_to(check_markdown.REPO).as_posix() for p in check_markdown.markdown_files()}
    assert {"README.md", "docs/PROTOCOL.md", ".claude/skills/release/SKILL.md"} <= names
    assert not any(n.startswith(("wiki/", "openinterp/builder/templates/")) for n in names)
    assert not any(n.startswith(".claude/skills/asd-ste100/") for n in names)


def test_rendered_experiment_and_index_templates_pass(tmp_path, record):
    """Two experiments on one unit also fill the history rows of the unit's page."""
    wiki = tmp_path / "wiki"
    write_experiment_page(record, wiki)
    probes = [
        t.model_copy(update={"passed": False, "measured": True})
        for t in record.verdict.adversarial_test_results
    ]
    update = {
        "status": "NEEDS_REVISION",
        "timestamp": record.verdict.timestamp + timedelta(hours=1),
        "adversarial_test_results": probes,
    }
    verdict = record.verdict.model_copy(update=update)
    second = write_experiment_page(AuditRecord(submission=record.submission, verdict=verdict), wiki)
    assert "Failed probes:" in second.read_text()
    pages = sorted(wiki.rglob("*.md"))
    assert len(pages) == 4  # home, unit page, two experiments
    problems = check_markdown.style_problems(pages)
    assert problems == [], "\n".join(problems)


@pytest.mark.parametrize(
    "fields",
    [
        {"outcome": "default_on", "active_share": 0.9},
        {"outcome": "dead", "proven_dead": True},
        {"outcome": "dead", "proven_dead": False},
    ],
)
def test_rendered_stop_template_passes(tmp_path, record, fields):
    wiki = tmp_path / "wiki"
    provenance = ExperimentProvenance(
        dtype="float32",
        corpus="corpus.jsonl",
        corpus_sha256="ab" * 32,
        n_snippets=1,
        tokens_scanned=100,
        selection="manual",
        harness_version="test",
    )
    draw = UnauditableDraw(
        identifier=record.submission.identifier,
        provenance=provenance,
        timestamp=datetime(2026, 11, 2, 12, tzinfo=UTC),
        **fields,
    )
    page = write_stop_page(draw, wiki)
    assert check_markdown.style_problems([page]) == [], page.read_text()
