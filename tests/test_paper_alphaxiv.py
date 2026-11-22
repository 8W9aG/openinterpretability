"""A published paper's alphaXiv package and URL (scripts/papers.py submit and alphaxiv, #5259)."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import papers  # noqa: E402

PAPER = (
    "# A title\n\nThe abstract has $R^2$ = 0.5\nand a [dataset]"
    "(https://huggingface.co/datasets/org/set).\n\n## Results\n\nOne.\n"
)
URL = "https://www.alphaxiv.org/abs/2611.00001"


def _published(tmp_path, publish=True):
    root = tmp_path / "papers"
    (root / "p").mkdir(parents=True)
    (root / "README.md").write_text("- 2026-11-01: [A title](p/README.md)\n")
    (root / "p" / "README.md").write_text(PAPER)
    (root / "p" / "results.jsonl").write_text('{"x": 1}\n')
    papers.new("p", "#12 (more seeds).", root)
    if publish:
        papers.publish("p", root, today="2026-11-20")
    return root


def _pdf(tmp_path):
    dist = tmp_path / "dist"
    (dist / "research" / "p").mkdir(parents=True)
    (dist / "research" / "p" / "paper.pdf").write_bytes(b"%PDF-1.4 test")
    return dist


def test_submit_writes_the_pdf_and_every_field(tmp_path):
    root = _published(tmp_path)
    package = papers.submit("p", root, _pdf(tmp_path), tmp_path / "out")
    assert (package / "paper.pdf").read_bytes() == b"%PDF-1.4 test"
    text = (package / "submission.md").read_text()
    assert "- **Title:** A title" in text
    assert "- **Authors, in order:** Will Sackfield; Open Interpretability" in text
    assert f"- **Licence:** {papers.LICENCE}" in text and "- **Version:** 1" in text
    assert "The abstract has $R^2$ = 0.5 and a [dataset]" in text  # the maths stays
    assert "- https://huggingface.co/datasets/org/set" in text
    assert "/blob/main/papers/p/results.jsonl" in text and "- https://doi.org/" in text


def test_submit_refuses_a_preprint_and_a_missing_pdf(tmp_path):
    root = _published(tmp_path, publish=False)
    with pytest.raises(SystemExit, match="not published"):
        papers.submit("p", root, _pdf(tmp_path), tmp_path / "out")
    papers.publish("p", root, today="2026-11-20")
    with pytest.raises(SystemExit, match="run make build"):
        papers.submit("p", root, tmp_path / "no-dist", tmp_path / "out")


def test_an_addendum_is_a_new_version(tmp_path):
    root = _published(tmp_path)
    readme = root / "p" / "README.md"
    readme.write_text(readme.read_text() + "\n## Addenda\n\n### 2026-12-01\n\nA fix.\n")
    package = papers.submit("p", root, _pdf(tmp_path), tmp_path / "out")
    assert "- **Version:** 2" in (package / "submission.md").read_text()


def test_record_keeps_the_hash_and_replaces_an_old_url(tmp_path):
    root = _published(tmp_path)
    papers.record("p", URL, root)
    papers.record("p", URL + "v2", root)
    cff = yaml.safe_load((root / "p" / "CITATION.cff").read_text())
    kinds = [(i["type"], i["value"]) for i in cff["identifiers"]]
    assert kinds[0][0] == "other" and kinds[1:] == [("url", URL + "v2")]
    assert papers.check(root) == []


def test_record_refuses_a_bad_url_and_a_preprint(tmp_path):
    root = _published(tmp_path)
    with pytest.raises(SystemExit, match="is not https://www.alphaxiv.org/abs/"):
        papers.record("p", "https://arxiv.org/abs/2611.00001", root)
    other = _published(tmp_path / "second", publish=False)
    with pytest.raises(SystemExit, match="not published"):
        papers.record("p", URL, other)


def test_check_finds_a_bad_or_early_alphaxiv_url(tmp_path):
    root = _published(tmp_path, publish=False)
    path = root / "p" / "CITATION.cff"
    cff = yaml.safe_load(path.read_text())
    cff["identifiers"] = [{"type": "url", "value": "https://example.com/x",
                           "description": papers.ALPHAXIV}]  # fmt: skip
    papers.write_citation(root / "p", cff)
    found = papers.check(root)
    assert any("a preprint has an alphaXiv URL" in p for p in found)
    assert any("is not https://www.alphaxiv.org/abs/<id>" in p for p in found)


def test_the_site_links_the_alphaxiv_page(tmp_path):
    from openinterpretability.builder.compiler import compile_site

    root = _published(tmp_path)
    compile_site([], tmp_path / "site", papers_dir=root)
    page = (tmp_path / "site" / "research" / "p" / "index.html").read_text()
    assert "alphaXiv</a>" not in page
    papers.record("p", URL, root)
    compile_site([], tmp_path / "site2", papers_dir=root)
    page = (tmp_path / "site2" / "research" / "p" / "index.html").read_text()
    assert f'href="{URL}" rel="noopener">alphaXiv</a>' in page


def test_main_runs_both_commands(tmp_path, monkeypatch, capsys):
    root = _published(tmp_path)
    monkeypatch.setattr(papers, "PAPERS", root)
    monkeypatch.setattr(papers, "submit", lambda slug: tmp_path / "out" / slug)
    monkeypatch.setattr(papers, "record", lambda slug, url: print(f"recorded {slug} {url}"))
    monkeypatch.setattr(sys, "argv", ["papers.py", "submit", "p"])
    assert papers.main() == 0
    monkeypatch.setattr(sys, "argv", ["papers.py", "alphaxiv", "p", URL])
    assert papers.main() == 0
    out = capsys.readouterr().out
    assert "upload it from the project account" in out and f"recorded p {URL}" in out
    monkeypatch.setattr(papers, "muse_issue", lambda slug: f"issue for {slug}")
    monkeypatch.setattr(sys, "argv", ["papers.py", "submit", "p", "--issue"])
    assert papers.main() == 0
    assert "issue for p" in capsys.readouterr().out


def test_muse_issue_opens_one_labelled_issue_with_the_steps(tmp_path):
    root = _published(tmp_path)
    calls = []

    def run(*args, stdin=None):
        calls.append((args, stdin))
        return "[]" if args[1] == "list" else "https://github.com/o/r/issues/9\n"

    assert papers.muse_issue("p", root, run) == "https://github.com/o/r/issues/9"
    (args, body) = calls[-1]
    assert args[:2] == ("issue", "create")
    assert "Post p v1 to the alphaXiv AI Track" in args
    assert [args[i + 1] for i, a in enumerate(args) if a == "--label"] == list(papers.MUSE_LABELS)
    assert "For Muse" in body and "papers.py submit p" in body and "papers.py alphaxiv p" in body
    assert "- **Title:** A title" in body  # the form's fields follow the steps


def test_muse_issue_is_not_opened_twice_or_for_a_preprint(tmp_path):
    root = _published(tmp_path)
    title = "Post p v1 to the alphaXiv AI Track"
    calls = []

    def run(*args, stdin=None):
        calls.append(args)
        return f'[{{"title": "{title}"}}]'

    assert papers.muse_issue("p", root, run) == f"open already: {title}"
    assert all(c[1] == "list" for c in calls)
    with pytest.raises(SystemExit, match="not published"):
        papers.muse_issue("p", _published(tmp_path / "second", publish=False), run)


def test_muse_issue_calls_gh_by_default(tmp_path, monkeypatch):
    import subprocess

    root = _published(tmp_path)
    seen = []

    def fake(args, input=None, **kwargs):
        seen.append(args)
        out = "[]" if args[2] == "list" else "url\n"
        return subprocess.CompletedProcess(args, 0, stdout=out)

    monkeypatch.setattr(subprocess, "run", fake)
    assert papers.muse_issue("p", root) == "url"
    assert seen[0][:3] == ["gh", "issue", "list"] and seen[1][:3] == ["gh", "issue", "create"]
