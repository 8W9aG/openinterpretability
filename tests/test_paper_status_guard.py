"""Paper status lines name only open issues (scripts/papers.py check, #5395)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("papers_script", SCRIPTS / "papers.py")
papers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(papers)


def _papers(tmp_path, status: str) -> Path:
    root = tmp_path / "papers"
    (root / "p").mkdir(parents=True)
    (root / "README.md").write_text("# Papers\n\n- 2026-11-19: [P](p/README.md)\n")
    (root / "p" / "README.md").write_text(f"# P\n\nThe abstract.\n\n{status}\n\n## Introduction\n")
    return root


def test_a_closing_issue_on_the_line_fails_and_a_cited_one_passes(tmp_path):
    root = _papers(tmp_path, "**Status:** Preprint. Work left: #12 (more seeds), "
                             "#13 (rewrite in the paper form of #4202).")  # fmt: skip
    assert papers.work_left(root) == {"p": (12, 13)}  # #4202 is a citation, in brackets
    (problem,) = papers.named({12}, root)
    assert problem.startswith("p: #12") and "publish p" in problem
    assert papers.named({4202, 99}, root) == []  # a cited issue, and one not on the line


def test_a_published_paper_names_no_work(tmp_path):
    root = _papers(tmp_path, "**Status:** Published 2026-11-20. Changes since are in the addenda.")
    assert papers.work_left(root) == {} and papers.named({12}, root) == []


def test_closed_asks_github_for_each_issue(monkeypatch):
    states = {12: "CLOSED", 13: "OPEN"}

    def run(args, **_):
        return SimpleNamespace(stdout=json.dumps({"state": states[int(args[3])]}))

    monkeypatch.setattr("subprocess.run", run)
    assert papers.closed({12, 13}) == {12}
