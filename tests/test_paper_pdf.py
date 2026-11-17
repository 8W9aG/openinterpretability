"""The PDF of each paper (openinterpretability/builder/paper_pdf.py, #3140), with a fake Chrome."""

import stat
import sys

from openinterpretability.builder import paper_pdf
from openinterpretability.builder.compiler import compile_site

PAPER = "# A title\n\nThe abstract.\n\n## Introduction\n\nText.\n\n## References\n\n- A.\n"

# A stand-in for Chrome: writes a tiny PDF to the --print-to-pdf path, and records its args.
FAKE = """#!{python}
import sys
args = sys.argv[1:]
open("{log}", "w").write("\\n".join(args))
out = next(a.split("=", 1)[1] for a in args if a.startswith("--print-to-pdf="))
open(out, "wb").write(b"%PDF-1.4 fake")
"""


def _fake_chrome(tmp_path):
    path = tmp_path / "chrome"
    log = tmp_path / "chrome.log"
    path.write_text(FAKE.format(python=sys.executable, log=log))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path, log


def _site(tmp_path, pdfs):
    papers = tmp_path / "papers"
    (papers / "p").mkdir(parents=True)
    (papers / "p" / "README.md").write_text(PAPER, encoding="utf-8")
    (papers / "README.md").write_text("- [A title](p/README.md)\n", encoding="utf-8")
    out = tmp_path / "site"
    compile_site([], out, papers_dir=papers, paper_pdfs=pdfs)
    return out / "research" / "p"


def test_chrome_comes_from_the_environment_first(tmp_path, monkeypatch):
    browser, _ = _fake_chrome(tmp_path)
    monkeypatch.setenv(paper_pdf.CHROME_ENV, str(browser))
    assert paper_pdf.chrome() == str(browser)
    monkeypatch.setenv(paper_pdf.CHROME_ENV, str(tmp_path / "missing"))
    assert paper_pdf.chrome() is None


def test_a_build_prints_each_paper_and_links_its_pdf(tmp_path, monkeypatch):
    browser, log = _fake_chrome(tmp_path)
    monkeypatch.setenv(paper_pdf.CHROME_ENV, str(browser))
    folder = _site(tmp_path, pdfs=True)
    assert (folder / "paper.pdf").read_bytes().startswith(b"%PDF")
    assert 'href="paper.pdf"' in (folder / "index.html").read_text(encoding="utf-8")
    assert not (folder / paper_pdf.PRINT_PAGE).exists()  # the print page is deleted
    args = log.read_text().splitlines()
    assert "--force-prefers-reduced-motion" in args and "--headless=new" in args


def test_the_print_page_has_the_abstract_and_numbered_sections(tmp_path, monkeypatch):
    kept = {}
    real = paper_pdf.print_pdf

    def keep(page, pdf, browser):
        kept["html"] = page.read_text(encoding="utf-8")
        return real(page, pdf, browser)

    browser, _ = _fake_chrome(tmp_path)
    monkeypatch.setenv(paper_pdf.CHROME_ENV, str(browser))
    monkeypatch.setattr(paper_pdf, "print_pdf", keep)
    _site(tmp_path, pdfs=True)
    html = kept["html"]
    assert "<h1>A title</h1>" in html and 'content: "Abstract"' in html
    assert "counter-increment: section" in html and 'id="introduction"' in html
    assert "@bottom-center" in html  # page numbers


def test_no_chrome_means_no_pdf_and_no_link(tmp_path, monkeypatch):
    monkeypatch.setenv(paper_pdf.CHROME_ENV, str(tmp_path / "missing"))
    folder = _site(tmp_path, pdfs=True)
    assert not (folder / "paper.pdf").exists()
    assert 'href="paper.pdf"' not in (folder / "index.html").read_text(encoding="utf-8")


def test_pdfs_are_off_unless_asked(tmp_path, monkeypatch):
    browser, log = _fake_chrome(tmp_path)
    monkeypatch.setenv(paper_pdf.CHROME_ENV, str(browser))
    folder = _site(tmp_path, pdfs=False)
    assert not (folder / "paper.pdf").exists() and not log.exists()
