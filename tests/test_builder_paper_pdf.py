"""The paper PDF helpers: finding Chrome, printing as root, failed prints and odd figures."""

import base64
import subprocess
from pathlib import Path

from openinterpretability.builder import paper_pdf


def test_chrome_is_the_first_usual_path_that_exists(monkeypatch, tmp_path):
    present = tmp_path / "Chrome"
    present.write_text("")
    monkeypatch.delenv(paper_pdf.CHROME_ENV)
    monkeypatch.setattr(paper_pdf, "CHROME_PATHS", (str(tmp_path / "absent"), str(present)))
    monkeypatch.setattr(paper_pdf.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert paper_pdf.chrome() == str(present)


def test_chrome_falls_back_to_the_first_name_on_the_path(monkeypatch, tmp_path):
    found = {"chromium": "/usr/bin/chromium", "chromium-browser": "/usr/bin/chromium-browser"}
    monkeypatch.delenv(paper_pdf.CHROME_ENV)
    monkeypatch.setattr(paper_pdf, "CHROME_PATHS", (str(tmp_path / "absent"),))
    monkeypatch.setattr(paper_pdf.shutil, "which", found.get)
    assert paper_pdf.chrome() == "/usr/bin/chromium"


def test_no_chrome_anywhere_gives_none(monkeypatch, tmp_path):
    monkeypatch.delenv(paper_pdf.CHROME_ENV)
    monkeypatch.setattr(paper_pdf, "CHROME_PATHS", (str(tmp_path / "absent"),))
    monkeypatch.setattr(paper_pdf.shutil, "which", lambda name: None)
    assert paper_pdf.chrome() is None


def test_printing_as_root_turns_off_the_sandbox(monkeypatch, tmp_path):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        Path(args[-2].removeprefix("--print-to-pdf=")).write_bytes(b"%PDF-1.7\n")

    monkeypatch.setattr(paper_pdf.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(paper_pdf.subprocess, "run", run)
    page = tmp_path / "page.html"
    page.write_text("<p>hi</p>")
    assert paper_pdf.print_pdf(page, tmp_path / "paper.pdf", "chrome")
    assert calls[0][:2] == ["chrome", "--no-sandbox"]


def test_a_print_that_times_out_writes_no_pdf(monkeypatch, tmp_path):
    def run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(paper_pdf.subprocess, "run", run)
    page = tmp_path / "page.html"
    page.write_text("<p>hi</p>")
    assert paper_pdf.print_pdf(page, tmp_path / "paper.pdf", "chrome") is False
    assert not (tmp_path / "paper.pdf").exists()


def test_a_chrome_that_cannot_start_writes_no_pdf(monkeypatch, tmp_path):
    def run(args, **kwargs):
        raise OSError("exec format error")

    monkeypatch.setattr(paper_pdf.subprocess, "run", run)
    page = tmp_path / "page.html"
    page.write_text("<p>hi</p>")
    assert paper_pdf.print_pdf(page, tmp_path / "paper.pdf", "chrome") is False


def test_text_without_an_svg_tag_prints_unchanged():
    text = "<p>not a figure ²</p>"
    assert paper_pdf.print_svg(text) == text


def test_a_missing_figure_keeps_its_link_and_a_present_one_is_inlined(tmp_path):
    (tmp_path / "media").mkdir()
    (tmp_path / "media/here.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    body = '<img src="media/gone.svg"><img src="media/here.svg">'
    out = paper_pdf.print_body(body, tmp_path)
    assert '<img src="media/gone.svg">' in out
    inlined = out.split('<img src="data:image/svg+xml;base64,')[1].split('"')[0]
    assert (
        base64.b64decode(inlined)
        .decode()
        .startswith('<svg xmlns="http://www.w3.org/2000/svg"><style>@font-face')
    )
