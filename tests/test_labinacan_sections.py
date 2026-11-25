"""labinacan's sections: Changelog, the literature review, Research and Posts (#6628)."""

from labinacan.site import sections


def test_research_reads_its_papers_with_the_repo_url_and_makes_each_folder(
    tmp_path, monkeypatch, capsys
):
    from types import SimpleNamespace

    from labinacan.site import paper_pdf

    paper, seen, pages = SimpleNamespace(slug="p", href="p/index.html", interactive=False), {}, []
    monkeypatch.setattr(sections, "load_papers", lambda d, url: seen.update(url=url) or [paper])
    monkeypatch.setattr(sections, "copy_assets", lambda *a: [])
    monkeypatch.setattr(paper_pdf, "write_pdf", lambda env, folder, *a: seen.update(pdf=folder))

    class Render:  # writes no file, so research/ doesn't exist before the paper's folder
        common, output_dir, env = {"repo_url": "https://x.org/r"}, tmp_path / "out", None

        def __call__(self, href, template, **kw):
            pages.append((href, template))

    found = sections.render_research(Render(), tmp_path, SimpleNamespace(pages=[]), False, {})
    assert found == [paper] and seen["url"] == "https://x.org/r"  # repo links in the papers
    assert seen["pdf"] == tmp_path / "out" / "research" / "p" and seen["pdf"].is_dir()
    assert pages == [
        ("research/index.html", "research.html"),
        ("research/p/index.html", "paper.html"),
    ]
    assert capsys.readouterr().out == ""  # no PDFs asked for: no warning about Chrome


def test_research_tells_the_index_what_exists_and_warns_without_chrome(
    tmp_path, monkeypatch, capsys
):
    from types import SimpleNamespace

    from labinacan.site import paper_pdf

    monkeypatch.setattr(sections, "load_papers", lambda d, url: [])
    flags = {"explained": True, "reviewed": False}
    monkeypatch.setattr(paper_pdf, "chrome", lambda: None)
    calls = []

    class Render:
        common, output_dir, env = {"repo_url": "r"}, tmp_path, None

        def __call__(self, href, template, **kw):
            calls.append((href, template, kw))

    sections.render_research(Render(), tmp_path, SimpleNamespace(pages=[]), True, flags)
    assert calls == [("research/index.html", "research.html",
                      {"papers": [], "explained": True, "reviewed": False})]  # fmt: skip
    warning = "No Chrome found ($OPENINTERPRETABILITY_CHROME): the papers get no PDFs.\n"
    assert capsys.readouterr().out == warning
    monkeypatch.setattr(paper_pdf, "chrome", lambda: "/bin/chrome")
    sections.render_research(Render(), tmp_path, SimpleNamespace(pages=[]), True, flags)
    assert capsys.readouterr().out == ""  # a browser: no warning


def _recorded(monkeypatch, tmp_path, versions, reviews):
    """A renderer and stand-ins for each section, all writing to one ``calls`` list."""
    calls = []
    monkeypatch.setattr(
        sections, "load_versions", lambda url: calls.append(("versions", url)) or versions
    )
    monkeypatch.setattr(
        sections, "load_reviews", lambda url: calls.append(("load", url)) or reviews
    )
    monkeypatch.setattr(
        sections, "render_reviews", lambda r, found: calls.append(("reviews", found))
    )

    def research(render, papers_dir, report, pdfs, index):
        calls.append(("research", papers_dir, report, pdfs, index))
        return ["paper"]

    monkeypatch.setattr(sections, "render_research", research)
    feeds = lambda out, papers, found, folder, common: calls.append(("feeds", papers, found))  # noqa: E731
    monkeypatch.setattr(sections, "write_research_feeds", feeds)
    monkeypatch.setattr(sections, "render_posts", lambda r, d, report: calls.append(("posts", d)))

    class Render:
        common, output_dir = {"repo_url": "r"}, tmp_path

        def __call__(self, href, template, **kw):
            calls.append((href, template, kw))

    return Render(), calls


def test_the_projects_own_sections_come_first_then_each_common_section(monkeypatch, tmp_path):
    from types import SimpleNamespace

    render, calls = _recorded(monkeypatch, tmp_path, ["v1"], ["review"])
    own = (lambda r: calls.append("about"), lambda r: calls.append("explanations"))
    report = SimpleNamespace(pages=[])
    sections.render_sections(render, report, own, papers_dir=tmp_path / "p",
                             posts_dir=tmp_path / "q", pdfs=True,
                             research_index={"explained": True})  # fmt: skip
    assert calls == [
        "about", "explanations",
        ("versions", "r"),
        ("changelog.html", "changelog.html", {"versions": ["v1"]}),
        ("load", "r"),
        ("reviews", ["review"]),
        ("research", tmp_path / "p", report, True, {"explained": True, "reviewed": True}),
        ("feeds", ["paper"], ["review"]),
        ("posts", tmp_path / "q"),
    ]  # fmt: skip


def test_an_install_with_no_changelog_or_reviews_writes_neither(monkeypatch, tmp_path):
    from types import SimpleNamespace

    render, calls = _recorded(monkeypatch, tmp_path, [], [])
    report = SimpleNamespace(pages=[])
    sections.render_sections(render, report)
    assert calls == [("versions", "r"), ("load", "r"), ("reviews", []),
                     ("research", None, report, False, {"reviewed": False}),
                     ("feeds", ["paper"], []), ("posts", None)]  # fmt: skip


class _Pages:
    """A renderer that records each page: its path, its template and what it gets."""

    def __init__(self, output_dir):
        self.common = {"repo_url": "https://x.org/r", "site_url": "https://x.org/",
                       "site_title": "X"}  # fmt: skip
        self.output_dir, self.env, self.pages = output_dir, "env", []

    def __call__(self, href, template, **kw):
        self.pages.append((href, template, kw))


def test_posts_reads_each_post_and_writes_its_page_assets_and_feed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    post = SimpleNamespace(href="hi/index.html", draft=False, interactive=False)
    draft = SimpleNamespace(href="drafts/d/index.html", draft=True, interactive=False)
    seen = {}

    def load(folder, url, source):
        seen["load"] = (folder, url, source)
        return [post, draft]

    def assets(folder, item, out, section):
        return [f"{section}:{item.href}:{out == tmp_path}"]

    def feed(listed, url, title):
        seen["feed"] = (listed, url, title)
        return "é"

    monkeypatch.setattr(sections, "load_papers", load)
    monkeypatch.setattr(sections, "copy_assets", assets)
    monkeypatch.setattr(sections, "posts_feed", feed)
    render, report = _Pages(tmp_path), SimpleNamespace(pages=[])
    (tmp_path / "posts").mkdir()  # the index's page makes it in a real build
    sections.render_posts(render, tmp_path / "src", report)
    assert seen["load"] == (tmp_path / "src", "https://x.org/r", "posts")
    assert render.pages == [
        ("posts/index.html", "posts.html", {"posts": [post]}),  # a draft is never listed
        ("posts/hi/index.html", "post.html", {"post": post}),
        ("posts/drafts/d/index.html", "post.html", {"post": draft}),  # but it is built
    ]
    assert report.pages == ["posts:hi/index.html:True", "posts:drafts/d/index.html:True"]
    assert seen["feed"] == ([post], "https://x.org/", "X")
    assert (tmp_path / sections.POSTS_FEED).read_bytes() == "é".encode()


def test_research_passes_each_paper_to_its_assets_pdf_and_page(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from labinacan.site import paper_pdf

    paper = SimpleNamespace(slug="p", href="p/index.html", interactive=False)
    seen, chromes = {}, []
    monkeypatch.setattr(
        sections, "load_papers", lambda d, url: seen.update(load=(d, url)) or [paper]
    )
    monkeypatch.setattr(sections, "copy_assets",
                        lambda d, item, out: [(d, item, out)])  # fmt: skip
    monkeypatch.setattr(paper_pdf, "chrome", lambda: chromes.append(1) or "chrome")

    def write_pdf(env, folder, item, common, browser):
        seen["pdf"] = (env, folder, item, common, browser)
        return "p.pdf"

    monkeypatch.setattr(paper_pdf, "write_pdf", write_pdf)
    (tmp_path / "research" / "p").mkdir(parents=True)  # a folder left by an earlier build
    render, report = _Pages(tmp_path), SimpleNamespace(pages=[])
    sections.render_research(render, tmp_path / "papers", report, True, {})
    assert seen["load"] == (tmp_path / "papers", "https://x.org/r") and chromes == [1]
    assert report.pages == [(tmp_path / "papers", paper, tmp_path)]
    assert seen["pdf"] == ("env", tmp_path / "research" / "p", paper, render.common, "chrome")
    assert render.pages[-1] == (
        "research/p/index.html",
        "paper.html",
        {"paper": paper, "pdf": "p.pdf"},
    )
    sections.render_research(render, tmp_path / "papers", report, False, {})
    assert chromes == [1]  # no PDFs asked for: no browser looked for
    assert seen["pdf"][-1] is None
