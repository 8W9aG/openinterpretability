"""Each paper's and post's link to its X post (openinterpretability/social/links.py, #5840)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from openinterpretability.cli import cli
from openinterpretability.social import run as social
from openinterpretability.social.feed import Entry, Tweet
from openinterpretability.social.links import folder_of, link, write_records
from tests.test_social_media import _paper

SITE = "https://open-interpretability.com/"


def test_a_post_and_a_threads_head_are_recorded_once_whole(tmp_path):
    state = tmp_path / "x.json"
    social.save_state(state, social.State(feeds={"papers"}))
    single = Entry(f"{SITE}research/a/index.html", "A", f"{SITE}research/a/", "S.")
    thread = Entry(f"{SITE}research/b/index.html", "B", f"{SITE}research/b/", "S.", None,
                   (Tweet("One"), Tweet("Two")))  # fmt: skip
    ids = iter(["t1", "t2", "t3"])
    posted: list = []
    social.run(state, {"papers": [single, thread]}, lambda *a: next(ids), now=0, posted=posted)
    assert posted == [(single.id, "t1"), (thread.id, "t2")]  # the thread's head, not its tail


def test_a_linked_entry_is_not_new_again(tmp_path):
    # The link PR changes media/social.json, not the entry's id, so nothing posts again.
    state = tmp_path / "x.json"
    entry = Entry(f"{SITE}research/a/index.html", "A", f"{SITE}research/a/", "S.")
    social.save_state(state, social.State({entry.id}, feeds={"papers"}))
    assert social.run(state, {"papers": [entry]}, lambda *a: "never", now=0) == ["nothing new"]


def test_each_page_maps_to_its_folder_and_the_link_keeps_the_other_keys(tmp_path):
    assert folder_of(f"{SITE}posts/hello/index.html", tmp_path) == tmp_path / "posts" / "hello"
    (tmp_path / "papers" / "p" / "media").mkdir(parents=True)
    manifest = tmp_path / "papers" / "p" / "media" / "social.json"
    manifest.write_text('{"source": "media/f.svg", "sha256": "ab"}')
    posted = tmp_path / "posted.json"
    write_records(posted, [(f"{SITE}research/p/index.html", "7"),
                           (f"{SITE}research/gone/index.html", "8")])  # fmt: skip
    assert link(posted, tmp_path) == [manifest]  # a folder not in the repo is skipped
    assert json.loads(manifest.read_text()) == {
        "source": "media/f.svg", "sha256": "ab", "x_url": "https://x.com/i/status/7"}  # fmt: skip


def test_the_page_shows_the_link_once_the_paper_has_one(tmp_path):
    from openinterpretability.builder.compiler import compile_site

    papers = _paper(tmp_path, animated=False)
    out = tmp_path / "dist"
    compile_site([], out, papers_dir=papers)
    assert "data-x-link" not in (out / "research" / "p" / "index.html").read_text()
    (papers / "p" / "media" / "social.json").write_text('{"x_url": "https://x.com/i/status/7"}')
    compile_site([], out, papers_dir=papers)
    page = (out / "research" / "p" / "index.html").read_text()
    assert 'href="https://x.com/i/status/7" rel="noopener" data-x-link>Thread on X' in page


def test_post_records_what_it_posted_and_link_writes_it(tmp_path, monkeypatch):
    from tests.test_cli_social import KEYS, FakeX

    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("openinterpretability.social.poster.XPoster", FakeX)
    FakeX.sent, FakeX.error = [], None
    feed = tmp_path / "papers.xml"
    feed.write_text('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
                    f"<id>{SITE}research/p/index.html</id><title>T</title>"
                    f'<link href="{SITE}research/p/index.html"/><summary>S.</summary>'
                    "</entry></feed>")  # fmt: skip
    state = tmp_path / "x.json"
    social.save_state(state, social.State(feeds={"papers", "posts"}))
    record = tmp_path / "posted.json"
    args = ["social", "post", "--papers", str(feed), "--posts", str(tmp_path / "none.xml"),
            "--state", str(state), "--record", str(record)]  # fmt: skip
    assert CliRunner().invoke(cli, args).exit_code == 0
    (row,) = json.loads(record.read_text())
    assert row["entry"] == f"{SITE}research/p/index.html" and row["url"].startswith("https://x")
    (tmp_path / "papers" / "p").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["social", "link", str(record)])
    assert result.output.strip() == "papers/p/media/social.json"


def test_nothing_posted_writes_no_record(tmp_path):
    state = tmp_path / "x.json"
    social.save_state(state, social.State(feeds={"papers", "posts"}))
    record = tmp_path / "posted.json"
    args = ["social", "post", "--papers", str(tmp_path / "none.xml"), "--posts",
            str(tmp_path / "none.xml"), "--state", str(state), "--dry-run", "--record",
            str(record)]  # fmt: skip
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert not Path(record).exists()
