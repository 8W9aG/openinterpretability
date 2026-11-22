"""Posting published papers and site posts to X, with no network (#4463, #5533)."""

from __future__ import annotations

import pytest

from openinterpretability.builder.research_feeds import published_feed
from openinterpretability.social import run as social
from openinterpretability.social.feed import Entry, read_feed
from openinterpretability.social.message import (
    LIMIT,
    compose,
    length,
    paper_post,
    site_post,
)
from openinterpretability.social.poster import oauth_header

SITE = "https://open-interpretability.com/"


def _post(n: int, summary: str = "A first paragraph.") -> Entry:
    url = f"{SITE}posts/p{n}/index.html"
    return Entry(url, f"Post {n}", url, summary)


def _feeds(*posts: Entry, papers=()) -> dict:
    return {"papers": list(papers), "posts": list(posts)}


@pytest.fixture
def poster():
    sent: list[str] = []

    def post(text: str, media=None) -> str:
        sent.append(text)
        post.media.append(media)  # type: ignore[attr-defined]
        return str(len(sent))

    post.media = []  # type: ignore[attr-defined]

    post.sent = sent  # type: ignore[attr-defined]
    return post


def test_the_first_run_seeds_the_state_and_posts_nothing(tmp_path, poster):
    state = tmp_path / "x.json"
    lines = social.run(state, _feeds(_post(1), _post(2)), poster, now=0)
    assert poster.sent == [] and "seeded 2" in lines[0]
    assert social.load_state(state).posted == {_post(1).id, _post(2).id}


def test_new_posts_are_posted_once_and_a_rerun_posts_nothing(tmp_path, poster):
    state = tmp_path / "x.json"
    social.run(state, _feeds(_post(1)), poster, now=0)  # seed
    social.run(state, _feeds(_post(2), _post(1)), poster, now=10)
    assert poster.sent == [f"Post 2\n\nA first paragraph.\n{_post(2).link}"]
    assert social.run(state, _feeds(_post(2), _post(1)), poster, now=20) == ["nothing new"]
    assert len(poster.sent) == 1


def test_a_verified_unit_never_posts():
    from click.testing import CliRunner

    from openinterpretability.cli import cli

    shown = CliRunner().invoke(cli, ["social", "post", "--help"]).output
    assert "--experiments" not in shown and "--papers" in shown and "--posts" in shown
    assert set(social.MESSAGES) == {"papers", "posts"}  # no feed of VERIFIED units


def test_a_state_from_before_5533_seeds_the_posts_feed(tmp_path, poster):
    state = tmp_path / "x.json"
    state.write_text('{"posted": ["https://x.org/old-paper"], "times": []}')  # no "feeds" key
    lines = social.run(state, _feeds(_post(1)), poster, now=0)
    assert poster.sent == [] and lines[0].startswith("seeded 1 entries of the new feed posts")
    assert social.load_state(state).feeds == {"papers", "posts"}
    social.run(state, _feeds(_post(2), _post(1)), poster, now=10)
    assert len(poster.sent) == 1 and poster.sent[0].startswith("Post 2")


def test_a_seen_entry_posts_once_with_repost(tmp_path, poster):
    state = tmp_path / "x.json"
    social.run(state, _feeds(_post(1)), poster, now=0)  # seed: post 1 is seen
    again = frozenset({_post(1).id, "https://example.org/gone"})
    lines = social.run(state, _feeds(_post(1)), poster, now=10, repost=again)
    assert lines[0] == "not in the feeds: https://example.org/gone"
    assert len(poster.sent) == 1 and poster.sent[0].startswith("Post 1")
    assert _post(1).id in social.load_state(state).posted
    assert social.run(state, _feeds(_post(1)), poster, now=20) == ["nothing new"]


def test_repost_on_a_dry_run_changes_no_state(tmp_path, poster):
    state = tmp_path / "x.json"
    social.run(state, _feeds(_post(1)), poster, now=0)
    before = state.read_text()
    lines = social.run(state, _feeds(_post(1)), None, now=10, repost=frozenset({_post(1).id}))
    assert lines[0].startswith("would post (") and state.read_text() == before


def _paper(published: bool):
    status = SimpleStatus(published="2026-11-20" if published else None)
    return SimplePaper("A bracket neuron", "It reads brackets. More text.", "p/index.html", status)


class SimpleStatus:
    def __init__(self, published):
        self.published, self.preprint = published, published is None


class SimplePaper:
    def __init__(self, title, summary, href, status):
        self.title, self.summary, self.href, self.status = title, summary, href, status
        self.social_image = self.social_video = None
        self.hero_alt = ""


def test_a_preprint_waits_and_is_posted_once_published(tmp_path, poster):
    state, feed = tmp_path / "x.json", tmp_path / "published.xml"
    social.run(state, _feeds(), poster, now=0)  # seed, with no published paper yet
    feed.write_text(published_feed([_paper(published=False)], SITE, "OI"))
    assert read_feed(feed) == []  # a preprint is not in the feed, so nothing records it
    feed.write_text(published_feed([_paper(published=True)], SITE, "OI"))
    social.run(state, _feeds(papers=read_feed(feed)), poster, now=10)
    assert len(poster.sent) == 1 and poster.sent[0].startswith(
        "A bracket neuron\n\nIt reads brackets."
    )
    social.run(state, _feeds(papers=read_feed(feed)), poster, now=20)
    assert len(poster.sent) == 1


def test_a_draft_post_never_posts(tmp_path, poster):
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.research_feeds import POSTS_FEED

    posts = tmp_path / "posts"
    (posts / "d").mkdir(parents=True)
    (posts / "d" / "README.md").write_text("# A draft\n\nNot yet.\n")
    (posts / "README.md").write_text("# Posts\n\n- Draft: [A draft](d/README.md)\n")
    compile_site([], tmp_path / "dist", posts_dir=posts)
    feed = read_feed(tmp_path / "dist" / POSTS_FEED)
    assert feed == []  # the build leaves drafts out of the feed that social post reads
    state = tmp_path / "x.json"
    social.run(state, _feeds(), poster, now=0)
    social.run(state, _feeds(*feed), poster, now=10)
    assert poster.sent == []


def test_the_daily_limit_holds_the_rest_for_later(tmp_path, poster, monkeypatch):
    monkeypatch.setattr(social, "PER_DAY", 1)
    state = tmp_path / "x.json"
    social.run(state, _feeds(), poster, now=0)
    social.run(state, _feeds(_post(1), _post(2)), poster, now=10)
    assert len(poster.sent) == 1
    social.run(state, _feeds(_post(1), _post(2)), poster, now=10 + social.DAY)
    assert len(poster.sent) == 2  # the held one goes out the next day


def test_a_dry_run_prints_and_changes_no_state(tmp_path, poster):
    state = tmp_path / "x.json"
    social.run(state, _feeds(), poster, now=0)
    before = state.read_text()
    lines = social.run(state, _feeds(_post(3)), None, now=10)
    assert lines[0].startswith("would post (") and state.read_text() == before
    assert social.run(tmp_path / "none.json", _feeds(_post(3)), None, now=0)[0].startswith("seeded")
    assert not (tmp_path / "none.json").exists()


def test_posts_fit_and_are_cut_at_a_word_boundary():
    long = "Fires on the closing bracket of a nested call " * 20
    text = compose("Post 4", long, _post(4).link)
    assert length(text) <= LIMIT and "…" in text
    body = text.splitlines()[0]
    assert body.removesuffix("…").split()[-1] in long.split()  # whole words only
    paper = paper_post(Entry("i", "T", f"{SITE}p.html", "One sentence. Another."))
    assert paper == f"T\n\nOne sentence.\n{SITE}p.html"
    post = site_post(Entry("s", "T", f"{SITE}s.html", "One paragraph. All of it."))
    assert post == f"T\n\nOne paragraph. All of it.\n{SITE}s.html"


def test_the_oauth_header_is_signed_and_stable():
    keys = {
        "X_API_KEY": "ck",
        "X_API_SECRET": "cs",
        "X_ACCESS_TOKEN": "at",
        "X_ACCESS_SECRET": "as",
    }
    one = oauth_header("POST", "https://api.x.com/2/tweets", keys, "nonce", "1")
    assert one == oauth_header("POST", "https://api.x.com/2/tweets", keys, "nonce", "1")
    assert one.startswith("OAuth ") and 'oauth_signature_method="HMAC-SHA1"' in one
    assert one != oauth_header("POST", "https://api.x.com/2/tweets", keys, "nonce", "2")


def test_a_literature_review_is_never_posted(tmp_path):
    from click.testing import CliRunner

    from openinterpretability.cli import cli

    none = str(tmp_path / "none.xml")
    args = ["social", "post", "--papers", none, "--posts", none, "--preview", "5"]
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0 and out.output == ""  # no default reads research/literature.xml
    args = ["social", "post", "--state", str(tmp_path / "s.json")]
    missing = CliRunner().invoke(cli, args, env={"X_API_KEY": ""})
    assert "secrets are not set" in missing.output
