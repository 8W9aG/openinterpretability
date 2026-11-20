"""Posting new results to X, with no network (openinterpretability/social, #4463)."""

from __future__ import annotations

import pytest

from openinterpretability.builder.research_feeds import published_feed
from openinterpretability.social import run as social
from openinterpretability.social.feed import Entry, read_feed
from openinterpretability.social.message import (
    LIMIT,
    experiment_post,
    length,
    paper_post,
)
from openinterpretability.social.poster import oauth_header

SITE = "https://open-interpretability.com/"


def _unit(n: int, explanation: str = "Fires on dates.") -> Entry:
    url = f"{SITE}models/gpt2/L1/N{n}-positive/x.html"
    return Entry(url, f"gpt2/L1/N{n}-positive VERIFIED", url, explanation)


@pytest.fixture
def poster():
    sent: list[str] = []

    def post(text: str) -> str:
        sent.append(text)
        return str(len(sent))

    post.sent = sent  # type: ignore[attr-defined]
    return post


def test_the_first_run_seeds_the_state_and_posts_nothing(tmp_path, poster):
    state = tmp_path / "x.json"
    lines = social.run(state, [_unit(1), _unit(2)], [], SITE, poster, now=0)
    assert poster.sent == [] and "seeded 2" in lines[0]
    assert social.load_state(state).posted == {_unit(1).id, _unit(2).id}


def test_new_entries_are_posted_once_and_a_rerun_posts_nothing(tmp_path, poster):
    state = tmp_path / "x.json"
    social.run(state, [_unit(1)], [], SITE, poster, now=0)  # seed
    social.run(state, [_unit(2), _unit(1)], [], SITE, poster, now=10)
    assert len(poster.sent) == 1 and "N2-positive VERIFIED" in poster.sent[0]
    assert social.run(state, [_unit(2), _unit(1)], [], SITE, poster, now=20) == ["nothing new"]
    assert len(poster.sent) == 1


def _paper(published: bool):
    status = SimpleStatus(published="2026-11-20" if published else None)
    return SimplePaper("A bracket neuron", "It reads brackets. More text.", "p/index.html", status)


class SimpleStatus:
    def __init__(self, published):
        self.published, self.preprint = published, published is None


class SimplePaper:
    def __init__(self, title, summary, href, status):
        self.title, self.summary, self.href, self.status = title, summary, href, status


def test_a_preprint_waits_and_is_posted_once_published(tmp_path, poster):
    state, feed = tmp_path / "x.json", tmp_path / "published.xml"
    social.run(state, [], [], SITE, poster, now=0)  # seed, with no published paper yet
    feed.write_text(published_feed([_paper(published=False)], SITE, "OI"))
    assert read_feed(feed) == []  # a preprint is not in the feed, so nothing records it
    feed.write_text(published_feed([_paper(published=True)], SITE, "OI"))
    social.run(state, [], read_feed(feed), SITE, poster, now=10)
    assert len(poster.sent) == 1 and poster.sent[0].startswith(
        "A bracket neuron\n\nIt reads brackets."
    )
    social.run(state, [], read_feed(feed), SITE, poster, now=20)
    assert len(poster.sent) == 1


def test_many_new_units_get_one_digest(tmp_path, poster):
    state = tmp_path / "x.json"
    social.run(state, [], [], SITE, poster, now=0)
    many = [_unit(n) for n in range(social.PER_RUN + 2)]
    social.run(state, many, [], SITE, poster, now=10)
    assert poster.sent == [f"{social.PER_RUN + 2} new VERIFIED units in the registry.\n{SITE}"]
    assert social.load_state(state).posted >= {e.id for e in many}


def test_the_daily_limit_holds_the_rest_for_later(tmp_path, poster, monkeypatch):
    monkeypatch.setattr(social, "PER_DAY", 1)
    state = tmp_path / "x.json"
    social.run(state, [], [], SITE, poster, now=0)
    social.run(state, [_unit(1), _unit(2)], [], SITE, poster, now=10)
    assert len(poster.sent) == 1
    social.run(state, [_unit(1), _unit(2)], [], SITE, poster, now=10 + social.DAY)
    assert len(poster.sent) == 2  # the held one goes out the next day


def test_a_dry_run_prints_and_changes_no_state(tmp_path, poster):
    state = tmp_path / "x.json"
    social.run(state, [], [], SITE, poster, now=0)
    before = state.read_text()
    lines = social.run(state, [_unit(3)], [], SITE, None, now=10)
    assert lines[0].startswith("would post:") and state.read_text() == before
    assert social.run(tmp_path / "none.json", [_unit(3)], [], SITE, None, now=0)[0].startswith(
        "seeded"
    )
    assert not (tmp_path / "none.json").exists()


def test_posts_fit_and_are_cut_at_a_word_boundary():
    long = "Fires on the closing bracket of a nested call " * 20
    text = experiment_post(_unit(4, long))
    assert length(text) <= LIMIT and "…" in text
    body = text.splitlines()[0]
    assert body.removesuffix("…").split()[-1] in long.split()  # whole words only
    paper = paper_post(Entry("i", "T", f"{SITE}p.html", "One sentence. Another."))
    assert paper == f"T\n\nOne sentence.\n{SITE}p.html"


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
    args = ["social", "post", "--experiments", none, "--papers", none, "--preview", "5"]
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0 and out.output == ""  # no default reads research/literature.xml
    args = ["social", "post", "--state", str(tmp_path / "s.json")]
    missing = CliRunner().invoke(cli, args, env={"X_API_KEY": ""})
    assert "secrets are not set" in missing.output
