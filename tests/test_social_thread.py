"""X threads for papers and posts: the file, its checks, the feed and the posting (#5833)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openinterpretability.builder.research import _thread
from openinterpretability.builder.research_feeds import published_feed
from openinterpretability.social import run as social
from openinterpretability.social.feed import Entry, Media, Tweet, read_feed
from openinterpretability.social.thread import THREAD, Prepared, load_thread, prepare, problems
from openinterpretability.social.thread import Tweet as ThreadTweet
from tests.test_social_poster import _Scripted

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://open-interpretability.com/"
README = (
    "# A paper\n\nFirst.\n\n## Method\n\n![A curve](media/c.svg)\n\n![](media/d.svg)\n\n## End\n"
)
ANCHORS = {"Method": "method", "End": "end"}


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "README.md").write_text(README)
    return tmp_path


def _tweets(n: int = 2, **last) -> list[ThreadTweet]:
    first = ThreadTweet("Method", "The method. {link}")
    end = ThreadTweet(last.get("section", "End"), last.get("text", "Read it: {link}"),
                      last.get("media", ()))  # fmt: skip
    return [first, *[ThreadTweet("Method", f"Step {i}.") for i in range(n - 2)], end]


def test_a_thread_that_keeps_every_rule_has_no_problem(folder):
    assert problems(folder, _tweets(), ANCHORS, {}) == []


@pytest.mark.parametrize(
    ("tweets", "drawn", "found"),
    [
        (_tweets(11), {}, "a thread has 2 to 10 tweets, not 11"),
        (_tweets(text="x " * 140 + "{link}"), {}, "over 280"),
        (_tweets(section="Nowhere"), {}, "tweet 2: no section 'Nowhere' in the README"),
        (_tweets(text="No link."), {}, "tweet 2 must link to the page: {link}"),
        (_tweets(text="Call 020 7946 0958. {link}"), {}, "the site screens out"),
        (_tweets(media=("e",)), {}, "tweet 2: no figure 'e' in the README"),
        (_tweets(media=("d",)), {"d": ["social-d.png"]}, "has no alt text"),
        (_tweets(media=("c",)), {}, "isn't drawn"),
        (_tweets(media=("c",) * 5), {"c": ["social-c.png"]}, "at most 4 images"),
        (_tweets(media=("c", "c")), {"c": ["social-c.png", "social-c.mp4"]}, "one video alone"),
    ],
)  # fmt: skip
def test_each_rule_names_the_tweet_that_breaks_it(folder, tweets, drawn, found):
    assert any(found in line for line in problems(folder, tweets, ANCHORS, drawn))


def test_a_thread_file_that_isnt_a_list_of_tweets_is_refused(tmp_path):
    (tmp_path / THREAD).write_text("text: alone\n")
    with pytest.raises(ValueError, match="a list of tweets"):
        load_thread(tmp_path)
    (tmp_path / THREAD).write_text("- section: Method\n")
    with pytest.raises(ValueError, match="needs 'text'"):
        load_thread(tmp_path)


def test_an_animated_figure_posts_its_video(folder):
    drawn = {"c": ["social-c.mp4", "social-c.png"]}
    (tweet,) = prepare(folder, [ThreadTweet("Method", "Hi", ("c",))], ANCHORS, drawn)
    assert tweet.media == (("social-c.mp4", "video/mp4", "A curve"),)


def test_the_build_fails_on_a_broken_thread_and_skips_a_folder_without_one(folder):
    assert _thread(folder, (("method", "Method"),)) == ()
    (folder / THREAD).write_text("- section: Method\n  text: Only one.\n")
    with pytest.raises(ValueError, match="a thread has 2 to 10 tweets, not 1"):
        _thread(folder, (("method", "Method"),))


def test_every_thread_in_the_repo_keeps_the_rules():
    from openinterpretability.builder.research import _read_paper

    found = sorted(ROOT.glob("papers/*/thread.yaml")) + sorted(ROOT.glob("posts/*/thread.yaml"))
    assert found, "the repo has at least one thread (#5833)"
    for path in found:
        paper = _read_paper(path.parent, "https://github.com/o/r", path.parent.parent.name)
        assert 2 <= len(paper.thread) <= 10


def test_the_feed_carries_the_thread_and_reads_it_back(tmp_path):
    from types import SimpleNamespace

    status = SimpleNamespace(published="2026-11-20", preprint=False)
    thread = (Prepared("Hook. {link}", "method", (("social-c.png", "image/png", "A curve"),)),
              Prepared("See {section_link}", "end"))  # fmt: skip
    paper = SimpleNamespace(title="A paper", href="p/index.html", summary="It works.",
                            status=status, social_image=None, social_video=None, hero_alt="",
                            thread=thread)  # fmt: skip
    feed = tmp_path / "published.xml"
    feed.write_text(published_feed([paper], SITE, "Site"))
    (entry,) = read_feed(feed)
    page = f"{SITE}research/p/index.html"
    assert entry.thread == (
        Tweet(f"Hook. {page}", (Media(f"{SITE}research/p/media/social-c.png", "image/png",
                                      "A curve"),)),
        Tweet(f"See {page}#end"),
    )  # fmt: skip


def _thread_entry(n: int = 3) -> Entry:
    tweets = tuple(Tweet(f"Tweet {i}") for i in range(1, n + 1))
    return Entry(f"{SITE}research/p/index.html", "A paper", f"{SITE}research/p/", "S", None, tweets)


class _Poster:
    """Posts each tweet, replying to the one before; fails on the tweet it is told to."""

    def __init__(self, fail_at: int | None = None):
        self.calls, self.fail_at = [], fail_at

    def __call__(self, text, media=None, reply_to=None):
        if len(self.calls) + 1 == self.fail_at:
            self.fail_at = None
            raise OSError("X is down")
        self.calls.append((text, reply_to))
        return f"id{len(self.calls)}"


def test_a_thread_posts_each_tweet_as_a_reply_to_the_one_before(tmp_path):
    state, entry = tmp_path / "x.json", _thread_entry()
    social.save_state(state, social.State(feeds={"papers"}))
    post = _Poster()
    lines = social.run(state, {"papers": [entry]}, post, now=0)
    assert post.calls == [("Tweet 1", None), ("Tweet 2", "id1"), ("Tweet 3", "id2")]
    assert lines == ["posted id1: Tweet 1", "posted id2: Tweet 2", "posted id3: Tweet 3"]
    saved = social.load_state(state)
    assert entry.id in saved.posted and saved.threads == {} and len(saved.times) == 3


def test_a_failed_tweet_resumes_from_the_next_one_and_never_repeats(tmp_path):
    state, entry = tmp_path / "x.json", _thread_entry()
    social.save_state(state, social.State(feeds={"papers"}))
    post = _Poster(fail_at=2)
    with pytest.raises(OSError):
        social.run(state, {"papers": [entry]}, post, now=0)
    assert social.load_state(state).threads == {entry.id: ["id1"]}
    social.run(state, {"papers": [entry]}, post, now=10)
    assert [text for text, _ in post.calls] == ["Tweet 1", "Tweet 2", "Tweet 3"]
    assert post.calls[1] == ("Tweet 2", "id1")  # it replies to the tweet already posted
    assert entry.id in social.load_state(state).posted


def test_a_thread_waits_until_the_whole_of_it_fits_todays_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(social, "PER_DAY", 4)
    state, entry = tmp_path / "x.json", _thread_entry(3)
    social.save_state(state, social.State(times=[0.0, 1.0], feeds={"papers"}))
    post = _Poster()
    lines = social.run(state, {"papers": [entry]}, post, now=5)
    assert post.calls == [] and lines[0].startswith("daily limit reached")


def test_a_dry_run_shows_each_tweet_with_its_length_and_media():
    media = Media("https://x.org/m.png", "image/png", "A curve")
    entry = Entry("e", "T", "l", "S", None, (Tweet("One https://x.org/p", (media,)), Tweet("Two")))
    text = social.preview(social.planned(social.State(), {"papers": [entry]})[0][1])
    assert text.splitlines()[:4] == ["would post a thread of 2:", "--- 1/2 (27 of 280)",
                                     "One https://x.org/p", "[image/png: A curve]"]  # fmt: skip


def test_a_reply_with_several_figures_carries_them_all():
    x = _Scripted([{"data": {"id": "m1"}}, {}, {"data": {"id": "m2"}}, {}, {"data": {"id": "t2"}}])
    x.upload = lambda url, kind, alt: {"a": "m1", "b": "m2"}[url]  # no network
    media = (Media("a", "image/png", "A"), Media("b", "image/png", "B"))
    x.replies = [{"data": {"id": "t2"}}]
    assert x.post("Two", media, reply_to="t1") == "t2"
    assert x.calls[-1][2] == {"text": "Two", "media": {"media_ids": ["m1", "m2"]},
                              "reply": {"in_reply_to_tweet_id": "t1"}}  # fmt: skip
