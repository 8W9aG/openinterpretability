"""The posting run of the social feeds (labinacan.social.run)."""

from __future__ import annotations

from labinacan.social import run as social
from labinacan.social.feed import Entry


def test_a_dry_run_seeds_a_new_feed_in_memory_and_leaves_the_state_file_alone(tmp_path):
    state = tmp_path / "x.json"
    before = '{"posted": ["https://x.org/old-paper"], "times": []}'  # it has seen only papers
    state.write_text(before)
    post = Entry("https://x.org/p1", "Post 1", "https://x.org/p1", "A first paragraph.")
    lines = social.run(state, {"papers": [], "posts": [post]}, None, now=0)
    assert lines == ["seeded 1 entries of the new feed posts; posted nothing", "nothing new"]
    assert state.read_text() == before


def test_a_preview_shows_a_tweets_figure_and_nothing_without_one():
    from types import SimpleNamespace

    from labinacan.social.run import preview

    figure = SimpleNamespace(type="image/png", alt="A curve")
    assert preview([("Hi", figure)]).endswith("Hi\n[image/png: A curve]")
    assert preview([("Hi", None)]).endswith(":\nHi")


def test_the_first_run_says_that_a_repost_waits_for_a_state(tmp_path):
    # #6631: --repost on a first run only seeds, and says so.
    from labinacan.social.feed import Entry
    from labinacan.social.run import run

    paper = Entry("p1", "A paper", "https://x.org/p", "It reads brackets.")
    lines = run(
        tmp_path / "x.json", {"papers": [paper]}, lambda t, m: "1", now=0, repost=frozenset({"p1"})
    )
    assert lines[-1] == "--repost waits for a state: the first run only seeds"
    assert run(tmp_path / "y.json", {"papers": [paper]}, lambda t, m: "1", now=0)[-1] != lines[-1]
