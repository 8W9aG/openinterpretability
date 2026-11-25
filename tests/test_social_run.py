"""The posting run of the social feeds (labinacan/social/run.py)."""

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
