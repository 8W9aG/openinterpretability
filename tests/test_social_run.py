"""The posting run of the social feeds (social/run.py)."""

from __future__ import annotations

from openinterpretability.social import run as social
from openinterpretability.social.feed import Entry


def test_a_dry_run_seeds_a_new_feed_in_memory_and_leaves_the_state_file_alone(tmp_path):
    state = tmp_path / "x.json"
    before = '{"posted": ["https://x.org/old-paper"], "times": []}'  # it has seen only papers
    state.write_text(before)
    post = Entry("https://x.org/p1", "Post 1", "https://x.org/p1", "A first paragraph.")
    lines = social.run(state, {"papers": [], "posts": [post]}, None, now=0)
    assert lines == ["seeded 1 entries of the new feed posts; posted nothing", "nothing new"]
    assert state.read_text() == before
