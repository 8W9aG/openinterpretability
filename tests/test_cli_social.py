"""``open-interpretability social post``: the preview, a post through X, and a failed post."""

import pytest
from click.testing import CliRunner

from openinterpretability.builder.research_feeds import published_feed
from openinterpretability.cli import cli
from openinterpretability.social import poster as x
from openinterpretability.social import run as social
from tests.test_social import SITE, _paper

KEYS = {name: "k" for name in x.SECRETS}


@pytest.fixture
def feed(tmp_path):
    path = tmp_path / "published.xml"
    path.write_text(published_feed([_paper(published=True)], SITE, "OI"))
    return path


def _args(tmp_path, feed, *more):
    none = str(tmp_path / "none.xml")
    return ["social", "post", "--papers", str(feed), "--posts", none, *more]


def test_the_preview_prints_each_post_with_its_length(tmp_path, feed):
    out = CliRunner().invoke(cli, _args(tmp_path, feed, "--preview", "1"))
    assert out.exit_code == 0, out.output
    head, text = out.output.split("\n", 1)
    assert head.startswith("would post (") and head.endswith(" of 280):")
    assert text.startswith("A bracket neuron\n\nIt reads brackets.")


class FakeX:
    """X's poster, with what it would post kept here and no network."""

    sent: list[str] = []
    error: OSError | None = None

    def __init__(self, keys):
        assert keys == KEYS

    def post(self, text, media=None):
        if FakeX.error is not None:
            raise FakeX.error
        FakeX.sent.append(text)
        return "1"


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    """A state that has seen no entry yet, the X secrets set, and the fake poster."""
    state = tmp_path / "x.json"
    social.run(state, {"papers": [], "posts": []}, lambda text, media=None: "0", now=0)
    monkeypatch.setattr(x, "XPoster", FakeX)
    monkeypatch.setattr(FakeX, "sent", [])
    monkeypatch.setattr(FakeX, "error", None)
    return state


def test_a_new_entry_is_posted_through_x(tmp_path, feed, seeded):
    out = CliRunner().invoke(cli, _args(tmp_path, feed, "--state", str(seeded)), env=KEYS)
    assert out.exit_code == 0, out.output
    assert len(FakeX.sent) == 1 and FakeX.sent[0].startswith("A bracket neuron")
    assert out.output == "posted 1: A bracket neuron\n"


def test_a_failed_post_is_a_clean_error(tmp_path, feed, seeded):
    FakeX.error = OSError("HTTP 503")
    out = CliRunner().invoke(cli, _args(tmp_path, feed, "--state", str(seeded)), env=KEYS)
    assert out.exit_code == 1 and "posting failed: HTTP 503" in out.output
