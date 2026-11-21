"""Posting to X: refused figures, video processing and text-only posts (social/poster.py)."""

import io

import pytest

from openinterpretability.social import poster
from openinterpretability.social.poster import XPoster

KEYS = {"X_API_KEY": "ck", "X_API_SECRET": "cs", "X_ACCESS_TOKEN": "at", "X_ACCESS_SECRET": "as"}


class _Scripted(XPoster):
    """A poster whose API calls get the given replies, in order, and are recorded."""

    def __init__(self, replies):
        self.waits = []
        super().__init__(KEYS, sleep=self.waits.append)
        self.replies, self.calls = list(replies), []

    def _call(self, method, path, body=None, query=None):
        self.calls.append((method, path, body))
        return self.replies.pop(0)


def test_only_a_web_address_is_fetched():
    with pytest.raises(ValueError, match="not a web address: file:///etc/passwd"):
        _Scripted([]).upload("file:///etc/passwd", "image/png", "")


def test_a_figure_larger_than_the_limit_is_refused(monkeypatch):
    monkeypatch.setattr(poster, "LARGEST", 4)
    monkeypatch.setattr(poster.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"x" * 9))
    with pytest.raises(ValueError, match="is larger than 4 bytes"):
        _Scripted([]).upload("https://site/f.png", "image/png", "")


def test_a_video_that_takes_too_long_to_process_is_refused():
    pending = {
        "data": {
            "id": "v1",
            "processing_info": {"state": "pending", "check_after_secs": poster.WAIT + 1},
        }
    }
    x = _Scripted([{"data": {"id": "v1"}}, {}, pending])
    with pytest.raises(ValueError, match=f"X took longer than {poster.WAIT} s"):
        x._video(b"abc", "video/mp4")
    assert x.waits == []


def test_a_video_that_x_cannot_process_is_refused():
    failed = {"data": {"id": "v1", "processing_info": {"state": "failed", "error": "codec"}}}
    x = _Scripted([{"data": {"id": "v1"}}, {}, failed])
    with pytest.raises(ValueError, match="X could not process the video: .*codec"):
        x._video(b"abc", "video/mp4")


def test_a_post_with_no_figure_is_text_alone():
    x = _Scripted([{"data": {"id": "t1"}}])
    assert x.post("A paper") == "t1"
    assert x.calls == [("POST", poster.TWEETS, {"text": "A paper"})]
