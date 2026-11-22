"""Head figures on X and in link cards: drawing, feeds, cards and uploads (#5534)."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from openinterpretability.builder.social_media import stale
from openinterpretability.social.feed import Media, read_feed
from openinterpretability.social.poster import SEGMENT, XPoster

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"X_API_KEY": "ck", "X_API_SECRET": "cs", "X_ACCESS_TOKEN": "at", "X_ACCESS_SECRET": "as"}


def test_every_drawn_head_figure_is_up_to_date():
    from openinterpretability.builder.research import _ENTRY, DRAFT

    # A draft never posts, so it draws its head figure when it is published (#5851).
    index = (ROOT / "posts" / "README.md").read_text(encoding="utf-8")
    drafts = {slug for date, slug in _ENTRY.findall(index) if date == DRAFT}
    posts = [f for f in ROOT.glob("posts/*/README.md") if f.parent.name not in drafts]
    folders = [*ROOT.glob("papers/*/README.md"), *posts]
    problems = {str(f.parent.relative_to(ROOT)): stale(f.parent) for f in folders}
    assert not {k: v for k, v in problems.items() if v}, "run scripts/social_images.py"


def test_a_changed_figure_is_stale(tmp_path):
    from openinterpretability.builder.social_media import write_manifest

    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "c.svg").write_text("<svg/>")
    (tmp_path / "media" / "social.png").write_bytes(b"png")
    (tmp_path / "README.md").write_text("# T\n\n![C](media/c.svg)\n")
    write_manifest(tmp_path, "media/c.svg", ["social.png"])
    assert stale(tmp_path) is None
    (tmp_path / "media" / "c.svg").write_text("<svg><g/></svg>")
    assert "changed since it was drawn" in stale(tmp_path)


def _paper(root: Path, animated: bool) -> Path:
    papers = root / "papers"
    folder = papers / "p"
    (folder / "media").mkdir(parents=True)
    (folder / "media" / "f.svg").write_text("<svg/>")
    (folder / "media" / "social.png").write_bytes(b"png")
    if animated:
        (folder / "media" / "social.mp4").write_bytes(b"mp4")
    (folder / "README.md").write_text(
        "# A paper\n\nIts summary.\n\n**Status:** Published 2026-11-20. Final.\n\n"
        "![A curve that climbs](media/f.svg)\n"
    )
    (papers / "README.md").write_text("# Papers\n\n- 2026-11-20: [A paper](p/README.md)\n")
    return papers


@pytest.mark.parametrize("animated", [False, True])
def test_the_feed_and_the_link_card_carry_the_drawn_figure(tmp_path, animated):
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.research_feeds import PUBLISHED_FEED

    out = tmp_path / "dist"
    compile_site([], out, papers_dir=_paper(tmp_path, animated))
    [entry] = read_feed(out / PUBLISHED_FEED)
    name, kind = ("social.mp4", "video/mp4") if animated else ("social.png", "image/png")
    assert entry.media.url.endswith(f"research/p/media/{name}") and entry.media.type == kind
    assert entry.media.alt == "A curve that climbs"
    page = (out / "research" / "p" / "index.html").read_text()
    assert 'research/p/media/social.png">' in page and "assets/og.png" not in page
    assert (out / "research" / "p" / "media" / name).exists()


class _FakeX(BaseHTTPRequestHandler):
    """X's media and post endpoints, and a site that serves the drawn files."""

    calls: list = []
    files: dict = {}
    fail_upload = False

    def log_message(self, *args):  # quiet
        pass

    def _reply(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802 (the handler's interface)
        if self.path.startswith("/site/"):
            data = self.files[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.calls.append(("GET", self.path, None))
        self._reply({"data": {"id": "v1", "processing_info": {"state": "succeeded"}}})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"null")
        self.calls.append(("POST", self.path, body))
        assert self.headers["Authorization"].startswith("OAuth ")
        if self.path == "/2/media/upload":
            if self.fail_upload:
                return self._reply({"title": "Payment Required"}, status=402)
            return self._reply({"data": {"id": "i1"}})
        if self.path == "/2/media/upload/initialize":
            return self._reply({"data": {"id": "v1"}})
        if self.path.endswith("/finalize"):
            info = {"state": "pending", "check_after_secs": 1}
            return self._reply({"data": {"id": "v1", "processing_info": info}})
        if self.path == "/2/tweets":
            return self._reply({"data": {"id": "t1"}})
        return self._reply({"data": {}})


@pytest.fixture
def fake_x():
    _FakeX.calls, _FakeX.files, _FakeX.fail_upload = [], {}, False
    server = HTTPServer(("127.0.0.1", 0), _FakeX)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_an_image_uploads_with_its_alt_text_and_attaches(fake_x):
    _FakeX.files["/site/social.png"] = b"\x89PNG"
    poster = XPoster(KEYS, api=fake_x)
    media = Media(f"{fake_x}/site/social.png", "image/png", "A curve")
    assert poster.post("A paper", media) == "t1"
    paths = [c[1] for c in _FakeX.calls]
    assert paths == ["/2/media/upload", "/2/media/metadata", "/2/tweets"]
    upload, alt, tweet = (c[2] for c in _FakeX.calls)
    assert base64.b64decode(upload["media"]) == b"\x89PNG"
    assert upload["media_category"] == "tweet_image"
    assert alt == {"id": "i1", "metadata": {"alt_text": {"text": "A curve"}}}
    assert tweet == {"text": "A paper", "media": {"media_ids": ["i1"]}}


def test_a_video_uploads_in_segments_and_waits_for_processing(fake_x):
    video = b"x" * (SEGMENT + 10)  # two segments
    _FakeX.files["/site/social.mp4"] = video
    slept = []
    poster = XPoster(KEYS, api=fake_x, sleep=slept.append)
    poster.post("A paper", Media(f"{fake_x}/site/social.mp4", "video/mp4", ""))
    paths = [c[1] for c in _FakeX.calls]
    assert paths == ["/2/media/upload/initialize", "/2/media/upload/v1/append",
                     "/2/media/upload/v1/append", "/2/media/upload/v1/finalize",
                     "/2/media/upload?command=STATUS&media_id=v1", "/2/tweets"]  # fmt: skip
    init, first, second = (c[2] for c in _FakeX.calls[:3])
    assert init == {"media_category": "tweet_video", "media_type": "video/mp4",
                    "total_bytes": len(video)}  # fmt: skip
    assert (first["segment_index"], second["segment_index"]) == (0, 1)
    assert base64.b64decode(first["media"]) + base64.b64decode(second["media"]) == video
    assert slept == [1]
    assert _FakeX.calls[-1][2]["media"] == {"media_ids": ["v1"]}


def test_a_failed_upload_posts_the_text_alone(fake_x, capsys):
    _FakeX.files["/site/social.png"] = b"\x89PNG"
    _FakeX.fail_upload = True
    poster = XPoster(KEYS, api=fake_x)
    assert poster.post("A paper", Media(f"{fake_x}/site/social.png", "image/png", "")) == "t1"
    assert _FakeX.calls[-1][2] == {"text": "A paper"}
    assert "media upload failed" in capsys.readouterr().err


def test_the_status_query_is_signed():
    from openinterpretability.social.poster import oauth_header

    url = "https://api.x.com/2/media/upload"
    plain = oauth_header("GET", url, KEYS, "n", "1")
    with_query = oauth_header("GET", url, KEYS, "n", "1", {"command": "STATUS", "media_id": "v1"})
    assert plain != with_query  # query parameters enter the signature
