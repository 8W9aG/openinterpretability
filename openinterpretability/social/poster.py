"""The client that posts to X (issue #4463).

:class:`XPoster` posts with ``POST https://api.x.com/2/tweets``, signed with OAuth 1.0a in
user context (HMAC-SHA1), as the account's access token allows. A post can carry the drawn
head figure of its paper or post (#5534). The poster downloads it from the site, and
uploads it with X's v2 media API. An image takes one request. A video takes initialize,
append and finalize, then a status poll. Then the poster sets the figure's alt text.

The signature is built here from the standard library, so the package gains no dependency.
A JSON body is not part of an OAuth 1.0a signature. ``social.run`` takes any function from
text to a post id, so a Bluesky or Mastodon client can take this one's place.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.x.com"
TWEETS = "/2/tweets"  # under API
SECRETS = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")
SEGMENT = 4 * 1024 * 1024  # a video's upload segment: X allows 4 MB in a JSON body
LARGEST = 64 * 1024 * 1024  # the largest drawn figure the poster downloads
WAIT = 300  # seconds to wait for X to process a video, at most
ALT = 1000  # X's limit on alt text


def _quote(value: str) -> str:
    """``value`` percent-encoded as OAuth 1.0a asks, with only ``~`` left as it is.

    Args:
        value: The text to encode.
    """
    return urllib.parse.quote(value, safe="~")


def oauth_header(method: str, url: str, keys: dict[str, str], nonce: str, stamp: str,
                 query: dict[str, str] | None = None) -> str:  # fmt: skip
    """The OAuth 1.0a ``Authorization`` header for a request with a JSON body or none.

    A JSON body isn't part of the signature. Query parameters are, so pass them as
    ``query``, and ``url`` without them.

    Args:
        method: The HTTP method, such as ``POST``.
        url: The request's URL, with no query.
        keys: The four X secrets, by name (:data:`SECRETS`).
        nonce: A value used once.
        stamp: The time in seconds, as text.
        query: The request's query parameters, or None.

    Returns:
        The header's value, which starts ``OAuth`` and holds the signature.

    Examples:
        >>> keys = dict(zip(SECRETS, ("ck", "cs", "at", "as")))
        >>> header = oauth_header("POST", TWEETS, keys, "n", "1")
        >>> header.startswith('OAuth oauth_consumer_key="ck"')
        True
        >>> header == oauth_header("POST", TWEETS, keys, "n", "1")  # stable for one nonce
        True
    """
    params = {"oauth_consumer_key": keys["X_API_KEY"], "oauth_nonce": nonce,
              "oauth_signature_method": "HMAC-SHA1", "oauth_timestamp": stamp,
              "oauth_token": keys["X_ACCESS_TOKEN"], "oauth_version": "1.0"}  # fmt: skip
    signed = {**params, **(query or {})}
    pairs = "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(signed.items()))
    base = "&".join((method.upper(), _quote(url), _quote(pairs)))
    key = f"{_quote(keys['X_API_SECRET'])}&{_quote(keys['X_ACCESS_SECRET'])}"
    digest = hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()
    params["oauth_signature"] = base64.b64encode(digest).decode()
    return "OAuth " + ", ".join(f'{_quote(k)}="{_quote(v)}"' for k, v in sorted(params.items()))


def keys_from_env() -> dict[str, str] | None:
    """The four X secrets from the environment, or None when any is missing.

    Returns:
        The secrets by name, or None when any of the four is unset or empty.

    Examples:
        >>> import os
        >>> for name in SECRETS:
        ...     os.environ.pop(name, None)
        >>> keys_from_env() is None
        True
    """
    keys = {name: os.environ.get(name, "") for name in SECRETS}
    return keys if all(keys.values()) else None


class XPoster:
    """Posts to X as the account that the access token belongs to."""

    def __init__(self, keys: dict[str, str], api: str = API, sleep=time.sleep):
        """Keep the secrets that sign each request.

        Args:
            keys: The four X secrets, by name (:data:`SECRETS`).
            api: The API's root URL. Tests give a local server.
            sleep: How to wait between status polls. Tests give one that doesn't.
        """
        self.keys, self.api, self.sleep = keys, api.rstrip("/"), sleep

    def _call(self, method: str, path: str, body: dict | None = None,
              query: dict[str, str] | None = None) -> dict:  # fmt: skip
        """Send one signed request with a JSON body, and return the JSON reply.

        Args:
            method: ``GET`` or ``POST``.
            path: The path under :attr:`api`, such as ``/2/tweets``.
            body: The JSON body, or None.
            query: The query parameters, or None.

        Returns:
            The reply's JSON, or an empty dict for an empty reply.
        """
        url = self.api + path
        header = oauth_header(method, url, self.keys, secrets.token_hex(16),
                              str(int(time.time())), query)  # fmt: skip
        full = f"{url}?{urllib.parse.urlencode(query)}" if query else url
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": header, "Content-Type": "application/json"}
        request = urllib.request.Request(full, data=data, method=method, headers=headers)  # noqa: S310
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 (the API's URL)
            raw = response.read()
        return json.loads(raw) if raw else {}

    def upload(self, url: str, kind: str, alt: str) -> str:
        """Download a drawn figure from the site, upload it to X, and return its media id.

        Args:
            url: The figure's URL on the site.
            kind: ``image/png`` or ``video/mp4``.
            alt: The figure's alt text.

        Returns:
            X's media id.

        Raises:
            ValueError: When the URL isn't a web address, the file is too large, or X can't
                process the video.
        """
        # The URL comes from the site's own feed, which this repo's build wrote. Only a web
        # address is fetched, never a file: or other scheme.
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"not a web address: {url}")
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            data = response.read(LARGEST + 1)
        if len(data) > LARGEST:
            raise ValueError(f"{url} is larger than {LARGEST} bytes")
        if not kind.startswith("video/"):
            media = {"media": base64.b64encode(data).decode(), "media_category": "tweet_image"}
            media_id = self._call("POST", "/2/media/upload", media)["data"]["id"]
        else:
            media_id = self._video(data, kind)
        if alt:
            alt_text = {"id": media_id, "metadata": {"alt_text": {"text": alt[:ALT]}}}
            self._call("POST", "/2/media/metadata", alt_text)
        return media_id

    def _video(self, data: bytes, kind: str) -> str:
        """Upload a video in segments and wait until X has processed it.

        Args:
            data: The video's bytes.
            kind: Its type, such as ``video/mp4``.

        Returns:
            X's media id.

        Raises:
            ValueError: When X can't process the video, or takes longer than :data:`WAIT`.
        """
        start = {"media_category": "tweet_video", "media_type": kind, "total_bytes": len(data)}
        media_id = self._call("POST", "/2/media/upload/initialize", start)["data"]["id"]
        for index, at in enumerate(range(0, len(data), SEGMENT)):
            part = base64.b64encode(data[at : at + SEGMENT]).decode()
            self._call("POST", f"/2/media/upload/{media_id}/append",
                       {"media": part, "segment_index": index})  # fmt: skip
        info = self._call("POST", f"/2/media/upload/{media_id}/finalize")["data"]
        waited = 0
        busy = ("pending", "in_progress")
        while (state := (info.get("processing_info") or {}).get("state")) in busy:
            delay = int(info["processing_info"].get("check_after_secs", 1))
            if waited + delay > WAIT:
                raise ValueError(f"X took longer than {WAIT} s to process the video")
            self.sleep(delay)
            waited += delay
            query = {"command": "STATUS", "media_id": media_id}
            info = self._call("GET", "/2/media/upload", query=query)["data"]
        if state == "failed":
            raise ValueError(f"X could not process the video: {info['processing_info']}")
        return media_id

    def post(self, text: str, media=None) -> str:
        """Publish ``text`` as a post, with its drawn figure when there is one; return its id.

        A failed upload doesn't stop the post, which goes out as text alone. The failure is
        printed, so the deploy's log shows it. A failed post raises.

        Args:
            text: The post.
            media: The :class:`~openinterpretability.social.feed.Media` to attach, or None.

        Returns:
            The id that X gives the new post.
        """
        body: dict = {"text": text}
        if media is not None:
            try:
                body["media"] = {"media_ids": [self.upload(media.url, media.type, media.alt)]}
            except (OSError, KeyError, ValueError) as error:  # an upload's failure, not the post's
                print(f"media upload failed, posting text alone: {error}", file=sys.stderr)
        return self._call("POST", TWEETS, body)["data"]["id"]
