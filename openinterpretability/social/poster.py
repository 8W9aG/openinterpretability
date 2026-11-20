"""The client that posts to X (issue #4463).

:class:`XPoster` posts with ``POST https://api.x.com/2/tweets``, signed with OAuth 1.0a in
user context (HMAC-SHA1), as the account's access token allows. The signature is built here
from the standard library, so the package gains no dependency. A JSON body is not part of
an OAuth 1.0a signature. ``social.run`` takes any function from text to a post id, so a
Bluesky or Mastodon client can take this one's place.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
import urllib.request

TWEETS = "https://api.x.com/2/tweets"
SECRETS = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")


def _quote(value: str) -> str:
    """``value`` percent-encoded as OAuth 1.0a asks, with only ``~`` left as it is.

    Args:
        value: The text to encode.
    """
    return urllib.parse.quote(value, safe="~")


def oauth_header(method: str, url: str, keys: dict[str, str], nonce: str, stamp: str) -> str:
    """The OAuth 1.0a ``Authorization`` header for a request with no query or form fields.

    Args:
        method: The HTTP method, such as ``POST``.
        url: The request's URL.
        keys: The four X secrets, by name (:data:`SECRETS`).
        nonce: A value used once.
        stamp: The time in seconds, as text.

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
    pairs = "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(params.items()))
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

    def __init__(self, keys: dict[str, str]):
        """Keep the secrets that sign each request.

        Args:
            keys: The four X secrets, by name (:data:`SECRETS`).
        """
        self.keys = keys

    def post(self, text: str) -> str:
        """Publish ``text`` as a post; return its id. An HTTP error raises.

        Args:
            text: The post.

        Returns:
            The id that X gives the new post.
        """
        header = oauth_header(
            "POST", TWEETS, self.keys, secrets.token_hex(16), str(int(time.time()))
        )
        request = urllib.request.Request(
            TWEETS, data=json.dumps({"text": text}).encode(), method="POST",
            headers={"Authorization": header, "Content-Type": "application/json"},
        )  # fmt: skip
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (a fixed https URL)
            return json.load(response)["data"]["id"]
