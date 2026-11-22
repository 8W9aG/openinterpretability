"""A paper's or a post's X thread: one tweet per section, written with it (#5833).

``thread.yaml`` in a paper's or a post's folder lists the thread's tweets, in order. The
agent that writes the paper or post writes it in the same PR (the ``write-paper`` skill,
``docs/posts.md``). No person approves it: :func:`problems` is the gate, and the build fails
on any problem. Each tweet has:

- ``section``: the heading of the section it is about, as written in the ``README.md``.
- ``text``: the blurb. ``{link}`` becomes the page's URL, and ``{section_link}`` the
  section's. Each link counts as 23 characters, as X counts it.
- ``media`` (optional): the ids of the section's figures. A figure's id is its file's stem,
  ``trace`` for ``media/trace.svg``. ``scripts/social_images.py`` draws each one, and its
  alt text is the figure's.

The rules: :data:`FEWEST` to :data:`MOST` tweets, each within X's 280. The first and the last
tweet link to the page. At most :data:`IMAGES` images, or one video, in a tweet. The text
passes the site's screen for personal data and taken-down passages unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from openinterpretability.social.message import LIMIT, length

THREAD = "thread.yaml"
FEWEST, MOST = 2, 10  # tweets in a thread
IMAGES = 4  # X's limit on images in one tweet; a video goes alone
_FIGURE = re.compile(r"!\[([^\]]*)\]\(media/([\w.-]+)\.svg\)")


@dataclass(frozen=True)
class Tweet:
    """One tweet of a thread, as ``thread.yaml`` gives it."""

    section: str
    text: str
    media: tuple[str, ...] = ()


def load_thread(folder: Path) -> list[Tweet] | None:
    """The thread of ``folder``, or None when it has no ``thread.yaml``.

    Args:
        folder: A paper's or a post's folder.

    Returns:
        The tweets, in order.

    Raises:
        ValueError: The file isn't a list of tweets with a section and a text.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> load_thread(folder) is None
        True
        >>> _ = (folder / THREAD).write_text("- section: Method\\n  text: How. {link}\\n")
        >>> load_thread(folder)
        [Tweet(section='Method', text='How. {link}', media=())]
    """
    import yaml  # transformers needs PyYAML, so it is always installed

    path = folder / THREAD
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(t, dict) for t in data):
        raise ValueError(f"{path}: a thread is a list of tweets")
    try:
        return [Tweet(str(t["section"]), str(t["text"]).strip(),
                      tuple(str(m) for m in t.get("media") or ())) for t in data]  # fmt: skip
    except KeyError as exc:
        raise ValueError(f"{path}: every tweet needs {exc.args[0]!r}") from None


def figures(folder: Path) -> dict[str, str]:
    """The SVG figures of ``folder``'s ``README.md``: each id and its alt text.

    Args:
        folder: A paper's or a post's folder.

    Returns:
        The alt text of each figure, by id (the file's stem).

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "README.md").write_text("# T\\n\\n![A curve](media/trace.svg)\\n")
        >>> figures(folder)
        {'trace': 'A curve'}
    """
    readme = folder / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    return {name: alt for alt, name in _FIGURE.findall(text)}


def resolve(text: str, page: str, anchor: str | None) -> str:
    """``text`` with ``{link}`` as the page's URL and ``{section_link}`` as the section's.

    Args:
        text: The tweet's text.
        page: The page's URL.
        anchor: The section's id on the page, or None.

    Returns:
        The text to post.

    Examples:
        >>> resolve("Read it: {section_link}", "https://x.org/p/", "method")
        'Read it: https://x.org/p/#method'
    """
    section = f"{page}#{anchor}" if anchor else page
    return text.replace("{section_link}", section).replace("{link}", page)


def problems(folder: Path, tweets: list[Tweet], anchors: dict[str, str],
             drawn: dict[str, list[str]]) -> list[str]:  # fmt: skip
    """Every rule that the thread breaks; an empty list when it keeps them all.

    Args:
        folder: The paper's or post's folder, whose ``README.md`` holds the figures.
        tweets: The thread.
        anchors: Each section's id on the page, by its heading.
        drawn: The files drawn for each figure, by id (``social_media.figure_files``).

    Returns:
        One line per problem, naming the tweet.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "README.md").write_text("# T\\n\\n## Method\\n")
        >>> problems(folder, [Tweet("Method", "Hello")], {"Method": "method"}, {})
        ['a thread has 2 to 10 tweets, not 1', 'tweet 1 must link to the page: {link}']
    """
    from openinterpretability.builder.pii import redact_text

    found = [] if FEWEST <= len(tweets) <= MOST else [
        f"a thread has {FEWEST} to {MOST} tweets, not {len(tweets)}"]  # fmt: skip
    alts = figures(folder)
    for n, tweet in enumerate(tweets, 1):
        text = resolve(tweet.text, "https://x.org/", anchors.get(tweet.section))
        if length(text) > LIMIT:
            found.append(f"tweet {n} is {length(text)} long, over {LIMIT}")
        if tweet.section not in anchors:
            found.append(f"tweet {n}: no section {tweet.section!r} in the README")
        if n in (1, len(tweets)) and "{link}" not in tweet.text:
            found.append(f"tweet {n} must link to the page: {{link}}")
        if redact_text(tweet.text) != tweet.text:
            found.append(f"tweet {n} holds text that the site screens out")
        found += _media_problems(n, tweet.media, alts, drawn)
    return found


def _media_problems(n: int, ids: tuple[str, ...], alts: dict[str, str],
                    drawn: dict[str, list[str]]) -> list[str]:  # fmt: skip
    """The problems with one tweet's figures.

    Args:
        n: The tweet's number, for the message.
        ids: Its figures' ids.
        alts: Each figure's alt text, by id.
        drawn: The files drawn for each figure, by id.

    Returns:
        One line per problem.
    """
    found = []
    for fig in ids:
        if fig not in alts:
            found.append(f"tweet {n}: no figure {fig!r} in the README")
        elif not alts[fig].strip():
            found.append(f"tweet {n}: figure {fig!r} has no alt text")
        elif not drawn.get(fig):
            found.append(f"tweet {n}: figure {fig!r} isn't drawn: run scripts/social_images.py")
    videos = sum(any(f.endswith(".mp4") for f in drawn.get(fig, [])) for fig in ids)
    if len(ids) > IMAGES or (videos and len(ids) > 1):
        found.append(f"tweet {n}: at most {IMAGES} images, or one video alone")
    return found


@dataclass(frozen=True)
class Prepared:
    """A checked tweet, ready for the feed: its text, its section's id and its drawn media."""

    text: str  # still with {link} and {section_link}, which the feed fills in
    anchor: str | None
    media: tuple[tuple[str, str, str], ...] = ()  # (name in media/, type, alt) of each figure


def prepare(folder: Path, tweets: list[Tweet], anchors: dict[str, str],
            drawn: dict[str, list[str]]) -> tuple[Prepared, ...]:  # fmt: skip
    """The thread's tweets with their sections' ids and their figures' drawn files.

    A figure that animates posts its video, and any other figure its image.

    Args:
        folder: The paper's or post's folder.
        tweets: The checked thread (:func:`problems` found nothing).
        anchors: Each section's id on the page, by its heading.
        drawn: The files drawn for each figure, by id.

    Returns:
        One :class:`Prepared` per tweet.

    Examples:
        >>> import tempfile
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "README.md").write_text("# T\\n\\n![A curve](media/c.svg)\\n")
        >>> prepare(folder, [Tweet("M", "Hi", ("c",))], {"M": "m"}, {"c": ["social-c.png"]})
        (Prepared(text='Hi', anchor='m', media=(('social-c.png', 'image/png', 'A curve'),)),)
    """
    alts = figures(folder)
    out = []
    for tweet in tweets:
        media = []
        for fig in tweet.media:
            files = drawn.get(fig, [])
            video = next((f for f in files if f.endswith(".mp4")), None)
            name = video or next(f for f in files if f.endswith(".png"))
            media.append((name, "video/mp4" if video else "image/png", alts[fig]))
        out.append(Prepared(tweet.text, anchors.get(tweet.section), tuple(media)))
    return tuple(out)
