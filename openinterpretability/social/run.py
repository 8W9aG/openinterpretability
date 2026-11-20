"""Which entries to post, within the limits, and the state that remembers them (issue #4463).

The state is a small JSON file. It holds the id of every entry that is posted or seen, the
time of each post, and the feeds it has seen. An entry is *new* when its id isn't in the
state; the clock never decides that. The first run, with no state, records every entry in
the feeds and posts nothing. So a new poster doesn't post fifty old entries at once. A feed
that is new to the state seeds the same way, so adding a feed posts nothing old (#5533).

- Each published paper and each site post gets its own post (#5533). VERIFIED units don't
  post.
- At most :data:`PER_DAY` posts go out in any 24 hours. The X API's free tier allows 50 a
  day by its current docs, and 17 by older ones. This stays under the lower figure. When
  the budget is spent, the rest isn't recorded, and it posts on a later run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from openinterpretability.social.feed import Entry
from openinterpretability.social.message import paper_post, site_post

PER_DAY = 15
DAY = 24 * 60 * 60
# Each feed that posts, by name, and how its entries become posts. Papers go first.
MESSAGES: dict[str, Callable[[Entry], str]] = {"papers": paper_post, "posts": site_post}
# The feeds that a state saved before #5533 had seen: it has no "feeds" key.
LEGACY_FEEDS = frozenset({"papers"})


@dataclass
class State:
    """The ids already posted or seen, the times of the posts made, and the feeds seen."""

    posted: set[str] = field(default_factory=set)
    times: list[float] = field(default_factory=list)
    feeds: set[str] = field(default_factory=lambda: set(LEGACY_FEEDS))


def load_state(path: Path) -> State | None:
    """The state at ``path``, or None when there is none yet.

    A state saved before #5533 has no ``feeds`` key. It counts as having seen the papers
    feed only, so the posts feed seeds on its first run.

    Args:
        path: The state's JSON file.

    Returns:
        The state, or None when the file doesn't exist.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "x.json"
        >>> load_state(path) is None
        True
        >>> save_state(path, State({"u2", "u1"}, [3.0], {"papers", "posts"}))
        >>> state = load_state(path)
        >>> sorted(state.posted), state.times, sorted(state.feeds)  # a set's order varies
        (['u1', 'u2'], [3.0], ['papers', 'posts'])
        >>> _ = path.write_text('{"posted": ["u1"], "times": []}')  # from before #5533
        >>> sorted(load_state(path).feeds)
        ['papers']
    """
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    feeds = set(data.get("feeds", LEGACY_FEEDS))
    return State(set(data.get("posted", [])), list(data.get("times", [])), feeds)


def save_state(path: Path, state: State) -> None:
    """Write ``state`` to ``path``.

    Args:
        path: The state's JSON file. Its folder is made if it is missing.
        state: The state to write.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "a" / "x.json"
        >>> save_state(path, State({"u1"}, [1.0], {"papers"}))
        >>> print(path.read_text(), end="")
        {
         "feeds": [
          "papers"
         ],
         "posted": [
          "u1"
         ],
         "times": [
          1.0
         ]
        }
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"feeds": sorted(state.feeds), "posted": sorted(state.posted),
            "times": sorted(state.times)}  # fmt: skip
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def planned(state: State, feeds: dict[str, list[Entry]]) -> list[tuple[str, str]]:
    r"""Each post to make, with the id it covers: papers first, then site posts.

    Args:
        state: What was posted or seen already.
        feeds: Each feed's entries, by name (a key of :data:`MESSAGES`).

    Returns:
        One ``(id, text)`` pair per post.

    Examples:
        >>> post = Entry("p1", "Hello", "https://x.org/p", "The first post.")
        >>> planned(State({"old"}), {"posts": [post]})
        [('p1', 'Hello\n\nThe first post.\nhttps://x.org/p')]
    """
    return [(e.id, MESSAGES[name](e)) for name in MESSAGES for e in feeds.get(name, [])
            if e.id not in state.posted]  # fmt: skip


def run(state_path: Path, feeds: dict[str, list[Entry]], post: Callable[[str], str] | None,
        now: float, repost: frozenset[str] = frozenset()) -> list[str]:  # fmt: skip
    """Post what is new and record it; return a line per action.

    With ``post`` None (a dry run), print the posts and change no state. ``repost`` names
    entries to post again although the state has them, for example the ones that the first
    run recorded as seen (#5527). They post within the same limits, and the state keeps them
    as posted only after the post succeeds.

    Args:
        state_path: The state's JSON file. With none yet, every entry is recorded as seen.
        feeds: Each feed's entries, by name: ``papers`` and ``posts``.
        post: A function that posts a text and returns its id, or None for a dry run.
        now: The time in seconds, for the daily limit.
        repost: The ids of entries to post although the state has them.

    Returns:
        One line per action, such as ``posted <id>: <first line>``, or ``["nothing new"]``.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "x.json"
        >>> paper = Entry("p1", "A bracket neuron", "https://x.org/p", "It reads brackets.")
        >>> run(path, {"papers": [paper]}, lambda text: "42", now=0)  # the first run
        ['seeded 1 entries; posted nothing']
        >>> new = Entry("s1", "Hello", "https://x.org/s", "The first post.")
        >>> run(path, {"papers": [paper], "posts": [new]}, lambda text: "43", now=10)
        ['seeded 1 entries of the new feed posts; posted nothing', 'nothing new']
        >>> newer = Entry("s2", "Again", "https://x.org/t", "The second post.")
        >>> run(path, {"papers": [paper], "posts": [newer, new]}, lambda text: "44", now=20)
        ['posted 44: Again']
    """
    entries = [e for found in feeds.values() for e in found]
    state = load_state(state_path)
    if state is None:
        if post is not None:
            save_state(state_path, State({e.id for e in entries}, [], set(feeds)))
        lines = [f"seeded {len(entries)} entries; posted nothing"]
        return lines + ["--repost waits for a state: the first run only seeds"] * bool(repost)
    lines = []
    for name in sorted(set(feeds) - state.feeds):  # a feed new to the state seeds (#5533)
        state.posted.update(e.id for e in feeds[name])
        state.feeds.add(name)
        if post is not None:
            save_state(state_path, state)
        lines.append(f"seeded {len(feeds[name])} entries of the new feed {name}; posted nothing")
    lines += [f"not in the feeds: {i}" for i in sorted(repost - {e.id for e in entries})]
    state.posted -= repost  # in memory only: the state saves an id again once it posts
    budget = PER_DAY - sum(1 for t in state.times if now - t < DAY)
    count = 0
    for entry_id, text in planned(state, feeds):
        if budget <= 0:
            lines.append("daily limit reached; the rest posts on a later run")
            break
        count += 1
        budget -= 1
        if post is None:
            lines.append(f"would post:\n{text}")
            continue
        lines.append(f"posted {post(text)}: {text.splitlines()[0]}")
        state.posted.add(entry_id)
        state.times.append(now)
        save_state(state_path, state)  # after each post, so a later failure loses nothing
    return lines if count else [*lines, "nothing new"]
