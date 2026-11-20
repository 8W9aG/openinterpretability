"""Which entries to post, within the limits, and the state that remembers them (issue #4463).

The state is a small JSON file: the id of every entry that is posted or seen, and the time
of each post. An entry is *new* when its id isn't in the state; the clock never decides
that. The first run, with no state, records every entry in the feeds and posts nothing. So a
new poster doesn't post fifty old units at once.

- Each published paper gets its own post.
- New VERIFIED experiments get one post each, up to :data:`PER_RUN`. More than that in one
  run get one digest post that links to the site.
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
from openinterpretability.social.message import digest_post, experiment_post, paper_post

PER_RUN = 3
PER_DAY = 15
DAY = 24 * 60 * 60


@dataclass
class State:
    """The ids already posted or seen, and the times of the posts made."""

    posted: set[str] = field(default_factory=set)
    times: list[float] = field(default_factory=list)


def load_state(path: Path) -> State | None:
    """The state at ``path``, or None when there is none yet.

    Args:
        path: The state's JSON file.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "x.json"
        >>> load_state(path) is None
        True
        >>> save_state(path, State({"u2", "u1"}, [3.0]))
        >>> state = load_state(path)
        >>> sorted(state.posted), state.times  # a set's order varies from run to run
        (['u1', 'u2'], [3.0])
    """
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return State(set(data.get("posted", [])), list(data.get("times", [])))


def save_state(path: Path, state: State) -> None:
    """Write ``state`` to ``path``.

    Args:
        path: The state's JSON file. Its folder is made if it is missing.
        state: The state to write.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "a" / "x.json"
        >>> save_state(path, State({"u1"}, [1.0]))
        >>> print(path.read_text(), end="")
        {
         "posted": [
          "u1"
         ],
         "times": [
          1.0
         ]
        }
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"posted": sorted(state.posted), "times": sorted(state.times)}
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def planned(state: State, experiments: list[Entry], papers: list[Entry],
            site: str) -> list[tuple[list[str], str]]:  # fmt: skip
    r"""Each post to make, with the ids it covers: papers first, then experiments.

    Args:
        state: What was posted or seen already.
        experiments: The VERIFIED experiments' feed entries.
        papers: The published papers' feed entries.
        site: The site's link, for a digest post.

    Examples:
        >>> unit = Entry("u1", "gpt2/L1/N5-positive VERIFIED", "https://x.org/u", "Fires.")
        >>> planned(State({"old"}), [unit], [], "https://x.org/")
        [(['u1'], 'gpt2/L1/N5-positive VERIFIED: Fires.\nhttps://x.org/u')]
    """
    posts = [([e.id], paper_post(e)) for e in papers if e.id not in state.posted]
    fresh = [e for e in experiments if e.id not in state.posted]
    if len(fresh) > PER_RUN:
        posts.append(([e.id for e in fresh], digest_post(len(fresh), site)))
    else:
        posts += [([e.id], experiment_post(e)) for e in fresh]
    return posts


def run(state_path: Path, experiments: list[Entry], papers: list[Entry], site: str,
        post: Callable[[str], str] | None, now: float) -> list[str]:  # fmt: skip
    """Post what is new and record it; return a line per action.

    With ``post`` None (a dry run), print the posts and change no state.

    Args:
        state_path: The state's JSON file. With none yet, every entry is recorded as seen.
        experiments: The VERIFIED experiments' feed entries.
        papers: The published papers' feed entries.
        site: The site's link, for a digest post.
        post: A function that posts a text and returns its id, or None for a dry run.
        now: The time in seconds, for the daily limit.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "x.json"
        >>> unit = Entry("u1", "gpt2/L1/N5-positive VERIFIED", "https://x.org/u", "Fires.")
        >>> run(path, [unit], [], "https://x.org/", lambda text: "42", now=0)  # first run
        ['seeded 1 entries; posted nothing']
        >>> new = Entry("u2", "gpt2/L1/N6-positive VERIFIED", "https://x.org/v", "Reads.")
        >>> run(path, [new, unit], [], "https://x.org/", lambda text: "42", now=10)
        ['posted 42: gpt2/L1/N6-positive VERIFIED: Reads.']
    """
    state = load_state(state_path)
    if state is None:
        if post is not None:
            save_state(state_path, State({e.id for e in [*experiments, *papers]}))
        return [f"seeded {len(experiments) + len(papers)} entries; posted nothing"]
    lines = []
    budget = PER_DAY - sum(1 for t in state.times if now - t < DAY)
    for ids, text in planned(state, experiments, papers, site):
        if budget <= 0:
            lines.append("daily limit reached; the rest posts on a later run")
            break
        if post is None:
            lines.append(f"would post:\n{text}")
            budget -= 1
            continue
        lines.append(f"posted {post(text)}: {text.splitlines()[0]}")
        state.posted.update(ids)
        state.times.append(now)
        save_state(state_path, state)  # after each post, so a later failure loses nothing
        budget -= 1
    return lines or ["nothing new"]
