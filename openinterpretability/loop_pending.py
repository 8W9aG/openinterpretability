"""Records that a loop's open PRs add, counted before they merge (#5826).

The loop lands a PR without waiting for CI. So the PR's records reach ``main`` later, or
never, because a contributor's PR waits for a maintainer. Each iteration starts from a
fresh ``origin/main``. Without help, the next draw would not see the records of the run so
far, and nor would the coverage meter of ``explain --run``. A VERIFIED unit would then be
drawn again.

:class:`PendingRecords` keeps the files that each landed PR added under ``wiki/``. While
the loop chooses its next draw, or measures, it puts them back as untracked files. Then it
removes them, so no experiment's commit holds another PR's records. A file that ``main``
holds already, because its PR merged, is left as it is.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from openinterpretability.core.paths import DEFAULT_WIKI_DIR


@dataclass
class PendingRecords:
    """The ``wiki/`` files of the loop's open PRs, by path relative to the worktree."""

    files: dict[str, bytes] = field(default_factory=dict)

    def keep(self, base: str = "origin/main") -> list[str]:
        """Keep the files that the checked-out branch adds under ``wiki/``, against ``base``.

        Args:
            base: The ref that the branch's PR merges into.

        Returns:
            The paths kept, relative to the worktree.

        Examples:
            >>> import os, tempfile
            >>> here = os.getcwd()
            >>> os.chdir(tempfile.mkdtemp())
            >>> git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c",
            ...     "user.email=t@t", *a], check=True, capture_output=True)
            >>> _ = git("init", "-q", "-b", "main"), git("commit", "-q", "--allow-empty", "-m", "a")
            >>> record = Path(DEFAULT_WIKI_DIR) / "rec.json"
            >>> _ = record.parent.mkdir(), record.write_text("{}")
            >>> _ = git("add", "."), git("commit", "-q", "-m", "b")
            >>> pending = PendingRecords()
            >>> pending.keep(base="HEAD~1"), pending.files
            (['wiki/rec.json'], {'wiki/rec.json': b'{}'})
            >>> os.chdir(here)
        """
        added = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=A", f"{base}...HEAD", "--",
             f"{DEFAULT_WIKI_DIR}/"],
            capture_output=True, text=True, check=True,
        ).stdout.split()  # fmt: skip
        for name in added:
            self.files[name] = Path(name).read_bytes()
        return added

    @contextmanager
    def applied(self) -> Iterator[list[Path]]:
        """Write the kept files that the worktree lacks, and remove them again on exit.

        Yields:
            The paths written, which are removed when the block ends.

        Examples:
            >>> import os, tempfile
            >>> here = os.getcwd()
            >>> os.chdir(tempfile.mkdtemp())
            >>> with PendingRecords({"wiki/a/rec.json": b"{}"}).applied() as written:
            ...     print(written, Path("wiki/a/rec.json").read_bytes())
            [PosixPath('wiki/a/rec.json')] b'{}'
            >>> Path("wiki/a/rec.json").exists()
            False
            >>> os.chdir(here)
        """
        written: list[Path] = []
        try:
            for name, data in self.files.items():
                path = Path(name)
                if path.exists():  # its PR merged, or another run wrote it
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                written.append(path)
            yield written
        finally:
            for path in written:
                path.unlink(missing_ok=True)
