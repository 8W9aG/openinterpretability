"""Before ``explain`` runs or records: reuse the best explanation of the prompt (#4503).

An explanation that already reaches the coverage asked for is never made again. ``explain``
reads the prompt's explanations first, and takes the one with the highest coverage it
measured (:func:`~openinterpretability.core.explanation_records.best`). Then:

- **done:** it reaches the target and has the level asked for. Nothing runs, and nothing
  is written. The command says which explanation it is, with its link.
- **level:** it reaches the target, but lacks the level (a long form, say). No experiments
  run. The new explanation only adds the level, and names the one it reuses.
- **run:** none reaches the target. The run goes on, and the new explanation names the best
  one so far as the one it builds on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Reuse:
    """What ``explain`` does about the prompt's earlier explanations (module docstring)."""

    action: str  # done, level or run
    previous: str | None = None  # the best earlier explanation's id
    message: str = ""

    @property
    def fields(self) -> dict:
        """The new record's ``previous`` and ``reused``."""
        return {"previous": self.previous, "reused": self.action == "level"}


def look_first(wiki_dir: Path, model: str, prompt: str, position: int, target: float | None,
               level: str, writes: bool) -> Reuse:  # fmt: skip
    """The :class:`Reuse` of an ``explain`` that writes, said aloud; a look reuses nothing."""
    import click

    from openinterpretability.explain_plan import TARGET

    if not writes:
        return Reuse("run")
    reuse = decide(wiki_dir, model, prompt, position, TARGET if target is None else target, level)
    if reuse.message:
        click.echo(reuse.message)
    return reuse


def decide(wiki_dir: Path, model: str, prompt: str, position: int, target: float,
           level: str) -> Reuse:  # fmt: skip
    """The :class:`Reuse` for an explanation of ``prompt`` to ``target``, at ``level``."""
    from openinterpretability.core.explanation_records import LEVELS, best, prompt_records

    found = best(prompt_records(wiki_dir, model, prompt, position))
    if found is None:
        return Reuse("run")
    coverage = found.prompt_coverage or 0.0
    said = f"{href(found)} reached {coverage:.1%} coverage, at the {found.level} level"
    if coverage < target:
        return Reuse("run", found.id, f"Builds on {said}; the target is {target:.0%}.")
    if LEVELS.index(found.level) >= LEVELS.index(level):
        return Reuse("done", found.id, f"Already explained: {said}, so nothing runs "
                     f"for the {target:.0%} target.")  # fmt: skip
    return Reuse("level", found.id, f"Reuses {said}. No experiments run; this only adds the "
                 f"{level} level.")  # fmt: skip


def href(record) -> str:
    """The explanation's page on the site."""
    from openinterpretability.builder.hrefs import explanation_href
    from openinterpretability.core.identifiers import model_slug
    from openinterpretability.core.project import project_links

    page = explanation_href(model_slug(record.model_name), record.prompt_id, record.id)
    return project_links()["site_url"] + page
