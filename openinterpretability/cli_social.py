"""``open-interpretability social``: post new results to social networks (issue #4463).

``social post`` reads the site's feeds of VERIFIED experiments and published papers, and
posts each new entry once (``openinterpretability/social``). The deploy workflow runs it
after each sync, with the X secrets. Without them it posts nothing and exits cleanly.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from openinterpretability.cli import cli


@cli.group(name="social")
def social_group() -> None:
    """Post new VERIFIED experiments and published papers (#4463)."""


def _entries(path: Path | None) -> list:
    from openinterpretability.social.feed import read_feed

    return read_feed(path) if path is not None and path.exists() else []


@social_group.command(name="post")
@click.option("--experiments", type=click.Path(path_type=Path), default=Path("dist/feed.xml"),
              show_default=True, help="The feed of VERIFIED experiments.")  # fmt: skip
@click.option("--papers", type=click.Path(path_type=Path),
              default=Path("dist/research/published.xml"), show_default=True,
              help="The feed of published papers only.")  # fmt: skip
@click.option("--state", type=click.Path(path_type=Path), default=Path(".social/x.json"),
              show_default=True, help="The ids posted so far.")  # fmt: skip
@click.option("--site", default=None, help="The link of a digest post. Default: the site.")
@click.option("--dry-run", is_flag=True, help="Print the posts; post nothing, change no state.")
@click.option(
    "--preview",
    type=int,
    default=0,
    help="Print the posts of the newest N entries of each feed, whatever the state.",
)
def post_command(experiments, papers, state, site, dry_run, preview) -> None:
    """Post each new entry of the feeds once, within the per-run and daily limits.

    Args:
        experiments: The feed of VERIFIED experiments.
        papers: The feed of published papers.
        state: The JSON file of the ids posted or seen so far.
        site: The link for a digest post, or None for the site's own.
        dry_run: Print the posts, and post nothing.
        preview: Print the posts of this many newest entries of each feed, or 0.

    Raises:
        click.ClickException: When a post fails, for example on an HTTP or network error.
    """
    from openinterpretability.core.project import site_url
    from openinterpretability.social.message import experiment_post, length, paper_post
    from openinterpretability.social.poster import XPoster, keys_from_env
    from openinterpretability.social.run import run

    found, published = _entries(experiments), _entries(papers)
    if preview:
        posts = [paper_post(e) for e in published[:preview]]
        posts += [experiment_post(e) for e in found[:preview]]
        for text in posts:
            click.echo(f"--- {length(text)} of 280\n{text}")
        return
    keys = keys_from_env()
    if keys is None and not dry_run:
        click.echo("The X secrets are not set, so nothing is posted.")
        return
    poster = None if dry_run else XPoster(keys).post
    try:
        lines = run(state, found, published, site or site_url(), poster, time.time())
    except OSError as error:  # an HTTP or network error: posts so far stay recorded
        raise click.ClickException(f"posting failed: {error}") from error
    click.echo("\n".join(lines))
