"""``open-interpretability social``: post new results to social networks (issue #4463).

``social post`` reads the site's feeds of published papers and site posts, and posts each
new entry once (``openinterpretability/social``). The deploy workflow runs it
after each sync, with the X secrets. Without them it posts nothing and exits cleanly.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from openinterpretability.cli import cli


@cli.group(name="social")
def social_group() -> None:
    """Post new published papers and site posts (#4463, #5533)."""


def _entries(path: Path | None) -> list:
    from openinterpretability.social.feed import read_feed

    return read_feed(path) if path is not None and path.exists() else []


@social_group.command(name="post")
@click.option("--papers", type=click.Path(path_type=Path), default=None,
              help="The feed of published papers. Default: the build's.")  # fmt: skip
@click.option("--posts", type=click.Path(path_type=Path), default=None,
              help="The feed of site posts, never drafts. Default: the build's.")  # fmt: skip
@click.option("--state", type=click.Path(path_type=Path), default=Path(".social/x.json"),
              show_default=True, help="The ids posted so far.")  # fmt: skip
@click.option("--dry-run", is_flag=True, help="Print the posts; post nothing, change no state.")
@click.option(
    "--preview",
    "newest",
    type=int,
    default=0,
    help="Print the posts of the newest N entries of each feed, whatever the state.",
)
@click.option("--repost", multiple=True,
              help="Post this entry id although the state has it, as the first run seeds "
              "every entry (#5527). Repeat for more.")  # fmt: skip
@click.option("--record", type=click.Path(path_type=Path), default=None,
              help="Write the entries that this run posts, and their post URLs, here, for "
              "`social link` (#5840).")  # fmt: skip
def post_command(papers, posts, state, dry_run, newest, repost, record) -> None:
    """Post each new published paper and site post once (#5533, #6081).

    Args:
        papers: The feed of published papers, or None for the build's.
        posts: The feed of site posts, or None for the build's.
        state: The JSON file of the ids posted or seen so far.
        dry_run: Print the posts, and post nothing.
        newest: Print the posts of this many newest entries of each feed, or 0.
        repost: Entry ids to post although the state has them.
        record: Where to write this run's posts, or None.

    Raises:
        click.ClickException: When a post fails, for example on an HTTP or network error.
    """
    from openinterpretability.builder.research_feeds import POSTS_FEED, PUBLISHED_FEED
    from openinterpretability.social.poster import XPoster, keys_from_env
    from openinterpretability.social.run import State, planned, preview, run

    papers = papers or Path("dist") / PUBLISHED_FEED  # the paths the build writes
    posts = posts or Path("dist") / POSTS_FEED
    feeds = {"papers": _entries(papers), "posts": _entries(posts)}
    if newest:  # each tweet with its length and media, threads too (#5833)
        for _, tweets in planned(State(), {k: v[:newest] for k, v in feeds.items()}):
            click.echo(preview(tweets))
        return
    keys = keys_from_env()
    if keys is None and not dry_run:
        click.echo("The X secrets are not set, so nothing is posted.")
        return
    poster, posted = None if dry_run else XPoster(keys).post, []
    try:
        lines = run(state, feeds, poster, time.time(), frozenset(repost), posted)
    except OSError as error:  # an HTTP or network error: posts so far stay recorded
        raise click.ClickException(f"posting failed: {error}") from error
    finally:
        if record is not None and posted:  # what posted, even if a later post failed
            from openinterpretability.social.links import write_records

            write_records(record, posted)
    click.echo("\n".join(lines))


@social_group.command(name="link")
@click.argument("posted", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def link_command(posted) -> None:
    """Write each posted entry's X URL into its folder's media/social.json (#5840).

    POSTED is the file that `social post --record` wrote. The deploy runs this, and opens
    a PR with the files it changes.

    \f
    Args:
        posted: The file of this run's posts.
    """
    from openinterpretability.social.links import link

    for path in link(posted):
        click.echo(path.as_posix())
