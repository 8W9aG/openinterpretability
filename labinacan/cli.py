"""The commands that any project's site needs: ``deploy`` and ``social`` (#6628).

A project mounts them on its own CLI (:func:`site_commands`). ``deploy`` syncs the built site
to Cloudflare R2. ``social post`` reads the site's feeds of published papers and site
posts, and posts each new entry once (``labinacan/social``). The deploy workflow runs it
after each sync, with the X secrets. Without them it posts nothing and exits cleanly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import click


@click.group(name="social")
def social_group() -> None:
    """Post new published papers and site posts (#4463, #5533)."""


def _entries(path: Path | None) -> list:
    from labinacan.social.feed import read_feed

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
    from labinacan.site.research_feeds import POSTS_FEED, PUBLISHED_FEED
    from labinacan.social.poster import XPoster, keys_from_env
    from labinacan.social.run import State, planned, preview, run

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
            from labinacan.social.links import write_records

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
    from labinacan.social.links import link

    for path in link(posted):
        click.echo(path.as_posix())


def deploy_command(site_url_env: str, site_url: Callable[[], str | None]) -> click.Command:
    """The ``deploy`` command, with the project's site URL as the default.

    Args:
        site_url_env: The environment variable that sets the site's URL.
        site_url: Returns the project's site URL, or None.

    Returns:
        The command.

    Examples:
        >>> deploy_command("LAB_SITE_URL", lambda: None).name
        'deploy'
    """

    @click.command(name="deploy")
    @click.option("--dist-dir", default=Path("dist"), show_default=True,
                  type=click.Path(exists=True, file_okay=False, path_type=Path))  # fmt: skip
    @click.option("--bucket", required=True, help="R2 bucket name.")
    @click.option("--prefix", default="", help="Key prefix inside the bucket.")
    @click.option("--delete", is_flag=True, help="Delete remote objects missing locally.")
    @click.option("--dry-run", is_flag=True, help="Show what would change without uploading.")
    @click.option(
        "--site-url",
        envvar=site_url_env,
        default=site_url,
        help="Public URL of the bucket root (default: the project's site URL). With CF_ZONE_ID "
        "and CF_API_TOKEN set, the changed URLs are purged from Cloudflare's cache.",
    )
    def deploy(dist_dir, bucket, prefix, delete, dry_run, site_url):
        """Delta-sync the built site to Cloudflare R2 (a no-op when nothing changed).

        \f
        Args:
            dist_dir: The built site.
            bucket: The R2 bucket.
            prefix: The key prefix inside the bucket.
            delete: Delete remote objects that the site no longer has.
            dry_run: Show the changes, and upload nothing.
            site_url: The public URL of the bucket's root, for the cache purge.
        """
        _deploy(dist_dir, bucket, prefix, delete, dry_run, site_url)

    return deploy


def _deploy(dist_dir, bucket, prefix, delete, dry_run, site_url) -> None:
    """Sync ``dist_dir`` to the bucket, then purge the changed URLs from Cloudflare's cache.

    Args:
        dist_dir: The built site.
        bucket: The R2 bucket.
        prefix: The key prefix inside the bucket.
        delete: Delete remote objects that the site no longer has.
        dry_run: Show the changes, and upload nothing.
        site_url: The public URL of the bucket's root, for the cache purge.
    """
    import os

    from labinacan.sync.r2 import (
        R2Config,
        make_client,
        purge_cloudflare_cache,
        purge_targets,
        sync_directory,
    )

    client = make_client(R2Config.from_env())
    plan = sync_directory(dist_dir, bucket, client, prefix=prefix, delete=delete,
                          dry_run=dry_run, log=click.echo)  # fmt: skip
    verb = "Would upload" if dry_run else "Uploaded"
    click.echo(
        f"{verb} {len(plan.uploads)}, unchanged {len(plan.unchanged)}, "
        f"deleted {len(plan.deletes)}, derived skipped {len(plan.skipped_derived)}"
    )
    zone, token = os.environ.get("CF_ZONE_ID"), os.environ.get("CF_API_TOKEN")
    if not (site_url and zone and token):
        return
    urls = purge_targets(plan, site_url, prefix)
    if not urls:
        click.echo("Nothing changed; no cache purge.")
    elif dry_run:
        click.echo(f"Would purge {len(urls)} URLs")
    else:
        purge_cloudflare_cache(zone, token, urls)
        click.echo(f"Purged {len(urls)} URLs from the Cloudflare cache")


def site_commands(site_url_env: str, site_url: Callable[[], str | None]) -> list[click.Command]:
    """The commands to mount on a project's CLI: ``deploy`` and the ``social`` group.

    Args:
        site_url_env: The environment variable that sets the site's URL.
        site_url: Returns the project's site URL, or None.

    Returns:
        The commands.

    Examples:
        >>> [c.name for c in site_commands("LAB_SITE_URL", lambda: None)]
        ['deploy', 'social']
    """
    return [deploy_command(site_url_env, site_url), social_group]
