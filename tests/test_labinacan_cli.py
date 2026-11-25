"""labinacan's commands: ``deploy`` and the mounting of ``social`` (#6628)."""

import click
from click.testing import CliRunner

from labinacan import cli as lab
from labinacan.sync import r2
from labinacan.sync.r2 import SyncPlan


def test_site_commands_give_deploy_the_projects_site_url():
    def url():
        return "https://x.org/"

    deploy, social = lab.site_commands("LAB_URL", url)
    option = next(p for p in deploy.params if p.name == "site_url")
    assert (option.envvar, option.default) == ("LAB_URL", url)
    assert social is lab.social_group


def _deploy(monkeypatch, tmp_path, env, urls):
    """Run ``deploy`` with a stand-in for R2 and Cloudflare; return what each got."""
    seen = {}
    monkeypatch.setattr(r2.R2Config, "from_env", classmethod(lambda cls: "config"))
    monkeypatch.setattr(r2, "make_client", lambda config: seen.update(client=config) or "client")

    def sync(dist, bucket, client, *, prefix, delete, dry_run, log):
        seen["sync"] = (dist, bucket, client, prefix, delete, dry_run, log)
        return SyncPlan(uploads=["p/a.html"])

    monkeypatch.setattr(r2, "sync_directory", sync)

    def targets(plan, site_url, prefix):
        seen["targets"] = (site_url, prefix)
        return urls

    def purge(zone, token, found):
        seen["purge"] = (zone, token, found)

    monkeypatch.setattr(r2, "purge_targets", targets)
    monkeypatch.setattr(r2, "purge_cloudflare_cache", purge)
    group = click.Group(commands=lab.site_commands("LAB_URL", lambda: "https://x.org/"))
    args = ["deploy", "--dist-dir", str(tmp_path), "--bucket", "b", "--prefix", "p/", "--delete"]
    result = CliRunner().invoke(group, args, env=env)
    return result, seen


def test_deploy_syncs_then_purges_the_changed_urls(monkeypatch, tmp_path):
    env = {"CF_ZONE_ID": "z", "CF_API_TOKEN": "t"}
    result, seen = _deploy(monkeypatch, tmp_path, env, ["https://x.org/p/a.html"])
    assert result.exit_code == 0, result.output
    assert seen["client"] == "config"
    assert seen["sync"] == (tmp_path, "b", "client", "p/", True, False, click.echo)
    assert seen["targets"] == ("https://x.org/", "p/")
    assert seen["purge"] == ("z", "t", ["https://x.org/p/a.html"])
    assert result.output == (
        "Uploaded 1, unchanged 0, deleted 0, derived skipped 0\n"
        "Purged 1 URLs from the Cloudflare cache\n"
    )


def test_deploy_says_when_nothing_needs_a_purge(monkeypatch, tmp_path):
    result, seen = _deploy(monkeypatch, tmp_path, {"CF_ZONE_ID": "z", "CF_API_TOKEN": "t"}, [])
    assert result.output.endswith("Nothing changed; no cache purge.\n") and "purge" not in seen


def test_deploy_purges_nothing_without_both_cloudflare_secrets(monkeypatch, tmp_path):
    for env in ({"CF_API_TOKEN": "t"}, {"CF_ZONE_ID": "z"}):
        result, seen = _deploy(monkeypatch, tmp_path, env, ["u"])
        assert result.exit_code == 0 and "targets" not in seen and "purge" not in seen
