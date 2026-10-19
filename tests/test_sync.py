import pytest

from openinterp.core.exceptions import SyncError
from openinterp.sync.r2 import (
    IMMUTABLE_CACHE,
    SHORT_CACHE,
    R2Config,
    cache_control_for,
    content_type_for,
    file_md5,
    sync_directory,
)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("index.html", SHORT_CACHE),
        ("neurons/m/L1/N2-positive.html", SHORT_CACHE),
        ("data/registry.parquet", SHORT_CACHE),
        ("data/manifest.json", SHORT_CACHE),
        ("data/snapshots/registry-0123456789abcdef.parquet", IMMUTABLE_CACHE),
        ("pagefind/pagefind-ui.js", SHORT_CACHE),
        ("pagefind/pagefind-entry.json", SHORT_CACHE),
        ("pagefind/fragment/en_3f2a1b9.pf_fragment", IMMUTABLE_CACHE),
        ("pagefind/wasm.en.pagefind", IMMUTABLE_CACHE),
        ("assets/app.4f9c2e1d.js", IMMUTABLE_CACHE),
    ],
)
def test_cache_control(key, expected):
    assert cache_control_for(key) == expected


def test_content_types():
    assert content_type_for("a/b.parquet") == "application/vnd.apache.parquet"
    assert content_type_for("index.html").startswith("text/html")


def test_config_from_env():
    cfg = R2Config.from_env(
        {"CF_ACCOUNT_ID": "acct", "R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s"}
    )
    assert cfg.endpoint_url == "https://acct.r2.cloudflarestorage.com"
    with pytest.raises(SyncError, match="R2_SECRET_ACCESS_KEY"):
        R2Config.from_env({"CF_ACCOUNT_ID": "acct", "R2_ACCESS_KEY_ID": "k"})


@pytest.fixture
def dist(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "index.html").write_text("<html>new</html>")
    (tmp_path / "data" / "registry.parquet").write_bytes(b"PAR1")
    (tmp_path / "about.html").write_text("same")
    return tmp_path


@pytest.fixture
def client(mocker, dist):
    client = mocker.Mock()
    remote = [
        {"Key": "about.html", "ETag": f'"{file_md5(dist / "about.html")}"'},
        {"Key": "index.html", "ETag": '"stale"'},
        {"Key": "old.html", "ETag": '"x"'},
    ]
    client.get_paginator.return_value.paginate.return_value = [{"Contents": remote}]
    return client


def test_delta_sync(dist, client):
    plan = sync_directory(dist, "bucket", client, delete=True)
    assert plan.unchanged == ["about.html"]
    assert plan.uploads == ["data/registry.parquet", "index.html"]  # assets before HTML
    assert plan.deletes == ["old.html"]

    calls = {c.kwargs["Key"]: c.kwargs for c in client.put_object.call_args_list}
    assert calls["index.html"]["CacheControl"] == SHORT_CACHE
    assert calls["index.html"]["ContentType"].startswith("text/html")
    assert calls["data/registry.parquet"]["ContentType"] == "application/vnd.apache.parquet"
    client.delete_objects.assert_called_once()


def test_dry_run_and_no_delete(dist, client):
    plan = sync_directory(dist, "bucket", client, dry_run=True)
    assert plan.deletes == []
    client.put_object.assert_not_called()
    client.delete_objects.assert_not_called()


def test_prefix(dist, client):
    client.get_paginator.return_value.paginate.return_value = [{}]
    plan = sync_directory(dist, "bucket", client, prefix="site")
    assert plan.uploads == ["site/data/registry.parquet", "site/about.html", "site/index.html"]
    assert client.get_paginator.return_value.paginate.call_args.kwargs["Prefix"] == "site/"


def test_client_errors_wrapped(dist, mocker):
    client = mocker.Mock()
    client.get_paginator.side_effect = RuntimeError("403")
    with pytest.raises(SyncError, match="403"):
        sync_directory(dist, "bucket", client)


def _remote(client, dist, keys):
    client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": k, "ETag": f'"{file_md5(dist / k)}"'} for k in keys]}
    ]


@pytest.fixture
def site(tmp_path):
    for rel, body in {
        "index.html": "<html>v1</html>",
        "data/snapshots/registry-0123456789abcdef.parquet": "PAR1",
        "pagefind/pagefind-entry.json": '{"hash": "en_aaaaaaa"}',
        "pagefind/pagefind.en_bbbbbbbbbb.pf_meta": "meta",
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return tmp_path


def test_unchanged_site_is_a_noop(site, mocker):
    client = mocker.Mock()
    _remote(client, site, [str(p.relative_to(site)) for p in site.rglob("*") if p.is_file()])
    plan = sync_directory(site, "bucket", client, delete=True)
    assert not plan.changed
    client.put_object.assert_not_called()
    client.delete_objects.assert_not_called()


def test_derived_only_changes_are_skipped(site, mocker):
    """A re-run of Pagefind on unchanged HTML must not upload or delete anything."""
    client = mocker.Mock()
    _remote(client, site, ["index.html", "data/snapshots/registry-0123456789abcdef.parquet"])
    client.get_paginator.return_value.paginate.return_value[0]["Contents"] += [
        {"Key": "pagefind/pagefind-entry.json", "ETag": '"old"'},
        {"Key": "pagefind/pagefind.en_cccccccccc.pf_meta", "ETag": '"old"'},
    ]
    plan = sync_directory(site, "bucket", client, delete=True)
    assert not plan.changed
    assert "pagefind/pagefind.en_cccccccccc.pf_meta" in plan.skipped_derived
    client.put_object.assert_not_called()


def test_derived_follows_source_changes(site, mocker):
    client = mocker.Mock()
    _remote(client, site, ["data/snapshots/registry-0123456789abcdef.parquet"])
    client.get_paginator.return_value.paginate.return_value[0]["Contents"] += [
        {"Key": "index.html", "ETag": '"stale"'},
        {"Key": "pagefind/pagefind.en_cccccccccc.pf_meta", "ETag": '"old"'},
        {"Key": "data/snapshots/registry-ffffffffffffffff.parquet", "ETag": '"old"'},
    ]
    plan = sync_directory(site, "bucket", client, delete=True)
    # Content-addressed first, then other assets, HTML last.
    assert plan.uploads == [
        "pagefind/pagefind.en_bbbbbbbbbb.pf_meta",
        "pagefind/pagefind-entry.json",
        "index.html",
    ]
    assert plan.deletes == ["pagefind/pagefind.en_cccccccccc.pf_meta"]  # old snapshot is kept


def test_purge_targets_only_short_cache():
    from openinterp.sync.r2 import SyncPlan, purge_targets

    plan = SyncPlan(
        uploads=[
            "data/snapshots/registry-0123456789abcdef.parquet",
            "data/registry.parquet",
            "index.html",
            "neurons/m/L1/N2-positive.html",
        ],
        deletes=["neurons/m/L1/N3-positive.html"],
    )
    assert purge_targets(plan, "https://interp.example.org/") == [
        "https://interp.example.org/data/registry.parquet",
        "https://interp.example.org/index.html",
        "https://interp.example.org/",
        "https://interp.example.org/neurons/m/L1/N2-positive.html",
        "https://interp.example.org/neurons/m/L1/N3-positive.html",
    ]
    assert purge_targets(SyncPlan(), "https://interp.example.org") == []


def test_purge_batches_and_errors(mocker):
    import io
    import json

    from openinterp.sync.r2 import purge_cloudflare_cache

    responses = []

    def opener(request, timeout):
        responses.append(json.loads(request.data))
        assert request.headers["Authorization"] == "Bearer tok"
        return io.BytesIO(b'{"success": true}')

    purge_cloudflare_cache("zone", "tok", [f"https://x.org/{i}" for i in range(65)], opener=opener)
    assert [len(r["files"]) for r in responses] == [30, 30, 5]

    purge_cloudflare_cache("zone", "tok", [], opener=opener)  # empty: no request
    assert len(responses) == 3

    with pytest.raises(SyncError, match="purge failed"):
        purge_cloudflare_cache(
            "zone",
            "tok",
            ["https://x.org/a"],
            opener=lambda r, timeout: io.BytesIO(b'{"success": false, "errors": ["bad"]}'),
        )
