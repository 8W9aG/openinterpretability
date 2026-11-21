"""Syncing the site to R2 and purging Cloudflare's cache (sync/r2.py)."""

import pytest

from openinterpretability.core.exceptions import SyncError
from openinterpretability.sync.r2 import (
    R2Config,
    file_md5,
    make_client,
    purge_cloudflare_cache,
    sync_directory,
)


def test_the_client_talks_to_the_accounts_endpoint():
    client = make_client(R2Config("acct", "key", "secret"))  # no request is made
    assert client.meta.endpoint_url == "https://acct.r2.cloudflarestorage.com"
    assert client.meta.region_name == "auto"


def test_a_missing_build_directory_is_an_error(tmp_path, mocker):
    with pytest.raises(SyncError, match="is not a directory; run `open-interpretability build`"):
        sync_directory(tmp_path / "dist", "bucket", mocker.Mock())


def test_a_dry_run_logs_deletes_but_deletes_nothing(tmp_path, mocker):
    (tmp_path / "index.html").write_text("<html></html>")
    client = mocker.Mock()
    remote = [{"Key": "index.html", "ETag": f'"{file_md5(tmp_path / "index.html")}"'},
              {"Key": "old.html", "ETag": '"x"'}]  # fmt: skip
    client.get_paginator.return_value.paginate.return_value = [{"Contents": remote}]
    logged = []
    plan = sync_directory(tmp_path, "bucket", client, delete=True, dry_run=True, log=logged.append)
    assert plan.deletes == ["old.html"] and logged == ["delete  old.html"]
    client.delete_objects.assert_not_called()


def test_a_sync_error_from_the_client_is_not_wrapped_again(tmp_path, mocker):
    (tmp_path / "index.html").write_text("<html></html>")
    client = mocker.Mock()
    client.get_paginator.side_effect = SyncError("no bucket")
    with pytest.raises(SyncError) as raised:
        sync_directory(tmp_path, "bucket", client)
    assert str(raised.value) == "no bucket"


def test_a_purge_request_that_fails_is_a_sync_error():
    def opener(request, timeout):
        raise OSError("connection refused")

    with pytest.raises(SyncError, match="Cloudflare cache purge failed: connection refused"):
        purge_cloudflare_cache("zone", "tok", ["https://x.org/a"], opener=opener)
