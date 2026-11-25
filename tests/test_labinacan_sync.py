"""labinacan's sync to R2: keys under a prefix, cache rules, deletes and their batches (#6628)."""

from labinacan.sync.r2 import IMMUTABLE_CACHE, SHORT_CACHE, file_md5, sync_directory


class FakeR2:
    """A bucket that records each list, upload and delete."""

    def __init__(self, remote):
        self.remote, self.calls = remote, []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):  # noqa: N803 (boto3's names)
        self.calls.append(("list", Bucket, Prefix))
        return [{"Contents": [{"Key": k, "ETag": f'"{e}"'} for k, e in self.remote.items()]}]

    def put_object(self, **kw):
        self.calls.append(("put", kw["Bucket"], kw["Key"], kw["CacheControl"]))

    def delete_objects(self, **kw):
        self.calls.append(("delete", kw["Bucket"], kw["Delete"]))


def _site(tmp_path):
    site = tmp_path / "dist"
    (site / "data" / "snapshots").mkdir(parents=True)
    (site / "index.html").write_text("home")
    (site / "data" / "snapshots" / "r.parquet").write_text("rows")
    return site


def _unchanged(site):
    """The bucket's ETags for the site's two files as they are: nothing to upload."""
    snapshot = site / "data" / "snapshots" / "r.parquet"
    return {
        "index.html": file_md5(site / "index.html"),
        "data/snapshots/r.parquet": file_md5(snapshot),
    }


def test_a_prefix_gets_its_slash_and_each_key_its_cache_rule(tmp_path):
    site, logged = _site(tmp_path), []
    remote = {"p/old.html": "0", "p/data/snapshots/old.parquet": "0"}  # the second is protected
    client = FakeR2(remote)
    plan = sync_directory(site, "b", client, prefix="p", delete=True, log=logged.append)
    assert client.calls[0] == ("list", "b", "p/")
    assert logged == [
        f"upload  p/data/snapshots/r.parquet  [{IMMUTABLE_CACHE}]",
        f"upload  p/index.html  [{SHORT_CACHE}]",
        "delete  p/old.html",
    ]
    assert client.calls[1:] == [
        ("put", "b", "p/data/snapshots/r.parquet", IMMUTABLE_CACHE),
        ("put", "b", "p/index.html", SHORT_CACHE),
        ("delete", "b", {"Objects": [{"Key": "p/old.html"}], "Quiet": True}),
    ]
    assert plan.deletes == ["p/old.html"]


def test_a_dry_run_logs_and_changes_nothing(tmp_path):
    client = FakeR2({"old.html": "0"})
    sync_directory(_site(tmp_path), "b", client, delete=True, dry_run=True)
    assert [c[0] for c in client.calls] == ["list"]


def test_deletes_go_in_batches_of_a_thousand(tmp_path):
    site = _site(tmp_path)
    remote = {f"gone/{i:04}.html": "0" for i in range(1001)}
    remote |= _unchanged(site)
    client = FakeR2(remote)
    sync_directory(site, "b", client, delete=True)
    batches = [c[2]["Objects"] for c in client.calls if c[0] == "delete"]
    assert [len(b) for b in batches] == [1000, 1]
    assert batches[1] == [{"Key": "gone/1000.html"}]


def test_a_change_to_derived_files_alone_is_skipped_and_logged(tmp_path):
    site, logged = _site(tmp_path), []
    (site / "pagefind").mkdir()
    (site / "pagefind" / "x.js").write_text("index")
    remote = _unchanged(site)
    client = FakeR2(remote)
    sync_directory(site, "b", client, log=logged.append)
    assert logged == ["skip    pagefind/x.js  (derived; no source file changed)"]
    assert [c[0] for c in client.calls] == ["list"]


def test_a_prefix_that_ends_in_a_slash_keeps_one(tmp_path):
    client = FakeR2({})
    sync_directory(_site(tmp_path), "b", client, prefix="p/", dry_run=True)
    assert client.calls == [("list", "b", "p/")]
