"""Delta-sync a built ``dist/`` directory to a Cloudflare R2 bucket over the S3 API.

Only files whose MD5 differs from the remote ETag are uploaded (``put_object``
always produces single-part uploads, whose ETag is the content MD5). Because the
builder is deterministic, an unchanged registry syncs as a no-op: nothing is
uploaded, deleted or purged.

* Upload order: content-addressed files, then other assets, then HTML, so a page
  never references an object that has not landed yet.
* Derived prefixes (``pagefind/``) are only synced when a source file changed. The
  Pagefind index is not byte-stable across runs, so syncing it alone would churn.
* Protected prefixes (``data/snapshots/``) are never deleted: snapshot URLs are
  permanent.
* With a Cloudflare zone configured, only changed short-cache URLs are purged.

Credentials come from ``CF_ACCOUNT_ID``, ``R2_ACCESS_KEY_ID`` and
``R2_SECRET_ACCESS_KEY``; purging uses ``CF_ZONE_ID`` and ``CF_API_TOKEN``.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from openinterp.core.exceptions import SyncError

SHORT_CACHE = "public, max-age=3600"
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
DERIVED_PREFIXES = ("pagefind/",)
PROTECTED_PREFIXES = ("data/snapshots/",)
PURGE_BATCH = 30  # Cloudflare's per-request limit for purge-by-URL

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".parquet": "application/vnd.apache.parquet",
    ".wasm": "application/wasm",
    ".js": "text/javascript",
    ".css": "text/css",
    ".svg": "image/svg+xml",
}
# Pagefind's index shards are content-addressed; its entry files (pagefind.js,
# pagefind-entry.json, pagefind-ui.*) are not and must stay on the short cache.
HASHED_NAME = re.compile(r"[_.-][0-9a-f]{7,}\.[A-Za-z0-9_]+$")
PAGEFIND_SHARD_SUFFIXES = {".pf_fragment", ".pf_index", ".pf_filter", ".pf_meta", ".pagefind"}


def cache_control_for(key: str) -> str:
    """Cache policy: immutable only for content-addressed files, one hour for everything else."""
    path = PurePosixPath(key)
    if key.startswith("data/snapshots/") or path.suffix in PAGEFIND_SHARD_SUFFIXES:
        return IMMUTABLE_CACHE
    if path.suffix not in {".html", ".json"} and HASHED_NAME.search(path.name):
        return IMMUTABLE_CACHE
    return SHORT_CACHE


def content_type_for(key: str) -> str:
    suffix = PurePosixPath(key).suffix
    return CONTENT_TYPES.get(suffix) or mimetypes.guess_type(key)[0] or "application/octet-stream"


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> R2Config:
        names = ("CF_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
        missing = [n for n in names if not env.get(n)]
        if missing:
            raise SyncError(f"Missing environment variables: {', '.join(missing)}")
        return cls(*(env[n] for n in names))


def make_client(config: R2Config) -> Any:
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name="auto",
    )


@dataclass
class SyncPlan:
    uploads: list[str] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped_derived: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.uploads or self.deletes)


def local_files(dist_dir: Path, prefix: str = "") -> dict[str, Path]:
    dist_dir = Path(dist_dir)
    return {
        prefix + path.relative_to(dist_dir).as_posix(): path
        for path in sorted(dist_dir.rglob("*"))
        if path.is_file() and not path.name.startswith(".")
    }


def remote_etags(client: Any, bucket: str, prefix: str = "") -> dict[str, str]:
    etags: dict[str, str] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            etags[obj["Key"]] = obj["ETag"].strip('"')
    return etags


def _upload_order(key: str) -> tuple[int, str]:
    if key.endswith(".html"):
        return (2, key)
    return (0 if cache_control_for(key) == IMMUTABLE_CACHE else 1, key)


def plan_sync(
    local: Mapping[str, Path],
    remote: Mapping[str, str],
    *,
    delete: bool = False,
    prefix: str = "",
    derived: Sequence[str] = DERIVED_PREFIXES,
    protected: Sequence[str] = PROTECTED_PREFIXES,
) -> SyncPlan:
    plan = SyncPlan()

    def rel(key: str) -> str:
        return key.removeprefix(prefix)

    for key, path in local.items():
        (plan.unchanged if remote.get(key) == file_md5(path) else plan.uploads).append(key)
    if delete:
        plan.deletes = sorted(
            k for k in set(remote) - set(local) if not rel(k).startswith(tuple(protected))
        )

    # Derived files only follow a change to a source file; on their own they are churn.
    is_derived = lambda k: rel(k).startswith(tuple(derived))  # noqa: E731
    if not any(not is_derived(k) for k in plan.uploads + plan.deletes):
        plan.skipped_derived = sorted(plan.uploads + plan.deletes)
        plan.unchanged += plan.uploads
        plan.uploads, plan.deletes = [], []

    plan.uploads.sort(key=lambda k: _upload_order(rel(k)))
    return plan


def sync_directory(
    dist_dir: Path,
    bucket: str,
    client: Any,
    *,
    prefix: str = "",
    delete: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = lambda _msg: None,
) -> SyncPlan:
    dist_dir = Path(dist_dir)
    if not dist_dir.is_dir():
        raise SyncError(f"{dist_dir} is not a directory; run `openinterp build` first")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    local = local_files(dist_dir, prefix)
    try:
        plan = plan_sync(local, remote_etags(client, bucket, prefix), delete=delete, prefix=prefix)
        for key in plan.skipped_derived:
            log(f"skip    {key}  (derived; no source file changed)")
        for key in plan.uploads:
            log(f"upload  {key}  [{cache_control_for(key.removeprefix(prefix))}]")
            if not dry_run:
                with local[key].open("rb") as body:
                    client.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=body,
                        ContentType=content_type_for(key),
                        CacheControl=cache_control_for(key.removeprefix(prefix)),
                    )
        for start in range(0, len(plan.deletes), 1000):  # DeleteObjects takes <= 1000 keys
            batch = plan.deletes[start : start + 1000]
            for key in batch:
                log(f"delete  {key}")
            if not dry_run:
                client.delete_objects(
                    Bucket=bucket, Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True}
                )
    except SyncError:
        raise
    except Exception as exc:  # botocore raises ClientError, EndpointConnectionError, ...
        raise SyncError(f"Sync to bucket {bucket!r} failed: {exc}") from exc
    return plan


def purge_targets(plan: SyncPlan, site_url: str, prefix: str = "") -> list[str]:
    """Public URLs whose cached copies are stale after ``plan`` ran.

    Content-addressed (immutable) objects get new names when they change, so only
    short-cache objects need purging.
    """
    base = site_url.rstrip("/") + "/"
    urls = []
    for key in plan.uploads + plan.deletes:
        rel = key.removeprefix(prefix.rstrip("/") + "/" if prefix else "")
        if cache_control_for(rel) == IMMUTABLE_CACHE:
            continue
        urls.append(base + rel)
        if rel == "index.html" or rel.endswith("/index.html"):
            urls.append(base + rel.removesuffix("index.html"))
    return urls


def purge_cloudflare_cache(
    zone_id: str,
    api_token: str,
    urls: Sequence[str],
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    """Purge ``urls`` from Cloudflare's edge cache (no-op for an empty list)."""
    for start in range(0, len(urls), PURGE_BATCH):
        request = urllib.request.Request(
            f"https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache",
            data=json.dumps({"files": list(urls[start : start + PURGE_BATCH])}).encode(),
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener(request, timeout=30) as response:
                body = json.load(response)
        except Exception as exc:
            raise SyncError(f"Cloudflare cache purge failed: {exc}") from exc
        if not body.get("success"):
            raise SyncError(f"Cloudflare cache purge failed: {body.get('errors')}")
