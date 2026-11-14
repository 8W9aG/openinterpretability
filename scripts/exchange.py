"""Publish what papers and analyses read from ``exchange/``, and check that they do (#3255).

``exchange/`` is git-ignored working space. Anything in it that a paper, an analysis or a
script reads must be on a public Hugging Face dataset, at a pinned revision. That dataset is
the one that ``data/exchange-manifest.json`` names. Then others can reproduce the result.

    uv run python scripts/exchange.py stage            # copy the cited inputs, hash them
    uv run python scripts/exchange.py upload           # publish the staged copy, pin it
    uv run python scripts/exchange.py fetch [dir ...]  # download pinned files to exchange/
    uv run python scripts/exchange.py check            # CI: every cited input is pinned

``stage`` writes the copies to a staging folder. In text files it rewrites the local home
directory to ``~``, so no published file holds a local path. The manifest hashes those
copies, and ``fetch`` checks them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

MANIFEST = Path("data/exchange-manifest.json")
EXCHANGE = Path("exchange")
STAGING = Path("exchange/.published")
REPO = "sackfield/open-interpretability-exchange"
# The files that cite inputs: papers, analyses, scripts and the package.
CITING = ("papers", "docs", "scripts", "openinterp", ".claude")
NAME = re.compile(r"exchange/([A-Za-z0-9._-]+)")
# Names under exchange/ that are only outputs or working folders, never cited inputs. A
# command writes them, and no analysis reads them. `bench` also holds the benchmark's
# private test probes, so it is never published.
OUTPUTS = {
    "bench", "bench-dev", "code-detection", "gpt2-spd", "language-detection",
    "spd-up-neurons.json", ".published",
}  # fmt: skip
OUTPUT_FILE = re.compile(r"^(ev|review|score)\.|\.(prompt\.md|response\.json)$")
# Inputs that stay unpublished, each with why, and how to make them again.
UNPUBLISHED = {
    "probe-comparison/*.pile.jsonl": "pile-10k excerpts: the Pile corpus is not published "
    "(docs/experiments.md). scripts/probe_comparison.py writes them again from pile-10k.",
}
TEXT = {".json", ".jsonl", ".md", ".txt", ".log", ".csv"}
CARD = """---
license: other
pretty_name: Open Interpretability exchange inputs
---

# Open Interpretability: inputs of papers and analyses

The files that papers, analyses and scripts of
[Open Interpretability](https://github.com/8W9aG/openinterpretability) read from its
git-ignored `exchange/` folder. Examples are SPD checkpoints and training logs, reviewer
prompts and probe sets, and corpus samples. `data/exchange-manifest.json` in the
repository pins this dataset's revision and each file's SHA-256, and
`scripts/exchange.py fetch` downloads and checks them.

Text in these files quotes the corpora that the scans read, under their own terms. Examples
are pile-10k (bigscience-bloom-rail-1.0, each Pile component under its own) and
FineWeb-Edu (ODC-BY). Model outputs and code-generated files are under the repository's MIT
licence. Pile corpus files themselves are not included.
"""


def cited() -> set[str]:
    """The names under ``exchange/`` that tracked papers, analyses and scripts read."""
    out = subprocess.run(["git", "grep", "-ohE", NAME.pattern, "--", *CITING],
                         capture_output=True, text=True, check=False).stdout  # fmt: skip
    names = {m.group(1).rstrip(".") for m in NAME.finditer(out)}
    return {n for n in names if n not in OUTPUTS and not OUTPUT_FILE.search(n)}


def write_manifest(manifest: dict) -> None:
    """Write ``manifest`` with one line per folder, so that it stays short to review."""
    groups: dict[str, dict[str, str]] = {}
    for rel, digest in sorted(manifest["files"].items()):
        folder, _, name = rel.partition("/")
        groups.setdefault(folder, {})[name] = digest
    head = {k: v for k, v in manifest.items() if k != "files"}
    lines = [
        f"  {json.dumps(k)}: {json.dumps(v, sort_keys=True)}," for k, v in sorted(head.items())
    ]
    body = [f"    {json.dumps(f)}: {json.dumps(g, sort_keys=True)}" for f, g in groups.items()]
    text = "{\n" + "\n".join(lines) + '\n  "folders": {\n' + ",\n".join(body) + "\n  }\n}\n"
    MANIFEST.write_text(text, encoding="utf-8")


def read_manifest() -> dict:
    """The manifest, with ``files`` as ``{path: sha256}``; empty if there is none."""
    if not MANIFEST.exists():
        return {"revision": None, "files": {}}
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    folders = data.pop("folders", {})
    files = {f"{f}/{n}" if n else f: d for f, g in folders.items() for n, d in g.items()}
    return {**data, "files": files}


def _excluded(rel: str) -> bool:
    return any(Path(rel).match(pattern) for pattern in UNPUBLISHED)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(home: str) -> dict:
    """Copy every cited input to the staging folder, with ``home`` rewritten to ``~``."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    files: dict[str, str] = {}
    for name in sorted(cited()):
        source = EXCHANGE / name
        paths = [source] if source.is_file() else sorted(source.rglob("*"))
        paths = [path for path in paths if path.is_file()]
        for path in paths:
            rel = path.relative_to(EXCHANGE).as_posix()
            if _excluded(rel):
                continue
            target = STAGING / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in TEXT:  # bytes, so that nothing but the home folder changes
                target.write_bytes(path.read_bytes().replace(home.encode(), b"~"))
            else:
                shutil.copyfile(path, target)
            files[rel] = _sha256(target)
    (STAGING / "README.md").write_text(CARD, encoding="utf-8")
    return {"repo": REPO, "revision": None, "unpublished": UNPUBLISHED, "files": files}


def missing() -> list[str]:
    """Cited names that are neither in the manifest nor unpublished on purpose."""
    published = {rel.split("/", 1)[0] for rel in read_manifest()["files"]}
    kept_back = {pattern.split("/", 1)[0] for pattern in UNPUBLISHED}
    return sorted(cited() - published - kept_back)


def upload() -> str:
    """Publish the staging folder to the public dataset; return the new commit's id."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    message = "Inputs that papers and analyses read (#3255)"
    info = api.upload_folder(
        repo_id=REPO, repo_type="dataset", folder_path=str(STAGING), commit_message=message
    )
    return info.oid


def fetch(names: list[str]) -> int:
    """Download the pinned files of ``names`` (default: all) into ``exchange/``, checked."""
    from huggingface_hub import hf_hub_download

    manifest = read_manifest()
    bad = 0
    for rel, digest in manifest["files"].items():
        if names and rel.split("/", 1)[0] not in names:
            continue
        local = Path(hf_hub_download(manifest["repo"], rel, repo_type="dataset",
                                     revision=manifest["revision"]))  # fmt: skip
        target = EXCHANGE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        if _sha256(target) != digest:
            print(f"hash mismatch: {rel}")
            bad += 1
    return 1 if bad else 0


def main() -> int:
    """Run one subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage")
    sub.add_parser("upload")
    sub.add_parser("check")
    fetching = sub.add_parser("fetch")
    fetching.add_argument("names", nargs="*")
    opts = parser.parse_args()
    if opts.command == "stage":
        manifest = stage(str(Path.home()))
        write_manifest(manifest)
        print(f"staged {len(manifest['files'])} files in {STAGING}; manifest has no revision yet")
        return 0
    if opts.command == "upload":
        manifest = read_manifest()
        manifest["revision"] = upload()
        write_manifest(manifest)
        print(f"published {REPO} at {manifest['revision']}")
        return 0
    if opts.command == "fetch":
        return fetch(opts.names)
    gaps = missing()
    if MANIFEST.exists() and read_manifest()["revision"] is None:
        print(f"{MANIFEST} has no revision: run scripts/exchange.py upload")
        return 1
    for name in gaps:
        print(f"exchange/{name} is cited but not published: run `exchange.py stage`, then `upload`")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
