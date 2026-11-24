"""The site's stylesheet (labinacan/site/css.py): the CLI and its download."""

import hashlib
import io
import runpy
import sys
from pathlib import Path

import pytest

from labinacan.site import css
from openinterpretability.builder import css as site_css

SYSTEM = ("Linux", "x86_64")
BINARY = b"#!/bin/sh\necho tailwind\n"


@pytest.fixture
def pinned(monkeypatch):
    """This machine's Tailwind CLI pinned to the hash of ``BINARY``, so a test can make it."""
    name = "tailwindcss-test"
    sha = hashlib.sha256(BINARY).hexdigest()
    monkeypatch.setattr(css, "BINARIES", {SYSTEM: (name, sha)})
    monkeypatch.setattr(css.platform, "system", lambda: SYSTEM[0])
    monkeypatch.setattr(css.platform, "machine", lambda: SYSTEM[1])
    return name


def test_the_cli_is_downloaded_once_checked_and_made_runnable(tmp_path, monkeypatch, pinned):
    fetched = []

    def urlopen(url, timeout):
        fetched.append(url)
        return io.BytesIO(BINARY)

    monkeypatch.setattr(css.urllib.request, "urlopen", urlopen)
    cache = tmp_path / "cache"
    path = css.binary(cache)
    assert path == cache / f"{pinned}-{css.VERSION}" and path.read_bytes() == BINARY
    assert path.stat().st_mode & 0o111  # executable
    url = f"https://github.com/tailwindlabs/tailwindcss/releases/download/v{css.VERSION}/{pinned}"
    assert fetched == [url]
    assert css.binary(cache) == path and len(fetched) == 1  # from the cache the second time


@pytest.fixture
def site(tmp_path):
    site = tmp_path / "dist"
    site.mkdir()
    (site / "index.html").write_text('<p class="text-sm mt-3">x</p>')
    return site


@pytest.fixture
def tailwind(tmp_path, monkeypatch, pinned):
    """The pinned CLI in the cache, and a stand-in for running it that records what it read."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{pinned}-{css.VERSION}").write_bytes(BINARY)
    seen = {}

    def run(args, check, capture_output):
        conf = Path(args[args.index("-c") + 1])
        seen["args"], seen["check"] = args, check
        seen["source"] = Path(args[args.index("-i") + 1]).read_text()
        seen["config"] = conf.read_text()
        seen["classes"] = Path(conf.parent / "classes.txt").read_text()
        Path(args[args.index("-o") + 1]).write_text(".text-sm{font-size:.875rem}")

    monkeypatch.setattr(css.subprocess, "run", run)
    return cache, seen


def test_build_writes_the_stylesheet_from_the_classes_the_pages_use(site, tailwind):
    cache, seen = tailwind
    out = css.build(site, cache, {"brand-700": "#047857"})
    assert out == site.resolve() / css.STYLESHEET
    assert out.read_text() == ".text-sm{font-size:.875rem}"
    assert seen["classes"] == "mt-3 text-sm"
    assert seen["args"][0] == str(cache / f"tailwindcss-test-{css.VERSION}")
    assert "--minify" in seen["args"] and '"content"' in seen["config"]
    assert '"brand-700": "#047857"' in seen["config"]  # the project's colours (#6628)


def test_main_builds_the_site_it_is_given(site, tailwind, monkeypatch, capsys):
    cache, _ = tailwind
    monkeypatch.setattr(sys, "argv", ["css", "--site", str(site), "--cache", str(cache)])
    site_css.main()
    out = site.resolve() / css.STYLESHEET
    assert capsys.readouterr().out == f"Wrote {out} ({out.stat().st_size:,} bytes)\n"


# runpy warns that the module is already imported, which is harmless here.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_running_the_module_reads_its_arguments(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["css", "--help"])  # argparse exits before any build
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("openinterpretability.builder.css", run_name="__main__")
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    assert "Build the site's stylesheet once" in out and "--site" in out and "--cache" in out


def test_this_repos_main_passes_its_colours_and_its_cache(monkeypatch):
    from openinterpretability.builder import palette
    from openinterpretability.core.legacy_names import cache_base

    seen = []
    monkeypatch.setattr(css, "main", lambda colours, cache: seen.append((colours, cache)))
    site_css.main()
    assert seen == [(palette.tailwind_colors(), cache_base() / "tailwind")]


def test_build_passes_the_cli_its_config_input_and_output(site, tailwind):
    """#6628: the build's files and arguments, exactly, now that the project passes colours."""
    import json

    cache, seen = tailwind
    out = css.build(site, cache, {"brand-700": "#047857"})
    conf = seen["args"][seen["args"].index("-c") + 1]
    source = seen["args"][seen["args"].index("-i") + 1]
    assert Path(conf).name == "tailwind.config.js" and Path(source).name == "input.css"
    assert seen["args"][1:] == ["-c", conf, "-i", source, "-o", str(out), "--minify"]
    assert seen["check"] is True  # a failed CLI fails the build
    assert seen["source"] == "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n"
    body = seen["config"].removeprefix("module.exports = ").removesuffix(";\n")
    assert json.loads(body)["content"] == [str(Path(conf).parent / "classes.txt")]
    assert seen["config"] == css.config(json.loads(body)["content"], {"brand-700": "#047857"})


def test_build_makes_a_missing_site_folder_and_reuses_an_existing_one(tmp_path, tailwind):
    cache, _ = tailwind
    site = tmp_path / "new" / "site"  # neither folder exists yet
    out = css.build(site, cache, {})
    assert out == site.resolve() / css.STYLESHEET and out.is_file()
    assert css.build(site, cache, {}) == out  # the second build finds the folders there


def test_the_config_is_indented_one_space():
    text = css.config(["c.txt"], {})
    assert text.startswith('module.exports = {\n "content": [\n  "c.txt"\n ],')


def test_main_defaults_to_dist_and_the_given_cache(monkeypatch, capsys, tmp_path):
    seen = []

    def build(site, cache, colours):
        seen.append((site, cache, colours))
        out = tmp_path / "site.css"
        out.write_text("x")
        return out

    monkeypatch.setattr(css, "build", build)
    monkeypatch.setattr(sys, "argv", ["css"])
    css.main({"a": "#000000"}, tmp_path / "cache")
    assert seen == [(Path("dist"), tmp_path / "cache", {"a": "#000000"})]
