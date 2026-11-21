"""The core CLI (openinterpretability/cli.py): model loading, devices, reevaluate and deploy."""

import os
import runpy
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from openinterpretability import cli as cli_module
from openinterpretability.cli import cli
from openinterpretability.core.exceptions import OpenInterpretabilityError
from openinterpretability.core.schemas import AuditRecord, NeuronIdentifier
from openinterpretability.harness import devices
from tests.conftest import FakeTokenizer
from tests.test_signing import auto_signing, keys  # noqa: F401  (fixtures)

# -- loading a model ------------------------------------------------------------------------


@pytest.fixture
def local_gpt2(tiny_gpt2, tmp_path, monkeypatch):
    """A tiny GPT-2 saved to a folder, so the loader reads real weights without a download."""
    import transformers

    folder = tmp_path / "tiny-gpt2"
    tiny_gpt2.save_pretrained(folder)
    # The tokenizer is the one download a local folder would still need.
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    return str(folder)


def test_a_local_model_loads_as_given_and_names_its_runner(local_gpt2, capsys):
    runner = cli_module._load_runner(local_gpt2, "cpu", "float32")
    assert runner.model_name == local_gpt2
    assert runner.model_revision is None  # not a Hub repo: no commit to pin
    assert runner.device.type == "cpu" and runner.n_layers == 2
    assert f"Loading {local_gpt2} on cpu" in capsys.readouterr().err
    assert devices.choice() == "--device cpu"


def test_a_hub_model_pins_the_commit_of_its_snapshot(monkeypatch, tmp_path):
    import huggingface_hub

    snapshot = tmp_path / "snapshots" / "0123abcd"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snapshot))
    assert cli_module._resolve_revision("org/model", "main") == "0123abcd"

    def offline(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    assert cli_module._resolve_revision("org/model", "v1") == "v1"


def test_an_oversubscribed_cpu_is_told_how_many_threads_torch_uses(local_gpt2, monkeypatch, capsys):
    monkeypatch.setattr("openinterpretability.harness.cpu_speed.apply_cpu_threads", lambda: 3)
    cli_module._load_runner(local_gpt2, "cpu", "float32")
    assert "The CPU is oversubscribed, so torch uses 3 threads." in capsys.readouterr().err


def test_without_a_qualified_model_scans_stay_off_the_ane(local_gpt2, capsys):
    devices.force_ane()  # as --device ane
    runner = cli_module._load_runner(local_gpt2, "cpu", "float32")
    assert "the ANE needs coremltools" in capsys.readouterr().err
    assert getattr(runner, "scan_backend", None) is None


def test_a_qualified_model_runs_its_scans_on_the_ane(local_gpt2, monkeypatch):
    from openinterpretability.harness.ane.backend import AneScan

    backend = object()  # the ANE itself is a device CI lacks
    monkeypatch.setattr(AneScan, "for_model", classmethod(lambda cls, *a: backend))
    devices.force_ane()
    runner = cli_module._load_runner(local_gpt2, "cpu", "float32")
    assert runner.scan_backend is backend


def test_a_streamed_model_scans_on_a_free_cpu_not_the_planned_ane(local_gpt2, monkeypatch, capsys):
    monkeypatch.setattr(
        devices, "_planned", devices.DevicePlan("cpu", "the GPU is busy", scan="ane")
    )
    monkeypatch.setattr(devices, "cpu_load_percent", lambda: 10.0)
    monkeypatch.setattr("openinterpretability.harness.memory.is_streamed", lambda model: True)
    runner = cli_module._load_runner(local_gpt2, "cpu", "float32")
    why = "the model streams and the CPU is free (load 10% of its cores)"
    assert f"Scans run on the CPU, not the ANE, because {why}." in capsys.readouterr().err
    assert devices.choice() == f"--device cpu; scans on the cpu, because {why}"
    assert getattr(runner, "scan_backend", None) is None


@pytest.mark.parametrize("ane", [False, True])
def test_a_retry_after_a_gpu_failure_uses_the_ane_only_when_it_can(monkeypatch, ane):
    monkeypatch.setenv(devices.FORCE_ENV, "cpu")
    monkeypatch.setattr(devices, "ane_possible", lambda: ane)
    assert cli_module.resolve_device("auto") == "cpu"
    assert devices.scans_on_ane() is ane
    assert devices.choice() == "cpu: a retry after the GPU failed"


@pytest.mark.parametrize("ane", [False, True])
def test_a_stuck_mps_falls_back_to_the_cpu_and_the_ane_only_when_it_can(monkeypatch, ane):
    monkeypatch.setattr(devices, "mps_stuck", lambda: "MPS didn't start")
    monkeypatch.setattr(devices, "ane_possible", lambda: ane)
    assert cli_module.resolve_device("mps") == "cpu"
    assert devices.scans_on_ane() is ane
    assert devices.choice() == "mps: cpu, because MPS didn't start"


# -- writing records ------------------------------------------------------------------------


def test_a_record_that_cannot_be_signed_is_kept_with_a_warning(
    auto_signing,  # noqa: F811
    keys,  # noqa: F811
    record,
    capsys,
):
    from openinterpretability.builder.wiki import archive_path
    from openinterpretability.signing import signature_path
    from tests.test_signing import signer_line

    auto_signing.write_text(signer_line("will", keys[0], 'valid-before="20200101Z"'))  # expired
    cli_module._write_wiki(record, Path("wiki"), kind="review")
    archived = archive_path(Path("wiki"), record)
    assert archived.exists() and not signature_path(archived).exists()
    assert f"warning: could not sign {archived}" in capsys.readouterr().err


def test_progress_bars_stay_on_for_a_terminal_unless_quiet(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    cli_module.quiet_libraries(False)
    assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ


# -- reevaluate -----------------------------------------------------------------------------


@pytest.fixture
def toy_record(record, toy_runner, monkeypatch, tmp_path):
    """A reviewed record of a toy-model pole, and the toy model loaded in its place."""
    ident = NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")
    submission = record.submission.model_copy(update={"identifier": ident, "model_num_layers": 3})
    old = AuditRecord(submission=submission, verdict=record.verdict)
    loads = []

    def load(model_name, *_a, **_k):
        loads.append(model_name)
        return toy_runner

    monkeypatch.setattr(cli_module, "_load_runner", load)
    path = tmp_path / "old.json"
    path.write_text(old.model_dump_json())
    return path, loads


def _invoke(*args):
    return CliRunner().invoke(cli, [str(a) for a in args])


def test_reevaluate_needs_out_when_it_skips_the_registry(toy_record):
    path, loads = toy_record
    result = _invoke("reevaluate", "--record", path, "--wiki-dir", "")
    assert result.exit_code != 0 and "--out is required" in result.output
    assert not loads


def test_reevaluate_refuses_a_record_without_a_verdict(toy_record, tmp_path):
    path, loads = toy_record
    old = AuditRecord.model_validate_json(path.read_text())
    bare = tmp_path / "bare.json"
    bare.write_text(old.model_copy(update={"verdict": None}).model_dump_json())
    result = _invoke("reevaluate", "--record", bare, "--out", tmp_path / "new.json")
    assert result.exit_code != 0 and "has no verdict to re-evaluate" in result.output
    assert not loads


def test_reevaluate_measures_the_stored_probes_again(toy_record, tmp_path):
    path, loads = toy_record
    out = tmp_path / "new.json"
    result = _invoke("reevaluate", "--record", path, "--out", out, "--wiki-dir", "")
    assert result.exit_code == 0, result.output
    assert loads == ["toy"]
    old = AuditRecord.model_validate_json(path.read_text())
    new = AuditRecord.model_validate_json(out.read_text())
    assert result.output.startswith(f"VERIFIED -> {new.verdict.status} — ")
    assert f"Wrote {out}" in result.output
    # The same probes, measured again; the old reviewer's model stays with them.
    stored = [r.prompt for r in old.verdict.adversarial_test_results]
    assert [r.prompt for r in new.verdict.adversarial_test_results][: len(stored)] == stored
    assert new.verdict.reviewer_agent_model == old.verdict.reviewer_agent_model
    assert new.submission.identifier == old.submission.identifier


def test_reevaluate_archives_the_original_and_the_new_record(toy_record, tmp_path):
    path, _ = toy_record
    wiki = tmp_path / "wiki"
    result = _invoke("reevaluate", "--record", path, "--wiki-dir", wiki)
    assert result.exit_code == 0, result.output
    assert result.output.count("Experiment: ") == 2  # the original, then the new record
    assert len(list((wiki / "experiments").rglob("*.json"))) == 2


def test_a_record_whose_page_exists_is_not_written_again(record, tmp_path, capsys):
    wiki = tmp_path / "wiki"
    cli_module._write_wiki(record, wiki, kind="backfill")
    assert "Experiment: " in capsys.readouterr().out
    pages = sorted(wiki.rglob("*.md"))
    cli_module._write_wiki(record, wiki, kind="backfill")  # pages are write-once
    assert capsys.readouterr().out == ""
    assert sorted(wiki.rglob("*.md")) == pages


# -- deploy ---------------------------------------------------------------------------------


@pytest.fixture
def r2(tmp_path, monkeypatch, mocker):
    """A built site, R2 credentials, and a fake bucket and cache in place of Cloudflare's."""
    from openinterpretability.sync import r2 as sync_r2

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>")
    for name in ("CF_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "x")
    monkeypatch.delenv("CF_ZONE_ID", raising=False)
    monkeypatch.delenv("CF_API_TOKEN", raising=False)
    client = mocker.Mock()
    client.get_paginator.return_value.paginate.return_value = [{"Contents": []}]
    monkeypatch.setattr(sync_r2, "make_client", lambda config: client)
    purged = []
    monkeypatch.setattr(
        sync_r2, "purge_cloudflare_cache", lambda zone, token, urls: purged.append(list(urls))
    )
    args = ["deploy", "--dist-dir", dist, "--bucket", "b", "--site-url", "https://x.org"]
    return dist, client, purged, args


def _cache_keys(monkeypatch):
    monkeypatch.setenv("CF_ZONE_ID", "zone")
    monkeypatch.setenv("CF_API_TOKEN", "token")


def test_deploy_uploads_and_skips_the_purge_without_cloudflare_keys(r2):
    _, client, purged, args = r2
    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    assert "Uploaded 1, unchanged 0, deleted 0, derived skipped 0" in result.output
    assert client.put_object.call_args.kwargs["Key"] == "index.html"
    assert purged == []


def test_deploy_purges_the_changed_urls(r2, monkeypatch):
    _, _, purged, args = r2
    _cache_keys(monkeypatch)
    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    assert purged and f"Purged {len(purged[0])} URLs from the Cloudflare cache" in result.output
    assert all(url.startswith("https://x.org/") for url in purged[0])


def test_a_dry_run_deploy_only_says_what_it_would_purge(r2, monkeypatch):
    _, client, purged, args = r2
    _cache_keys(monkeypatch)
    result = _invoke(*args, "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Would upload 1" in result.output and "Would purge" in result.output
    assert purged == [] and not client.put_object.called


def test_deploy_with_nothing_changed_purges_nothing(r2, monkeypatch):
    from openinterpretability.sync.r2 import file_md5

    dist, client, purged, args = r2
    _cache_keys(monkeypatch)
    etag = f'"{file_md5(dist / "index.html")}"'
    client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "index.html", "ETag": etag}]}
    ]
    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    assert "Uploaded 0, unchanged 1" in result.output
    assert "Nothing changed; no cache purge." in result.output and purged == []


# -- the console entry point ----------------------------------------------------------------


def test_main_prints_a_package_error_as_text(monkeypatch):
    def fails(**_kwargs):
        raise OpenInterpretabilityError("the registry is locked")

    monkeypatch.setattr(cli_module, "cli", fails)
    with pytest.raises(SystemExit) as exit_:
        cli_module.main()
    assert exit_.value.code == "error: the registry is locked"


def test_main_reraises_an_error_that_no_gpu_caused(monkeypatch):
    def fails(**_kwargs):
        raise ValueError("a bad value")

    monkeypatch.setattr(cli_module, "cli", fails)
    with pytest.raises(ValueError, match="a bad value"):
        cli_module.main()


def test_main_runs_the_command_again_after_a_gpu_failure(monkeypatch):
    retried = []

    def fails(**_kwargs):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(cli_module, "cli", fails)
    monkeypatch.setattr(devices, "gpu_failed", lambda exc: "MPS")
    monkeypatch.setattr(devices, "retry", lambda exc, gpu: retried.append(gpu))
    with pytest.raises(RuntimeError):
        cli_module.main()
    assert retried == ["MPS"]


# runpy warns that the module is already imported, which is harmless here.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_running_the_module_runs_the_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["open-interpretability", "--version"])
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module("openinterpretability.cli", run_name="__main__")
    assert exit_.value.code == 0
    assert "version" in capsys.readouterr().out
