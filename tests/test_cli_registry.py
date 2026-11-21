"""Registry-maintenance commands: `wiki`, `stats`, `next` and `ledger`."""

import json
from datetime import timedelta

from click.testing import CliRunner

from openinterpretability.builder.wiki import write_experiment_page
from openinterpretability.cli import cli
from openinterpretability.cli_registry import registry_stats
from openinterpretability.core.ledger import DrawEntry, write_entry
from openinterpretability.core.paths import experiments_dir
from openinterpretability.core.schemas import ExperimentProvenance
from openinterpretability.harness.layer_cache import ENV as CACHE_ENV

MODEL = "Qwen/Qwen2.5-0.5B"


def _run(*args):
    result = CliRunner().invoke(cli, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result.output


def test_wiki_imports_loose_records_as_experiments(record, tmp_path):
    loose, wiki = tmp_path / "loose", tmp_path / "wiki"
    loose.mkdir()
    (loose / "audit.json").write_text(record.model_dump_json())
    out = _run("wiki", "--import-dir", loose, "--wiki-dir", wiki)
    assert "Wiki: " in out and f"1 new experiments; indexes rebuilt in {wiki}" in out
    assert list(experiments_dir(wiki).rglob("*.json"))
    # Without --import-dir it only rebuilds the indexes.
    assert f"0 new experiments; indexes rebuilt in {wiki}" in _run("wiki", "--wiki-dir", wiki)


def test_stats_counts_the_registry_and_says_what_the_errata_set_aside(record, tmp_path):
    write_experiment_page(record, tmp_path)
    out = _run("stats", "--wiki-dir", tmp_path)
    assert "Registry: 1 experiments on 1 units; VERIFIED 1/1 (100%)" in out
    assert f"  {MODEL}: 1 experiments" in out and "Protocol versions: < 0.2 ×1" in out
    assert "Set aside by errata" not in out
    text = registry_stats([record], set_aside=2)
    assert text.endswith("Set aside by errata, and not counted: 2 experiments")


def _next(wiki, seed, *extra):
    return _run("next", "--wiki-dir", wiki, "--no-claims", "--epsilon", "0", "--seed", seed,
                "--model", MODEL, *extra)  # fmt: skip


def test_next_names_no_skills_for_a_draw_that_needs_none(record, tmp_path):
    write_experiment_page(record, tmp_path)
    out = _next(tmp_path, 3)  # this seed draws a reevaluation, which names no skill
    assert "Next experiment: reevaluate on" in out
    assert "Skills:" not in out


def test_next_says_when_the_layer_of_a_neuron_audit_is_in_the_scan_cache(
    record, tmp_path, monkeypatch
):
    write_experiment_page(record, tmp_path / "wiki")
    cache = tmp_path / "cache"
    monkeypatch.setenv(CACHE_ENV, str(cache))
    out = _next(tmp_path / "wiki", 1)  # this seed draws the opposite pole, at layer 3
    assert "--neuron 117 --pole negative" in out and "Scan cache:" not in out
    meta = cache / MODEL.replace("/", "--") / "L3" / "k" / "meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"fields": {"model": MODEL, "layer": 3}}))
    out = _next(tmp_path / "wiki", 1)
    assert "Scan cache: layer 3 is cached, so the audit replays its scan" in out


def test_next_lists_the_explore_share_of_each_model(tmp_path):
    out = _run("next", "--wiki-dir", tmp_path, "--no-claims", "--seed", 1, "--top", 2,
               "--model", MODEL, "--model", "EleutherAI/pythia-70m")  # fmt: skip
    assert "Explore draws by model (neurons covered):" in out
    assert "EleutherAI/pythia-70m  0 covered" in out and f"{MODEL}  0 covered" in out
    assert "← smallest model below the verified target" in out
    # With one model there is no share to list.
    one = _run("next", "--wiki-dir", tmp_path, "--no-claims", "--seed", 1, "--top", 2,
               "--model", MODEL)  # fmt: skip
    assert "Explore draws by model" not in one


def _random(record, seed):
    prov = ExperimentProvenance(dtype="float32", corpus="c", n_snippets=1, tokens_scanned=1,
                                selection="random", harness_version="t", seed=seed)  # fmt: skip
    return record.model_copy(
        update={"submission": record.submission.model_copy(update={"provenance": prov})}
    )


def test_ledger_prints_each_draw_and_the_verified_share_of_explore_draws(record, tmp_path):
    when = record.verdict.timestamp - timedelta(hours=1)
    audit = f"--model {MODEL} --random --seed 42"
    write_entry(DrawEntry(timestamp=when, seed=1, kind="explore", model=MODEL, audit=audit),
                tmp_path)  # fmt: skip
    out = _run("ledger", "--wiki-dir", tmp_path)
    assert "explore" in out and f"{MODEL} (seed 42): none" in out
    assert "1 draws: " in out and "Explore draws: 1, 0 VERIFIED." in out
    assert "VERIFIED share" not in out  # no explore draw has finished

    write_experiment_page(_random(record, 42), tmp_path)
    unreviewed = experiments_dir(tmp_path) / "unreviewed.json"
    unreviewed.write_text(record.model_copy(update={"verdict": None}).model_dump_json())
    out = _run("ledger", "--wiki-dir", tmp_path)
    assert f"{MODEL} (seed 42): VERIFIED" in out and "Explore draws: 1, 1 VERIFIED." in out
    assert "VERIFIED share of explore draws: 100.0% of all, 100.0% of finished." in out


def test_ledger_with_no_explore_draw_prints_no_share(record, tmp_path):
    slug = record.submission.identifier.slug
    when = record.verdict.timestamp - timedelta(hours=1)
    write_entry(DrawEntry(timestamp=when, seed=2, kind="revise", model=MODEL, slug=slug),
                tmp_path)  # fmt: skip
    write_experiment_page(record, tmp_path)
    out = _run("ledger", "--wiki-dir", tmp_path)
    assert f"revise             {slug}: VERIFIED" in out
    assert "Explore draws: 0, 0 VERIFIED." in out and "VERIFIED share" not in out
