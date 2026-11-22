"""The benchmark commands (cli_benchmark.py): freeze, measure, seal and score."""

import json
import re

from click.testing import CliRunner

from openinterpretability import benchmark as bench
from openinterpretability.agents.judge import DetectionAnswer, ExplanationJudge
from openinterpretability.builder.wiki import write_experiment_page
from openinterpretability.cli import cli
from tests.test_benchmark import _sealable, balanced, make_record


def _invoke(*args):
    return CliRunner().invoke(cli, ["benchmark", *map(str, args)])


def test_freeze_writes_a_version_once_unless_forced(tmp_path, record):
    wiki = tmp_path / "wiki"
    for i in (1, 2, 3):
        write_experiment_page(make_record(record, idx=i, results=balanced(str(i))), wiki)
    args = ["freeze", "--wiki-dir", wiki, "--version", "v1", "--out", tmp_path / "bench"]
    out_dir = tmp_path / "bench" / "v1"

    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    manifest = bench.read_manifest(out_dir)
    dev, test = manifest["units"]["dev"], manifest["units"]["test"]
    assert dev + test == 3
    assert result.output.strip() == (
        f"Wrote {out_dir}: {dev} dev units ({manifest['dev_probes']} probes), "
        f"{test} test units awaiting fresh probes"
    )

    again = _invoke(*args)
    assert again.exit_code == 1
    assert f"{out_dir} exists; versions are frozen (use a new version)" in again.output
    assert _invoke(*args, "--force").exit_code == 0


def test_measure_refuses_a_unit_outside_the_test_split(tmp_path, record):
    rec = make_record(record, idx=1, results=balanced("1"))
    bench.write(bench.freeze([rec], "v1", test_fraction=0.0), tmp_path / "v1")
    probes = tmp_path / "probes.json"
    probes.write_text(json.dumps({"probes": []}))
    slug = rec.submission.identifier.slug
    result = _invoke(
        "measure", "--bench", tmp_path / "v1", "--wiki-dir", tmp_path, "--unit", slug,
        "--probes", probes, "--labels", tmp_path / "test.private.jsonl",
    )  # fmt: skip
    assert result.exit_code == 1
    assert f"{slug} is not a test unit of {tmp_path / 'v1'}" in result.output
    assert not (tmp_path / "test.private.jsonl").exists()


def test_seal_publishes_the_hash_or_says_why_it_cannot(tmp_path, record):
    _, rows = _sealable(tmp_path, record)
    labels = tmp_path / "test.private.jsonl"
    bench.write_jsonl(labels, rows[:6])  # the second unit has no labels
    result = _invoke("seal", "--bench", tmp_path, "--labels", labels)
    assert result.exit_code == 1 and "have no labels" in result.output

    bench.write_jsonl(labels, rows)
    result = _invoke("seal", "--bench", tmp_path, "--labels", labels)
    assert result.exit_code == 0, result.output
    manifest = bench.read_manifest(tmp_path)
    assert result.output.strip() == f"Sealed 12 test probes: {manifest['test_labels_sha256']}"


def test_score_of_the_test_split_needs_sealed_labels(tmp_path, record):
    _sealable(tmp_path, record)
    result = _invoke(
        "score", "--bench", tmp_path, "--split", "test", "--explanations", "registry",
        "--judge-model", "openai/judge", "--out", tmp_path / "o.jsonl",
    )  # fmt: skip
    assert result.exit_code == 1 and "no sealed test labels yet" in result.output


def test_score_asks_the_llm_judge_and_warns_when_it_lists_every_text(tmp_path, record, monkeypatch):
    rec = make_record(record, idx=1, results=balanced("one"), text="Fires on fire.")
    bench.write(bench.freeze([rec], "v1", test_fraction=0.0), tmp_path / "v1")
    asked = []

    def completion(self, response_model, prompt):
        # The LLM call: this judge lists every numbered text.
        asked.append((self.model, response_model))
        numbers = [int(n) for n in re.findall(r"^(\d+)\. ", prompt, re.MULTILINE)]
        return DetectionAnswer(active=numbers)

    monkeypatch.setattr(ExplanationJudge, "_completion", completion)
    out = tmp_path / "results.jsonl"
    result = _invoke(
        "score", "--bench", tmp_path / "v1", "--explanations", "registry",
        "--judge-model", "openai/judge", "--out", out,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert asked == [("openai/judge", DetectionAnswer)]
    [row] = bench.read_jsonl(out)
    assert row["balanced_accuracy"] == 0.5 and row["listed_share"] == 1.0
    slug = rec.submission.identifier.slug
    assert (
        f"Warning: the judge listed at least 90% of the texts for 1 unit(s): {slug}."
    ) in result.output
