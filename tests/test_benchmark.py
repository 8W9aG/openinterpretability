import json
import re
from datetime import timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from openinterp import benchmark as bench
from openinterp.cli import cli
from openinterp.core.schemas import AdversarialTestResult, AuditRecord, ExperimentProvenance


def probe(text, *, active, predicted=None, kind=None, measured=True):
    """A reviewer probe whose measured activity is ``active``."""
    predicted = active if predicted is None else predicted
    kind = kind or ("positive_holdout" if predicted else "hard_negative")
    return AdversarialTestResult(
        kind=kind,
        prompt=text,
        passed=predicted == active,
        expected_active=predicted,
        measured=measured,
    )


def make_record(
    record, *, idx, selection="random", hours=0, results=(), status="VERIFIED", text=None
):
    sub = record.submission
    ident = sub.identifier.model_copy(update={"neuron_idx": idx})
    prov = ExperimentProvenance(
        dtype="float32",
        corpus="test",
        n_snippets=1,
        tokens_scanned=1,
        selection=selection,
        harness_version="0",
    )
    update = {"identifier": ident, "provenance": prov}
    if text:
        update["natural_language_explanation"] = text
    verdict = record.verdict.model_copy(
        update={
            "status": status,
            "adversarial_test_results": list(results),
            "timestamp": record.verdict.timestamp + timedelta(hours=hours),
        }
    )
    return AuditRecord(submission=sub.model_copy(update=update), verdict=verdict)


def balanced(prefix, n=3):
    """``n`` active and ``n`` inactive probes, the active ones containing "fire"."""
    return [probe(f"{prefix} fire {i}", active=True) for i in range(n)] + [
        probe(f"{prefix} calm {i}", active=False) for i in range(n)
    ]


def test_the_split_is_stable_salted_and_near_its_fraction():
    slugs = [f"m/L0/N{i}-positive" for i in range(2000)]
    splits = [bench.split_of(s, "v1") for s in slugs]
    assert splits == [bench.split_of(s, "v1") for s in slugs]
    assert 0.17 < splits.count("test") / len(slugs) < 0.23
    assert splits != [bench.split_of(s, "v2") for s in slugs]


def test_probe_labels_are_the_measured_activity(record):
    rec = make_record(
        record,
        idx=1,
        results=[
            probe("a", active=True),  # predicted active, passed
            probe("b", active=True, predicted=False),  # predicted inactive, failed
            probe("c", active=False, predicted=True),
            probe("d", active=True, measured=False),  # a one-token probe of a floor unit
            probe("e", active=True, kind="counterfactual"),
        ],
    )
    assert bench.probe_labels(rec) == [
        ("a", True, "positive_holdout"),
        ("b", True, "hard_negative"),
        ("c", False, "positive_holdout"),
    ]


def test_freeze_takes_randomly_drawn_poles_with_both_labels(record):
    records = [
        make_record(record, idx=1, results=balanced("one", 4), status="REJECTED", text="Old."),
        # A follow-up of a random draw is manual, and adds probes; "one fire 0" disagrees.
        make_record(
            record,
            idx=1,
            selection="manual",
            hours=1,
            results=[probe("one fire 0", active=False), probe("one new", active=False)],
            text="Fires on fire.",
        ),
        make_record(record, idx=2, selection="manual", results=balanced("two")),
        make_record(record, idx=3, results=[probe(f"t{i}", active=False) for i in range(9)]),
    ]
    frozen = bench.freeze(records, "v1", test_fraction=0.0)
    assert [u.neuron for u in frozen.units] == [1]
    assert frozen.manifest["selection"]["skipped_units"] == {
        "not_random_neuron_pole": 1,
        "too_few_probes": 1,
    }
    texts = sorted(p["text"] for p in frozen.dev_probes)
    assert "one fire 0" not in texts and "one new" in texts and len(texts) == 8
    unit = frozen.units[0]
    assert unit.records == [r.verdict.timestamp.strftime("%Y%m%dT%H%M%SZ") for r in records[:2]]
    assert unit.threshold == pytest.approx(0.25 * record.submission.max_activation)
    [explanation] = frozen.explanations["dev"]
    assert explanation["explanation"] == "Fires on fire."  # the newest VERIFIED


def test_scores_count_missing_units_at_chance_and_resume():
    probes = {
        "u1": [{"text": f"fire {i}", "active": True} for i in range(3)]
        + [{"text": f"calm {i}", "active": False} for i in range(3)],
        "u2": [{"text": "fire", "active": False}, {"text": "calm", "active": True}],
    }
    calls = []

    def judge(unit, explanation, texts):
        calls.append(texts)
        return ["fire" in t for t in texts]

    rows = bench.score(probes, {"u1": "Fires on fire.", "u2": "Fires on fire."}, judge)
    assert [r["balanced_accuracy"] for r in rows] == [1.0, 0.0]
    again = bench.score(probes, {"u1": "x", "u2": "x"}, judge, done={r["unit"]: r for r in rows})
    assert again == rows and len(calls) == 2  # nothing judged twice
    units = [bench.Unit(unit=u, model="m", model_revision=None, layer=0, neuron=0,
                        pole="positive", split="dev") for u in ("u1", "u2", "u3")]  # fmt: skip
    summary = bench.summarize(rows[:1], units)
    assert summary["explained"] == 1
    assert summary["mean_balanced_accuracy"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)


def test_probes_are_shown_in_an_order_that_hides_their_labels():
    probes = [{"text": f"p{i}", "active": i < 10} for i in range(20)]
    order = bench.shuffled(probes, "u")
    assert order == bench.shuffled(list(reversed(probes)), "u")
    assert [p["active"] for p in order] != sorted((p["active"] for p in order), reverse=True)


def test_sealed_test_labels_must_match_their_hash(tmp_path, record):
    records = [make_record(record, idx=i, results=balanced(str(i))) for i in (1, 2)]
    bench.write(bench.freeze(records, "v1", test_fraction=1.0), tmp_path)
    labels = tmp_path / "test.private.jsonl"
    slugs = sorted(bench.read_units(tmp_path))
    rows = [{"unit": s, "text": f"{s} {w} {i}", "active": w == "fire", "kind": "hard_negative"}
            for s in slugs for w in ("fire", "calm") for i in range(3)]  # fmt: skip
    bench.write_jsonl(labels, rows[:6])
    with pytest.raises(ValueError, match="have no labels"):
        bench.seal(tmp_path, labels)
    bench.write_jsonl(labels, rows)
    manifest = bench.seal(tmp_path, labels)
    assert manifest["test_probes"] == 12
    assert set(bench.load_labels(tmp_path, "test", labels)) == set(slugs)
    labels.write_text(labels.read_text().replace('"active": true', '"active": false', 1))
    with pytest.raises(ValueError, match="does not match"):
        bench.load_labels(tmp_path, "test", labels)


def test_the_score_command_writes_rows_and_a_summary(tmp_path, record, monkeypatch):
    import openinterp.cli_benchmark as cli_benchmark

    records = [make_record(record, idx=1, results=balanced("one"), text="Fires on fire.")]
    bench.write(bench.freeze(records, "v1", test_fraction=0.0), tmp_path / "v1")
    monkeypatch.setattr(
        cli_benchmark, "_judge", lambda model: lambda u, e, ts: ["fire" in t for t in ts]
    )
    out = tmp_path / "results.jsonl"
    args = ["benchmark", "score", "--bench", str(tmp_path / "v1"), "--explanations", "registry"]
    result = CliRunner().invoke(cli, [*args, "--judge-model", "openai/judge", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert bench.read_jsonl(out)[0]["balanced_accuracy"] == 1.0
    summary = (tmp_path / "results.summary.json").read_text()
    assert '"mean_balanced_accuracy": 1.0' in summary


def test_judges_can_answer_every_prompt_at_once(tmp_path, record):
    records = [
        make_record(record, idx=i, results=balanced(str(i)), text="Fires on fire.") for i in (1, 2)
    ]
    bench.write(bench.freeze(records, "v1", test_fraction=0.0), tmp_path / "v1")
    prompts, out = tmp_path / "prompts", tmp_path / "results.jsonl"
    args = ["benchmark", "score", "--bench", str(tmp_path / "v1"), "--explanations", "registry"]
    args += ["--judge-model", "external/judge", "--out", str(out), "--prompts-dir", str(prompts)]

    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "2 unit(s) wait for a judge's answer" in result.output
    files = sorted(prompts.glob("*.prompt.md"))
    assert len(files) == 2 and not (tmp_path / "results.summary.json").exists()
    assert "Fires on fire." in files[0].read_text()

    def answer(prompt: Path, numbers) -> None:
        texts = prompt.read_text().split("# Prompt")[1]
        listed = re.findall(r"^(\d+)\. (.*)$", texts, re.MULTILINE)
        chosen = [int(n) for n, text in listed if "fire" in text] if numbers is None else numbers
        prompt.with_name(prompt.name.replace(".prompt.md", ".json")).write_text(
            json.dumps({"active": chosen})
        )

    answer(files[0], None)
    answer(files[1], [99])  # names a text that doesn't exist
    result = CliRunner().invoke(cli, args)
    assert "1 unit(s) wait" in result.output and len(bench.read_jsonl(out)) == 1
    assert "only" in next(prompts.glob("*.error.txt")).read_text()
    assert next(prompts.glob("*.rejected.json"))

    answer(files[1], None)
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert [r["balanced_accuracy"] for r in bench.read_jsonl(out)] == [1.0, 1.0]
    summary = json.loads((tmp_path / "results.summary.json").read_text())
    assert summary["judge_model"] == "external/judge" and summary["explained"] == 2
    assert not list(prompts.glob("*.error.txt"))


def test_prompt_files_need_an_external_judge(tmp_path, record):
    bench.write(bench.freeze([make_record(record, idx=1, results=balanced("1"))], "v1",
                             test_fraction=0.0), tmp_path / "v1")  # fmt: skip
    args = ["benchmark", "score", "--bench", str(tmp_path / "v1"), "--explanations", "registry",
            "--judge-model", "openai/judge", "--out", str(tmp_path / "o.jsonl"),
            "--prompts-dir", str(tmp_path / "p")]  # fmt: skip
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2 and "needs an external/<name> judge" in result.output


def test_measure_appends_fresh_test_labels_from_the_model(
    tmp_path, record, toy_runner, monkeypatch
):
    import json

    import openinterp.cli as cli_module
    from openinterp.core.identifiers import NeuronIdentifier
    from openinterp.core.paths import experiments_dir

    ident = NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")
    sub = record.submission.model_copy(update={"identifier": ident, "first_token_floor": False})
    rec = make_record(record.model_copy(update={"submission": sub}), idx=2, results=balanced("x"))
    slug = rec.submission.identifier.slug
    wiki = tmp_path / "wiki"
    stamp = rec.verdict.timestamp.strftime("%Y%m%dT%H%M%SZ")
    (experiments_dir(wiki) / slug).mkdir(parents=True)
    (experiments_dir(wiki) / slug / f"{stamp}.json").write_text(rec.model_dump_json())
    bench.write(bench.freeze([rec], "v1", test_fraction=1.0), tmp_path / "v1")
    texts = ["abcde", "hello world", "zzzz", "aaaa bbbb"]
    probes = [{"text": t, "expected_active": True, "rationale": "r"} for t in texts]
    probes_file = tmp_path / "probes.json"
    probes_file.write_text(json.dumps({"probes": probes * 3}))
    monkeypatch.setattr(cli_module, "_load_unit_runner", lambda *a, **k: toy_runner)

    def run(labels):
        args = ["benchmark", "measure", "--bench", str(tmp_path / "v1"), "--wiki-dir", str(wiki)]
        args += ["--unit", slug, "--probes", str(probes_file), "--labels", str(labels)]
        return CliRunner().invoke(cli, args)

    assert run(tmp_path / "labels.jsonl").exit_code != 0  # would not be ignored by git
    labels = tmp_path / "test.private.jsonl"
    result = run(labels)
    assert result.exit_code == 0, result.output
    rows = bench.read_jsonl(labels)
    assert len(rows) == 12 and {r["unit"] for r in rows} == {slug}
    assert {r["active"] for r in rows} == {True, False}  # the texts tell labels apart
    threshold = 0.25 * sub.max_activation
    for row in rows:
        strength = toy_runner.max_pole_strength(row["text"], 1, 2, "positive")
        assert row["active"] == (strength >= threshold)


def _units(n, split="dev"):
    return [bench.Unit(unit=f"m/L{i // 10}/N{i}-positive", model="m", model_revision=None,
                       layer=i // 10, neuron=i, pole="positive", split=split)
            for i in range(n)]  # fmt: skip


def test_a_shuffled_order_is_a_fixed_random_sample_of_the_split():
    units = _units(200)
    assert bench.ordered(reversed(units), "slug", "v1") == sorted(units, key=lambda u: u.unit)
    shuffled = bench.ordered(units, "shuffled", "v1")
    assert shuffled == bench.ordered(list(reversed(units)), "shuffled", "v1")  # input order
    assert shuffled != bench.ordered(units, "shuffled", "v2")  # salted by the version
    first = shuffled[:50]
    assert len({u.layer for u in first}) >= 15  # spread over the 20 layers, not layers 0-4
    # One more unit in the split doesn't move the others relative to each other.
    grown = bench.ordered([*units, *_units(201)[200:]], "shuffled", "v1")
    assert [u for u in grown if u.neuron < 200] == shuffled
    with pytest.raises(ValueError, match="unknown order"):
        bench.ordered(units, "random", "v1")


def test_a_partial_run_says_whether_it_is_a_random_sample(tmp_path, record, monkeypatch):
    import openinterp.cli_benchmark as cli_benchmark

    records = [make_record(record, idx=i, results=balanced(str(i)), text="Fires on fire.")
               for i in range(1, 5)]  # fmt: skip
    bench.write(bench.freeze(records, "v1", test_fraction=0.0), tmp_path / "v1")
    monkeypatch.setattr(
        cli_benchmark, "_judge", lambda model: lambda u, e, ts: ["fire" in t for t in ts]
    )
    args = ["benchmark", "score", "--bench", str(tmp_path / "v1"), "--explanations", "registry",
            "--judge-model", "openai/judge", "--limit", "2"]  # fmt: skip
    for order, sample in (("slug", False), ("shuffled", True)):
        out = tmp_path / f"{order}.jsonl"
        result = CliRunner().invoke(cli, [*args, "--order", order, "--out", str(out)])
        assert result.exit_code == 0, result.output
        assert ("not a random sample" in result.output) is not sample
        summary = json.loads((tmp_path / f"{order}.summary.json").read_text())
        assert (summary["order"], summary["random_sample"]) == (order, sample)
        assert (summary["units"], summary["split_units"]) == (2, 4)
