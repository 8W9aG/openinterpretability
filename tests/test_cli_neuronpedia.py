"""The `open-interpretability neuronpedia-review` command, end to end on a tiny dictionary.

Neuronpedia's answers come from its on-disk cache, so no test reaches the network.
"""

import json
from dataclasses import replace

from click.testing import CliRunner

from openinterpretability.agents.reviewer import AdversarialProbe, ProbeSet
from openinterpretability.cli import cli
from openinterpretability.core.paths import experiments_dir
from openinterpretability.core.schemas import FeatureIdentifier, VerificationVerdict
from openinterpretability.harness import dictionaries
from tests.test_feature_audit import (  # noqa: F401  (fixtures)
    FEATURE,
    TEXTS,
    _record,
    runner,
    scan,
    spec,
)


def _with(record, ident=None, **verdict):
    """A copy of ``record`` with another identifier or other verdict fields."""
    sub = record.submission.model_copy(update={"identifier": ident or FEATURE})
    return record.model_copy(
        update={"submission": sub, "verdict": record.verdict.model_copy(update=verdict)}
    )


def _archive(wiki, records):
    root = experiments_dir(wiki)
    root.mkdir(parents=True)
    for i, record in enumerate(records):
        (root / f"r{i}.json").write_text(record.model_dump_json())


def _cache(root, feature_idx, explanations):
    path = root / "tiny" / "1-tc" / f"{feature_idx}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(explanations))


def test_the_review_writes_prompts_then_measures_the_probes_then_reuses_the_verdict(
    runner,  # noqa: F811
    scan,  # noqa: F811
    spec,  # noqa: F811
    record,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(
        dictionaries.DICTIONARIES, "tiny-tc", replace(spec, neuronpedia="tiny/{layer}-tc")
    )
    cache = tmp_path / "np"
    monkeypatch.setenv("OPENINTERPRETABILITY_NEURONPEDIA_CACHE", str(cache))
    _cache(cache, 5, [{"description": "repeated letters", "explainer": "gpt-4o-mini"}])
    _cache(cache, 6, [{"description": "a human's words", "explainer": "human"}])
    _cache(cache, 8, [{"description": "vowels", "explainer": "gpt-4o-mini"}])

    feature = _record(runner, scan)
    older = feature.verdict.timestamp.replace(year=2000)
    human_only = FeatureIdentifier(
        model_name="tiny-gpt2", layer=1, dictionary="tiny-tc", feature_idx=6
    )
    unhosted = FeatureIdentifier(
        model_name="tiny-gpt2", layer=1, dictionary="nowhere", feature_idx=5
    )
    second_latent = FEATURE.model_copy(update={"feature_idx": 8})
    wiki, work, out = tmp_path / "wiki", tmp_path / "work", tmp_path / "out" / "np-review"
    _archive(
        wiki,
        [
            feature,
            _with(feature, timestamp=older),  # an older review of the same latent
            feature,  # the same review again, at the same time
            _with(feature, second_latent),
            _with(feature, human_only),  # Neuronpedia has only a human explanation
            _with(feature, unhosted),  # Neuronpedia does not host the dictionary
            _with(feature, FEATURE.model_copy(update={"feature_idx": 7}), metrics=None),
            record,  # a neuron, not a dictionary latent
        ],
    )
    args = ["neuronpedia-review", "--wiki-dir", str(wiki), "--work-dir", str(work),
            "--out", str(out)]  # fmt: skip
    unit, unit8 = work / FEATURE.slug, work / second_latent.slug

    first = CliRunner().invoke(cli, args)
    assert first.exit_code == 0, first.output
    assert f"2 latents await probe sets under {work}." in first.output
    assert (unit / "review.prompt.md").exists() and (unit / "submission.json").exists()
    assert not (work / human_only.slug).exists()  # no machine explanation, so no row
    row, row8 = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
    assert (row["slug"], row8["slug"]) == (FEATURE.slug, second_latent.slug)
    assert row["neuronpedia"] is None and row8["neuronpedia"] is None
    assert row["neuronpedia_explanation"] == "repeated letters"
    assert row["url"] == "https://www.neuronpedia.org/tiny/1-tc/5"
    assert out.with_suffix(".md").exists()

    # A second run before the probes keeps the prompt the reviewer may be reading.
    (unit / "review.prompt.md").write_text("being answered")
    again = CliRunner().invoke(cli, args)
    assert again.exit_code == 0 and "2 latents await probe sets" in again.output
    assert (unit / "review.prompt.md").read_text() == "being answered"

    probes = [
        AdversarialProbe(text=t, expected_active=i % 2 == 0, rationale="r")
        for i, t in enumerate(TEXTS + ["xyzxyz", "plain", "abab", "nope!"])
    ]
    for where in (unit, unit8):
        (where / "review.probes.json").write_text(ProbeSet(probes=probes).model_dump_json())
    loads = []

    def load(ident, device, dtype, revision=None):
        loads.append(ident.model_name)
        return runner

    monkeypatch.setattr("openinterpretability.cli._load_unit_runner", load)
    second = CliRunner().invoke(cli, args)
    assert second.exit_code == 0, second.output
    assert "await probe sets" not in second.output
    assert loads == ["tiny-gpt2"]  # both latents are measured with one model load
    assert (unit8 / "verdict.json").exists()
    verdict = VerificationVerdict.model_validate_json((unit / "verdict.json").read_text())
    rows = out.with_suffix(".jsonl").read_text()
    row = json.loads(rows.splitlines()[0])
    assert row["reviewer_model"] == "external/claude-sonnet-5-5"
    assert row["neuronpedia"]["balanced_accuracy"] == verdict.metrics.balanced_accuracy

    # The third run reads the measured verdict and loads no model.
    third = CliRunner().invoke(cli, args)
    assert third.exit_code == 0, third.output
    assert loads == ["tiny-gpt2"]
    assert out.with_suffix(".jsonl").read_text() == rows


def test_the_review_stops_when_neuronpedia_cannot_be_read(
    runner,  # noqa: F811
    scan,  # noqa: F811
    spec,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(
        dictionaries.DICTIONARIES, "tiny-tc", replace(spec, neuronpedia="tiny/{layer}-tc")
    )
    # The request failed: the network is the boundary that this test replaces.
    monkeypatch.setattr("openinterpretability.crossref.latent_explanations", lambda *a, **k: None)
    wiki = tmp_path / "wiki"
    _archive(wiki, [_record(runner, scan)])
    result = CliRunner().invoke(
        cli,
        [
            "neuronpedia-review",
            "--wiki-dir",
            str(wiki),
            "--work-dir",
            str(tmp_path / "w"),
            "--out",
            str(tmp_path / "o"),
        ],  # fmt: skip
    )
    assert result.exit_code == 1
    assert "could not read Neuronpedia for" in result.output and "try again later" in result.output
