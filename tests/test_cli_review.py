"""The review command's helpers: sisters, aligned latents, option checks and warnings."""

from types import SimpleNamespace

import click
import pytest

from openinterpretability import crossref
from openinterpretability.agents.reviewer import AdversarialProbe
from openinterpretability.cli_review import (
    aligned_latents_text,
    check_review_options,
    sisters_text,
    warn_self_review,
    warn_unequal_lengths,
)
from openinterpretability.core.identifiers import NeuronIdentifier
from tests.conftest import FakeTokenizer
from tests.test_sisters import _record, _sister_row


def test_sisters_text_shows_a_studied_sister_and_says_so(tmp_path, monkeypatch, record, capsys):
    from openinterpretability.builder.wiki import write_experiment_page

    monkeypatch.chdir(tmp_path)  # the sister rows are read from docs/analyses/ here
    sister = _record(record, "o/m", 2, 9, "negative")
    sister = sister.model_copy(update={"submission": sister.submission.model_copy(
        update={"natural_language_explanation": "closing quotes"})})  # fmt: skip
    write_experiment_page(sister, tmp_path / "wiki")
    pole = NeuronIdentifier(model_name="a/m", layer=1, neuron_idx=5, pole="positive")
    rows = tmp_path / "docs/analyses/sister-neurons.jsonl"
    rows.parent.mkdir(parents=True)
    matches = [(2, 9, "negative", 0.8, pole.slug, 0.8)]
    rows.write_text(_sister_row(pole.slug, "o/m", matches).model_dump_json() + "\n")
    text = sisters_text(pole, tmp_path / "wiki", "reviewer")
    assert "- o--m/L2/N9-negative (r +0.80, sister; VERIFIED): closing quotes" in text
    err = capsys.readouterr().err
    # The count is not checked: it counts the fence lines too (reported on #5855).
    assert err.startswith("Showing the reviewer ") and err.endswith(" studied sister neurons\n")


def test_sisters_text_is_empty_when_the_wiki_is_skipped(record):
    assert sisters_text(record.submission.identifier, "", "proposer") == ""


def _alignment_rows(tmp_path, slug):
    side = {"matches": [{"feature_idx": 7, "cosine": 0.6}], "chance": 0.1, "percentile": 0.99}
    row = crossref.AlignmentRow(slug=slug, dictionary="res-jb", repo="r", revision="v",
                                path="p", model_revision=None, writes=side)  # fmt: skip
    path = tmp_path / "docs/analyses/dictionary-alignment.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(row.model_dump_json() + "\n")


def test_aligned_latents_text_describes_each_aligned_latent(tmp_path, monkeypatch, submission):
    monkeypatch.chdir(tmp_path)
    _alignment_rows(tmp_path, submission.identifier.slug)
    asked = []

    def explanations(row, feature_idx, **kwargs):  # Neuronpedia, the network boundary
        asked.append(feature_idx)
        return [crossref.Explanation(description="dates in prose", explainer="gpt-test")]

    monkeypatch.setattr(crossref, "neuronpedia_explanations", explanations)
    (tmp_path / "wiki").mkdir()
    text = aligned_latents_text(submission, tmp_path / "wiki")
    assert asked == [7]
    assert "- res-jb latent 7 writes along this pole (cosine +0.60; chance 0.10): " in text
    assert "dates in prose [Neuronpedia, gpt-test]" in text


def test_aligned_latents_text_is_empty_and_warns_when_the_pole_has_none(
    tmp_path, monkeypatch, submission, capsys
):
    monkeypatch.chdir(tmp_path)
    assert aligned_latents_text(submission, "") == ""
    slug = submission.identifier.slug
    assert capsys.readouterr().err == f"warning: no aligned latents for {slug}\n"


def test_skipping_the_registry_needs_an_out_file(tmp_path):
    with pytest.raises(click.UsageError, match="--out is required when --wiki-dir '' skips"):
        check_review_options("", None, None, "", None, tmp_path / "corpus.jsonl")


def test_a_self_review_is_warned_about(submission, capsys):
    warn_self_review(submission.submitting_agent_model, submission)
    err = capsys.readouterr().err
    assert err.startswith("warning: openai/gpt-test both proposed and reviewed this")
    warn_self_review("external/other-model", submission)
    assert capsys.readouterr().err == ""


def _probes(active_text, inactive_text):
    return [AdversarialProbe(text=t, expected_active=a, rationale="r")
            for t, a in ((active_text, True), (inactive_text, False))]  # fmt: skip


def test_probe_groups_of_unequal_length_are_warned_about(submission, capsys):
    runner = SimpleNamespace(tokenizer=FakeTokenizer())
    warn_unequal_lengths(_probes("a" * 30, "b" * 10), runner, submission)
    err = capsys.readouterr().err
    assert err.startswith("warning: The median token lengths of the active and inactive ")
    assert "differ 3.0×" in err
    warn_unequal_lengths(_probes("a" * 10, "b" * 10), runner, submission)
    assert capsys.readouterr().err == ""


def test_no_length_warning_without_a_tokenizer(submission, capsys):
    runner = SimpleNamespace(tokenizer=None)
    warn_unequal_lengths(_probes("a" * 30, "b"), runner, submission)
    assert capsys.readouterr().err == ""
