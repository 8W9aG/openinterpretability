import json

import pytest

from openinterpretability import planner_models
from openinterpretability.core.identifiers import HeadIdentifier, NeuronIdentifier
from openinterpretability.planner_models import ModelCoverage, model_coverage


def test_coverage_of_a_model_of_unknown_size_shows_no_total():
    assert str(ModelCoverage("a/x", 1200, None, 7)) == "1,200 covered, 7 verified"
    assert str(ModelCoverage("a/x", 3, 1000, 1)) == "3 covered, 1/1,000 verified (0.10%)"


def test_an_unreadable_config_gives_no_config_and_no_heads(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", offline)
    assert planner_models.model_config("a/x", download=False) is None
    assert planner_models.model_heads("a/x", download=False) is None


def test_dead_and_stopped_units_outside_the_pool_or_not_neurons_are_not_counted(monkeypatch):
    monkeypatch.setattr(planner_models, "model_dims", lambda model, download=True: (2, 10))
    other = NeuronIdentifier(model_name="b/y", layer=0, neuron_idx=1, pole="positive")
    head = HeadIdentifier(model_name="a/x", layer=0, head_idx=2)
    coverage = model_coverage(["a/x"], [], [], dead=[other, head], stopped=[other, head])
    assert coverage == {"a/x": ModelCoverage("a/x", 0, 20, 0)}


# The config readers, with the Hugging Face Hub faked (#6810).


def fake_hub(monkeypatch, tmp_path, config, cached=True):
    """Fake hf_hub_download and the cache; return the calls made to each."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    calls = {"download": [], "cache": []}

    def download(model, name):
        calls["download"].append((model, name))
        return str(path)

    def cache(model, name):
        calls["cache"].append((model, name))
        return str(path) if cached else object()  # an object: cached as missing

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", cache)
    return calls


def test_a_config_is_downloaded_or_read_from_the_cache_only(monkeypatch, tmp_path):
    calls = fake_hub(monkeypatch, tmp_path, {"n_layer": 2})
    assert planner_models.model_config("a/x") == {"n_layer": 2}
    assert calls == {"download": [("a/x", "config.json")], "cache": []}
    assert planner_models.model_config("a/x", download=False) == {"n_layer": 2}
    assert calls["download"] == [("a/x", "config.json")]  # no second download
    assert calls["cache"] == [("a/x", "config.json")]


def test_a_config_cached_as_missing_is_no_config(monkeypatch, tmp_path):
    fake_hub(monkeypatch, tmp_path, {"n_layer": 2}, cached=False)
    assert planner_models.model_config("a/x", download=False) is None
    assert planner_models.model_dims("a/x", download=False) is None


@pytest.mark.parametrize(
    ("config", "dims", "heads"),
    [
        ({"num_hidden_layers": 24, "intermediate_size": 4864, "num_attention_heads": 14},
         (24, 4864), (24, 14)),
        ({"n_layer": 12, "n_inner": None, "n_embd": 768, "n_head": 12}, (12, 3072), (12, 12)),
        ({"n_layer": 6, "n_inner": 1000, "n_head": 8}, (6, 1000), (6, 8)),
        ({"n_layer": 6}, None, None),
        ({"n_head": 8, "n_embd": 512}, None, None),
    ],
)  # fmt: skip
def test_dims_and_heads_come_from_either_naming_of_the_config(
    monkeypatch, tmp_path, config, dims, heads
):
    calls = fake_hub(monkeypatch, tmp_path, config)
    assert planner_models.config_dims(config) == dims
    assert planner_models.model_dims("a/x", download=False) == dims
    assert calls == {"download": [], "cache": [("a/x", "config.json")]}
    assert planner_models.model_heads("b/y") == heads
    assert calls["download"] == [("b/y", "config.json")]
    assert planner_models.model_dims("c/z") == dims  # downloads by default too
    assert planner_models.model_heads("d/w", download=False) == heads
    assert [m for m, _ in calls["download"]] == ["b/y", "c/z"]


def test_coverage_reads_a_models_size_from_the_cache_only(monkeypatch):
    asked = []
    monkeypatch.setattr(planner_models, "model_dims",
                        lambda model, **kw: asked.append((model, kw)) or (2, 10))  # fmt: skip
    assert model_coverage(["a/x"], [], [])["a/x"].total == 20
    assert asked == [("a/x", {"download": False})]


def test_records_outside_the_pool_or_of_heads_are_skipped_not_the_rest(monkeypatch, record):
    monkeypatch.setattr(planner_models, "model_dims", lambda model, **kw: (2, 10))
    model = record.submission.identifier.model_name
    elsewhere = record.model_copy(update={"submission": record.submission.model_copy(
        update={"identifier": record.submission.identifier.model_copy(
            update={"model_name": "b/y"})})})  # fmt: skip
    head = record.model_copy(update={"submission": record.submission.model_copy(
        update={"identifier": HeadIdentifier(model_name=model, layer=0, head_idx=2)})})  # fmt: skip
    coverage = model_coverage([model], [elsewhere, head, record], [])[model]
    assert coverage.covered == 1


def test_each_verified_neuron_counts_once(monkeypatch, record):
    monkeypatch.setattr(planner_models, "model_dims", lambda model, **kw: (2, 10))
    ident = record.submission.identifier
    moved = ident.model_copy(update={"neuron_idx": ident.neuron_idx + 1})
    sub = record.submission.model_copy(update={"identifier": moved})
    other = record.model_copy(update={"submission": sub})
    assert record.verdict.status == "VERIFIED"
    coverage = model_coverage([ident.model_name], [record, other, record], [])[ident.model_name]
    assert (coverage.covered, coverage.verified) == (2, 2)


def test_each_neuron_proven_dead_counts_once_as_verified(monkeypatch):
    monkeypatch.setattr(planner_models, "model_dims", lambda model, **kw: (2, 10))
    dead = [NeuronIdentifier(model_name="a/x", layer=0, neuron_idx=i, pole="positive")
            for i in (1, 2, 2)]  # fmt: skip
    coverage = model_coverage(["a/x"], [], [], dead=dead)["a/x"]
    assert (coverage.covered, coverage.verified) == (2, 2)
