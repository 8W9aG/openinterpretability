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
