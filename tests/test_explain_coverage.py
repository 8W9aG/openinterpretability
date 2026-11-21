"""Coverage gains of planned experiments (openinterpretability/explain_coverage.py)."""

from types import SimpleNamespace

from openinterpretability import explain_coverage as ec
from openinterpretability import explain_plan as ep


def test_no_gain_is_measured_when_ablating_every_unit_leaves_the_answer_as_it_was(monkeypatch):
    # The model's log-probability is the same with and without the ablation.
    asked = []
    monkeypatch.setattr(ec, "keep_hooks", lambda runner, unit_means, units: list(units))
    monkeypatch.setattr(
        ec, "_log_prob", lambda r, p, t, pos, hooks=None: asked.append(hooks) or -1.0
    )
    experiments = [ep.PlannedExperiment(target=name, entity="pole", share=0.1, status="unstudied",
                                        draw="explore", rationale="", commands=[], units=[name])
                   for name in ("a", "b")]  # fmt: skip
    planned = SimpleNamespace(experiments=experiments, groups=[])
    ec.add_gains(None, "p", None, 0, -1, [("k", 0.5, "unstudied", True)], planned)
    assert [e.target for e in planned.experiments] == ["a", "b"]
    assert [e.gain for e in planned.experiments] == [None, None]
    assert asked == [None, [], ["k"]]  # clean, all ablated, the explained unit kept: no more
