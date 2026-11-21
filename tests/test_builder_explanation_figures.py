"""An explanation page's figures, when the record lacks what one needs (#4479)."""

from openinterpretability.builder import explanation_figures as ef
from tests.test_explanation_figures import _data


def test_a_figure_with_no_data_is_not_drawn():
    data = _data().model_copy(update={"coverage": None, "reading": {}, "ablation": []})
    assert ef.coverage(data, [0.1, 0.2]) is None
    assert ef.reading(data) is None
    assert ef.ablation(data) is None
    assert [d.name for d in ef.figures(data, "x")] == ["graph", "waterfall"]
