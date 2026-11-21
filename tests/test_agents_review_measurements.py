"""The reviewer's wiring readout of a unit that never fires (agents/review_measurements.py)."""

from openinterpretability.agents.review_measurements import _activation_wiring


def test_a_unit_with_no_positive_threshold_gets_no_wiring_and_runs_no_model(mocker):
    # A dead unit has scale 0, so its threshold is 0: no prompt can fire it.
    runner, submission = mocker.Mock(), mocker.Mock()
    assert _activation_wiring(submission, runner, [], 0.0) is None
    assert runner.mock_calls == [] and submission.mock_calls == []
