"""Re-judging older VERIFIED records under today's rule (scripts/protocol_mix.py)."""

import importlib.util
from pathlib import Path

from tests.test_verdict_regression import judged  # noqa: F401  (fixture)

_SPEC = importlib.util.spec_from_file_location(
    "protocol_mix", Path(__file__).parents[1] / "scripts" / "protocol_mix.py"
)
protocol_mix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol_mix)


def _with_policy(record, **policy):
    old = record.verdict.policy.model_copy(update={"protocol_version": "0.7", **policy})
    return record.model_copy(update={"verdict": record.verdict.model_copy(update={"policy": old})})


def test_an_older_record_is_rejudged_under_todays_rule(judged):  # noqa: F811
    today = protocol_mix.ReviewPolicy()
    status, failed = protocol_mix.rejudge(_with_policy(judged), today)
    assert status == judged.verdict.status and failed == sorted(
        g for g, ok in judged.verdict.metrics.gates.items() if not ok
    )


def test_other_measurement_settings_need_a_new_measurement(judged):  # noqa: F811
    today = protocol_mix.ReviewPolicy()
    assert protocol_mix.rejudge(_with_policy(judged, activation_threshold=0.5), today) is None
    no_metrics = judged.model_copy(
        update={"verdict": judged.verdict.model_copy(update={"metrics": None})}
    )
    assert protocol_mix.rejudge(no_metrics, today) is None
