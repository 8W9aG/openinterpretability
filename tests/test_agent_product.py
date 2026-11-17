"""The product and host recorded with each run (issue #3235)."""

import pytest

from openinterpretability.harness import environment


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for var in ("CLAUDECODE", environment.PRODUCT_ENV, *(v for v, _ in environment.HOSTS)):
        monkeypatch.delenv(var, raising=False)


def test_claude_code_is_detected_and_a_named_product_wins(monkeypatch):
    assert environment.agent_product() is None
    monkeypatch.setenv("CLAUDECODE", "1")
    assert environment.agent_product() == "claude-code"
    monkeypatch.setenv(environment.PRODUCT_ENV, "muse-spark")
    assert environment.agent_product() == "muse-spark"


def test_a_hosted_environment_is_named_and_local_is_none(monkeypatch):
    assert environment.host() is None
    monkeypatch.setenv("CI", "true")
    assert environment.host() == "ci"
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-x")
    assert environment.host() == "colab"  # the first one set wins


def test_a_run_records_them(tiny_gpt2, monkeypatch):
    from openinterpretability.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    env = environment.run_environment(NeuronHookRunner(tiny_gpt2, FakeTokenizer()))
    assert (env.agent_product, env.host) == ("claude-code", "github-actions")


def test_the_parquet_export_reads_the_verdicts_environment(record):
    from openinterpretability.builder.parquet_export import _product
    from openinterpretability.core.run_environment import RunEnvironment

    env = RunEnvironment(device="cpu", dtype="float32", openinterp_version="t",
                         torch_version="t", agent_product="claude-code", host="colab")  # fmt: skip
    rec = record.model_copy(
        update={"verdict": record.verdict.model_copy(update={"environment": env})}
    )
    assert _product(rec) == {"agent_product": "claude-code", "host": "colab"}
