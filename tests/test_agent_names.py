import pytest

from openinterp.core.agent_names import canonical_agent, same_agent, spelling_problem


@pytest.mark.parametrize(
    "model, name",
    [
        ("external/deepseek-flash[1m]", "deepseek-flash"),
        ("external/deepseek-flash", "deepseek-flash"),
        ("external/gemini/gemini-3.8-flash", "gemini-3.8-flash"),
        ("anthropic/claude-opus-5-5 (Claude Code session)", "claude-opus-5-5"),
        ("external/Claude-Opus-5-5", "claude-opus-5-5"),
        ("external/muse-spark-1.3-contributor", "muse-spark-1.3"),
        ("external/muse-spark", "muse-spark-1.3"),
        ("external/muse-spark-3.1-contributor", "muse-spark-1.3"),
        ("openai/gpt-test", "gpt-test"),
    ],
)
def test_canonical_agent(model, name):
    assert canonical_agent(model) == name


def test_same_agent():
    assert same_agent("external/glm-5.3-flash", "external/glm-5.3-flash[1m]")
    assert not same_agent("external/glm-5.3-flash", "external/glm-4.7-flash")
    assert not same_agent("external/glm-5.3-flash", None)


@pytest.mark.parametrize(
    "model, expected",
    [
        ("external/deepseek-flash", None),
        ("anthropic/claude-opus-5-5", None),  # a LiteLLM string names what was called
        ("external/deepseek-flash[1m]", "external/deepseek-flash"),
        ("external/gemini/gemini-3.8-flash", "external/gemini-3.8-flash"),
        ("external/claude-opus-5-5 (Claude Code session)", "external/claude-opus-5-5"),
        ("external/muse-spark-1.3-contributor", "external/muse-spark-1.3"),
    ],
)
def test_spelling_problem(model, expected):
    problem = spelling_problem(model)
    assert problem is None if expected is None else f"spell it {expected} " in problem


def test_recorded_model_rejects_new_spellings(monkeypatch):
    from openinterp.agents.base import MODEL_ALIASES_ENV, recorded_model

    monkeypatch.delenv(MODEL_ALIASES_ENV, raising=False)
    assert recorded_model("external/deepseek-flash") == "external/deepseek-flash"
    with pytest.raises(ValueError, match="spell it external/deepseek-flash"):
        recorded_model("external/deepseek-flash[1m]")
    monkeypatch.setenv(MODEL_ALIASES_ENV, "claude-opus-5-5=glm-5.3-flash[1m]")
    with pytest.raises(ValueError, match="spell it external/glm-5.3-flash"):
        recorded_model("external/claude-opus-5-5")
