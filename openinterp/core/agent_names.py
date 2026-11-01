"""One name per agent model, however a record spelled it.

``submitting_agent_model``, ``reviewer_agent_model`` and ``judge_model`` are free text, and
sessions spell one model several ways: ``external/deepseek-flash[1m]`` and
``external/deepseek-flash``, ``external/gemini/gemini-3.8-flash`` and
``external/gemini-3.8-flash``, ``anthropic/claude-opus-5-5 (Claude Code session)`` and
``external/claude-opus-5-5``. Records are signed, so the site build maps them to one name
with :func:`canonical_agent` rather than rewriting them, and :func:`spelling_problem`
rejects new spellings at submission.
"""

from __future__ import annotations

import re

EXTERNAL_PREFIX = "external/"

# Spellings that name the same model, after the generic clean-up in canonical_agent.
# Muse Spark sessions report their version inconsistently, and "-contributor" is its
# pricing tier (openinterp.core.pricing.FAMILIES), not part of the model's name; every
# run so far was 1.3 at the Contributor tier.
ALIASES: dict[str, str] = {
    "muse-spark": "muse-spark-1.3",
    "muse-spark-1.3-contributor": "muse-spark-1.3",
    "muse-spark-1.3.3": "muse-spark-1.3",
    "muse-spark-1.5-contributor": "muse-spark-1.3",
    "muse-spark-3.1-contributor": "muse-spark-1.3",
}

# Shown on an agent's page when its recorded name is doubtful.
NOTES: dict[str, str] = {
    "claude-3-5-sonnet-20241022": (
        "Self-reported by one Claude Code session on 2026-10-26, and probably wrong: this "
        "model was retired long before. Its records can't say which model actually ran."
    ),
}


def canonical_agent(model: str) -> str:
    """The model's one name: ``external/deepseek-flash[1m]`` → ``deepseek-flash``.

    Drops the route (``external/``, a LiteLLM provider such as ``anthropic/``), a context
    suffix (``[1m]``), a parenthesised remark and case, then applies :data:`ALIASES`.
    """
    name = re.sub(r"\s*\(.*\)\s*$", "", model.strip()).removeprefix(EXTERNAL_PREFIX)
    name = re.sub(r"\[[^\]]*\]$", "", name.rsplit("/", 1)[-1]).strip().lower()
    return ALIASES.get(name, name) or "unknown"


def same_agent(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and canonical_agent(a) == canonical_agent(b)


def spelling_problem(model: str) -> str | None:
    """Why ``model`` isn't a name a new record may carry, or ``None`` when it is.

    An external answerer must be ``external/<name>`` with ``<name>`` already canonical. A
    LiteLLM string is what gets called, so it already names the model and passes.
    """
    if not model.startswith(EXTERNAL_PREFIX):
        return None
    expected = EXTERNAL_PREFIX + canonical_agent(model)
    if model != expected:
        return f"spell it {expected} (no provider, context size, remark or version alias)"
    return None
