"""One name per agent model, however a record spelled it.

``submitting_agent_model``, ``reviewer_agent_model`` and ``judge_model`` are free text.
Sessions spell one model in several ways. For example:

- ``external/deepseek-flash[1m]`` and ``external/deepseek-flash``.
- ``external/gemini/gemini-3.8-flash`` and ``external/gemini-3.8-flash``.
- ``anthropic/claude-opus-5-5 (Claude Code session)`` and ``external/claude-opus-5-5``.

Records are signed, so the site build does not rewrite them. It maps each spelling to one
name with :func:`canonical_agent`. :func:`spelling_problem` rejects new spellings at
submission.
"""

from __future__ import annotations

import re

EXTERNAL_PREFIX = "external/"

# Spellings that name the same model, after the generic clean-up in canonical_agent.
# One 2026-10-26 session ran Grok Build but reported itself as Claude 3.5 Sonnet.
ALIASES: dict[str, str] = {
    "claude-3-5-sonnet-20241022": "grok-build-0.1",
}

# Model families whose sessions report their version inconsistently. canonical_agent maps
# every name that starts with the key to the value. Muse Spark sessions have recorded
# muse-spark, muse-spark-3, muse-spark-1-3-contributor and others. "-contributor" is the
# Muse Spark pricing tier (openinterp.core.pricing.FAMILIES), not part of the model's
# name. Every run so far was 1.3 at the Contributor tier.
FAMILY_ALIASES: dict[str, str] = {
    "muse-spark": "muse-spark-1.3",
}

# Shown on an agent's page when its recorded name is doubtful.
NOTES: dict[str, str] = {
    "grok-build-0.1": (
        "One session on 2026-10-26 ran this model but recorded itself as "
        "claude-3-5-sonnet-20241022; its records are signed, so the name is remapped here."
    ),
}


def canonical_agent(model: str) -> str:
    """The model's one name: ``external/deepseek-flash[1m]`` → ``deepseek-flash``.

    Drops the route (``external/``, a LiteLLM provider such as ``anthropic/``), a context
    suffix (``[1m]``), a parenthesised remark and case. Then applies :data:`FAMILY_ALIASES`
    and :data:`ALIASES`.
    """
    name = re.sub(r"\s*\(.*\)\s*$", "", model.strip()).removeprefix(EXTERNAL_PREFIX)
    name = re.sub(r"\[[^\]]*\]$", "", name.rsplit("/", 1)[-1]).strip().lower()
    for family, canonical in FAMILY_ALIASES.items():
        if name.startswith(family):
            return canonical
    return ALIASES.get(name, name) or "unknown"


def same_agent(a: str | None, b: str | None) -> bool:
    """True when both names are set and name one model after ``ALIASES`` and spelling fixes."""
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
