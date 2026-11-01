"""Abstract LLM agent with structured outputs.

Two backends: a LiteLLM model string (routed through instructor), or ``external/<name>``,
where any LLM, agent or person answers a prompt file written to disk.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ValidationError

from openinterp.core import costs
from openinterp.core.agent_names import EXTERNAL_PREFIX, spelling_problem
from openinterp.core.exceptions import AgentError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("OPENINTERP_AGENT_MODEL", "anthropic/claude-opus-5-5")
MODEL_ALIASES_ENV = "OPENINTERP_MODEL_ALIASES"


def recorded_model(model: str | None) -> str | None:
    """The name to record for ``model``, after ``$OPENINTERP_MODEL_ALIASES``.

    For external answerers behind a proxy, e.g. Claude Code routed through a LiteLLM
    proxy to Grok, whose subagents report the Claude model they think they are. The
    variable holds ``reported=actual`` pairs separated by commas
    (``claude-haiku-5-5=grok-4-fast,claude-opus-5-5=grok-4``), or one bare name that
    replaces every external model. Only ``external/<name>`` is rewritten: a LiteLLM
    string is what gets called, so it already names the model. Raises ``ValueError`` for
    an external name spelled other than the site would show it
    (:func:`openinterp.core.agent_names.spelling_problem`), so one model keeps one name.
    """
    spec = os.environ.get(MODEL_ALIASES_ENV, "").strip()
    if not model or not model.startswith(EXTERNAL_PREFIX):
        return model
    if not spec:
        return _checked(model)
    name = model.removeprefix(EXTERNAL_PREFIX)
    aliases, default = {}, None
    for entry in filter(None, (e.strip() for e in spec.split(","))):
        reported, sep, actual = entry.partition("=")
        if not sep:
            default = entry.removeprefix(EXTERNAL_PREFIX)
        elif reported.strip() and actual.strip():
            aliases[reported.strip().removeprefix(EXTERNAL_PREFIX)] = actual.strip()
        else:
            raise ValueError(f"${MODEL_ALIASES_ENV}: malformed entry {entry!r}")
    actual = aliases.get(name, default or name)
    return _checked(EXTERNAL_PREFIX + actual.removeprefix(EXTERNAL_PREFIX))


def _checked(model: str) -> str:
    if problem := spelling_problem(model):
        raise ValueError(f"{model!r}: {problem}")
    return model


class LLMAgent:
    """Base class for agents that return validated Pydantic objects.

    ``model`` is either a LiteLLM model string (``"anthropic/..."``, ``"openai/..."``,
    ``"ollama/..."``) or ``"external/<name>"``. An external model is answered through
    files in ``exchange_dir``: the agent writes ``<Agent>.prompt.md`` (with the response's
    JSON schema) and waits for ``<Agent>.response.json``. The full model string is what
    gets recorded, so name the model that actually answered, e.g. ``external/gpt-5``.
    Pass ``client`` to inject a pre-built instructor client (useful in tests).
    """

    system_prompt: ClassVar[str] = ""
    role: ClassVar[str] = "agent"  # the stage its calls are timed as

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        temperature: float | None = None,
        max_retries: int = 2,
        client: Any | None = None,
        exchange_dir: Path | None = None,
        timeout: float | None = None,
        poll_interval: float = 2.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = client
        self.exchange_dir = exchange_dir or Path(
            os.environ.get("OPENINTERP_EXCHANGE_DIR", "exchange")
        )
        self.timeout = timeout or float(os.environ.get("OPENINTERP_EXCHANGE_TIMEOUT", 3600))
        self.poll_interval = poll_interval

    @property
    def client(self) -> Any:
        if self._client is None:
            import instructor
            import litellm

            self._client = instructor.from_litellm(litellm.completion)
        return self._client

    @classmethod
    def render_prompt(
        cls, response_model: type[BaseModel], prompt: str, response_path: Path | str
    ) -> str:
        """The full prompt for an external answerer: system prompt, task and JSON schema."""
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        return (
            f"# System\n\n{cls.system_prompt}\n\n# Prompt\n\n{prompt}\n\n"
            f"# Response\n\nWrite a JSON object matching this schema to `{response_path}`:\n\n"
            f"```json\n{schema}\n```\n"
        )

    def structured(self, response_model: type[T], prompt: str) -> T:
        with costs.stage(self.role):
            if self.model.startswith(EXTERNAL_PREFIX):
                result = self._external(response_model, prompt)
                chars = len(self.system_prompt) + len(prompt)
                call = costs.LLMCall(
                    agent=self.role, model=self.model, external=True, prompt_chars=chars
                )
                costs.add_call(call)
                return result
            return self._completion(response_model, prompt)

    def _completion(self, response_model: type[T], prompt: str) -> T:
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        responses: list[Any] = []  # one raw response per attempt, for usage and cost
        client, hook = self.client, responses.append
        if hasattr(client, "on"):  # instructor's hooks; a stub client may lack them
            client.on("completion:response", hook)
        try:
            return client.chat.completions.create(
                model=self.model,
                response_model=response_model,
                max_retries=self.max_retries,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
            )
        except Exception as exc:  # instructor/litellm raise many provider-specific types
            raise AgentError(
                f"{type(self).__name__} ({self.model}) failed to produce "
                f"{response_model.__name__}: {exc}"
            ) from exc
        finally:
            if hasattr(client, "off"):
                client.off("completion:response", hook)
            chars = len(self.system_prompt) + len(prompt)
            costs.add_call(
                costs.LLMCall(
                    agent=self.role,
                    model=self.model,
                    prompt_chars=chars,
                    **costs.usage_of(responses),
                )
            )

    def _external(self, response_model: type[T], prompt: str) -> T:
        """Write the prompt to disk and wait for a valid JSON response from anyone."""
        name = type(self).__name__
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.exchange_dir / f"{name}.prompt.md"
        response_path = self.exchange_dir / f"{name}.response.json"
        error_path = self.exchange_dir / f"{name}.error.txt"
        for stale in (response_path, error_path):
            stale.unlink(missing_ok=True)
        _write_whole(prompt_path, self.render_prompt(response_model, prompt, response_path))
        print(f"Waiting for {response_path} (prompt: {prompt_path})…", file=sys.stderr)
        deadline = time.monotonic() + self.timeout
        seen = None  # the last unparseable response, rejected only once it stops changing
        while time.monotonic() < deadline:
            text = response_path.read_text() if response_path.exists() else ""
            if text.strip():
                try:
                    result = response_model.model_validate_json(text)
                except ValidationError as exc:
                    if text != seen:  # possibly caught mid-write: look again next poll
                        seen = text
                        time.sleep(self.poll_interval)
                        continue
                    seen = None
                    _write_whole(error_path, str(exc))
                    response_path.unlink()
                    print(f"Invalid response, see {error_path}", file=sys.stderr)
                else:
                    error_path.unlink(missing_ok=True)
                    return result
            time.sleep(self.poll_interval)
        raise AgentError(f"{name} ({self.model}) got no response at {response_path}.")


def _write_whole(path: Path, text: str) -> None:
    """Write ``text`` so a reader polling for ``path`` never sees it half-written."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(text)
    partial.replace(path)
