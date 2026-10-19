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

from openinterp.core.exceptions import AgentError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("OPENINTERP_AGENT_MODEL", "anthropic/claude-opus-5-5")
EXTERNAL_PREFIX = "external/"


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

    def structured(self, response_model: type[T], prompt: str) -> T:
        if self.model.startswith(EXTERNAL_PREFIX):
            return self._external(response_model, prompt)
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            return self.client.chat.completions.create(
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

    def _external(self, response_model: type[T], prompt: str) -> T:
        """Write the prompt to disk and wait for a valid JSON response from anyone."""
        name = type(self).__name__
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.exchange_dir / f"{name}.prompt.md"
        response_path = self.exchange_dir / f"{name}.response.json"
        error_path = self.exchange_dir / f"{name}.error.txt"
        for stale in (response_path, error_path):
            stale.unlink(missing_ok=True)
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        prompt_path.write_text(
            f"# System\n\n{self.system_prompt}\n\n# Prompt\n\n{prompt}\n\n"
            f"# Response\n\nWrite a JSON object matching this schema to `{response_path}`:\n\n"
            f"```json\n{schema}\n```\n"
        )
        print(f"Waiting for {response_path} (prompt: {prompt_path})…", file=sys.stderr)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if response_path.exists():
                try:
                    result = response_model.model_validate_json(response_path.read_text())
                except ValidationError as exc:
                    error_path.write_text(str(exc))
                    response_path.unlink()
                    print(f"Invalid response, see {error_path}", file=sys.stderr)
                else:
                    error_path.unlink(missing_ok=True)
                    return result
            time.sleep(self.poll_interval)
        raise AgentError(f"{name} ({self.model}) got no response at {response_path}.")
