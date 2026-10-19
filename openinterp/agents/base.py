"""Abstract LLM agent with structured outputs (LiteLLM routing + instructor validation)."""

from __future__ import annotations

import os
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from openinterp.core.exceptions import AgentError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("OPENINTERP_AGENT_MODEL", "anthropic/claude-opus-5-5")


class LLMAgent:
    """Base class for agents that return validated Pydantic objects.

    ``model`` is any LiteLLM model string (``"anthropic/..."``, ``"openai/..."``,
    ``"ollama/..."``), so proposers and reviewers can run on different providers.
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
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import instructor
            import litellm

            self._client = instructor.from_litellm(litellm.completion)
        return self._client

    def structured(self, response_model: type[T], prompt: str) -> T:
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
