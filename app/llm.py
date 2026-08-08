"""Claude client for answer generation."""

from __future__ import annotations


class ClaudeLLM:  # pragma: no cover - requires an API key
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5") -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> tuple[str, int, int]:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in message.content
            if getattr(block, "type", "") == "text"
        )
        return text, message.usage.input_tokens, message.usage.output_tokens
