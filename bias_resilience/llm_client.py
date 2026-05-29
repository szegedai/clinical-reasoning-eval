"""Thin OpenAI-compatible LLM client with retry, timeout, and thinking-token capture."""
from __future__ import annotations

import time
from dataclasses import dataclass

import openai

from .config import ModelConfig, MAX_RETRIES, CALL_TIMEOUT_S


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    tokens_thinking: int    # 0 unless provider surfaces reasoning tokens
    latency_ms: float


class LLMClient:
    """Wraps openai.OpenAI for one (provider, model) pair."""

    def __init__(self, model_config: ModelConfig, *, max_retries: int = MAX_RETRIES):
        self._cfg = model_config
        self._max_retries = max_retries
        self._client = openai.OpenAI(
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            timeout=CALL_TIMEOUT_S,
            default_headers=model_config.extra_headers,
        )

    @property
    def model_id(self) -> str:
        return self._cfg.model_id

    @property
    def model_name(self) -> str:
        return self._cfg.name

    def call(self, messages: list[dict], *, temperature: float | None = None) -> LLMResponse:
        """Call the LLM and return a structured response.

        Retries on API errors with exponential backoff (2s → 4s → 8s).
        Captures thinking/reasoning tokens if the provider surfaces them.
        """
        temp = temperature if temperature is not None else self._cfg.temperature
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                t0 = time.monotonic()
                kwargs: dict = dict(
                    model=self._cfg.model_id,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self._cfg.max_tokens,
                )
                if self._cfg.extra_body:
                    kwargs["extra_body"] = self._cfg.extra_body

                response = self._client.chat.completions.create(**kwargs)
                latency_ms = (time.monotonic() - t0) * 1000

                text = response.choices[0].message.content or ""
                usage = response.usage

                tokens_in = getattr(usage, "prompt_tokens", 0) or 0
                tokens_out = getattr(usage, "completion_tokens", 0) or 0

                tokens_thinking = 0
                if usage:
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        tokens_thinking = getattr(details, "reasoning_tokens", 0) or 0

                return LLMResponse(
                    text=text,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    tokens_thinking=tokens_thinking,
                    latency_ms=latency_ms,
                )

            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** (attempt + 1))

        raise last_error  # type: ignore[misc]
