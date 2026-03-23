"""
OpenAI Adapter — Concrete LLMAdapter for OpenAI Chat Completions
=================================================================
PURPOSE:
  Wire OpenAI's Chat Completions API into LLMExecutor. Reads the API key
  from the OPENAI_API_KEY environment variable. No key is ever stored in
  source code.

SUPPORTED MODELS:
  - 'gpt-4o-mini' (default) — fast, cheap, capable; good for most tasks
  - 'gpt-4o'                — highest capability, higher cost
  - 'gpt-3.5-turbo'        — legacy, cheapest option

  Cost table (_COST_PER_1K_TOKENS below) covers blended input+output pricing.
  Update figures from https://openai.com/pricing during effectiveness reviews.

USAGE:
  from execution.openai_adapter import OpenAIAdapter
  from execution.llm_executor import LLMExecutor

  executor = LLMExecutor()
  executor.register('gpt4o_mini', OpenAIAdapter())        # default model
  executor.register('gpt4o',      OpenAIAdapter('gpt-4o'))

  result = executor.execute(prompt, allocation)
  # result: {'text': '...', 'tokens_used': int, 'cost_usd': float, ...}

ENVIRONMENT:
  OPENAI_API_KEY  — required; your OpenAI secret key (sk-...)
  OPENAI_BASE_URL — optional; override for proxies / Azure OpenAI endpoints
"""

from __future__ import annotations

import os

from execution.llm_executor import LLMAdapter

# ── Per-model blended cost (input + output) in USD per 1,000 tokens ──────────
# Update from https://openai.com/pricing during scheduled effectiveness reviews.
# Figures are blended estimates assuming roughly equal prompt and completion.
_COST_PER_1K_TOKENS: dict = {
    'gpt-4o-mini':   0.0003,   # $0.15/1M input + $0.60/1M output → ~$0.30/1M blended
    'gpt-4o':        0.00625,  # $2.50/1M input + $10.00/1M output → ~$6.25/1M blended
    'gpt-3.5-turbo': 0.00075,  # $0.50/1M input + $1.50/1M output  → ~$0.75/1M blended
}

_DEFAULT_MODEL = 'gpt-4o-mini'


class OpenAIAdapter(LLMAdapter):
    """
    LLMAdapter implementation for OpenAI Chat Completions.

    Args:
        model:       OpenAI model name (default: 'gpt-4o-mini').
        temperature: Sampling temperature passed to the API (default: 0.7).
        system:      Optional system prompt prepended to every call.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set when the adapter is
                          first used (checked lazily to allow import without key).
        ImportError:      If the 'openai' package is not installed.
    """

    def __init__(self, model: str = _DEFAULT_MODEL,
                 temperature: float = 0.7,
                 system: str = ''):
        self.model = model
        self.temperature = temperature
        self.system = system
        self._client = None  # Created lazily on first complete() call

    def _get_client(self):
        """Lazily initialise the OpenAI client."""
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use OpenAIAdapter. "
                "Install it with: pip install openai"
            ) from exc
        api_key = os.environ.get('OPENAI_API_KEY', '').strip()
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export your OpenAI secret key before running."
            )
        base_url = os.environ.get('OPENAI_BASE_URL') or None
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def complete(self, prompt: str, max_tokens: int, **kwargs) -> dict:
        """
        Call the OpenAI Chat Completions API and return a standardised result.

        Args:
            prompt:     The user prompt to send.
            max_tokens: Maximum completion tokens (maps to max_completion_tokens).
            **kwargs:   Extra parameters forwarded to the API call
                        (e.g. temperature override, seed, top_p).

        Returns:
            dict with:
              text:        str — generated text
              tokens_used: int — total tokens consumed (prompt + completion)
              cost_usd:    float — estimated cost for this call
        """
        client = self._get_client()

        messages = []
        if self.system:
            messages.append({'role': 'system', 'content': self.system})
        messages.append({'role': 'user', 'content': prompt})

        temperature = kwargs.pop('temperature', self.temperature)

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        text = ''
        if response.choices:
            text = response.choices[0].message.content or ''

        tokens_used = 0
        if response.usage:
            tokens_used = response.usage.total_tokens

        cost_per_1k = _COST_PER_1K_TOKENS.get(self.model, 0.002)
        cost_usd = round((tokens_used / 1_000.0) * cost_per_1k, 6)

        return {
            'text': text,
            'tokens_used': tokens_used,
            'cost_usd': cost_usd,
        }
