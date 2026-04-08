"""
OpenAI-Compatible Adapter — Concrete LLMAdapter for OpenAI-style Chat APIs
===========================================================================
PURPOSE:
  Wire OpenAI-style Chat Completions APIs into LLMExecutor. Works with OpenAI
  directly and with OpenAI-compatible endpoints (OpenRouter, xAI-compatible
  gateways, Ollama's local OpenAI endpoint, etc.).

SUPPORTED MODELS:
  Any model accepted by the target OpenAI-compatible endpoint.
  Known OpenAI models are mapped to model registry keys for cost calculation.

USAGE:
  from execution.openai_adapter import OpenAIAdapter
  from execution.llm_executor import LLMExecutor

  executor = LLMExecutor()
  executor.register('gpt4o_mini', OpenAIAdapter())        # OpenAI default model
  executor.register(
      'openrouter_claude',
      OpenAIAdapter(
          model='anthropic/claude-3.5-sonnet',
          provider='openrouter',
          api_key_env='OPENROUTER_API_KEY',
          base_url='https://openrouter.ai/api/v1',
      ),
  )

  result = executor.execute(prompt, allocation)
  # result: {
  #   'text': '...',
  #   'model': 'gpt-4o-mini',
  #   'provider': 'openai',
  #   'usage': {'input_tokens': int, 'output_tokens': int},
  #   'cost': {'input_usd': float, 'output_usd': float,
  #            'tool_usd': float, 'total_usd': float},
  #   'llm': 'gpt4o_mini',
  #   'skipped': False,
  # }

ENVIRONMENT (defaults):
  OPENAI_API_KEY  — required API key unless api_key is passed directly
  OPENAI_BASE_URL — optional base URL override
"""

from __future__ import annotations

import os
import time

from execution.llm_executor import LLMAdapter

# Maps OpenAI model names to their "provider:model" keys in models/registry.py.
# Add new models here when they are added to the models registry.
_MODEL_KEYS: dict = {
    'gpt-4o-mini':   'openai:gpt-4o-mini',
    'gpt-4o':        'openai:gpt-4o',
    'gpt-3.5-turbo': 'openai:gpt-3.5-turbo',
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
                 system: str = '',
                 provider: str = 'openai',
                 model_key: str = '',
                 api_key: str = '',
                 base_url: str = '',
                 api_key_env: str = 'OPENAI_API_KEY',
                 base_url_env: str = 'OPENAI_BASE_URL'):
        self.model = model
        self.temperature = temperature
        self.system = system
        self.provider = provider
        self.model_key = model_key
        self.api_key = api_key
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
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
        api_key = (self.api_key or os.environ.get(self.api_key_env, '')).strip()
        if not api_key:
            raise EnvironmentError(
                f"{self.api_key_env} environment variable is not set. "
                f"Export your {self.provider} API key before running."
            )
        base_url = self.base_url or os.environ.get(self.base_url_env) or None
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
              text:        str   — generated text
              model:       str   — model name used (e.g. 'gpt-4o-mini')
              provider:    str   — provider identifier ('openai')
              mode:        str   — call mode ('chat')
              latency_ms:  float — wall-clock time for the API call in ms
              usage:       dict  — {'input_tokens': int, 'output_tokens': int}
              cost:        dict  — {'input_usd': float, 'output_usd': float,
                                    'tool_usd': float, 'total_usd': float}
        """
        from models.pricing import calculate_cost

        client = self._get_client()

        messages = []
        if self.system:
            messages.append({'role': 'system', 'content': self.system})
        messages.append({'role': 'user', 'content': prompt})

        temperature = kwargs.pop('temperature', self.temperature)

        t_start = time.monotonic()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)

        text = ''
        if response.choices:
            text = response.choices[0].message.content or ''

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0

        usage = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        }

        model_key = self.model_key or _MODEL_KEYS.get(self.model, '')
        if model_key:
            cost = calculate_cost(model_key, usage)
        else:
            # Unknown model: return zero cost rather than silently fail.
            cost = {'input_usd': 0.0, 'output_usd': 0.0,
                    'tool_usd': 0.0, 'total_usd': 0.0}

        return {
            'text': text,
            'model': self.model,
            'provider': self.provider,
            'mode': 'chat',
            'latency_ms': latency_ms,
            'usage': usage,
            'cost': cost,
        }
