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

  Pricing is read from models/registry.py — there are no hardcoded figures
  in this file. Update pricing by editing models/registry.py only.

USAGE:
  from execution.openai_adapter import OpenAIAdapter
  from execution.llm_executor import LLMExecutor

  executor = LLMExecutor()
  executor.register('gpt4o_mini', OpenAIAdapter())        # default model
  executor.register('gpt4o',      OpenAIAdapter('gpt-4o'))

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

ENVIRONMENT:
  OPENAI_API_KEY  — required; your OpenAI secret key (sk-...)
  OPENAI_BASE_URL — optional; override for proxies / Azure OpenAI endpoints
"""

from __future__ import annotations

import os
import time

from execution.llm_executor import LLMAdapter

_PROVIDER = 'openai'

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

        model_key = _MODEL_KEYS.get(self.model, '')
        if model_key:
            cost = calculate_cost(model_key, usage)
        else:
            # Unknown model: return zero cost rather than silently fail.
            cost = {'input_usd': 0.0, 'output_usd': 0.0,
                    'tool_usd': 0.0, 'total_usd': 0.0}

        return {
            'text': text,
            'model': self.model,
            'provider': _PROVIDER,
            'mode': 'chat',
            'latency_ms': latency_ms,
            'usage': usage,
            'cost': cost,
        }

