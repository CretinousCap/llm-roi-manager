"""
LLM Executor — Execution Layer for LLM Provider Adapters
=========================================================
PURPOSE:
  Provide a thin adapter layer that bridges the routing/evaluation library
  with concrete LLM API clients. Callers register their provider adapters;
  the executor dispatches to the correct one based on the recommended LLM name.

HOW IT WORKS:
  1. Subclass LLMAdapter and implement complete() for your provider.
  2. Register the adapter: executor.register('claude', MyClaudeAdapter())
  3. Call executor.execute() with a prompt and an allocation dict (from
     token_allocator.allocate()). The executor routes to the right adapter,
     calls complete(), and returns a standardised result dict.

ADAPTER CONTRACT:
  complete() must return a dict with:
    text:     str   — generated text
    model:    str   — model name (e.g. 'gpt-4o-mini')
    provider: str   — provider identifier (e.g. 'openai')
    usage:    dict  — {'input_tokens': int, 'output_tokens': int}
    cost:     dict  — {'input_usd': float, 'output_usd': float,
                        'tool_usd': float, 'total_usd': float}

  The executor appends 'llm' and 'skipped' to produce the final result.

ADDING A NEW PROVIDER:
  Subclass LLMAdapter, implement complete(), then register the instance with
  LLMExecutor.register(). No changes to the router or registry are needed.

NOTE:
  This module contains zero LLM API client imports. API clients live in your
  application layer — you bring them to the adapter.

USAGE:
  executor = LLMExecutor()
  executor.register('claude', MyClaudeAdapter())
  result = executor.execute(
      prompt='Write a sort function in Python.',
      allocation={'preferred_llm': 'claude', 'token_budget': 2000},
  )
  # result: {
  #   'text': '...', 'model': 'claude-3-5-sonnet', 'provider': 'anthropic',
  #   'usage': {'input_tokens': 120, 'output_tokens': 192},
  #   'cost': {'input_usd': 0.00036, 'output_usd': 0.00288,
  #            'tool_usd': 0.0, 'total_usd': 0.00324},
  #   'llm': 'claude', 'skipped': False,
  # }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMAdapter(ABC):
    """
    Abstract base class for LLM provider adapters.

    Subclass this for each LLM provider (Claude, Codex, Gemini, …).
    The adapter is the only place in this library where provider-specific
    API client code lives.
    """

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int, **kwargs) -> dict:
        """
        Call the LLM and return a result dict.

        Args:
            prompt:     The prompt to send to the LLM.
            max_tokens: Maximum tokens to generate (from token_allocator).
            **kwargs:   Provider-specific parameters (temperature, etc.).

        Returns:
            dict with:
              text:     str  — generated text
              model:    str  — model name used (e.g. 'gpt-4o-mini')
              provider: str  — provider identifier (e.g. 'openai')
              usage:    dict — {'input_tokens': int, 'output_tokens': int}
              cost:     dict — {'input_usd': float, 'output_usd': float,
                                 'tool_usd': float, 'total_usd': float}
        """
        ...


class LLMExecutor:
    """
    Dispatches LLM calls to the correct registered adapter.

    One executor instance is typically shared across sessions. Adapters are
    registered at startup and reused.

    Example::

        executor = LLMExecutor()
        executor.register('claude', MyClaudeAdapter())
        result = executor.execute(
            prompt='Explain gradient descent.',
            allocation={'preferred_llm': 'claude', 'token_budget': 1500},
        )
    """

    def __init__(self, config: dict = None):
        from core import EXECUTOR_DEFAULTS
        self.cfg = {**EXECUTOR_DEFAULTS, **(config or {})}
        self._adapters: dict = {}

    def register(self, llm_name: str, adapter: LLMAdapter) -> None:
        """
        Register an LLM adapter.

        Args:
            llm_name: LLM identifier matching LLM_REGISTRY key (e.g. 'claude').
            adapter:  Concrete LLMAdapter instance for this provider.
        """
        if not isinstance(adapter, LLMAdapter):
            raise TypeError(
                f"adapter must be an LLMAdapter instance, "
                f"got {type(adapter).__name__}"
            )
        self._adapters[llm_name] = adapter

    def execute(self, prompt: str, allocation: dict,
                llm_override: Optional[str] = None,
                **kwargs) -> dict:
        """
        Execute a prompt using the recommended (or overridden) LLM.

        Args:
            prompt:       The prompt text to send.
            allocation:   Output of token_allocator.allocate() — must include
                          'preferred_llm' and 'token_budget'.
            llm_override: Force a specific LLM regardless of allocation.
            **kwargs:     Passed through to the adapter's complete() call.

        Returns:
            dict with:
              text:     str   — generated text (empty string if no adapter)
              model:    str   — model name used
              provider: str   — provider identifier
              usage:    dict  — {'input_tokens': int, 'output_tokens': int}
              cost:     dict  — {'input_usd': float, 'output_usd': float,
                                  'tool_usd': float, 'total_usd': float}
              llm:      str   — LLM registry name used
              skipped:  bool  — True if no adapter registered for this LLM
        """
        llm_name = llm_override or allocation.get('preferred_llm', '')
        max_tokens = allocation.get(
            'token_budget', self.cfg['default_max_tokens']
        )

        _zero_cost = {'input_usd': 0.0, 'output_usd': 0.0,
                      'tool_usd': 0.0, 'total_usd': 0.0}

        if llm_name not in self._adapters:
            return {
                'text': '',
                'model': '',
                'provider': '',
                'usage': {'input_tokens': 0, 'output_tokens': 0},
                'cost': _zero_cost,
                'llm': llm_name,
                'skipped': True,
            }

        adapter = self._adapters[llm_name]
        raw = adapter.complete(prompt, max_tokens=max_tokens, **kwargs)
        return {
            'text': raw.get('text', ''),
            'model': raw.get('model', ''),
            'provider': raw.get('provider', ''),
            'usage': raw.get('usage', {'input_tokens': 0, 'output_tokens': 0}),
            'cost': raw.get('cost', _zero_cost),
            'llm': llm_name,
            'skipped': False,
        }

    def available_llms(self) -> list:
        """Return list of registered LLM names."""
        return list(self._adapters.keys())
