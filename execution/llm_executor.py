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
    text:        str   — generated text
    model:       str   — model name (e.g. 'gpt-4o-mini')
    provider:    str   — provider identifier (e.g. 'openai')
    mode:        str   — call mode (e.g. 'chat', 'reasoning', 'image')
    latency_ms:  float — wall-clock time for the API call in milliseconds
    usage:       dict  — {'input_tokens': int, 'output_tokens': int}
    cost:        dict  — {'input_usd': float, 'output_usd': float,
                           'tool_usd': float, 'total_usd': float}

  The executor wraps the call, classifies any errors, attaches billing
  context, and produces the final standardised result.

STANDARDISED RESULT FORMAT:
  {
    'status':     'success' | 'error' | 'unavailable',
    'provider':   str,
    'model':      str,
    'mode':       str,           # e.g. 'chat', 'reasoning', 'image'
    'latency_ms': float,
    'usage':      {'input_tokens': int, 'output_tokens': int},
    'cost':       {'input_usd': float, 'output_usd': float,
                   'tool_usd': float, 'total_usd': float},
    'text':       str,
    'error':      {'type': str, 'message': str} | None,
    'billing':    {
                    'account_id':   str,   # stable identifier, e.g. 'acc_personal'
                    'account_name': str,   # human-readable name, e.g. 'personal'
                    'project_id':   str,   # stable identifier, e.g. 'proj_llm-roi'
                    'project_name': str,   # human-readable name, e.g. 'llm-roi'
                    'environment':  str,   # 'dev' | 'test' | 'prod'
                  },
    # Backward-compat aliases (kept for now):
    'llm':        str,
    'skipped':    bool,
    'tokens_used': int,
    'cost_usd':   float,
  }

BILLING CONTEXT (metadata parameter):
  Pass an optional metadata dict to execute() to attach billing information.
  New-style keys (preferred):
    metadata = {
        'account_id':   'acc_personal',    # stable identifier
        'account_name': 'personal',        # human-readable name
        'project_id':   'proj_llm-roi',    # stable identifier
        'project_name': 'llm-roi-manager', # human-readable name
        'environment':  'dev | test | prod',
    }
  Old-style keys (still accepted for backward compatibility):
    metadata = {
        'account':     'personal',         # → account_name='personal',
                                           #   account_id='acc_personal'
        'project':     'llm-roi-manager',  # → project_name='llm-roi-manager',
                                           #   project_id='proj_llm-roi-manager'
        'environment': 'dev',
    }
  Missing metadata (or missing individual keys) defaults to:
    account_id='acc_unknown', account_name='unknown',
    project_id='proj_default', project_name='default', environment='dev'.

ERROR CLASSIFICATION:
  All adapter exceptions are caught and classified via safe string matching
  (not SDK-specific types) into: 'quota', 'rate_limit', 'timeout', 'auth',
  or 'unknown'.

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
      metadata={'account_id': 'acc_personal', 'account_name': 'personal',
                'project_id': 'proj_llm-roi', 'project_name': 'llm-roi-manager',
                'environment': 'dev'},
  )
  # result: {
  #   'status': 'success',
  #   'text': '...', 'model': 'claude-3-5-sonnet', 'provider': 'anthropic',
  #   'mode': 'chat', 'latency_ms': 842.5,
  #   'usage': {'input_tokens': 120, 'output_tokens': 192},
  #   'cost': {'input_usd': 0.00036, 'output_usd': 0.00288,
  #            'tool_usd': 0.0, 'total_usd': 0.00324},
  #   'error': None,
  #   'billing': {'account_id': 'acc_personal', 'account_name': 'personal',
  #               'project_id': 'proj_llm-roi', 'project_name': 'llm-roi-manager',
  #               'environment': 'dev'},
  #   'llm': 'claude', 'skipped': False,
  #   'tokens_used': 312, 'cost_usd': 0.00324,
  # }
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional


def _classify_error(exc: Exception) -> str:
    """
    Classify an exception into a stable error type string.

    Uses safe string matching on the exception type name and message —
    not on SDK-specific exception classes — so this works across providers.

    Returns one of: 'quota', 'rate_limit', 'timeout', 'auth', 'unknown'.
    """
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    combined = f'{name} {msg}'

    if any(k in combined for k in (
        'quota', 'insufficient_quota', 'exceeded your current quota',
    )):
        return 'quota'
    if any(k in combined for k in (
        'rate_limit', 'ratelimit', 'rate limit', 'too many request',
        'requests per',
    )):
        return 'rate_limit'
    if any(k in combined for k in (
        'timeout', 'timed out', 'deadline exceeded', 'connection timeout',
    )):
        return 'timeout'
    if any(k in combined for k in (
        'auth', 'unauthorized', 'invalid api key', 'incorrect api key',
        'permission denied', '401', '403', 'api key',
    )):
        return 'auth'
    return 'unknown'


def _normalise_billing(metadata: Optional[dict]) -> dict:
    """
    Return a normalised billing dict with stable identifiers and defaults.

    Accepts both new-style keys (account_id / account_name / project_id /
    project_name) and old-style keys (account / project) for backward
    compatibility.  Old-style keys are promoted:
      account  → account_name = account,  account_id  = 'acc_{account}'
      project  → project_name = project,  project_id  = 'proj_{project}'

    Defaults when keys are absent:
      account_id='acc_unknown', account_name='unknown',
      project_id='proj_default', project_name='default', environment='dev'.
    """
    meta = metadata or {}

    # ── Account fields ────────────────────────────────────────────────────────
    if 'account_name' in meta or 'account_id' in meta:
        # New-style: at least one explicit new key present
        account_name = str(meta.get('account_name', 'unknown'))
        account_id   = str(meta.get('account_id',   f'acc_{account_name}'))
    elif 'account' in meta:
        # Old-style: derive both new fields from the legacy key
        account_name = str(meta['account'])
        account_id   = f'acc_{account_name}'
    else:
        account_name = 'unknown'
        account_id   = 'acc_unknown'

    # ── Project fields ────────────────────────────────────────────────────────
    if 'project_name' in meta or 'project_id' in meta:
        # New-style: at least one explicit new key present
        project_name = str(meta.get('project_name', 'default'))
        project_id   = str(meta.get('project_id',   f'proj_{project_name}'))
    elif 'project' in meta:
        # Old-style: derive both new fields from the legacy key
        project_name = str(meta['project'])
        project_id   = f'proj_{project_name}'
    else:
        project_name = 'default'
        project_id   = 'proj_default'

    return {
        'account_id':   account_id,
        'account_name': account_name,
        'project_id':   project_id,
        'project_name': project_name,
        'environment':  str(meta.get('environment', 'dev')),
    }


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
              text:        str   — generated text
              model:       str   — model name used (e.g. 'gpt-4o-mini')
              provider:    str   — provider identifier (e.g. 'openai')
              mode:        str   — call mode (e.g. 'chat', 'reasoning', 'image')
              latency_ms:  float — wall-clock time for the API call in ms
              usage:       dict  — {'input_tokens': int, 'output_tokens': int}
              cost:        dict  — {'input_usd': float, 'output_usd': float,
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
            metadata={'account_id': 'acc_personal', 'account_name': 'personal',
                      'project_id': 'proj_llm-roi', 'project_name': 'llm-roi-manager',
                      'environment': 'dev'},
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
                metadata: Optional[dict] = None,
                **kwargs) -> dict:
        """
        Execute a prompt using the recommended (or overridden) LLM.

        Args:
            prompt:       The prompt text to send.
            allocation:   Output of token_allocator.allocate() — must include
                          'preferred_llm' and 'token_budget'.
            llm_override: Force a specific LLM regardless of allocation.
            metadata:     Optional billing context dict. Accepted keys:
                            New-style (preferred):
                              account_id, account_name, project_id, project_name
                            Old-style (backward-compat):
                              account  → account_name + account_id derived
                              project  → project_name + project_id derived
                            environment (default 'dev')
                          Unrecognised keys are silently ignored.
            **kwargs:     Passed through to the adapter's complete() call.

        Returns:
            Standardised result dict:
              status:     'success' | 'error' | 'unavailable'
              provider:   str
              model:      str
              mode:       str    (e.g. 'chat')
              latency_ms: float
              usage:      {'input_tokens': int, 'output_tokens': int}
              cost:       {'input_usd': float, 'output_usd': float,
                           'tool_usd': float, 'total_usd': float}
              text:       str
              error:      {'type': str, 'message': str} or None
              billing:    {'account_id': str, 'account_name': str,
                           'project_id': str, 'project_name': str,
                           'environment': str}
              # Backward-compat aliases:
              llm:        str
              skipped:    bool
              tokens_used: int
              cost_usd:   float
        """
        llm_name = llm_override or allocation.get('preferred_llm', '')
        max_tokens = allocation.get(
            'token_budget', self.cfg['default_max_tokens']
        )
        billing = _normalise_billing(metadata)

        _zero_cost = {'input_usd': 0.0, 'output_usd': 0.0,
                      'tool_usd': 0.0, 'total_usd': 0.0}
        _zero_usage = {'input_tokens': 0, 'output_tokens': 0}

        if llm_name not in self._adapters:
            return {
                'status': 'unavailable',
                'provider': '',
                'model': '',
                'mode': '',
                'latency_ms': 0.0,
                'usage': _zero_usage,
                'cost': _zero_cost,
                'text': '',
                'error': {
                    'type': 'adapter_missing',
                    'message': (
                        f"No adapter registered for LLM '{llm_name}'. "
                        f"Call executor.register('{llm_name}', ...) "
                        f"before executing."
                    ),
                },
                'billing': billing,
                # Backward-compat aliases
                'llm': llm_name,
                'skipped': True,
                'tokens_used': 0,
                'cost_usd': 0.0,
            }

        adapter = self._adapters[llm_name]
        t_start = time.monotonic()
        try:
            raw = adapter.complete(prompt, max_tokens=max_tokens, **kwargs)
        except Exception as exc:
            latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)
            err_type = _classify_error(exc)
            return {
                'status': 'error',
                'provider': '',
                'model': '',
                'mode': '',
                'latency_ms': latency_ms,
                'usage': _zero_usage,
                'cost': _zero_cost,
                'text': '',
                'error': {
                    'type': err_type,
                    'message': str(exc),
                },
                'billing': billing,
                # Backward-compat aliases
                'llm': llm_name,
                'skipped': False,
                'tokens_used': 0,
                'cost_usd': 0.0,
            }

        # Use the adapter-reported latency when available; fall back to the
        # wall-clock time measured around the complete() call.
        fallback_latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)
        latency_ms = raw.get('latency_ms', fallback_latency_ms)

        usage = raw.get('usage', _zero_usage)
        cost = raw.get('cost', _zero_cost)
        tokens_used = (
            usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
        )

        return {
            'status': 'success',
            'provider': raw.get('provider', ''),
            'model': raw.get('model', ''),
            'mode': raw.get('mode', 'chat'),
            'latency_ms': round(float(latency_ms), 2),
            'usage': usage,
            'cost': cost,
            'text': raw.get('text', ''),
            'error': None,
            'billing': billing,
            # Backward-compat aliases
            'llm': llm_name,
            'skipped': False,
            'tokens_used': tokens_used,
            'cost_usd': cost.get('total_usd', 0.0),
        }

    def available_llms(self) -> list:
        """Return list of registered LLM names."""
        return list(self._adapters.keys())
