"""
Anthropic Adapter — Concrete LLMAdapter for Claude via Anthropic Messages API
==============================================================================
PURPOSE:
  Wire Anthropic Claude models into LLMExecutor without adding non-stdlib
  dependencies. Reads API key from ANTHROPIC_API_KEY by default.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

from execution.llm_executor import LLMAdapter
from models.pricing import calculate_cost

_MODEL_KEYS: dict = {
    'claude-3-5-sonnet-latest': 'anthropic:claude-3-5-sonnet',
    'claude-3-7-sonnet-latest': 'anthropic:claude-3-7-sonnet',
}


class AnthropicAdapter(LLMAdapter):
    """LLMAdapter implementation for Anthropic's Messages API."""

    def __init__(self,
                 model: str = 'claude-3-5-sonnet-latest',
                 temperature: float = 0.7,
                 system: str = '',
                 api_key: str = '',
                 api_key_env: str = 'ANTHROPIC_API_KEY',
                 base_url: str = 'https://api.anthropic.com/v1/messages'):
        self.model = model
        self.temperature = temperature
        self.system = system
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.base_url = base_url

    def complete(self, prompt: str, max_tokens: int, **kwargs) -> dict:
        """Call Anthropic Messages API and return standardised adapter output."""
        api_key = (self.api_key or os.environ.get(self.api_key_env, '')).strip()
        if not api_key:
            raise EnvironmentError(
                f"{self.api_key_env} environment variable is not set. "
                "Export your Anthropic API key before running."
            )

        payload = {
            'model': self.model,
            'max_tokens': int(max_tokens),
            'temperature': kwargs.pop('temperature', self.temperature),
            'messages': [{'role': 'user', 'content': prompt}],
        }
        if self.system:
            payload['system'] = self.system
        payload.update(kwargs)

        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            },
            method='POST',
        )

        t_start = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode('utf-8'))
        latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)

        text = ''
        content = body.get('content', [])
        if content and isinstance(content, list):
            text = ''.join(
                block.get('text', '') for block in content
                if isinstance(block, dict) and block.get('type') == 'text'
            )

        usage = body.get('usage', {})
        usage_out = {
            'input_tokens': int(usage.get('input_tokens', 0) or 0),
            'output_tokens': int(usage.get('output_tokens', 0) or 0),
        }

        model_key = _MODEL_KEYS.get(self.model, '')
        if model_key:
            cost = calculate_cost(model_key, usage_out)
        else:
            cost = {'input_usd': 0.0, 'output_usd': 0.0,
                    'tool_usd': 0.0, 'total_usd': 0.0}

        return {
            'text': text,
            'model': self.model,
            'provider': 'anthropic',
            'mode': 'chat',
            'latency_ms': latency_ms,
            'usage': usage_out,
            'cost': cost,
        }

