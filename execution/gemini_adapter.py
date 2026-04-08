"""
Gemini Adapter — Concrete LLMAdapter for Google Gemini generateContent API
===========================================================================
PURPOSE:
  Wire Gemini models into LLMExecutor with stdlib-only HTTP calls.
  Reads API key from GEMINI_API_KEY by default.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

from execution.llm_executor import LLMAdapter
from models.pricing import calculate_cost

_MODEL_KEYS: dict = {
    'gemini-pro': 'google:gemini-pro',
}


class GeminiAdapter(LLMAdapter):
    """LLMAdapter implementation for Gemini REST generateContent endpoint."""

    def __init__(self,
                 model: str = 'gemini-pro',
                 temperature: float = 0.7,
                 system: str = '',
                 api_key: str = '',
                 api_key_env: str = 'GEMINI_API_KEY',
                 base_url: str = 'https://generativelanguage.googleapis.com'):
        self.model = model
        self.temperature = temperature
        self.system = system
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip('/')

    def complete(self, prompt: str, max_tokens: int, **kwargs) -> dict:
        """Call Gemini generateContent API and return standardised output."""
        api_key = (self.api_key or os.environ.get(self.api_key_env, '')).strip()
        if not api_key:
            raise EnvironmentError(
                f"{self.api_key_env} environment variable is not set. "
                "Export your Gemini API key before running."
            )

        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': kwargs.pop('temperature', self.temperature),
                'maxOutputTokens': int(max_tokens),
            },
        }
        if self.system:
            payload['systemInstruction'] = {'parts': [{'text': self.system}]}
        payload.update(kwargs)

        endpoint = (
            f'{self.base_url}/v1beta/models/{self.model}:generateContent?'
            f'{urllib.parse.urlencode({"key": api_key})}'
        )
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        t_start = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode('utf-8'))
        latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)

        text = ''
        candidates = body.get('candidates', [])
        if candidates and isinstance(candidates, list):
            parts = (
                candidates[0].get('content', {}).get('parts', [])
                if isinstance(candidates[0], dict) else []
            )
            text = ''.join(
                part.get('text', '')
                for part in parts
                if isinstance(part, dict)
            )

        usage_meta = body.get('usageMetadata', {})
        usage = {
            'input_tokens': int(usage_meta.get('promptTokenCount', 0) or 0),
            'output_tokens': int(
                usage_meta.get('candidatesTokenCount', 0) or 0
            ),
        }

        model_key = _MODEL_KEYS.get(self.model, '')
        if model_key:
            cost = calculate_cost(model_key, usage)
        else:
            cost = {'input_usd': 0.0, 'output_usd': 0.0,
                    'tool_usd': 0.0, 'total_usd': 0.0}

        return {
            'text': text,
            'model': self.model,
            'provider': 'google',
            'mode': 'chat',
            'latency_ms': latency_ms,
            'usage': usage,
            'cost': cost,
        }

