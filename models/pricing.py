"""
Pricing Engine — Deterministic Cost Calculation from the Models Registry
=========================================================================
PURPOSE:
  Translate raw token usage into USD cost using per-model pricing from
  models/registry.py. This is the only place in the codebase where token
  counts are converted to dollar amounts.

HOW IT WORKS:
  calculate_cost() looks up the model entry in LLM_REGISTRY, multiplies
  input and output token counts by their respective per-1k prices, and
  returns a fully-itemised cost breakdown. Tool call pricing is reserved
  for future use (returns 0.0 when not provided).

DESIGN PRINCIPLES:
  - Deterministic: same inputs always produce the same output.
  - Side-effect free: no state mutation, no I/O.
  - Explicit: every cost component is named in the returned dict.
  - Replaceable: swap pricing data by updating models/registry.py alone.

USAGE:
  from models.pricing import calculate_cost

  cost = calculate_cost(
      model_key='openai:gpt-4o-mini',
      usage={'input_tokens': 500, 'output_tokens': 300},
  )
  # cost → {'input_usd': 7.5e-05, 'output_usd': 0.00018,
  #          'tool_usd': 0.0, 'total_usd': 0.0002550}
"""

from __future__ import annotations


def calculate_cost(model_key: str,
                   usage: dict,
                   tools: list = None) -> dict:
    """
    Calculate the USD cost of an LLM call from token usage.

    Args:
        model_key: Registry key in the form "provider:model"
                   (e.g. "openai:gpt-4o-mini"). Must exist in
                   models.registry.LLM_REGISTRY.
        usage:     Token usage dict:
                     {
                       "input_tokens":  int,   # prompt / context tokens
                       "output_tokens": int,   # completion tokens
                     }
        tools:     Reserved for future tool-call pricing. Pass a list of
                   tool-call records when available; currently always
                   contributes 0.0 to the cost.

    Returns:
        dict with:
          input_usd:  float — cost for input tokens
          output_usd: float — cost for output tokens
          tool_usd:   float — cost for tool calls (0.0 until implemented)
          total_usd:  float — sum of all components

    Raises:
        KeyError: If model_key is not found in LLM_REGISTRY.
        TypeError: If usage is not a dict.
    """
    from models.registry import LLM_REGISTRY

    if not isinstance(usage, dict):
        raise TypeError(
            f"usage must be a dict with 'input_tokens' and 'output_tokens', "
            f"got {type(usage).__name__}"
        )

    if model_key not in LLM_REGISTRY:
        raise KeyError(
            f"Model {model_key!r} not found in models registry. "
            f"Available models: {list(LLM_REGISTRY)}"
        )

    entry = LLM_REGISTRY[model_key]
    input_tokens: int = max(0, int(usage.get('input_tokens', 0)))
    output_tokens: int = max(0, int(usage.get('output_tokens', 0)))

    input_usd: float = (input_tokens / 1_000.0) * entry['input_price_per_1k']
    output_usd: float = (output_tokens / 1_000.0) * entry['output_price_per_1k']

    # Tool call pricing: placeholder — always 0.0 until per-tool rates are defined.
    tool_usd: float = 0.0

    total_usd: float = input_usd + output_usd + tool_usd

    return {
        'input_usd': round(input_usd, 8),
        'output_usd': round(output_usd, 8),
        'tool_usd': round(tool_usd, 8),
        'total_usd': round(total_usd, 8),
    }
