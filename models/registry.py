"""
Models Registry — Central Pricing and Capability Registry for LLMs
===================================================================
PURPOSE:
  Single source of truth for per-model pricing and capability metadata.
  All cost calculations must read from this registry; no pricing figures
  should be hardcoded anywhere else in the codebase.

HOW IT WORKS:
  LLM_REGISTRY is a plain dict keyed by "provider:model". Each entry
  carries:
    - Separate input and output prices per 1 000 tokens (USD).
    - Context window size.
    - Capability tags (used for future routing hints).
    - Model type (text, reasoning, image, …).
    - Optional human-readable notes.

HOW TO EXTEND:
  Add a new entry with the "provider:model" key. No other files need to
  change for the pricing engine to pick it up.

    LLM_REGISTRY['myprovider:my-model'] = {
        'provider': 'myprovider',
        'model': 'my-model',
        'input_price_per_1k': 0.001,    # USD per 1 000 input tokens
        'output_price_per_1k': 0.002,   # USD per 1 000 output tokens
        'context_window': 8_192,
        'capabilities': ['chat', 'code'],
        'type': 'text',
        'notes': 'Optional description.',
    }

PRICING SOURCES:
  - OpenAI:  https://openai.com/pricing
  - xAI:     https://x.ai/api
  - Google:  https://ai.google.dev/pricing
  - Mistral: https://mistral.ai/technology/#pricing

  Update figures during scheduled effectiveness reviews (see docs/SOP.md).
  Placeholder entries are marked in 'notes'; verify before production use.
"""

# ── Model Registry ─────────────────────────────────────────────────────────────
# Keys follow the "<provider>:<model>" convention.
# Prices are in USD per 1 000 tokens.
LLM_REGISTRY: dict = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    'openai:gpt-4o-mini': {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'input_price_per_1k': 0.000150,   # $0.150 / 1M input tokens
        'output_price_per_1k': 0.000600,  # $0.600 / 1M output tokens
        'context_window': 128_000,
        'capabilities': ['chat', 'code', 'reasoning', 'json'],
        'type': 'text',
        'notes': 'Fast, cheap, capable; recommended default for most tasks.',
    },
    'openai:gpt-4o': {
        'provider': 'openai',
        'model': 'gpt-4o',
        'input_price_per_1k': 0.002500,   # $2.50 / 1M input tokens
        'output_price_per_1k': 0.010000,  # $10.00 / 1M output tokens
        'context_window': 128_000,
        'capabilities': ['chat', 'code', 'reasoning', 'vision', 'json'],
        'type': 'text',
        'notes': 'Highest OpenAI capability; use when quality is critical.',
    },
    'openai:gpt-3.5-turbo': {
        'provider': 'openai',
        'model': 'gpt-3.5-turbo',
        'input_price_per_1k': 0.000500,   # $0.50 / 1M input tokens
        'output_price_per_1k': 0.001500,  # $1.50 / 1M output tokens
        'context_window': 16_385,
        'capabilities': ['chat', 'code'],
        'type': 'text',
        'notes': 'Legacy model; cheaper but lower capability than gpt-4o-mini.',
    },
    # ── xAI ───────────────────────────────────────────────────────────────────
    'xai:grok-4.20-reasoning': {
        'provider': 'xai',
        'model': 'grok-4.20-reasoning',
        'input_price_per_1k': 0.005000,
        'output_price_per_1k': 0.015000,
        'context_window': 131_072,
        'capabilities': ['chat', 'code', 'reasoning'],
        'type': 'reasoning',
        'notes': 'Placeholder pricing; verify official rates before production use.',
    },
    # ── Google ────────────────────────────────────────────────────────────────
    'google:gemini-pro': {
        'provider': 'google',
        'model': 'gemini-pro',
        'input_price_per_1k': 0.000125,
        'output_price_per_1k': 0.000375,
        'context_window': 32_760,
        'capabilities': ['chat', 'code', 'reasoning'],
        'type': 'text',
        'notes': 'Placeholder pricing; verify from Google AI pricing page.',
    },
    # ── Mistral ───────────────────────────────────────────────────────────────
    'mistral:mixtral': {
        'provider': 'mistral',
        'model': 'mixtral',
        'input_price_per_1k': 0.000700,
        'output_price_per_1k': 0.000700,
        'context_window': 32_768,
        'capabilities': ['chat', 'code', 'reasoning'],
        'type': 'text',
        'notes': 'Placeholder pricing; verify from Mistral AI pricing page.',
    },
}


def get_model(model_key: str) -> dict:
    """
    Look up a model entry by its "provider:model" key.

    Args:
        model_key: Registry key in the form "provider:model"
                   (e.g. "openai:gpt-4o-mini").

    Returns:
        The registry entry dict for that model.

    Raises:
        KeyError: If the model_key is not present in LLM_REGISTRY.
    """
    if model_key not in LLM_REGISTRY:
        raise KeyError(
            f"Model {model_key!r} not found in models registry. "
            f"Available models: {list(LLM_REGISTRY)}"
        )
    return LLM_REGISTRY[model_key]


def list_models(provider: str = None) -> list:
    """
    Return a list of registered model keys, optionally filtered by provider.

    Args:
        provider: If supplied, only keys for that provider are returned
                  (e.g. 'openai', 'google').

    Returns:
        Sorted list of model key strings.
    """
    if provider is None:
        return sorted(LLM_REGISTRY)
    return sorted(k for k in LLM_REGISTRY if LLM_REGISTRY[k]['provider'] == provider)
