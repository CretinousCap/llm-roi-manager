"""
Models package — Central model registry and pricing engine.

Exports:
  LLM_REGISTRY   — dict keyed by "provider:model" with pricing and capabilities.
  calculate_cost — deterministic cost calculation from token usage.
"""

from models.registry import LLM_REGISTRY
from models.pricing import calculate_cost

__all__ = ['LLM_REGISTRY', 'calculate_cost']
