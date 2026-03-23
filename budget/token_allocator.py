"""
Token Allocator — Confidence-Scaled Budget Management
======================================================
PURPOSE:
  Calculate the token budget for a task based on router confidence and
  task context — giving more tokens to high-confidence, well-matched tasks
  and fewer to ambiguous ones.

ADAPTED FROM:
  position_sizer.py in the HybridQuant trading system.
  ATR-based sizing + regime_confidence_scaling
      → base_token_budget + confidence_scaling

HOW IT WORKS:
  Base budget: BUDGET_DEFAULTS['base_token_budget'] (or task's budget_tokens hint)
  Confidence scaling: multiplied by confidence (min_confidence_scale to 1.0)

  High-confidence routing (clear code task → Codex) → full token budget.
  Ambiguous task → reduced budget.

  This is the "Kelly criterion lite" applied to token allocation:
  bet bigger (give more tokens) when the routing edge is clearer.

CONSTRAINTS:
  - Max budget: BUDGET_DEFAULTS['max_token_budget'] (hard cap)
  - Min budget: BUDGET_DEFAULTS['min_token_budget'] (minimum viable call)
"""

from __future__ import annotations


def allocate(context: dict, recommendation: dict,
             config: dict = None) -> dict:
    """
    Calculate token budget and estimated cost for a task.

    Analogous to position_sizer.compute() — scales allocation by confidence.

    Args:
        context:        Output of task_classifier.classify().
        recommendation: Output of llm_router.route().
        config:         Override BUDGET_DEFAULTS.

    Returns:
        dict with:
          token_budget:       int — recommended token limit for the LLM call
          confidence:         float — router confidence used for scaling
          scale_factor:       float — multiplier applied (min_confidence_scale – 1.0)
          estimated_cost_usd: float — estimated cost at token_budget usage
          preferred_llm:      str — carried through from recommendation
    """
    from core import BUDGET_DEFAULTS
    from core.registry import LLM_REGISTRY

    cfg = {**BUDGET_DEFAULTS, **(config or {})}

    confidence: float = recommendation.get('confidence', 0.5)
    preferred_llm: str = recommendation.get('preferred_llm', '')

    # ── Confidence scaling ────────────────────────────────────────────────────
    if cfg['use_confidence_scaling']:
        min_s = cfg['min_confidence_scale']
        scale_factor = min_s + (1.0 - min_s) * confidence
    else:
        scale_factor = 1.0

    # Use task's budget_tokens hint as base if provided
    task_hint = context.get('task', {}).get('budget_tokens', 0)
    base = task_hint if task_hint and task_hint > 0 else cfg['base_token_budget']

    raw_budget = base * scale_factor
    token_budget = int(
        max(cfg['min_token_budget'], min(cfg['max_token_budget'], raw_budget))
    )

    # ── Estimate cost ─────────────────────────────────────────────────────────
    # Look up blended cost from models registry via the routing entry's model_key.
    cost_per_1k = 0.0
    if preferred_llm in LLM_REGISTRY:
        model_key = LLM_REGISTRY[preferred_llm].get('model_key', '')
        if model_key:
            try:
                from models.registry import LLM_REGISTRY as MODELS_REGISTRY
                m = MODELS_REGISTRY.get(model_key, {})
                if m:
                    # Blended estimate: average of input and output pricing.
                    # Actual split is unknown at allocation time.
                    cost_per_1k = (
                        m.get('input_price_per_1k', 0.0)
                        + m.get('output_price_per_1k', 0.0)
                    ) / 2.0
            except (ImportError, KeyError, TypeError):
                pass  # Models registry unavailable — estimate at zero
    estimated_cost_usd = (token_budget / 1000.0) * cost_per_1k

    return {
        'token_budget': token_budget,
        'confidence': round(confidence, 4),
        'scale_factor': round(scale_factor, 4),
        'estimated_cost_usd': round(estimated_cost_usd, 6),
        'preferred_llm': preferred_llm,
    }
