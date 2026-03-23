"""
LLM Router — Context-Weighted Multi-LLM Selection
==================================================
PURPOSE:
  Take context scores from the task classifier and combine them with
  LLM agent scores (from LLM_REGISTRY or PerformanceTracker) into a
  single recommendation: which LLM to use and with how much confidence.

ADAPTED FROM:
  signal_combiner.py in the HybridQuant trading system.
  regime_trending/ranging → context affinity weights per LLM
  SIGNAL_REGISTRY → LLM_REGISTRY
  final_signal → recommendation dict

HOW IT WORKS:
  1. For each LLM in LLM_REGISTRY, look up its context_affinity
  2. Scale its weight by the relevant context dimension score
     (same as regime_trending scales Donchian weight in HybridQuant)
  3. Compute the weighted agent score for each LLM
  4. Return the highest-scoring LLM as the recommendation

WHY SOFT WEIGHTING (not hard rules):
  Hard rules ("always use Codex for code") break when task descriptions
  are ambiguous or when an LLM's effectiveness varies by sub-domain.
  Soft weighting means in ambiguous tasks, all LLMs compete on even footing
  and the one with the best historical effectiveness for that context wins.

ADDING A NEW LLM:
  1. Add the LLM to LLM_REGISTRY in core/registry.py
  2. The router picks it up automatically — no changes here needed
"""

from __future__ import annotations


def route(context: dict,
          tracker=None,
          weights: dict = None) -> dict:
    """
    Select the best LLM for a task given its context scores.

    Analogous to signal_combiner.compute() — produces a weighted
    recommendation from all registered LLM agents.

    Args:
        context: Output of task_classifier.classify() — must include
                 'context_scores' (dict) and 'confidence' (float).
        tracker: Optional PerformanceTracker instance. When provided,
                 its score() is used instead of (or in addition to) the
                 registry score_fn for LLMs that have enough historical data.
        weights: Override default_weight per LLM, e.g. {'claude': 1.5}.

    Returns:
        dict with:
          preferred_llm:     name of highest-scoring LLM (str or None)
          confidence:        normalised confidence 0.0 – 1.0
          llm_scores:        {llm_name: raw_score} for all registered LLMs
          effective_weights: {llm_name: weight} used in the blend
          context_dominant:  dominant context dimension from classifier
    """
    from core.registry import LLM_REGISTRY

    weights = weights or {}
    context_scores: dict = context.get('context_scores', {})
    dominant: str = context.get('dominant', '')

    effective_weights: dict = {}
    agent_scores: dict = {}
    weighted_scores: dict = {}

    for llm_name, reg in LLM_REGISTRY.items():
        affinity = reg.get('context_affinity', 'neutral')
        base_w = weights.get(llm_name, reg.get('default_weight', 1.0))

        # Scale weight by the context dimension matching this LLM's affinity.
        # Clip to 0 so negative context scores don't invert the weight.
        if affinity in context_scores:
            regime_scale = max(0.0, context_scores[affinity])
        elif affinity == 'neutral':
            regime_scale = 0.5  # Always partially active
        else:
            regime_scale = 0.0

        eff_w = base_w * regime_scale
        effective_weights[llm_name] = round(eff_w, 4)

        agent_score = _get_agent_score(llm_name, reg, context, tracker)
        agent_scores[llm_name] = round(agent_score, 4)
        weighted_scores[llm_name] = eff_w * agent_score

    # ── Find preferred LLM ───────────────────────────────────────────────────
    total_weight = sum(effective_weights.values())

    if total_weight == 0:
        return {
            'preferred_llm': None,
            'confidence': 0.0,
            'llm_scores': agent_scores,
            'effective_weights': effective_weights,
            'context_dominant': dominant,
        }

    # Composite score per LLM = weighted_score / own_weight (normalised)
    composite: dict = {}
    for llm_name in LLM_REGISTRY:
        w = effective_weights[llm_name]
        composite[llm_name] = weighted_scores[llm_name] / w if w > 0 else -1.0

    preferred_llm = max(composite, key=lambda k: composite[k])
    raw_confidence = composite[preferred_llm]  # already in -1..+1 range

    # Normalise to 0..1
    confidence = max(0.0, (raw_confidence + 1.0) / 2.0)

    return {
        'preferred_llm': preferred_llm,
        'confidence': round(confidence, 4),
        'llm_scores': agent_scores,
        'effective_weights': effective_weights,
        'context_dominant': dominant,
    }


def _get_agent_score(llm_name: str, reg: dict,
                     context: dict, tracker) -> float:
    """
    Get the agent score for an LLM on the current task.

    Preference order:
      1. PerformanceTracker (historical data — most accurate)
      2. registry score_fn (static scoring heuristic)
      3. Default 0.0 (neutral — let effective_weight alone decide)
    """
    if tracker is not None:
        score = tracker.score(llm_name, context)
        if score != 0.0:
            return score

    score_fn = reg.get('score_fn')
    if score_fn is not None:
        task = {'description': context.get('dominant', ''),
                'context_scores': context.get('context_scores', {})}
        return float(score_fn(task))

    return 0.0
