"""
LLM_REGISTRY — Plugin Registry for LLM Agents
===============================================
PURPOSE:
  Register LLM agents and their context affinities here.
  The router iterates this registry and skips any LLM whose score_fn is None.

ADAPTED FROM:
  SIGNAL_REGISTRY in signal_combiner.py (HybridQuant trading system).

HOW TO ADD A NEW LLM:
  1. Create agents/your_llm.py with a class implementing score(task) -> float
  2. Import it below and add an entry to LLM_REGISTRY
  3. The router picks it up automatically — no other changes needed
  4. Add a CHANGELOG.md entry

CONTEXT AFFINITIES:
  'code_generation'  — LLMs specialised in writing and completing code
  'content_writing'  — LLMs specialised in prose, blogs, marketing copy
  'data_analysis'    — LLMs strong on structured reasoning and data tasks
  'neutral'          — LLMs that perform well across all context types

WEIGHT GUIDELINES:
  default_weight = 1.0 is the baseline.
  Adjust during scheduled effectiveness reviews only (see docs/SOP.md).
  Max change: ±20% per review cycle. Document the rationale in CHANGELOG.md.
"""

# ── LLM Registry ──────────────────────────────────────────────────────────────
# Each entry key is a unique LLM identifier used throughout the system.
#
# score_fn: callable(task: dict) -> float  (-1.0 to +1.0)
#   Positive = LLM is a good fit for the task.
#   Negative = LLM is a poor fit.
#   None = placeholder; router will skip static scoring for this entry
#          (PerformanceTracker historical data will still be used if available).
LLM_REGISTRY: dict = {
    'claude': {
        'score_fn': None,               # Set to ClaudeAgent().score once implemented
        'context_affinity': 'neutral',  # Strong across all types
        'default_weight': 1.0,
        'cost_per_1k_tokens': 0.015,    # USD — update to current pricing
    },
    'codex': {
        'score_fn': None,               # Set to CodexAgent().score once implemented
        'context_affinity': 'code_generation',
        'default_weight': 1.0,
        'cost_per_1k_tokens': 0.002,
    },
    'gemini': {
        'score_fn': None,               # Set to GeminiAgent().score once implemented
        'context_affinity': 'content_writing',
        'default_weight': 1.0,
        'cost_per_1k_tokens': 0.001,
    },
    'gpt4o_mini': {
        'score_fn': None,               # Set to GPT4oMiniAgent().score once implemented
        'context_affinity': 'neutral',  # Capable across all context types
        'default_weight': 1.0,
        # Must stay in sync with _COST_PER_1K_TOKENS['gpt-4o-mini'] in
        # execution/openai_adapter.py — update both during pricing reviews.
        'cost_per_1k_tokens': 0.0003,   # Blended input+output; update from openai.com/pricing
    },
    # ── Future LLMs go here ───────────────────────────────────────────────────
    # 'gpt4': {
    #     'score_fn': GPT4Agent().score,
    #     'context_affinity': 'neutral',
    #     'default_weight': 1.0,
    #     'cost_per_1k_tokens': 0.030,
    # },
    # 'mistral': {
    #     'score_fn': MistralAgent().score,
    #     'context_affinity': 'data_analysis',
    #     'default_weight': 0.8,
    #     'cost_per_1k_tokens': 0.0007,
    # },
}
