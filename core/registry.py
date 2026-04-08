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
  3. Set model_key to the matching "provider:model" key in models/registry.py
  4. The router picks it up automatically — no other changes needed
  5. Add a CHANGELOG.md entry

CONTEXT AFFINITIES:
  'code_generation'  — LLMs specialised in writing and completing code
  'content_writing'  — LLMs specialised in prose, blogs, marketing copy
  'data_analysis'    — LLMs strong on structured reasoning and data tasks
  'neutral'          — LLMs that perform well across all context types

WEIGHT GUIDELINES:
  default_weight = 1.0 is the baseline.
  Adjust during scheduled effectiveness reviews only (see docs/SOP.md).
  Max change: ±20% per review cycle. Document the rationale in CHANGELOG.md.

PRICING:
  Do NOT add cost figures here. Pricing lives in models/registry.py.
  model_key links this routing entry to its pricing entry.
"""

# ── LLM Registry ──────────────────────────────────────────────────────────────
# Each entry key is a unique LLM identifier used throughout the system.
#
# score_fn: callable(task: dict) -> float  (-1.0 to +1.0)
#   Positive = LLM is a good fit for the task.
#   Negative = LLM is a poor fit.
#   None = placeholder; router will skip static scoring for this entry
#          (PerformanceTracker historical data will still be used if available).
#
# model_key: "provider:model" key into models/registry.py LLM_REGISTRY.
#   Used by token_allocator (cost estimation) and performance_tracker (ROI).
#   Set to '' for LLMs not yet registered in the models registry.
LLM_REGISTRY: dict = {
    'claude': {
        'score_fn': None,               # Set to ClaudeAgent().score once implemented
        'context_affinity': 'neutral',  # Strong across all types
        'default_weight': 1.0,
        'model_key': 'anthropic:claude-3-5-sonnet',
    },
    'codex': {
        'score_fn': None,               # Set to CodexAgent().score once implemented
        'context_affinity': 'code_generation',
        'default_weight': 1.0,
        'model_key': '',                # Set to e.g. 'openai:gpt-4o' when wired
    },
    'gemini': {
        'score_fn': None,               # Set to GeminiAgent().score once implemented
        'context_affinity': 'content_writing',
        'default_weight': 1.0,
        'model_key': 'google:gemini-pro',
    },
    'grok': {
        'score_fn': None,               # Set to GrokAgent().score once implemented
        'context_affinity': 'data_analysis',
        'default_weight': 1.0,
        'model_key': 'xai:grok-4.20-reasoning',
    },
    'qwen9b_ollama': {
        'score_fn': None,               # Set to QwenAgent().score once implemented
        'context_affinity': 'code_generation',
        'default_weight': 0.9,
        'model_key': 'ollama:qwen2.5-coder:9b',
    },
    'phi_ollama': {
        'score_fn': None,               # Set to PhiAgent().score once implemented
        'context_affinity': 'neutral',
        'default_weight': 0.9,
        'model_key': 'ollama:phi4',
    },
    'gpt4o_mini': {
        'score_fn': None,               # Set to GPT4oMiniAgent().score once implemented
        'context_affinity': 'neutral',  # Capable across all context types
        'default_weight': 1.0,
        'model_key': 'openai:gpt-4o-mini',
    },
    # ── Future LLMs go here ───────────────────────────────────────────────────
    # 'gpt4o': {
    #     'score_fn': GPT4oAgent().score,
    #     'context_affinity': 'neutral',
    #     'default_weight': 1.0,
    #     'model_key': 'openai:gpt-4o',
    # },
    # 'mistral': {
    #     'score_fn': MistralAgent().score,
    #     'context_affinity': 'data_analysis',
    #     'default_weight': 0.8,
    #     'model_key': 'mistral:mixtral',
    # },
}
