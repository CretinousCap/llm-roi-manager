"""Core package — default configs for all LLM ROI Manager modules.

Override any value by passing config= to the relevant classify/route/allocate
function. Only the keys you supply are overridden; all others keep their
defaults (dict-merge pattern used throughout this project).
"""

# ── Task Classifier defaults ──────────────────────────────────────────────────
TASK_DEFAULTS = {
    # Number of words to inspect for keyword density scoring
    'keyword_window': 50,
    # EMA smoothing factor for context scores across sequential tasks
    'context_ema_alpha': 0.3,
    # Minimum keyword matches to produce a non-zero context score
    'min_keyword_hits': 1,
}

# ── Session defaults ──────────────────────────────────────────────────────────
SESSION_DEFAULTS = {
    # Activate trailing quality lock once quality exceeds this value (0-1)
    'trail_activation_quality': 0.70,
    # Fraction of peak quality allowed to drop before flagging degradation
    # e.g. peak=0.90, offset=0.20 → flag if quality drops below 0.72
    'trail_offset': 0.20,
    # IterationGuard: review after this many LLM calls
    'max_iterations': 6,
    # IterationGuard: must exceed this quality to continue past max_iterations
    'min_continuation_quality': 0.50,
    # Whether to adapt trail offset based on task context confidence
    'context_adapt': True,
    # ±range added to trail_offset based on context confidence
    'context_offset_range': 0.05,
}

# ── Budget defaults ───────────────────────────────────────────────────────────
BUDGET_DEFAULTS = {
    # Base token budget per task when no budget_tokens hint is given
    'base_token_budget': 2000,
    # Hard cap regardless of confidence
    'max_token_budget': 8000,
    # Minimum regardless of confidence
    'min_token_budget': 500,
    # Scale budget by router confidence: True = Kelly-lite approach
    'use_confidence_scaling': True,
    # Minimum scale factor applied when router confidence is 0
    'min_confidence_scale': 0.4,
}

# ── Executor defaults ─────────────────────────────────────────────────────────
EXECUTOR_DEFAULTS = {
    # Token limit passed to adapters when allocation dict has no token_budget
    'default_max_tokens': 2000,
}

# ── Evaluator defaults ────────────────────────────────────────────────────────
EVALUATOR_DEFAULTS = {
    # Max tokens the judge LLM is allowed to produce (score only, not prose)
    'judge_max_tokens': 50,
    # Truncate response text sent to the judge to limit prompt size
    'max_response_chars': 4000,
}

# ── Result store defaults ─────────────────────────────────────────────────────
STORE_DEFAULTS = {
    # Default JSONL file path (relative to the working directory)
    'default_path': 'results/llm_results.jsonl',
}

# ── Performance tracker defaults ──────────────────────────────────────────────
TRACKER_DEFAULTS = {
    # Rolling window size for quality score history per LLM per task type
    'history_window': 20,
    # Exponential decay on older observations (1.0 = no decay, 0.9 = slight decay)
    'history_decay': 0.95,
    # Minimum observations before a score is considered reliable
    'min_observations': 3,
}
