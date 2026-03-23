"""
Task Classifier — Context Scoring
==================================
PURPOSE:
  Determine the context type of a task (code generation, content writing,
  data analysis, or a blend) so the router can weight LLMs appropriately.

ADAPTED FROM:
  regime_classifier.py in the HybridQuant trading system.
  regime_score → context_score per dimension
  regime_trending/ranging → context affinity weights for LLM routing

WHY SOFT SCORING (not hard categories):
  "Write a Python function" is clearly code_generation.
  "Analyse this dataset and write a summary report" is a blend of
  data_analysis AND content_writing.
  Hard categories would force an either/or choice and break on blended tasks.

  Soft scoring gives every task a continuous score per dimension (-1 to +1).
  The router then soft-weights each LLM by how well its affinity matches the
  task's dimension scores — exactly as regime weighting works in HybridQuant's
  signal combiner.

HOW IT WORKS:
  1. Tokenise the task description into keywords
  2. Count keyword density per context dimension
  3. Compute a raw score per dimension (positive = strong signal)
  4. Smooth with EMA if previous context scores are passed (optional)
  5. Return context_scores dict + overall confidence + structured task profile

TUNING:
  The primary tuning lever is the keyword vocabulary (_VOCAB dict below).
  Add terms to make the classifier more sensitive to domain-specific tasks.
  Threshold parameters (keyword_window, context_ema_alpha) are in TASK_DEFAULTS.
"""

from __future__ import annotations

import re


# ── Keyword vocabulary per context dimension ──────────────────────────────────
_VOCAB: dict = {
    'code_generation': [
        'function', 'class', 'method', 'implement', 'code', 'script',
        'algorithm', 'module', 'api', 'refactor', 'debug', 'test', 'unit test',
        'python', 'javascript', 'typescript', 'sql', 'bash', 'shell',
        'endpoint', 'parse', 'regex', 'loop', 'async', 'exception',
        'return', 'variable', 'import', 'library', 'package', 'build',
    ],
    'content_writing': [
        'write', 'blog', 'article', 'post', 'draft', 'content', 'copy',
        'email', 'newsletter', 'summary', 'paragraph', 'tone', 'audience',
        'headline', 'introduction', 'conclusion', 'story', 'narrative',
        'persuasive', 'engaging', 'describe', 'explain to', 'seo',
        'social media', 'tweet', 'caption', 'marketing',
    ],
    'data_analysis': [
        'analyse', 'analyze', 'dataset', 'data', 'table', 'csv', 'chart',
        'trend', 'correlation', 'distribution', 'statistics', 'aggregate',
        'group by', 'filter', 'outlier', 'mean', 'median', 'variance',
        'regression', 'model', 'predict', 'forecast', 'visualise',
        'visualize', 'plot', 'insight', 'report', 'metric',
    ],
}

# ── Profile heuristic keyword lists ───────────────────────────────────────────

# task_type
_TASK_TYPE_EDIT_KW = {
    'refactor', 'fix', 'debug', 'edit', 'update', 'change', 'correct',
    'revise', 'improve', 'adjust', 'rewrite', 'modify',
}
_TASK_TYPE_ANALYSIS_KW = {
    'analyse', 'analyze', 'evaluate', 'review', 'compare', 'assess',
    'understand', 'investigate', 'inspect', 'examine', 'audit',
}
_TASK_TYPE_PLANNING_KW = {
    'plan', 'design', 'outline', 'strategy', 'roadmap', 'architect',
    'structure', 'blueprint', 'schema', 'spec', 'specification',
}
_TASK_TYPE_TRANSFORMATION_KW = {
    'convert', 'transform', 'translate', 'migrate', 'reformat',
    'port', 'transcribe', 'reshape',
}

# complexity
_COMPLEXITY_HIGH_KW = {
    'complex', 'advanced', 'large', 'enterprise', 'distributed', 'optimise',
    'optimize', 'production', 'scalable', 'concurrent', 'multithreaded',
    'microservice', 'architecture',
}
_COMPLEXITY_LOW_KW = {
    'simple', 'basic', 'short', 'quick', 'minimal', 'small', 'brief',
    'trivial', 'straightforward',
}

# interaction_stage
_STAGE_REFINEMENT_KW = {
    'improve', 'refine', 'revise', 'fix', 'change', 'adjust', 'better',
    'update', 'rework', 'tweak',
}
_STAGE_FOLLOWUP_KW = {
    'also', 'additionally', 'next', 'then', 'continue', 'follow',
    'furthermore', 'moreover', 'in addition',
}

# structure
_STRUCTURE_ITERATIVE_KW = {
    'iterative', 'loop', 'repeated', 'step by step', 'each step',
    'incremental', 'iterate',
}
_STRUCTURE_MULTI_KW = {
    'multiple', 'several', 'list of', 'sections', 'parts', 'steps',
    'items', 'components', 'modules',
}

# latency_sensitivity
_LATENCY_HIGH_KW = {
    'immediately', 'real-time', 'realtime', 'live', 'instant', 'instantly',
    'fast', 'urgent', 'asap', 'now',
}

# risk
_RISK_HIGH_KW = {
    'production', 'deploy', 'deployment', 'database', 'security', 'auth',
    'authentication', 'payment', 'critical', 'sensitive', 'finance',
    'financial', 'compliance', 'gdpr', 'pii',
}

# Modality derived from dominant context dimension
_MODALITY_MAP: dict = {
    'code_generation': 'code',
    'content_writing': 'text',
    'data_analysis': 'analysis',
}


def classify(task: dict, config: dict = None,
             history: list = None) -> dict:
    """
    Score a task across context dimensions and return context scores.

    Analogous to regime_classifier.compute() — produces continuous scores
    per dimension that the router uses to weight each LLM.

    Args:
        task:    Task dict with at least 'description' (str) and optionally
                 'type_hint' (str) — one of 'code_generation',
                 'content_writing', 'data_analysis'.
        config:  Override defaults from core/__init__.py TASK_DEFAULTS.
        history: Optional list of previous context score dicts for EMA
                 smoothing across a sequence of related tasks.

    Returns:
        dict with:
          context_scores:  {'code_generation': float, 'content_writing': float,
                            'data_analysis': float}  each -1.0 to +1.0
          dominant:        name of the highest-scoring dimension
          confidence:      abs(dominant_score), 0.0 – 1.0
          raw_hits:        {'code_generation': int, ...}  keyword match counts
          profile:         structured task profile dict with keys:
                             modality, task_type, complexity,
                             interaction_stage, structure,
                             latency_sensitivity, risk
    """
    from core import TASK_DEFAULTS
    cfg = {**TASK_DEFAULTS, **(config or {})}

    description: str = task.get('description', '')
    type_hint: str = task.get('type_hint', '')

    # ── Step 1: Tokenise ─────────────────────────────────────────────────────
    text = (description + ' ' + type_hint).lower()
    tokens = re.findall(r"[a-z_]+", text)
    window = tokens[:cfg['keyword_window']]
    window_str = ' '.join(window)

    # ── Step 2: Count keyword hits per dimension ─────────────────────────────
    raw_hits: dict = {}
    for dimension, keywords in _VOCAB.items():
        hits = sum(1 for kw in keywords if kw in window_str)
        raw_hits[dimension] = hits

    total_hits = sum(raw_hits.values()) or 1  # avoid division by zero

    # ── Step 3: Compute raw score per dimension ──────────────────────────────
    # Score = (hits_for_dim - hits_for_others) / total_hits
    raw_scores: dict = {}
    for dimension in _VOCAB:
        own = raw_hits[dimension]
        others = total_hits - own
        raw_scores[dimension] = (own - others) / total_hits

    # Honour explicit type_hint: boost the hinted dimension
    if type_hint in _VOCAB:
        for dimension in raw_scores:
            if dimension == type_hint:
                raw_scores[dimension] = min(1.0, raw_scores[dimension] + 0.3)
            else:
                raw_scores[dimension] = max(-1.0, raw_scores[dimension] - 0.1)

    # ── Step 4: EMA smoothing across sequential tasks (optional) ─────────────
    # Prevents flip-flopping when task descriptions vary slightly between calls.
    context_scores = dict(raw_scores)
    if history:
        alpha = cfg['context_ema_alpha']
        last = history[-1].get('context_scores', {})
        for dim in context_scores:
            prev = last.get(dim, context_scores[dim])
            context_scores[dim] = alpha * context_scores[dim] + (1 - alpha) * prev

    # ── Step 5: Derive dominant dimension and confidence ─────────────────────
    dominant = max(context_scores, key=lambda d: context_scores[d])
    confidence = abs(context_scores[dominant])

    # ── Step 6: Build structured task profile ────────────────────────────────
    profile = _build_profile(text, dominant)

    return {
        'context_scores': context_scores,
        'dominant': dominant,
        'confidence': round(confidence, 4),
        'raw_hits': raw_hits,
        'profile': profile,
    }


# ── Profile builder ───────────────────────────────────────────────────────────

def _build_profile(text: str, dominant: str) -> dict:
    """
    Derive a structured task profile from the lowercased task text and
    the dominant context dimension. Uses keyword heuristics only — no LLM calls.

    Args:
        text:     Lowercased concatenation of description + type_hint.
        dominant: Winning context dimension from the classifier.

    Returns:
        dict with keys: modality, task_type, complexity, interaction_stage,
                        structure, latency_sensitivity, risk.
    """
    # modality
    modality = _MODALITY_MAP.get(dominant, 'text')

    # task_type — first match wins in priority order
    if any(kw in text for kw in _TASK_TYPE_ANALYSIS_KW):
        task_type = 'analysis'
    elif any(kw in text for kw in _TASK_TYPE_PLANNING_KW):
        task_type = 'planning'
    elif any(kw in text for kw in _TASK_TYPE_TRANSFORMATION_KW):
        task_type = 'transformation'
    elif any(kw in text for kw in _TASK_TYPE_EDIT_KW):
        task_type = 'edit'
    else:
        task_type = 'generation'

    # complexity
    if any(kw in text for kw in _COMPLEXITY_HIGH_KW):
        complexity = 'high'
    elif any(kw in text for kw in _COMPLEXITY_LOW_KW):
        complexity = 'low'
    else:
        complexity = 'medium'

    # interaction_stage
    if any(kw in text for kw in _STAGE_FOLLOWUP_KW):
        interaction_stage = 'followup'
    elif any(kw in text for kw in _STAGE_REFINEMENT_KW):
        interaction_stage = 'refinement'
    else:
        interaction_stage = 'initial'

    # structure
    if any(kw in text for kw in _STRUCTURE_ITERATIVE_KW):
        structure = 'iterative'
    elif any(kw in text for kw in _STRUCTURE_MULTI_KW):
        structure = 'multi_block'
    else:
        structure = 'single_output'

    # latency_sensitivity
    latency_sensitivity = (
        'high' if any(kw in text for kw in _LATENCY_HIGH_KW) else 'low'
    )

    # risk
    risk = 'high' if any(kw in text for kw in _RISK_HIGH_KW) else 'low'

    return {
        'modality': modality,
        'task_type': task_type,
        'complexity': complexity,
        'interaction_stage': interaction_stage,
        'structure': structure,
        'latency_sensitivity': latency_sensitivity,
        'risk': risk,
    }
