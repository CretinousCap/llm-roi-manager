"""
Performance Tracker — Per-LLM Effectiveness History
=====================================================
PURPOSE:
  Maintain a rolling history of quality scores for each LLM per task context
  type. Feed this data back into the router to improve recommendations over time.

ADAPTED FROM:
  The score decay patterns in donchian_trend.py and mean_reversion.py
  (HybridQuant trading system). Those modules maintain a running score that
  decays each bar without a new signal; this module maintains a running
  effectiveness estimate that decays over time without new data.

HOW IT WORKS:
  For each (llm, context_type) pair:
    1. Store quality scores in a rolling window (deque, bounded by history_window)
    2. Apply exponential decay to older observations so recent performance
       is weighted more heavily (like the 3-speed decay in mean_reversion.py)
    3. Expose score() — an effectiveness float (-1 to +1) the router can use
       as the score_fn for a registered LLM

USAGE:
  tracker = PerformanceTracker()
  tracker.record('claude', 'code_generation', quality_score=0.85)
  tracker.record('claude', 'code_generation', quality_score=0.78)
  score = tracker.score('claude', task={'dominant': 'code_generation'})
  # Returns float in -1..+1 representing effectiveness vs 0.5 baseline
"""

from __future__ import annotations

from collections import defaultdict, deque


class PerformanceTracker:
    """
    Tracks rolling effectiveness of each LLM per context type.

    Create one instance and share it across sessions. Records accumulate
    across the lifetime of the instance (or until reset() is called).

    score() returns a float (-1 to +1) suitable for use as a score_fn
    in LLM_REGISTRY. Positive = above 0.5 baseline, Negative = below baseline.
    """

    def __init__(self, config: dict = None):
        from core import TRACKER_DEFAULTS
        cfg = {**TRACKER_DEFAULTS, **(config or {})}
        self.cfg = cfg

        # _history[llm][context_type] = deque of (quality_score, weight)
        self._history: dict = defaultdict(lambda: defaultdict(deque))

    def record(self, llm: str, context_type: str,
               quality_score: float) -> None:
        """
        Record a quality observation for an LLM on a given context type.

        Args:
            llm:           LLM identifier (e.g. 'claude', 'codex').
            context_type:  Context dimension (e.g. 'code_generation').
            quality_score: Quality rating for this observation (0.0 – 1.0).
        """
        quality_score = max(0.0, min(1.0, quality_score))
        window = self.cfg['history_window']
        bucket = self._history[llm][context_type]

        # Decay all existing weights before appending the new observation
        decay = self.cfg['history_decay']
        decayed = deque(((q, w * decay) for q, w in bucket), maxlen=window)
        decayed.append((quality_score, 1.0))
        self._history[llm][context_type] = decayed

    def score(self, llm: str, task: dict) -> float:
        """
        Return an effectiveness score (-1 to +1) for this LLM on the task.

        The score is the weighted-average quality (0-1) rescaled to -1..+1
        relative to a neutral baseline of 0.5. LLMs with fewer than
        min_observations return 0.0 (neutral — let the registry weight decide).

        Args:
            llm:  LLM identifier.
            task: Task dict; uses 'dominant' key for context type lookup.
                  Also uses 'context_scores' for a blended lookup if present.

        Returns:
            float in -1.0 .. +1.0
        """
        dominant = task.get('dominant', '')
        context_scores = task.get('context_scores', {dominant: 1.0} if dominant else {})

        weighted_score = 0.0
        weight_total = 0.0

        for ctx_type, ctx_weight in context_scores.items():
            if ctx_weight <= 0:
                continue
            bucket = self._history[llm].get(ctx_type, deque())
            if len(bucket) < self.cfg['min_observations']:
                continue

            total_w = sum(w for _, w in bucket)
            if total_w == 0:
                continue
            avg_quality = sum(q * w for q, w in bucket) / total_w

            # Rescale 0..1 quality to -1..+1 centred at 0.5 baseline
            effectiveness = (avg_quality - 0.5) * 2.0
            weighted_score += effectiveness * ctx_weight
            weight_total += ctx_weight

        if weight_total == 0:
            return 0.0  # Insufficient data — neutral

        return max(-1.0, min(1.0, weighted_score / weight_total))

    def summary(self, llm: str) -> dict:
        """
        Return a summary dict of recorded observations for an LLM.

        Useful for effectiveness review reports.
        """
        result = {}
        for ctx_type, bucket in self._history[llm].items():
            if not bucket:
                continue
            scores = [q for q, _ in bucket]
            result[ctx_type] = {
                'observations': len(scores),
                'mean_quality': round(sum(scores) / len(scores), 4),
                'min_quality': round(min(scores), 4),
                'max_quality': round(max(scores), 4),
            }
        return result

    def reset(self, llm: str = None, context_type: str = None) -> None:
        """
        Clear history. Omit all args to clear everything.
        Pass llm to clear one LLM, llm+context_type for one bucket.
        """
        if llm is None:
            self._history.clear()
        elif context_type is None:
            self._history[llm].clear()
        else:
            self._history[llm][context_type].clear()
