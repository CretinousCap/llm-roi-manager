"""
Performance Tracker — Per-LLM Effectiveness History and ROI Engine
===================================================================
PURPOSE:
  Maintain a rolling history of quality scores and costs for each LLM per
  task context type. Feed this data back into the router to improve
  recommendations over time, including ROI-weighted routing.

ADAPTED FROM:
  The score decay patterns in donchian_trend.py and mean_reversion.py
  (HybridQuant trading system). Those modules maintain a running score that
  decays each bar without a new signal; this module maintains a running
  effectiveness estimate that decays over time without new data.

HOW IT WORKS:
  For each (llm, context_type) pair:
    1. Store quality scores and costs in a rolling window (deque, bounded by
       history_window). Each entry is a (quality_score, weight, cost_usd) tuple.
    2. Apply exponential decay to older observations so recent performance
       is weighted more heavily (like the 3-speed decay in mean_reversion.py).
    3. Expose score() — an effectiveness float (-1 to +1) the router can use
       as the score_fn for a registered LLM.
    4. Expose roi_score() — an ROI-adjusted effectiveness float (-1 to +1)
       that penalises LLMs whose actual cost exceeds their registry reference
       cost. Used by the router when cost data is available.

USAGE:
  tracker = PerformanceTracker()
  tracker.record('claude', 'code_generation', quality_score=0.85, cost_usd=0.012)
  tracker.record('claude', 'code_generation', quality_score=0.78, cost_usd=0.010)
  score = tracker.score('claude', task={'dominant': 'code_generation'})
  roi   = tracker.roi_score('claude', task={'dominant': 'code_generation'})
  # Both return float in -1..+1
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

        # _history[llm][context_type] = deque of (quality_score, weight, cost_usd)
        self._history: dict = defaultdict(lambda: defaultdict(deque))
        # _failures[llm][context_type] = count of non-success records
        self._failures: dict = defaultdict(lambda: defaultdict(int))
        # _latency_ms[llm][context_type] = deque of latency_ms floats
        self._latency_ms: dict = defaultdict(lambda: defaultdict(deque))

    def record(self, llm: str, context_type: str,
               quality_score: float, cost_usd: float = 0.0,
               status: str = 'success', latency_ms: float = 0.0) -> None:
        """
        Record a quality/cost observation for an LLM on a given context type.

        Args:
            llm:           LLM identifier (e.g. 'claude', 'codex').
            context_type:  Context dimension (e.g. 'code_generation').
            quality_score: Quality rating for this observation (0.0 – 1.0).
            cost_usd:      Actual cost of this LLM call in USD (default 0.0).
            status:        Execution status ('success', 'error', 'unavailable').
                           Records where status != 'success' are not included in
                           quality or cost history but do increment failure counts.
            latency_ms:    Wall-clock time for the API call in milliseconds.
                           Tracked regardless of status.
        """
        window = self.cfg['history_window']

        # Always track latency (regardless of status)
        latency_deque = self._latency_ms[llm][context_type]
        latency_deque.append(max(0.0, float(latency_ms)))
        if len(latency_deque) > window:
            latency_deque.popleft()

        if status != 'success':
            self._failures[llm][context_type] += 1
            return  # Do not record quality or cost for failed calls

        quality_score = max(0.0, min(1.0, quality_score))
        cost_usd = max(0.0, cost_usd)
        bucket = self._history[llm][context_type]

        # Decay all existing weights before appending the new observation
        decay = self.cfg['history_decay']
        decayed = deque(
            ((q, w * decay, c) for q, w, c in bucket), maxlen=window
        )
        decayed.append((quality_score, 1.0, cost_usd))
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

            total_w = sum(w for _, w, _ in bucket)
            if total_w == 0:
                continue
            avg_quality = sum(q * w for q, w, _ in bucket) / total_w

            # Rescale 0..1 quality to -1..+1 centred at 0.5 baseline
            effectiveness = (avg_quality - 0.5) * 2.0
            weighted_score += effectiveness * ctx_weight
            weight_total += ctx_weight

        if weight_total == 0:
            return 0.0  # Insufficient data — neutral

        return max(-1.0, min(1.0, weighted_score / weight_total))

    def roi_score(self, llm: str, task: dict) -> float:
        """
        Return an ROI-adjusted effectiveness score (-1 to +1) for this LLM.

        Extends score() by penalising LLMs whose actual average cost exceeds
        the registry reference cost (cost_per_1k_tokens). The penalty is
        proportional to the cost overrun, capped at the full effectiveness
        value so the result never falls below -1.

        Falls back to score() when:
          - There is insufficient data (returns 0.0 from score())
          - No cost data has been recorded (all costs are 0.0)
          - The LLM is not in LLM_REGISTRY

        Args:
            llm:  LLM identifier.
            task: Task dict; uses 'dominant' and 'context_scores' keys.

        Returns:
            float in -1.0 .. +1.0
        """
        base = self.score(llm, task)
        if base == 0.0:
            return 0.0  # Insufficient quality data — neutral

        dominant = task.get('dominant', '')
        context_scores = task.get(
            'context_scores', {dominant: 1.0} if dominant else {}
        )

        # Gather cost data across relevant context buckets
        total_cost = 0.0
        total_observations = 0
        for ctx_type, ctx_weight in context_scores.items():
            if ctx_weight <= 0:
                continue
            bucket = self._history[llm].get(ctx_type, deque())
            if len(bucket) < self.cfg['min_observations']:
                continue
            costs = [c for _, _, c in bucket]
            total_cost += sum(costs)
            total_observations += len(costs)

        if total_observations == 0 or total_cost == 0.0:
            return base  # No cost data — use quality effectiveness as-is

        avg_cost_usd = total_cost / total_observations

        # Look up reference cost from models registry via the model_key in
        # core registry. Falls back to 0.0 (no penalty) if not configured.
        try:
            from core.registry import LLM_REGISTRY as ROUTING_REGISTRY
            from models.registry import LLM_REGISTRY as MODELS_REGISTRY
            model_key = ROUTING_REGISTRY.get(llm, {}).get('model_key', '')
            ref_cost_per_1k = 0.0
            if model_key and model_key in MODELS_REGISTRY:
                m = MODELS_REGISTRY[model_key]
                # Blended estimate: average of input and output pricing.
                ref_cost_per_1k = (
                    m.get('input_price_per_1k', 0.0)
                    + m.get('output_price_per_1k', 0.0)
                ) / 2.0
        except (ImportError, KeyError, TypeError):
            ref_cost_per_1k = 0.0

        if ref_cost_per_1k <= 0.0:
            return base  # No reference cost — cannot normalise

        # Reference cost per call: registry stores cost_per_1k_tokens, so we
        # compare avg_cost_usd directly against it (baseline = 1k tokens/call).
        # ref_cost_per_1k > 0 is guaranteed by the guard on line above.
        ref_cost_per_call = ref_cost_per_1k

        # Penalty = fraction by which actual cost exceeds reference.
        # Positive overrun → reduce effectiveness; under-run → no bonus (cap at 0).
        overrun_fraction = max(0.0, (avg_cost_usd - ref_cost_per_call)
                               / ref_cost_per_call)

        # Scale the penalty by the effectiveness magnitude so high-quality
        # LLMs get a proportional reduction, not a flat cut.
        penalty = min(abs(base), overrun_fraction * abs(base))
        roi = base - penalty if base >= 0 else base + penalty

        return max(-1.0, min(1.0, round(roi, 4)))

    def summary(self, llm: str) -> dict:
        """
        Return a summary dict of recorded observations for an LLM.

        Useful for effectiveness review reports.
        Includes quality statistics, cost-per-quality-unit (ROI metric),
        failure counts, and average latency per context type.
        """
        # Collect all context types seen (quality history OR failures OR latency)
        all_contexts = (
            set(self._history[llm]) | set(self._failures[llm])
            | set(self._latency_ms[llm])
        )

        result = {}
        for ctx_type in all_contexts:
            bucket = self._history[llm].get(ctx_type, deque())
            failure_count = self._failures[llm].get(ctx_type, 0)
            latency_deque = self._latency_ms[llm].get(ctx_type, deque())

            # Skip context types with no recorded data at all
            if not bucket and failure_count == 0 and not latency_deque:
                continue

            mean_latency_ms = (
                round(sum(latency_deque) / len(latency_deque), 2)
                if latency_deque else None
            )

            if not bucket:
                # Only failures or latency recorded; no quality data yet
                result[ctx_type] = {
                    'observations': 0,
                    'mean_quality': None,
                    'min_quality': None,
                    'max_quality': None,
                    'total_cost_usd': 0.0,
                    'mean_cost_usd': None,
                    'cost_per_quality_unit': None,
                    'failure_count': failure_count,
                    'mean_latency_ms': mean_latency_ms,
                }
                continue

            scores = [q for q, _, _ in bucket]
            costs = [c for _, _, c in bucket]
            total_cost = sum(costs)
            total_quality = sum(scores)
            cost_per_quality = (
                round(total_cost / total_quality, 6)
                if total_quality > 0 else None
            )
            result[ctx_type] = {
                'observations': len(scores),
                'mean_quality': round(sum(scores) / len(scores), 4),
                'min_quality': round(min(scores), 4),
                'max_quality': round(max(scores), 4),
                'total_cost_usd': round(total_cost, 6),
                'mean_cost_usd': round(total_cost / len(costs), 6),
                'cost_per_quality_unit': cost_per_quality,
                'failure_count': failure_count,
                'mean_latency_ms': mean_latency_ms,
            }
        return result

    def reset(self, llm: str = None, context_type: str = None) -> None:
        """
        Clear history. Omit all args to clear everything.
        Pass llm to clear one LLM, llm+context_type for one bucket.
        """
        if llm is None:
            self._history.clear()
            self._failures.clear()
            self._latency_ms.clear()
        elif context_type is None:
            self._history[llm].clear()
            self._failures[llm].clear()
            self._latency_ms[llm].clear()
        else:
            self._history[llm][context_type].clear()
            self._failures[llm][context_type] = 0
            self._latency_ms[llm][context_type].clear()
