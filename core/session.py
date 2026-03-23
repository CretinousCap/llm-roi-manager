"""
WrapSession — Stateful Session Manager
=======================================
PURPOSE:
  Wrap a series of LLM interactions for a single task in a managed session
  that tracks quality, cost, and iteration count — then cleans up on exit.

ADAPTED FROM:
  TrailingProfitLock (trailing_stop.py) and TimeExit (time_exit.py) in the
  HybridQuant trading system.

HOW IT WORKS:

  Trailing quality lock (→ TrailingProfitLock):
    1. Track the peak quality score seen across all iterations in the session
    2. Once quality exceeds trail_activation_quality, the trail activates
    3. If current quality drops below peak * (1 - trail_offset), the session
       is flagged with quality_degraded = True
    4. Caller decides whether to stop or continue — the session only signals
       degradation, it does not force termination
    5. trail_offset widens when context confidence is high (more room) and
       tightens when confidence is low (lock quality faster)

  IterationGuard (→ TimeExit):
    1. Count iterations across the session
    2. After max_iterations, require quality > min_continuation_quality
    3. If quality is below threshold → guard.should_stop() returns True
    4. Targets "grinding sessions" that iterate without converging

USAGE:
  Use as a context manager. State is automatically cleaned up on __exit__:

    with WrapSession(task=task, context=context) as session:
        for _ in range(10):
            result = call_your_llm(...)
            session.record(llm='claude', quality_score=0.85,
                           tokens_used=1200, cost_usd=0.012)
            if session.quality_degraded:
                break
            if session.guard.should_stop(session.iteration, session.current_quality):
                break
        summary = session.summary()
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


class IterationGuard:
    """
    Prevents runaway sessions that iterate without converging.

    Analogous to TimeExit in the trading system: after max_iterations,
    the session must demonstrate sufficient quality to continue.
    """

    def __init__(self, config: dict = None):
        cfg = {
            'max_iterations': 6,
            'min_continuation_quality': 0.50,
        }
        cfg.update(config or {})
        self.cfg = cfg

    def should_stop(self, iteration: int, current_quality: float) -> bool:
        """
        Return True if the session should stop due to stale iteration.

        Args:
            iteration:       Number of LLM calls completed so far.
            current_quality: Most recent quality score (0.0 – 1.0).

        Returns:
            True if past the iteration limit and quality is below threshold.
        """
        return (iteration >= self.cfg['max_iterations']
                and current_quality < self.cfg['min_continuation_quality'])


class WrapSession:
    """
    Stateful context manager for a single LLM task session.

    Tracks:
      - peak_quality:       highest quality score seen (like peak price)
      - current_quality:    most recent quality score
      - quality_degraded:   True when quality drops too far below peak
      - cumulative_cost:    total USD spend across all iterations
      - cumulative_tokens:  total tokens used
      - iteration:          number of record() calls made
      - guard:              IterationGuard instance

    Create one WrapSession per task. Do not reuse across tasks.

    Example::

        with WrapSession(task=task, context=context) as session:
            result = call_llm(...)
            session.record(llm='claude', quality_score=0.9,
                           tokens_used=1200, cost_usd=0.012)
            if session.quality_degraded or session.guard.should_stop(
                    session.iteration, session.current_quality):
                break
        summary = session.summary()
    """

    def __init__(self, task: dict = None, context: dict = None,
                 config: dict = None):
        from core import SESSION_DEFAULTS
        cfg = {**SESSION_DEFAULTS, **(config or {})}
        self.cfg = cfg

        self.task = task or {}
        self.context = context or {}

        self.peak_quality: float = 0.0
        self.current_quality: float = 0.0
        self.quality_degraded: bool = False
        self.cumulative_cost: float = 0.0
        self.cumulative_tokens: int = 0
        self.iteration: int = 0
        self._history: list = []
        self._started_at: Optional[datetime] = None

        self.guard = IterationGuard(config={
            'max_iterations': cfg['max_iterations'],
            'min_continuation_quality': cfg['min_continuation_quality'],
        })

    # ── Context manager protocol ─────────────────────────────────────────────

    def __enter__(self) -> 'WrapSession':
        self._started_at = datetime.now(timezone.utc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._cleanup()
        return False  # Do not suppress exceptions

    def _cleanup(self) -> None:
        """Reset live state. History is retained so caller can inspect after exit."""
        self.peak_quality = 0.0
        self.current_quality = 0.0
        self.quality_degraded = False

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, llm: str, quality_score: float,
               tokens_used: int = 0, cost_usd: float = 0.0,
               metadata: dict = None) -> None:
        """
        Record the outcome of one LLM call within this session.

        Args:
            llm:           Name of the LLM used (e.g. 'claude', 'codex').
            quality_score: Effectiveness rating for this iteration (0.0 – 1.0).
                           Higher is better. Your evaluation logic provides this.
            tokens_used:   Number of tokens consumed.
            cost_usd:      Cost of this call in USD.
            metadata:      Any extra data to store with this record.
        """
        quality_score = max(0.0, min(1.0, quality_score))
        self.iteration += 1
        self.current_quality = quality_score
        self.cumulative_cost += cost_usd
        self.cumulative_tokens += tokens_used

        if quality_score > self.peak_quality:
            self.peak_quality = quality_score

        self._check_quality_trail()

        self._history.append({
            'iteration': self.iteration,
            'llm': llm,
            'quality_score': quality_score,
            'tokens_used': tokens_used,
            'cost_usd': cost_usd,
            'quality_degraded': self.quality_degraded,
            'metadata': metadata or {},
        })

    # ── Trailing quality lock ─────────────────────────────────────────────────

    def _check_quality_trail(self) -> None:
        """
        Flag quality_degraded if current quality has dropped too far below peak.

        Analogous to the trailing stop trigger in TrailingProfitLock.
        """
        cfg = self.cfg

        if self.peak_quality < cfg['trail_activation_quality']:
            return  # Trail not yet activated

        offset = cfg['trail_offset']
        if cfg['context_adapt'] and self.context:
            confidence = self.context.get('confidence', 0.0)
            offset += confidence * cfg['context_offset_range']
        offset = max(0.05, min(0.40, offset))

        trail_threshold = self.peak_quality * (1.0 - offset)
        if self.current_quality < trail_threshold:
            self.quality_degraded = True

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Return a summary of the session. Call after the with-block.
        """
        elapsed = None
        if self._started_at:
            elapsed = (datetime.now(timezone.utc) - self._started_at).total_seconds()

        return {
            'iterations': self.iteration,
            'peak_quality': self.peak_quality,
            'final_quality': self.current_quality,
            'quality_degraded': self.quality_degraded,
            'cumulative_cost_usd': self.cumulative_cost,
            'cumulative_tokens': self.cumulative_tokens,
            'elapsed_seconds': elapsed,
            'history': list(self._history),
        }
