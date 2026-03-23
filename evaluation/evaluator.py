"""
Evaluator — LLM-as-Judge Quality Scorer
=========================================
PURPOSE:
  Score the quality of an LLM's response to a task. Acts as the bridge
  between raw LLM output and the quality metrics consumed by
  PerformanceTracker and WrapSession.

HOW IT WORKS:
  Two evaluation modes:

  1. LLM-as-judge (preferred): Uses a judge LLM adapter (implements the same
     LLMAdapter interface as LLMExecutor adapters) to score the response.
     The judge is prompted with the task description and the response and
     asked to return a score 0–10. The score is normalised to 0.0–1.0.

  2. Heuristic fallback: When no judge adapter is provided, uses a
     length-saturation and keyword-overlap heuristic. This is a weak signal
     suitable only for development. Wire in a real judge LLM in production.

USAGE:
  # Heuristic-only mode
  evaluator = Evaluator()

  # LLM-as-judge mode
  evaluator = Evaluator(judge_adapter=MyClaudeAdapter())

  result = evaluator.evaluate(
      task={'description': 'Write a sort function in Python.'},
      response='def sort_list(lst): return sorted(lst)',
      context={'dominant': 'code_generation'},
  )
  # result['quality_score'] → float 0.0–1.0
  # result['method']        → 'llm_judge' | 'heuristic'
"""

from __future__ import annotations

import math
import re


_JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator. Score the following LLM response to the given task.

TASK:
{task_description}

RESPONSE:
{response_text}

Rate the response quality from 0 to 10, where:
  0-2:  Completely wrong, harmful, or irrelevant
  3-4:  Partially correct but with significant errors or omissions
  5-6:  Acceptable — addresses the task but could be improved
  7-8:  Good — correct and reasonably complete
  9-10: Excellent — thorough, correct, and well-presented

Respond with ONLY a single integer from 0 to 10. No explanation.\
"""

# Length at which the saturation curve reaches ~63% score (1 - 1/e).
# Responses shorter than this get penalised; longer ones approach 1.0 gradually.
_TARGET_LENGTH_CHARS = 500

# Words shorter than this are ignored during keyword-overlap scoring.
# Filters out stop words (a, an, the, to, …) that are not meaningful signals.
_MIN_SIGNIFICANT_WORD_LENGTH = 4


class Evaluator:
    """
    Scores LLM responses using a judge LLM or heuristic fallback.

    One Evaluator instance may be shared across sessions.

    Args:
        judge_adapter: Optional LLMAdapter used as the judge LLM.
                       If None, the heuristic scorer is used instead.
        config:        Override EVALUATOR_DEFAULTS.
    """

    def __init__(self, judge_adapter=None, config: dict = None):
        from core import EVALUATOR_DEFAULTS
        self.cfg = {**EVALUATOR_DEFAULTS, **(config or {})}
        self._judge = judge_adapter

    def evaluate(self, task: dict, response: str,
                 context: dict = None) -> dict:
        """
        Score a response and return quality metadata.

        Args:
            task:     Task dict with at least 'description' (str).
            response: The LLM's response text to evaluate.
            context:  Optional classify() output (used in heuristic mode).

        Returns:
            dict with:
              quality_score: float 0.0–1.0
              method:        'llm_judge' | 'heuristic'
              raw_score:     int 0–10 from judge, or None for heuristic
              confidence:    float 0.0–1.0 indicating evaluator confidence
        """
        if self._judge is not None:
            return self._judge_evaluate(task, response)
        return self._heuristic_evaluate(task, response, context or {})

    # ── LLM-as-judge ─────────────────────────────────────────────────────────

    def _judge_evaluate(self, task: dict, response: str) -> dict:
        """Score via a judge LLM adapter."""
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            task_description=task.get('description', ''),
            response_text=response[:self.cfg['max_response_chars']],
        )
        try:
            raw = self._judge.complete(
                prompt,
                max_tokens=self.cfg['judge_max_tokens'],
            )
            text = raw.get('text', '').strip()
            score_int = _parse_judge_score(text)
            quality = score_int / 10.0
            return {
                'quality_score': round(quality, 4),
                'method': 'llm_judge',
                'raw_score': score_int,
                'confidence': 0.9,
            }
        except Exception:
            # Judge call failed — fall back to heuristic
            return self._heuristic_evaluate(task, response, {})

    # ── Heuristic fallback ───────────────────────────────────────────────────

    def _heuristic_evaluate(self, task: dict, response: str,
                             context: dict) -> dict:
        """
        Score using length-saturation and keyword-overlap heuristics.

        This is a weak signal intended for development and testing only.
        Use a real judge LLM in production.

        Scoring:
          - Length score:   saturating curve peaking near 500 characters
          - Overlap score:  fraction of significant task words in the response
          - Quality:        equal-weighted blend of both components
        """
        response = response or ''
        task_desc = task.get('description', '')

        length_score = _length_score(len(response))
        overlap_score = _keyword_overlap(task_desc, response)

        quality = max(0.0, min(1.0, 0.5 * length_score + 0.5 * overlap_score))
        return {
            'quality_score': round(quality, 4),
            'method': 'heuristic',
            'raw_score': None,
            'confidence': 0.4,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_judge_score(text: str) -> int:
    """Extract integer 0–10 from judge response. Returns 5 (neutral) on failure."""
    for match in re.findall(r'\b(\d{1,2})\b', text):
        val = int(match)
        if 0 <= val <= 10:
            return val
    return 5  # Neutral fallback


def _length_score(char_count: int) -> float:
    """
    Saturating length score: 0.0 for empty, approaches 1.0 near _TARGET_LENGTH_CHARS.
    Uses 1 - exp(-n / target) so there is no hard cap.
    """
    if char_count == 0:
        return 0.0
    return round(1.0 - math.exp(-char_count / _TARGET_LENGTH_CHARS), 4)


def _keyword_overlap(task_desc: str, response: str) -> float:
    """
    Fraction of significant task words (length >= _MIN_SIGNIFICANT_WORD_LENGTH)
    present in the response. Returns 0.5 (neutral) when the task description
    has no significant words.
    """
    task_words = set(
        w.lower()
        for w in re.findall(
            rf'\b[a-zA-Z]{{{_MIN_SIGNIFICANT_WORD_LENGTH},}}\b', task_desc
        )
    )
    if not task_words:
        return 0.5
    response_lower = response.lower()
    hits = sum(1 for w in task_words if w in response_lower)
    return hits / len(task_words)
