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

  2. Heuristic fallback: When no judge adapter is provided, uses heuristics
     across four quality dimensions (correctness, clarity, completeness,
     efficiency). Weights are modality-aware:
       - code:     correctness 40 %, efficiency 30 %, completeness 20 %, clarity 10 %
       - text:     clarity 40 %, completeness 30 %, correctness 20 %, efficiency 10 %
       - analysis: correctness 40 %, completeness 30 %, clarity 20 %, efficiency 10 %
       - default:  25 % each

USAGE:
  # Heuristic-only mode
  evaluator = Evaluator()

  # LLM-as-judge mode
  evaluator = Evaluator(judge_adapter=MyClaudeAdapter())

  result = evaluator.evaluate(
      task={'description': 'Write a sort function in Python.'},
      response='def sort_list(lst): return sorted(lst)',
      context={'dominant': 'code_generation', 'profile': {'modality': 'code'}},
  )
  # result['quality_score'] → float 0.0–1.0
  # result['dimensions']    → {'correctness': float, 'clarity': float,
  #                             'completeness': float, 'efficiency': float}
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

# ── Modality-aware dimension weights ─────────────────────────────────────────
# Each dict must sum to 1.0.
_DIMENSION_WEIGHTS: dict = {
    'code': {
        'correctness': 0.40,
        'efficiency':  0.30,
        'completeness': 0.20,
        'clarity':     0.10,
    },
    'text': {
        'clarity':      0.40,
        'completeness': 0.30,
        'correctness':  0.20,
        'efficiency':   0.10,
    },
    'analysis': {
        'correctness':  0.40,
        'completeness': 0.30,
        'clarity':      0.20,
        'efficiency':   0.10,
    },
    'default': {
        'correctness':  0.25,
        'clarity':      0.25,
        'completeness': 0.25,
        'efficiency':   0.25,
    },
}


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
                      If a 'profile' key is present, modality-aware weights
                      are applied to the dimension scores.

        Returns:
            dict with:
              quality_score: float 0.0–1.0 (weighted average of dimensions)
              dimensions:    {'correctness': float, 'clarity': float,
                              'completeness': float, 'efficiency': float}
              method:        'llm_judge' | 'heuristic'
              raw_score:     int 0–10 from judge, or None for heuristic
              confidence:    float 0.0–1.0 indicating evaluator confidence
        """
        if self._judge is not None:
            return self._judge_evaluate(task, response, context or {})
        return self._heuristic_evaluate(task, response, context or {})

    # ── LLM-as-judge ─────────────────────────────────────────────────────────

    def _judge_evaluate(self, task: dict, response: str,
                        context: dict) -> dict:
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
            # Derive approximate dimensions from the single judge score
            dimensions = _uniform_dimensions(quality)
            return {
                'quality_score': round(quality, 4),
                'dimensions': dimensions,
                'method': 'llm_judge',
                'raw_score': score_int,
                'confidence': 0.9,
            }
        except Exception:
            # Judge call failed — fall back to heuristic
            return self._heuristic_evaluate(task, response, context)

    # ── Heuristic fallback ───────────────────────────────────────────────────

    def _heuristic_evaluate(self, task: dict, response: str,
                             context: dict) -> dict:
        """
        Score using length-saturation and keyword-overlap heuristics across
        four quality dimensions with modality-aware weighting.

        Dimensions:
          correctness:  keyword overlap between task and response
          clarity:      structural signals (sentence count, avg length)
          completeness: length-saturation curve
          efficiency:   conciseness ratio (completeness vs. length penalty)

        Weights depend on modality extracted from context['profile']['modality'].
        """
        response = response or ''
        task_desc = task.get('description', '')

        # ── Dimension scores ──────────────────────────────────────────────
        correctness = _keyword_overlap(task_desc, response)
        completeness = _length_score(len(response))
        clarity = _clarity_score(response)
        efficiency = _efficiency_score(len(response), correctness)

        dimensions = {
            'correctness':  round(correctness, 4),
            'clarity':      round(clarity, 4),
            'completeness': round(completeness, 4),
            'efficiency':   round(efficiency, 4),
        }

        # ── Modality-aware weighted average ──────────────────────────────
        modality = (context.get('profile') or {}).get('modality', 'default')
        weights = _DIMENSION_WEIGHTS.get(modality, _DIMENSION_WEIGHTS['default'])

        quality = sum(dimensions[dim] * weights[dim] for dim in dimensions)
        quality = max(0.0, min(1.0, quality))

        return {
            'quality_score': round(quality, 4),
            'dimensions': dimensions,
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


def _clarity_score(response: str) -> float:
    """
    Heuristic clarity score based on sentence structure.

    Signals used:
      - Presence of multiple sentences (structured vs. one-liner)
      - Average sentence length in a plausible range (10–30 words)
      - Presence of code blocks or bullet points (structured output)

    Returns a float 0.0–1.0.
    """
    if not response:
        return 0.0

    # Bonus for structured formatting (code blocks, bullets, numbered lists)
    has_structure = bool(
        re.search(r'```|^\s*[-*]\s|^\s*\d+\.\s', response, re.MULTILINE)
    )

    sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    sentences = [s for s in sentences if s]
    num_sentences = len(sentences)

    if num_sentences == 0:
        return 0.3 + (0.2 if has_structure else 0.0)

    words_per_sentence = [len(s.split()) for s in sentences]
    avg_words = sum(words_per_sentence) / num_sentences

    # Sentence-length score: peak at 15 words, falls off outside 5–40 range
    length_fit = max(0.0, 1.0 - abs(avg_words - 15) / 25)

    # Multi-sentence bonus (up to 3 sentences fully rewarded)
    multi_bonus = min(1.0, (num_sentences - 1) / 2) * 0.2

    base = 0.5 * length_fit + 0.3 * multi_bonus + (0.2 if has_structure else 0.0)
    return round(max(0.0, min(1.0, base)), 4)


def _efficiency_score(char_count: int, correctness: float) -> float:
    """
    Heuristic efficiency score: rewards responses that are correct without
    being excessively verbose.

    A response that is correct but very long gets a lower efficiency score
    than one that is equally correct but more concise.
    """
    if char_count == 0:
        return 0.0
    # Ideal length is _TARGET_LENGTH_CHARS; penalise beyond 3× that
    length_penalty = max(0.0, 1.0 - max(0.0, char_count - _TARGET_LENGTH_CHARS)
                         / (_TARGET_LENGTH_CHARS * 2))
    return round(max(0.0, min(1.0, 0.6 * correctness + 0.4 * length_penalty)), 4)


def _uniform_dimensions(quality: float) -> dict:
    """Return four equal dimensions that average to the given quality score."""
    return {
        'correctness':  round(quality, 4),
        'clarity':      round(quality, 4),
        'completeness': round(quality, 4),
        'efficiency':   round(quality, 4),
    }
