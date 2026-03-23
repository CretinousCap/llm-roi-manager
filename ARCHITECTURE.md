# LLM ROI Manager — Architecture Guide

**For humans and AI coding agents.** This explains how the system works, how to
modify it, and how to avoid breaking it.

---

## The 30-Second Version

Multiple LLM agents independently score each task. A task classifier determines
the context (code, content, analysis) and weights each LLM accordingly. A router
combines the weighted scores into a single recommendation. A stateful session
wrapper tracks quality and cost across the interaction and stops runaway
iterations.

No hard "LLM-X does code, LLM-Y does writing" rules. Soft, continuous weighting
that adapts as effectiveness data accumulates.

---

## How Data Flows

```
Task Description
    │
    ▼
┌─────────────────────────┐
│  Task Classifier        │  ← "Is this code, content, or analysis?"
│  (task_classifier.py)   │     Scores each dimension: -1 to +1
│                         │     Output: context_scores dict
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐   ┌──────────┐
│ LLM-A    │   │ LLM-B    │  ← Independent agents (one per LLM)
│ Agent    │   │ Agent    │     Each scores the task: -1 to +1
│          │   │          │     Positive = good fit, Negative = poor fit
└────┬─────┘   └────┬─────┘
     │               │
     ▼               ▼
┌─────────────────────────┐
│  LLM Router             │  ← Context-weighted blending
│  (llm_router.py)        │     Code task → more weight to code-specialist LLMs
│                         │     Content task → more weight to writing LLMs
│                         │     Output: recommendation dict
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐   ┌──────────┐
│  Token   │   │  Wrap    │  ← Token budget scales with confidence
│ Budget   │   │ Session  │     Session tracks quality, cost, peak score
│          │   │          │     IterationGuard kills runaway loops
└──────────┘   └──────────┘
```

---

## File Map

```
llm-roi-manager/
│
├── core/
│   ├── __init__.py             # TASK_DEFAULTS, SESSION_DEFAULTS, BUDGET_DEFAULTS
│   ├── session.py              # WrapSession + IterationGuard
│   └── registry.py             # LLM_REGISTRY
│
├── classifiers/
│   └── task_classifier.py      # classify(task) → context_scores
│
├── agents/
│   └── performance_tracker.py  # PerformanceTracker — per-LLM history
│
├── router/
│   └── llm_router.py           # route(context) → recommendation
│
└── budget/
    └── token_allocator.py      # allocate(context, recommendation) → token_budget
```

---

## The Task Classifier (Adapted from Regime Classifier)

**Problem:** "Write a function" and "Write a blog post" look similar as strings.
Fixed keyword rules break as soon as task phrasing varies.

**Solution:** Score each task across multiple context dimensions using keyword
density. Output is a continuous score (-1 to +1) per dimension, not a binary
category.

```
context_score['code_generation']   =  +0.8  (strongly code-like)
context_score['content_writing']   =  -0.4  (not writing)
context_score['data_analysis']     =  -0.2  (not analysis)
```

These scores feed the router's weighting, exactly as `regime_score` feeds the
signal combiner in HybridQuant.

---

## How LLM Weighting Works

The router does **soft weighting** — not hard switching.

```
context_score['code_generation'] = +0.7

  → Codex weight   = 1.0 * 0.7 = 0.70  (affinity = 'code_generation')
  → Claude weight  = 1.0 * 0.5 = 0.50  (affinity = 'neutral')
  → Gemini weight  = 1.0 * 0.0 = 0.00  (affinity = 'content_writing', no match)

  → Combined score dominated by Codex

context_score['code_generation'] = +0.1  (ambiguous)

  → All weights are low → router returns low confidence
  → token_allocator issues a reduced budget (Kelly-lite: less bet when edge is unclear)
  → WrapSession uses IterationGuard with a tight limit
```

---

## WrapSession — Stateful Session Management

**Analogous to `TrailingProfitLock` in HybridQuant.**

- **Trailing quality lock**: Once quality exceeds `trail_activation_quality`,
  the trail activates. If subsequent quality drops below
  `peak * (1 - trail_offset)`, the session flags `quality_degraded = True`.
- **IterationGuard**: After `max_iterations` calls, requires quality above
  `min_continuation_quality` to continue. Targets grinding sessions that
  aren't converging.

---

## Adding a New LLM

1. Create `agents/your_llm.py`
2. Implement `score(task: dict) -> float` (returns -1 to +1)
3. Add to `LLM_REGISTRY` in `core/registry.py`
4. The router picks it up automatically — no changes to router core
5. Update `CHANGELOG.md`

**Planned LLM agents:**
- `agents/claude.py` — strong on reasoning, code review, long context
- `agents/codex.py` — strong on code generation and completion
- `agents/gemini.py` — strong on multimodal, content, summarisation
- `agents/gpt4.py` — strong on general reasoning, instruction following

---

## Effectiveness Review Cadence

Analogous to the quarterly review in HybridQuant's SOP.

1. Pull metrics: quality scores, cost-per-task, task types via `PerformanceTracker.summary()`
2. Compare to previous review baseline
3. Identify: which LLM underperformed? For which task type?
4. Adjust `default_weight` in `LLM_REGISTRY` (max ±20% change, documented)
5. **Never** change `score_fn` logic during a review — that requires a full
   evaluation cycle (see `docs/SOP.md`)

---

## Architecture Parallels to HybridQuant Trading System

| Trading System | LLM ROI Manager |
|----------------|-----------------|
| `regime_classifier.py` | `classifiers/task_classifier.py` |
| `SIGNAL_REGISTRY` | `core/registry.py` — `LLM_REGISTRY` |
| `signal_combiner.py` | `router/llm_router.py` |
| `position_sizer.py` | `budget/token_allocator.py` |
| `trailing_stop.py` — `TrailingProfitLock` | `core/session.py` — `WrapSession` |
| `time_exit.py` — `TimeExit` | `core/session.py` — `IterationGuard` |
| `CHANGELOG` + quarterly review | `CHANGELOG.md` + effectiveness review cadence |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1.0 | 2026-03 | Initial multi-agent skeleton ported from HybridQuant patterns |
