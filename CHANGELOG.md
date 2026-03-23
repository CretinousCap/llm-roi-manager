# Changelog — LLM ROI Manager

All notable changes to the LLM ROI Manager.

Format: [Semantic Versioning](https://semver.org/)
- MAJOR: New architecture or fundamental routing change
- MINOR: New LLM agent, weight change, or parameter change from effectiveness review
- PATCH: Bug fix, infra change, no logic change

---

## [Unreleased]

### Added — Execution, Evaluation, and ROI Loop (v0.2.0)

- `execution/llm_executor.py`: `LLMAdapter` abstract base class + `LLMExecutor`
  dispatcher. Provides a thin adapter layer for wiring concrete LLM API clients
  into the library. Callers subclass `LLMAdapter`, implement `complete()`, then
  register with `LLMExecutor.register()`. No API clients in the library itself.

- `evaluation/evaluator.py`: `Evaluator` — LLM-as-judge quality scorer with
  heuristic fallback. Accepts an optional `judge_adapter` (any `LLMAdapter`);
  prompts the judge to score the response 0–10 and normalises to 0.0–1.0.
  When no judge is provided, falls back to a length-saturation + keyword-overlap
  heuristic suitable for development and testing.

- `storage/result_store.py`: `ResultStore` — thread-safe JSONL result
  persistence. Each execution result is stored as one JSON line. Supports
  `load_all()`, `load_by_llm()`, `load_by_context()`, `iter_records()`, and
  `record_count()` for offline analysis and ROI review.

- `agents/performance_tracker.py`: Extended `PerformanceTracker` into an ROI
  engine:
  - `record()` now accepts `cost_usd` (default 0.0 — backward compatible)
  - History tuples are now `(quality_score, weight, cost_usd)`
  - `roi_score()`: ROI-adjusted effectiveness (-1 to +1). Penalises LLMs whose
    actual average cost exceeds the registry reference `cost_per_1k_tokens`.
    Falls back to `score()` when no cost data is available.
  - `summary()` now includes `total_cost_usd`, `mean_cost_usd`, and
    `cost_per_quality_unit` per (llm, context_type) bucket.

- `router/llm_router.py`: `route()` now accepts `use_roi=True` parameter.
  When True, `_get_agent_score()` prefers `roi_score()` over `score()` when
  the tracker has cost data, enabling cost-aware LLM selection.

- `core/__init__.py`: Added `EXECUTOR_DEFAULTS`, `EVALUATOR_DEFAULTS`, and
  `STORE_DEFAULTS` for the three new modules.

### Architecture Decisions
- Execution layer uses an adapter pattern — zero LLM API imports inside the
  library; all provider-specific code lives in application-layer adapters.
- Evaluator uses LLM-as-judge (same adapter interface) with heuristic fallback
  so the evaluation pipeline works without a judge LLM during development.
- JSONL chosen for the result store: append-only, human-readable, no schema
  migration, trivially parseable by any data tool.
- ROI scoring is additive to existing quality scoring — `use_roi=False` reverts
  the router to the original quality-only effectiveness behaviour.

---

## [0.1.0] — 2026-03

### Added
- Initial multi-agent skeleton ported from HybridQuant trading system patterns
- `core/session.py`: `WrapSession` — stateful session manager with trailing
  quality lock and `IterationGuard` (adapted from `trailing_stop.py` and
  `time_exit.py`)
- `core/registry.py`: `LLM_REGISTRY` — plugin registry for LLM agents
  (adapted from `SIGNAL_REGISTRY` in `signal_combiner.py`)
- `core/__init__.py`: Default configs (`TASK_DEFAULTS`, `SESSION_DEFAULTS`,
  `BUDGET_DEFAULTS`, `TRACKER_DEFAULTS`)
- `classifiers/task_classifier.py`: Task context classifier — scores tasks
  on code/content/analysis dimensions (adapted from `regime_classifier.py`)
- `agents/performance_tracker.py`: Per-LLM effectiveness tracker — maintains
  rolling history of quality scores and costs
- `router/llm_router.py`: Multi-LLM context-weighted router (adapted from
  `signal_combiner.py`); iterates `LLM_REGISTRY`, scales weights by task
  context affinity, outputs recommendation with confidence
- `budget/token_allocator.py`: Token budget allocator — scales budget by
  router confidence (adapted from `position_sizer.py` regime-confidence
  scaling)
- `AGENTS.md`: AI coding agent instructions (Claude Code, Codex, Gemini,
  GitHub Copilot)
- `ARCHITECTURE.md`: Full system design with architecture parallels
- `docs/SOP.md`: Standard operating procedure for adding LLMs and
  effectiveness reviews

### Architecture Decisions
- Pure Python core — no LLM API client imports anywhere in `core/`,
  `classifiers/`, `agents/`, `router/`, or `budget/`; API calls happen
  above this library in the application layer
- Soft context-weighted routing (not hard "LLM-X for task-Y" rules)
- REGISTRY pattern: new LLMs added via config dict, not by modifying router
- `WrapSession` context manager: state is isolated per session, cleaned up
  on exit — no cross-session leakage
