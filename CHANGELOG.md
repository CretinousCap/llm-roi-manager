# Changelog — LLM ROI Manager

All notable changes to the LLM ROI Manager.

Format: [Semantic Versioning](https://semver.org/)
- MAJOR: New architecture or fundamental routing change
- MINOR: New LLM agent, weight change, or parameter change from effectiveness review
- PATCH: Bug fix, infra change, no logic change

---

## [Unreleased]

_(Add entries here before merging any logic changes)_

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
