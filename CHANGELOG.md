# Changelog — LLM ROI Manager

All notable changes to the LLM ROI Manager.

Format: [Semantic Versioning](https://semver.org/)
- MAJOR: New architecture or fundamental routing change
- MINOR: New LLM agent, weight change, or parameter change from effectiveness review
- PATCH: Bug fix, infra change, no logic change

---

## [Unreleased]

### Added — Multi-Provider Adapter Onboarding

- `execution/anthropic_adapter.py`: Added an Anthropic Messages API adapter
  (`AnthropicAdapter`) for direct Claude API usage via `ANTHROPIC_API_KEY`.
- `execution/gemini_adapter.py`: Added a Gemini REST adapter
  (`GeminiAdapter`) for direct Gemini API usage via `GEMINI_API_KEY`.
- `examples/run_task.py`: Added `--provider` presets for OpenAI, Anthropic,
  Gemini, OpenRouter-backed Claude/Grok/Gemini, and local Ollama
  (`qwen2.5-coder:9b`, `phi4`).
- `models/registry.py`: Added model entries for Anthropic Claude, OpenRouter
  presets, local Ollama models, and a fal.ai placeholder model key.
- `core/registry.py`: Added routing entries for Grok and local Ollama models
  and linked Claude to its Anthropic model key.
- `execution/openai_adapter.py`: Generalised adapter configuration so it can
  target OpenAI-compatible gateways by overriding provider, key env var, base
  URL, and model key.
- `README.md`: Documented how to run the example with ChatGPT/OpenAI, Claude,
  Gemini, Grok via OpenRouter, and local Ollama.

### Added — Execution Observability and Billing Context (v0.4.0)

**Execution observability (requirements 1–7):**

- `execution/llm_executor.py`: Standardised execution result format. All
  calls now return `status` ('success' | 'error' | 'unavailable'), `mode`,
  `latency_ms`, `error` (`{'type', 'message'}` or `None`). Backward-compat
  aliases `tokens_used`, `cost_usd`, and `skipped` are retained.
  `_classify_error()` classifies adapter exceptions into 'quota',
  'rate_limit', 'timeout', 'auth', or 'unknown' via safe string matching
  (no SDK-specific type imports). `_normalise_billing()` enforces billing
  defaults. Missing adapters now return `status='unavailable'` with
  `error.type='adapter_missing'` — no silent fallback.

- `execution/openai_adapter.py`: Added `import time`; `latency_ms` is now
  measured around the API call and returned in the result dict. `mode='chat'`
  is also returned to satisfy the standardised contract.

- `agents/performance_tracker.py`: `record()` gains `status` and
  `latency_ms` parameters. Records with `status != 'success'` increment
  a per-LLM-per-context failure counter and are excluded from quality/cost
  history. `_failures` and `_latency_ms` tracking dicts added. `summary()`
  now exposes `failure_count` and `mean_latency_ms` for every context bucket.
  `reset()` clears all three tracking structures.

**Billing context (requirement 8):**

- `execution/llm_executor.py`: `execute()` accepts an optional `metadata`
  dict (`account`, `project`, `environment`). Missing keys default to
  `account='unknown'`, `project='default'`, `environment='dev'`. The
  normalised `billing` dict is included in every execution result.

- `storage/result_store.py`: Added `load_by_billing_account(account)` and
  `load_by_billing_project(project)` helper methods for simple in-memory
  filtering of persisted records by billing context.

- `examples/run_task.py`: Defines `BILLING_METADATA` (`personal /
  llm-roi-manager / dev`). Passes it to `executor.execute()`. Prints billing
  context at step 6. Stored record now includes `status`, `mode`,
  `latency_ms`, `error`, and `billing`. Dry-run stub updated to include all
  new standardised fields. `tracker.record()` call updated with `status` and
  `latency_ms` arguments.

**Architecture decisions:**
- Billing context is tracking-only; it does not influence routing or adapter behaviour.
- API keys are never stored in billing context or results.
- Error classification uses string matching on `type(exc).__name__` and
  `str(exc)` — not on SDK exception types — so it is provider-agnostic.
- Latency is measured by the adapter (closest to the network call) and falls
  back to executor-level wall-clock time when the adapter omits it.

---

### Added — Models Registry and Pricing Engine (v0.3.0)

- `models/registry.py`: Central `LLM_REGISTRY` keyed by `"provider:model"`.
  Each entry declares `input_price_per_1k`, `output_price_per_1k`,
  `context_window`, `capabilities`, and `type`. Seeded with:
  `openai:gpt-4o-mini`, `openai:gpt-4o`, `openai:gpt-3.5-turbo`,
  `xai:grok-4.20-reasoning`, `google:gemini-pro`, `mistral:mixtral`.
  Helper functions: `get_model()`, `list_models()`.

- `models/pricing.py`: `calculate_cost(model_key, usage, tools=None) → dict`.
  Reads pricing exclusively from `models/registry.py`. Returns a fully
  itemised cost breakdown: `input_usd`, `output_usd`, `tool_usd`, `total_usd`.
  Raises `KeyError` for unknown models. Deterministic and side-effect free.

- `models/__init__.py`: Exports `LLM_REGISTRY` and `calculate_cost`.

### Changed — v0.3.0

- `execution/openai_adapter.py`: Removed `_COST_PER_1K_TOKENS` hardcoded dict.
  Now calls `calculate_cost()` from `models/pricing.py` using per-call
  `prompt_tokens` and `completion_tokens` from the OpenAI API response.
  Returns structured `usage` dict (`input_tokens`, `output_tokens`) and
  `cost` dict (`input_usd`, `output_usd`, `tool_usd`, `total_usd`) alongside
  `text`, `model`, and `provider`. Provider identity is now explicit.

- `execution/llm_executor.py`: Updated `LLMAdapter.complete()` contract and
  `LLMExecutor.execute()` return signature to match the new structured format.
  Both `usage` and `cost` dicts are passed through from adapters intact.

- `core/registry.py`: Replaced `cost_per_1k_tokens` with `model_key` in each
  routing entry. `model_key` is a `"provider:model"` reference into
  `models/registry.py`. This decouples routing configuration from pricing.

- `budget/token_allocator.py`: Cost estimation now reads from `models/registry.py`
  via the routing entry's `model_key`. Pre-call estimate uses the average of
  `input_price_per_1k` and `output_price_per_1k` (token split unknown at
  allocation time).

- `agents/performance_tracker.py`: `roi_score()` now reads reference cost from
  `models/registry.py` (via routing entry's `model_key`) instead of from
  `core/registry.py`. Blended cost reference is consistent with allocator.

- `examples/run_task.py`: Updated dry-run payload and print statements to
  reflect the new structured `usage` and `cost` dicts. Cost breakdown is now
  shown per component (input, output, total). Record stored to JSONL includes
  `model`, `provider`, `input_tokens`, `output_tokens`, and itemised cost fields.

### Architecture Decisions — v0.3.0

- **Single source of truth for pricing**: All USD figures live in
  `models/registry.py`. Adapters call `calculate_cost()`; they do not define
  costs.
- **Separate input/output prices**: The registry records input and output prices
  separately (matching how providers bill). Pre-call estimates blend them;
  post-call actuals use the exact split from the API response.
- **Explicit provider identity**: Every execution result now carries `model`
  and `provider` fields so downstream consumers (evaluator, store, tracker)
  can route or filter by provider without parsing strings.
- **Graceful degradation**: LLMs without a `model_key` (e.g. placeholder
  entries) produce `0.0` cost estimates instead of raising errors, keeping
  the pipeline runnable during incremental provider onboarding.

---

## [Unreleased]

### Added — Execution, Evaluation, and ROI Loop (v0.2.0)

- `execution/llm_executor.py`: `LLMAdapter` abstract base class + `LLMExecutor`
  dispatcher. Provides a thin adapter layer for wiring concrete LLM API clients
  into the library. Callers subclass `LLMAdapter`, implement `complete()`, then
  register with `LLMExecutor.register()`. No API clients in the library itself.

- `execution/openai_adapter.py`: `OpenAIAdapter` — concrete `LLMAdapter` for
  OpenAI Chat Completions. Reads `OPENAI_API_KEY` from the environment (never
  stored in source). Supports `gpt-4o-mini` (default), `gpt-4o`, and
  `gpt-3.5-turbo`. Cost is estimated from a per-model table and returned
  alongside `text` and `tokens_used`. The OpenAI client is initialised lazily
  so importing the module requires no key.

- `evaluation/evaluator.py`: `Evaluator` — LLM-as-judge quality scorer with
  heuristic fallback. Accepts an optional `judge_adapter` (any `LLMAdapter`);
  prompts the judge to score the response 0–10 and normalises to 0.0–1.0.
  When no judge is provided, falls back to a length-saturation + keyword-overlap
  heuristic suitable for development and testing.

- `storage/result_store.py`: `ResultStore` — thread-safe JSONL result
  persistence. Each execution result is stored as one JSON line. Supports
  `load_all()`, `load_by_llm()`, `load_by_context()`, `iter_records()`, and
  `record_count()` for offline analysis and ROI review.

- `examples/run_task.py`: End-to-end pipeline example. Runs the full loop —
  classify → route → allocate → execute → evaluate → store → track — with
  a single `gpt-4o-mini` call. Supports `--dry-run` to exercise the pipeline
  without an API key.

- `requirements.txt`: Declares `openai>=1.0.0` as the only non-stdlib
  dependency (required by `OpenAIAdapter`).

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

- `core/registry.py`: Added `gpt4o_mini` entry (affinity: `neutral`,
  cost: $0.0003/1k tokens) so the router can recommend and track it.

### Architecture Decisions
- Execution layer uses an adapter pattern — zero LLM API imports inside the
  library; all provider-specific code lives in application-layer adapters.
- `OpenAIAdapter` initialises the `openai.OpenAI` client lazily, so the module
  can be imported and tested without setting `OPENAI_API_KEY`.
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
