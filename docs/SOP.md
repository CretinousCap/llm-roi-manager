# LLM ROI Manager — Standard Operating Procedure

Version 1.0 — 2026-03

---

## 1. Overview

This SOP governs development, testing, and deployment of the LLM ROI Manager.
It applies to all code changes, configuration updates, and effectiveness review
cycles.

**Design Principles:**

- **Module independence:** Each core module (classifier, router, tracker, budget)
  has zero dependency on LLM API clients. Modules can be tested standalone.
- **Router as plumbing:** The router does not contain LLM API calls. It receives
  scores and weights them. API calls happen in the application layer above.
- **AI-assisted development:** GitHub Copilot, Claude Code, Codex, and Gemini
  all work on this repository. AI-generated code follows the same review
  pipeline as human-written code. See `AGENTS.md` for AI agent instructions.

---

## 2. Change Process

### 2.1 Adding a New LLM Agent

1. Create `agents/{llm_name}.py` — implement `score(task: dict) -> float`
2. Add to `LLM_REGISTRY` in `core/registry.py`
3. Add `CHANGELOG.md` entry under `## [Unreleased]`
4. Run smoke test:
   ```bash
   python -c "from router.llm_router import route; print('ok')"
   ```

### 2.2 Adjusting LLM Weights (Effectiveness Review)

Weights may only change during scheduled effectiveness reviews:

1. Pull metrics from `PerformanceTracker.summary()` for each LLM
2. Compare to previous review baseline
3. Identify underperforming (llm, context_type) pairs
4. Adjust `default_weight` in `LLM_REGISTRY` — **max ±20% per review**
5. Document rationale in `CHANGELOG.md` under `## [Unreleased]`
6. **Never** change `score_fn` logic during a review — that requires a full
   evaluation cycle (see Section 2.3)

### 2.3 Changing Core Module Logic

Changes to `task_classifier.py`, `llm_router.py`, `session.py`, or
`token_allocator.py` are **Major or Minor** version changes:

1. Document the problem being solved in a GitHub Issue
2. Implement change on a feature branch
3. Manually test with at least 10 representative tasks per context type
4. Evaluate: does the change improve routing accuracy? Document evidence.
5. Update `CHANGELOG.md` with full rationale
6. PR review required before merge

---

## 3. Versioning Policy

Follows Semantic Versioning (see `CHANGELOG.md`):

- **MAJOR** — Architecture change or fundamental routing change
- **MINOR** — New LLM agent, weight change from effectiveness review
- **PATCH** — Bug fix, infra change, no routing logic change

Every PR must include a `CHANGELOG.md` entry for non-trivial changes.

---

## 4. Effectiveness Review Cadence

Conduct a review every **4-6 weeks** or after 100+ recorded sessions:

1. Export `PerformanceTracker.summary()` for all LLMs
2. Calculate cost-per-quality-unit per LLM per context type
3. Identify: which LLM is overweighted vs its actual effectiveness?
4. Adjust `default_weight` values (max ±20%, documented)

---

## 5. Security

- LLM API keys are **never** stored in this repository
- API keys are injected via environment variables in the application layer
- `core/`, `classifiers/`, `agents/`, `router/`, `budget/` contain no secrets
- `.gitignore` excludes `.env`, `*.key`, and credential files
