# Agent Instructions — LLM ROI Manager

**For GitHub Copilot, Claude Code, Codex, Gemini, and all AI coding agents.**

This document tells you how to work in this repository: what it does, how it's
organised, what conventions to follow, and how to validate your changes.

---

## What This Repository Does

`llm-roi-manager` optimises which LLM (Claude, Codex, Gemini, GPT-4, etc.)
to use for a given task — code generation, content writing, data analysis, etc.
— based on measured quality, cost, and context, using a multi-agent
context-weighted architecture.

It does **not** do trading. It does not call LLMs directly. It is the
**routing and evaluation layer** that sits above your LLM API clients.

---

## Core Architecture Patterns

### 1 — Multi-Agent with a Combiner/Router

```
Task description
    ↓
Task Classifier  ← scores task on context dimensions (code / content / analysis)
    ↓
LLM_REGISTRY     ← each LLM entry declares its context affinities + weight
    ↓
LLM Router       ← context-weighted blending → single recommendation
    ↓
WrapSession      ← stateful wrapper (tracks quality, cost, iterations)
    ↓
Feedback         ← performance_tracker updates effectiveness history
```

Each LLM agent is a **pure Python module** in `agents/` with a single `score()`
method. Agents do not import each other — only the router does.

### 2 — REGISTRY Pattern

New LLMs are added via `LLM_REGISTRY` in `core/registry.py`, not by modifying
the router. The router iterates the registry and skips absent score functions.

```python
LLM_REGISTRY = {
    'my_new_llm': {
        'score_fn': my_agent.score,      # callable: (task: dict) -> float (-1..+1)
        'context_affinity': 'code_generation',
        'default_weight': 1.0,
        'cost_per_1k_tokens': 0.002,
    },
}
```

### 3 — WrapSession (Stateful Session Management)

`core/session.py` wraps each LLM interaction session. The pattern mirrors
`TrailingProfitLock` in the HybridQuant trading system:

- Tracks peak quality score (like peak price for trailing stop)
- Detects quality degradation across iterations
- Tracks cumulative cost and token usage
- `IterationGuard` kills runaway sessions (like `TimeExit` kills stale trades)
- Cleans up state on `__exit__` — no cross-session leakage

### 4 — Config Injection

Every function accepts an optional `config: dict`. Module-level `*_DEFAULTS`
dicts define the baseline. Override only what you need:

```python
context = classify(task, config={'code_weight': 1.5})
```

---

## Repository Conventions

### Python

- All core modules in `core/`, `classifiers/`, `agents/`, `router/`, `budget/`
  are **pure Python** — no LLM API client imports
- LLM API calls happen in your application code, above this library
- Every module has a docstring: **purpose**, **how it works**, **how to tune**
- Type hints on all public functions
- No external dependencies beyond the Python standard library

### Change Management

1. Logic changes → add a `## [Unreleased]` entry to `CHANGELOG.md` with
   rationale before merging
2. Parameter tweaks happen during scheduled effectiveness reviews (see
   `docs/SOP.md`), not ad-hoc
3. New LLMs go through `LLM_REGISTRY` — never modify router core logic directly
4. All changes must pass the import smoke test before merging

### Versioning

- **MAJOR** — New architecture or fundamental routing change
- **MINOR** — New LLM agent, weight change, or parameter change from review
- **PATCH** — Bug fix, infra change, no logic change

---

## How to Add a New LLM Agent

1. Create `agents/your_llm.py` with a class implementing `score(task: dict) -> float`
2. Add an entry to `LLM_REGISTRY` in `core/registry.py`
3. Add a `CHANGELOG.md` entry with rationale and expected effectiveness
4. Run the smoke test: `python -c "from router.llm_router import route; print('ok')"`

---

## Validation / Smoke Test

```bash
python -c "
from core import session, registry
from classifiers import task_classifier
from agents import performance_tracker
from router import llm_router
from budget import token_allocator
print('All modules importable — OK')
"
```

---

## File Map

```
llm-roi-manager/
│
├── AGENTS.md                   ← This file (AI agent instructions)
├── README.md                   ← Project overview and quick start
├── ARCHITECTURE.md             ← Full system design
├── CHANGELOG.md                ← Version history
│
├── core/
│   ├── __init__.py             # Default configs (*_DEFAULTS)
│   ├── session.py              # WrapSession + IterationGuard (stateful)
│   └── registry.py             # LLM_REGISTRY — plugin registry
│
├── classifiers/
│   ├── __init__.py
│   └── task_classifier.py      # Task context classifier (→ regime_classifier)
│
├── agents/
│   ├── __init__.py
│   └── performance_tracker.py  # Per-LLM effectiveness history tracker
│
├── router/
│   ├── __init__.py
│   └── llm_router.py           # Multi-LLM router (→ signal_combiner)
│
├── budget/
│   ├── __init__.py
│   └── token_allocator.py      # Token budget allocation (→ position_sizer)
│
└── docs/
    └── SOP.md                  # Standard operating procedure for updates
```
