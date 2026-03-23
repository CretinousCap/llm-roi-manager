# LLM ROI Manager

Optimises which LLM (Claude, Codex, Gemini, GPT-4, etc.) to use for each task,
based on measured effectiveness, cost, and task context — using the same
multi-agent, regime-weighted architecture as the HybridQuant trading system.

---

## The 30-Second Version

Multiple LLM agents compete for each task. A task classifier determines the
context (code generation, content writing, data analysis, etc.) and weights
each LLM accordingly. The router blends scores, picks the best option, and
tracks effectiveness over time so the system self-improves.

No hard "use LLM-X for task-Y" rules. Continuous, context-weighted routing.

---

## Quick Start

### Installation

```bash
pip install openai          # only external dependency (for OpenAIAdapter)
# or: pip install -r requirements.txt
```

### End-to-End Example (one real LLM call)

```bash
export OPENAI_API_KEY=sk-...
python examples/run_task.py

# Try the pipeline without an API key:
python examples/run_task.py --dry-run
```

Output:
```
[1] Classify  dominant='code_generation'  confidence=0.5
[2] Route     preferred='gpt4o_mini'  confidence=0.5
[3] Allocate  token_budget=1400  estimated_cost=$0.000420
[4] Execute   llm='gpt4o_mini'  tokens=187  cost=$0.000056
    response preview: 'def fib(n, memo={}):\n    ...'
[5] Evaluate  quality_score=0.74  method='heuristic'  confidence=0.4
[6] Store     path=results/llm_results.jsonl  total_records=1
[7] Track     observations=1  mean_quality=0.74  cost_per_quality_unit=7.6e-05
```

### Library Usage

```python
from core.session import WrapSession
from classifiers.task_classifier import classify
from router.llm_router import route
from budget.token_allocator import allocate

# Describe the task
task = {
    'description': 'Write a Python function to parse JSON from LLM responses',
    'type_hint': 'code_generation',   # optional — classifier will infer if omitted
    'budget_tokens': 2000,
}

# Classify the task context
context = classify(task)

# Route to the best LLM
recommendation = route(context)
print(recommendation['preferred_llm'])    # e.g. 'codex'
print(recommendation['confidence'])       # 0.0 – 1.0

# Allocate token budget (scales with confidence)
budget = allocate(context, recommendation)
print(budget['token_budget'])

# Wrap the session for state tracking
with WrapSession(task=task, context=context) as session:
    # ... call the recommended LLM here ...
    session.record(
        llm='codex',
        quality_score=0.85,    # 0.0 – 1.0 (your evaluation)
        tokens_used=1420,
        cost_usd=0.014,
    )
    if session.quality_degraded:
        pass  # re-route or stop
    summary = session.summary()
```

---

## Wiring a Real LLM

```python
from execution.openai_adapter import OpenAIAdapter
from execution.llm_executor import LLMExecutor

executor = LLMExecutor()
executor.register('gpt4o_mini', OpenAIAdapter())          # uses OPENAI_API_KEY
executor.register('gpt4o',      OpenAIAdapter('gpt-4o'))  # higher capability

result = executor.execute(prompt, allocation)
# result: {'text': '...', 'tokens_used': 187, 'cost_usd': 0.000056, 'llm': 'gpt4o_mini'}
```

---

## Architecture Parallels

This project directly ports the patterns from the HybridQuant trading system:

| Trading System | LLM ROI Manager |
|----------------|-----------------|
| Regime Classifier (z-scored ADX) | Task Classifier (context scoring) |
| Signal modules (Donchian, MR) | LLM agents (each scores a task) |
| Signal Combiner + SIGNAL_REGISTRY | LLM Router + LLM_REGISTRY |
| Position Sizer (ATR + regime) | Token Allocator (confidence-scaled budget) |
| TrailingProfitLock (stateful) | WrapSession (stateful, tracks peak quality) |
| TimeExit (stale trade killer) | IterationGuard (runaway iteration killer) |
| CHANGELOG + quarterly review | CHANGELOG + effectiveness review cadence |

See `ARCHITECTURE.md` for the full design.

---

## Project Status

Current version: **v0.2.0** — execution layer, OpenAI adapter, LLM-as-judge evaluator, JSONL result store, ROI engine.

See `CHANGELOG.md` for version history and planned work.
