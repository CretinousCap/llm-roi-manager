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

# Use your existing provider keys/subscriptions:
export ANTHROPIC_API_KEY=...
python examples/run_task.py --provider anthropic

export GEMINI_API_KEY=...
python examples/run_task.py --provider gemini

export OPENROUTER_API_KEY=...
python examples/run_task.py --provider openrouter_claude
python examples/run_task.py --provider openrouter_grok
python examples/run_task.py --provider openrouter_gemini

# Local Ollama (Qwen/Phi) via OpenAI-compatible endpoint:
python examples/run_task.py --provider ollama_qwen9b
python examples/run_task.py --provider ollama_phi
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
from execution.anthropic_adapter import AnthropicAdapter
from execution.gemini_adapter import GeminiAdapter
from execution.llm_executor import LLMExecutor

executor = LLMExecutor()
executor.register('gpt4o_mini', OpenAIAdapter())          # uses OPENAI_API_KEY
executor.register('gpt4o',      OpenAIAdapter('gpt-4o'))  # higher capability
executor.register('claude',     AnthropicAdapter())       # uses ANTHROPIC_API_KEY
executor.register('gemini',     GeminiAdapter())          # uses GEMINI_API_KEY

# OpenRouter / Ollama via OpenAI-compatible adapter:
executor.register(
    'grok',
    OpenAIAdapter(
        model='x-ai/grok-3-mini-beta',
        provider='openrouter',
        api_key_env='OPENROUTER_API_KEY',
        base_url='https://openrouter.ai/api/v1',
        model_key='openrouter:x-ai/grok-3-mini-beta',
    ),
)
executor.register(
    'qwen9b_ollama',
    OpenAIAdapter(
        model='qwen2.5-coder:9b',
        provider='ollama',
        api_key='ollama',
        base_url='http://localhost:11434/v1',
        model_key='ollama:qwen2.5-coder:9b',
    ),
)

result = executor.execute(prompt, allocation)
# result: {
#   'text': '...',
#   'model': 'gpt-4o-mini',
#   'provider': 'openai',
#   'usage': {'input_tokens': 120, 'output_tokens': 67},
#   'cost': {'input_usd': 1.8e-05, 'output_usd': 4.02e-05,
#            'tool_usd': 0.0, 'total_usd': 5.82e-05},
#   'llm': 'gpt4o_mini',
# }
```

> `fal.ai`: if your chosen fal.ai endpoint is OpenAI-compatible, wire it with
> `OpenAIAdapter(provider='fal', api_key_env='FAL_KEY', base_url='...')`.
> Otherwise, add a dedicated `LLMAdapter` for that endpoint.

---

## Models Registry and Pricing Engine

All pricing lives in `models/registry.py`. Each model is registered under a
`"provider:model"` key with separate input and output prices.

### Inspect pricing

```python
from models.registry import get_model, list_models
from models.pricing import calculate_cost

# List all registered models
print(list_models())            # ['google:gemini-pro', 'mistral:mixtral', ...]
print(list_models('openai'))    # ['openai:gpt-3.5-turbo', 'openai:gpt-4o', ...]

# Get a model entry
entry = get_model('openai:gpt-4o-mini')
print(entry['input_price_per_1k'])   # 0.00015 (USD per 1k input tokens)
print(entry['output_price_per_1k'])  # 0.0006

# Calculate cost from token usage
cost = calculate_cost(
    model_key='openai:gpt-4o-mini',
    usage={'input_tokens': 800, 'output_tokens': 400},
)
# → {'input_usd': 0.00012, 'output_usd': 0.00024,
#    'tool_usd': 0.0, 'total_usd': 0.00036}
```

### Adding a new model

1. Add a pricing entry to `models/registry.py`:

   ```python
   'myprovider:my-model': {
       'provider': 'myprovider',
       'model': 'my-model',
       'input_price_per_1k': 0.001,
       'output_price_per_1k': 0.002,
       'context_window': 32_768,
       'capabilities': ['chat', 'code'],
       'type': 'text',
       'notes': 'My new model.',
   }
   ```

2. Add a routing entry to `core/registry.py` (set `model_key` to the key above):

   ```python
   'my_model': {
       'score_fn': None,
       'context_affinity': 'neutral',
       'default_weight': 1.0,
       'model_key': 'myprovider:my-model',
   }
   ```

3. Subclass `LLMAdapter` and implement `complete()` for your provider.
4. Register it: `executor.register('my_model', MyAdapter())`
5. Add a `CHANGELOG.md` entry.

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

Current version: **v0.3.0** — models registry, centralised pricing engine,
structured cost output per execution call.

Previous: v0.2.0 — execution layer, OpenAI adapter, LLM-as-judge evaluator, JSONL result store, ROI engine.

See `CHANGELOG.md` for version history and planned work.
