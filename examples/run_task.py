#!/usr/bin/env python3
"""
run_task.py — Minimal end-to-end LLM ROI Manager example
==========================================================
Demonstrates the full pipeline with one real LLM call:

  task → classify → route → allocate → execute → evaluate → store → track

Usage:
  OPENAI_API_KEY=sk-...  python examples/run_task.py
  ANTHROPIC_API_KEY=...  python examples/run_task.py --provider anthropic
  GEMINI_API_KEY=...     python examples/run_task.py --provider gemini
  OPENROUTER_API_KEY=... python examples/run_task.py --provider openrouter_claude
  python examples/run_task.py --provider ollama_qwen9b --dry-run

The script prints each pipeline step with its output so you can see what
each module contributes.  Results are written to results/llm_results.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import uuid

# Allow running from the repo root or from inside examples/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from agents.performance_tracker import PerformanceTracker
from budget.token_allocator import allocate
from classifiers.task_classifier import classify
from evaluation.evaluator import Evaluator
from execution.llm_executor import LLMExecutor
from router.llm_router import route
from storage.result_store import ResultStore

# ── Task under test ───────────────────────────────────────────────────────────
TASK = {
    'description': (
        'Write a Python function that returns the nth Fibonacci number '
        'using recursion with memoization. Include a brief docstring.'
    ),
}

# ── Billing context ───────────────────────────────────────────────────────────
BILLING_METADATA = {
    'account_id':   'acc_personal',
    'account_name': 'personal',
    'project_id':   'proj_llm-roi-manager',
    'project_name': 'llm-roi-manager',
    'environment':  'dev',
}

# ── LLM wiring presets for this example ───────────────────────────────────────
_PROVIDER_PRESETS = {
    'openai': {
        'llm': 'gpt4o_mini',
        'model': 'gpt-4o-mini',
    },
    'anthropic': {
        'llm': 'claude',
        'model': 'claude-3-5-sonnet-latest',
    },
    'gemini': {
        'llm': 'gemini',
        'model': 'gemini-pro',
    },
    'openrouter_claude': {
        'llm': 'claude',
        'model': 'anthropic/claude-3.5-sonnet',
    },
    'openrouter_grok': {
        'llm': 'grok',
        'model': 'x-ai/grok-3-mini-beta',
    },
    'openrouter_gemini': {
        'llm': 'gemini',
        'model': 'google/gemini-2.0-flash-001',
    },
    'ollama_qwen9b': {
        'llm': 'qwen9b_ollama',
        'model': 'qwen2.5-coder:9b',
    },
    'ollama_phi': {
        'llm': 'phi_ollama',
        'model': 'phi4',
    },
}


def _build_executor(provider: str) -> tuple[LLMExecutor, str, str]:
    """Create and register one adapter based on a provider preset."""
    executor = LLMExecutor()
    preset = _PROVIDER_PRESETS[provider]
    llm_name = preset['llm']
    model = preset['model']

    if provider == 'anthropic':
        from execution.anthropic_adapter import AnthropicAdapter
        executor.register(llm_name, AnthropicAdapter(model=model))
    elif provider == 'gemini':
        from execution.gemini_adapter import GeminiAdapter
        executor.register(llm_name, GeminiAdapter(model=model))
    elif provider.startswith('openrouter_'):
        from execution.openai_adapter import OpenAIAdapter
        executor.register(
            llm_name,
            OpenAIAdapter(
                model=model,
                provider='openrouter',
                api_key_env='OPENROUTER_API_KEY',
                base_url='https://openrouter.ai/api/v1',
                model_key=f'openrouter:{model}',
            ),
        )
    elif provider.startswith('ollama_'):
        from execution.openai_adapter import OpenAIAdapter
        executor.register(
            llm_name,
            OpenAIAdapter(
                model=model,
                provider='ollama',
                api_key='ollama',
                base_url='http://localhost:11434/v1',
                model_key=f'ollama:{model}',
            ),
        )
    else:
        from execution.openai_adapter import OpenAIAdapter
        executor.register(llm_name, OpenAIAdapter(model=model))

    return executor, llm_name, model


def run(dry_run: bool = False, provider: str = 'openai') -> dict:
    """
    Execute the full pipeline and return the final record dict.

    Args:
        dry_run: When True, skip the real API call and use a placeholder
                 response so the rest of the pipeline can be exercised
                 without an API key.
    """
    # ── 0. Generate run identifiers ───────────────────────────────────────────
    task_id = str(uuid.uuid4())
    prompt_hash = hashlib.md5(TASK['description'].encode()).hexdigest()

    # ── 1. Classify the task ──────────────────────────────────────────────────
    context = classify(TASK)
    profile = context['profile']
    print(
        f'[1] Classify  dominant={context["dominant"]!r}  '
        f'confidence={context["confidence"]}'
    )
    print(
        f'    profile   modality={profile["modality"]!r}  '
        f'task_type={profile["task_type"]!r}  '
        f'complexity={profile["complexity"]!r}  '
        f'interaction_stage={profile["interaction_stage"]!r}  '
        f'structure={profile["structure"]!r}  '
        f'latency_sensitivity={profile["latency_sensitivity"]!r}  '
        f'risk={profile["risk"]!r}'
    )

    # ── 2. Route to best LLM (ROI-aware) ─────────────────────────────────────
    tracker = PerformanceTracker()
    recommendation = route(context, tracker=tracker, use_roi=True)
    preferred = recommendation['preferred_llm']
    print(
        f'[2] Route     preferred={preferred!r}  '
        f'confidence={recommendation["confidence"]}'
    )

    # ── 3. Allocate token budget ──────────────────────────────────────────────
    allocation = allocate(context, recommendation)
    print(
        f'[3] Allocate  token_budget={allocation["token_budget"]}  '
        f'estimated_cost=${allocation["estimated_cost_usd"]:.6f}'
    )

    # ── 4. Execute ────────────────────────────────────────────────────────────
    configured_model = _PROVIDER_PRESETS[provider]['model']
    configured_llm = _PROVIDER_PRESETS[provider]['llm']
    if dry_run:
        # Bypass real API call — use a canned response for testing
        from execution.llm_executor import _normalise_billing
        exec_result = {
            'status': 'success',
            'text': (
                'def fib(n, memo={}):\n'
                '    """Return the nth Fibonacci number with memoization."""\n'
                '    if n in memo:\n'
                '        return memo[n]\n'
                '    if n <= 1:\n'
                '        return n\n'
                '    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n'
                '    return memo[n]\n'
            ),
            'model': configured_model,
            'provider': provider,
            'mode': 'chat',
            'latency_ms': 0.0,
            'usage': {'input_tokens': 0, 'output_tokens': 0},
            'cost': {'input_usd': 0.0, 'output_usd': 0.0,
                     'tool_usd': 0.0, 'total_usd': 0.0},
            'error': None,
            'billing': _normalise_billing(BILLING_METADATA),
            'llm': configured_llm,
            'skipped': False,
            'tokens_used': 0,
            'cost_usd': 0.0,
        }
        print(f'[4] Execute   [DRY RUN — no API call made]')
    else:
        executor, configured_llm, _ = _build_executor(provider)
        llm_to_use = preferred if preferred in executor.available_llms() \
            else configured_llm
        if llm_to_use != preferred:
            print(
                f'[4] Execute   WARNING: router preferred {preferred!r} but no '
                f'adapter is registered for it. Falling back to {llm_to_use!r}. '
                f'Register an adapter with executor.register({preferred!r}, ...) '
                f'to use the router\'s recommendation directly.'
            )
        exec_result = executor.execute(
            TASK['description'],
            allocation,
            llm_override=llm_to_use,
            metadata=BILLING_METADATA,
        )
        cost = exec_result['cost']
        usage = exec_result['usage']
        print(
            f'[4] Execute   status={exec_result["status"]!r}  '
            f'llm={exec_result["llm"]!r}  '
            f'model={exec_result["model"]!r}  '
            f'provider={exec_result["provider"]!r}'
        )
        print(
            f'    latency   {exec_result["latency_ms"]:.1f} ms'
        )
        print(
            f'    tokens    input={usage["input_tokens"]}  '
            f'output={usage["output_tokens"]}'
        )
        print(
            f'    cost      input=${cost["input_usd"]:.8f}  '
            f'output=${cost["output_usd"]:.8f}  '
            f'total=${cost["total_usd"]:.8f}'
        )

    # Print a preview of the response
    response_text = exec_result['text']
    preview = response_text[:200].replace('\n', '\\n')
    print(f'    response preview: {preview!r}')
    response_chars = len(response_text)
    response_tokens = exec_result['usage']['output_tokens']

    # ── 5. Evaluate response quality ──────────────────────────────────────────
    # Using heuristic evaluator here. In production, pass a judge_adapter:
    #   evaluator = Evaluator(judge_adapter=OpenAIAdapter(model='gpt-4o'))
    evaluator = Evaluator()
    eval_result = evaluator.evaluate(TASK, response_text, context)
    dims = eval_result['dimensions']
    print(
        f'[5] Evaluate  quality_score={eval_result["quality_score"]}  '
        f'method={eval_result["method"]!r}  '
        f'confidence={eval_result["confidence"]}'
    )
    print(
        f'    dimensions  correctness={dims["correctness"]}  '
        f'clarity={dims["clarity"]}  '
        f'completeness={dims["completeness"]}  '
        f'efficiency={dims["efficiency"]}'
    )

    # ── 6. Store result ───────────────────────────────────────────────────────
    store = ResultStore()   # writes to results/llm_results.jsonl by default
    cost = exec_result['cost']
    usage = exec_result['usage']
    record = {
        'task_id': task_id,
        'prompt_hash': prompt_hash,
        'llm': exec_result['llm'],
        'model': exec_result['model'],
        'provider': exec_result['provider'],
        'status': exec_result['status'],
        'mode': exec_result['mode'],
        'latency_ms': exec_result['latency_ms'],
        'error': exec_result.get('error'),
        'task_description': TASK['description'],
        'context_dominant': context['dominant'],
        'context_confidence': context['confidence'],
        'task_profile': context['profile'],
        'router_preferred': preferred,
        'router_confidence': recommendation['confidence'],
        'token_budget': allocation['token_budget'],
        'input_tokens': usage['input_tokens'],
        'output_tokens': usage['output_tokens'],
        'response_chars': response_chars,
        'response_tokens': response_tokens,
        'cost_input_usd': cost['input_usd'],
        'cost_output_usd': cost['output_usd'],
        'cost_tool_usd': cost['tool_usd'],
        'cost_total_usd': cost['total_usd'],
        'quality_score': eval_result['quality_score'],
        'eval_dimensions': dims,
        'eval_method': eval_result['method'],
        'billing': exec_result['billing'],
        'dry_run': dry_run,
    }
    store.append(record)
    billing = exec_result['billing']
    print(
        f'[6] Store     path={store.path}  total_records={store.record_count()}'
    )
    print(
        f'    billing   account_id={billing["account_id"]!r}  '
        f'account_name={billing["account_name"]!r}  '
        f'project_id={billing["project_id"]!r}  '
        f'project_name={billing["project_name"]!r}  '
        f'environment={billing["environment"]!r}'
    )

    # ── 7. Update performance tracker ────────────────────────────────────────
    tracker.record(
        llm=exec_result['llm'],
        context_type=context['dominant'],
        quality_score=eval_result['quality_score'],
        cost_usd=exec_result['cost']['total_usd'],
        status=exec_result.get('status', 'success'),
        latency_ms=exec_result.get('latency_ms', 0.0),
    )
    summary = tracker.summary(exec_result['llm'])
    ctx = context['dominant']
    if ctx in summary:
        s = summary[ctx]
        print(
            f'[7] Track     observations={s["observations"]}  '
            f'mean_quality={s["mean_quality"]}  '
            f'cost_per_quality_unit={s["cost_per_quality_unit"]}'
        )

    print('\n✓ End-to-end pipeline complete.')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run the LLM ROI Manager end-to-end pipeline.'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Skip the real OpenAI API call and use a canned response.'
    )
    parser.add_argument(
        '--provider',
        default='openai',
        choices=sorted(_PROVIDER_PRESETS),
        help='Provider preset to run (openai, anthropic, gemini, openrouter_*, ollama_*).',
    )
    args = parser.parse_args()

    if not args.dry_run:
        required_env = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'openrouter_claude': 'OPENROUTER_API_KEY',
            'openrouter_grok': 'OPENROUTER_API_KEY',
            'openrouter_gemini': 'OPENROUTER_API_KEY',
            'ollama_qwen9b': '',
            'ollama_phi': '',
        }[args.provider]
        if required_env and not os.environ.get(required_env):
            print(
                f'Error: {required_env} is not set.\n'
                f'Set it with:  export {required_env}=...\n'
                'Or run with:  python examples/run_task.py --dry-run'
            )
            sys.exit(1)

    run(dry_run=args.dry_run, provider=args.provider)


if __name__ == '__main__':
    main()
