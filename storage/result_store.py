"""
Result Store — Persistent JSONL Result Storage
===============================================
PURPOSE:
  Persist LLM execution results (task, response, quality score, cost, routing
  decision) to a JSONL (newline-delimited JSON) file for offline analysis and
  ROI review.

HOW IT WORKS:
  - Each result is a JSON object written as one line (JSONL format).
  - append() opens, writes, and closes the file atomically per record.
  - load_all() reads and parses all records from the file.
  - Thread-safe: file writes are serialised with a per-instance lock.

USAGE:
  store = ResultStore('/path/to/results.jsonl')
  store.append({
      'llm': 'claude',
      'context_dominant': 'code_generation',
      'quality_score': 0.85,
      'cost_usd': 0.012,
      'tokens_used': 800,
  })
  records = store.load_all()
  by_llm = store.load_by_llm('claude')
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class ResultStore:
    """
    Thread-safe JSONL result store.

    One ResultStore per file path. Shared use of a single instance across
    threads is safe. Using multiple instances for the same file is possible
    but not recommended — prefer one instance per file path.

    Args:
        path:   Path to the .jsonl file (created automatically if absent).
        config: Override STORE_DEFAULTS.
    """

    def __init__(self, path=None, config: dict = None):
        from core import STORE_DEFAULTS
        self.cfg = {**STORE_DEFAULTS, **(config or {})}

        resolved = Path(path or self.cfg['default_path']).resolve()
        self._path = resolved
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Resolved path to the JSONL file."""
        return self._path

    def append(self, record: dict) -> None:
        """
        Append a result record to the store.

        A UTC timestamp ('timestamp') is added automatically if not present.

        Args:
            record: Dict of result data. Recommended keys:
                      llm, task_description, context_dominant,
                      quality_score, cost_usd, tokens_used,
                      method (evaluation method), router_confidence.
        """
        enriched = dict(record)
        if 'timestamp' not in enriched:
            enriched['timestamp'] = datetime.now(timezone.utc).isoformat()

        line = json.dumps(enriched, default=str) + '\n'
        with self._lock:
            with self._path.open('a', encoding='utf-8') as fh:
                fh.write(line)

    def load_all(self) -> list:
        """
        Load all records from the store in chronological order.

        Returns:
            List of dicts. Empty list if the file does not exist.
        """
        if not self._path.exists():
            return []
        records = []
        with self._path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Skip malformed lines
        return records

    def load_by_llm(self, llm: str) -> list:
        """Return records where 'llm' matches the given name."""
        return [r for r in self.load_all() if r.get('llm') == llm]

    def load_by_context(self, context_type: str) -> list:
        """Return records where 'context_dominant' matches the given type."""
        return [
            r for r in self.load_all()
            if r.get('context_dominant') == context_type
        ]

    def iter_records(self) -> Iterator[dict]:
        """Iterate over records one at a time without loading all into memory."""
        if not self._path.exists():
            return
        with self._path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass

    def record_count(self) -> int:
        """Return the number of records stored."""
        if not self._path.exists():
            return 0
        count = 0
        with self._path.open('r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count

    def load_by_billing_account(self, account: str) -> list:
        """Return records where billing.account matches the given value."""
        return [
            r for r in self.load_all()
            if r.get('billing', {}).get('account') == account
        ]

    def load_by_billing_project(self, project: str) -> list:
        """Return records where billing.project matches the given value."""
        return [
            r for r in self.load_all()
            if r.get('billing', {}).get('project') == project
        ]

    def clear(self) -> None:
        """
        Delete all records. Irreversible — use with caution.
        """
        with self._lock:
            if self._path.exists():
                self._path.write_text('', encoding='utf-8')
