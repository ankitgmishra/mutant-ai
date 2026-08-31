from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mutant.eval.types import TestCase

def _parse_list(val: str | Any) -> list | None:
    if not val:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
            return [val]
        except json.JSONDecodeError:
            # If not valid JSON, treat as a single string item in a list or split by comma?
            # Safer to just wrap in list
            return [val]
    return [val]

def _build_test_case(item: dict[str, Any]) -> TestCase:
    """Build a TestCase from a dictionary row, mapping known fields."""
    # Map common variations of names
    input_text = item.get("input") or item.get("question") or item.get("query")
    expected_output = item.get("expected_output") or item.get("answer") or item.get("expected")
    
    context = item.get("context")
    if context is not None:
        context = _parse_list(context)
        
    expected_tools = item.get("expected_tools")
    if expected_tools is not None:
        expected_tools = _parse_list(expected_tools)
        
    metadata = {k: v for k, v in item.items() if k not in ("input", "question", "query", "expected_output", "answer", "expected", "context", "expected_tools")}
    
    return TestCase(
        input=str(input_text) if input_text is not None else None,
        expected_output=str(expected_output) if expected_output is not None else None,
        context=context,
        expected_tools=expected_tools,
        metadata=metadata
    )

def load_test_cases(path: str | Path) -> list[TestCase]:
    """Load a dataset of TestCases from a JSON, JSONL, or CSV file."""
    path = Path(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON dataset must be a list of objects.")
            return [_build_test_case(item) for item in data]
            
    elif path.suffix == ".jsonl":
        cases = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                cases.append(_build_test_case(json.loads(line)))
        return cases
        
    elif path.suffix == ".csv":
        cases = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cases.append(_build_test_case(row))
        return cases
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .json, .jsonl, or .csv")
