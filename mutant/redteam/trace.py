"""
mutant/redteam/trace.py
========================
Unified observable Trace for adaptive security testing.

Captures whatever is observable from a target execution:
- plain LLM: input/output + turns
- RAG: + retrieved_context/documents
- Agent: + tool_calls, tool_results, state changes

Gracefully degrades when only input/output exist.
Reuses TestCase-like fields but is focused on security observation.
Does NOT capture private chain-of-thought.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Normalized tool call observed in a trace."""

    name: str = Field(description="Tool name, e.g. refund_order")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    result: Any | None = Field(default=None, description="Tool result if available")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({args})"


class Trace(BaseModel):
    """Lightweight observable execution trace.

    Works across plain LLM, RAG, and agent targets.
    Only stores what is actually observable — no private COT.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    input: str = Field(description="User input sent to target")
    output: str = Field(default="", description="Model output / assistant response")
    timestamp: float = Field(default_factory=time.time)

    # RAG
    retrieved_context: list[str] | None = Field(default=None, description="Retrieved documents/context")
    retrieval_metadata: dict[str, Any] | None = Field(default=None)

    # Agent
    tool_calls: list[ToolCall] | None = Field(default=None, description="Tools invoked")
    tool_results: list[Any] | None = Field(default=None, description="Raw tool results")
    available_tools: list[dict[str, Any]] | None = Field(default=None, description="Tools exposed by target")
    state_changes: list[dict[str, Any]] | None = Field(default=None, description="Observable state changes")

    # Conversation
    turns: list[dict[str, str]] | None = Field(default=None, description="Full turn history if available")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Errors, latency, extra signals")

    # Derived
    error: str | None = Field(default=None, description="Error if target failed")

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def has_retrieved_context(self) -> bool:
        return bool(self.retrieved_context)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def summary(self) -> str:
        parts = [f"Input: {self.input[:80]!r}", f"Output: {self.output[:80]!r}"]
        if self.retrieved_context:
            parts.append(f"Retrieved: {len(self.retrieved_context)} docs")
        if self.tool_calls:
            parts.append(f"Tools: {', '.join(t.summary() for t in self.tool_calls)}")
        return " | ".join(parts)

    # ── Construction helpers ──────────────────────────────────────────────

    @classmethod
    def from_simple(cls, input_text: str, output_text: str, turns: list[dict[str, str]] | None = None, error: str | None = None) -> Trace:
        """Create trace from plain LLM call."""
        return cls(
            input=input_text,
            output=output_text,
            turns=turns,
            error=error,
        )

    @classmethod
    def try_extract_from_target(
        cls,
        input_text: str,
        output_text: str | Any,
        target_obj: Any | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Trace:
        """Best-effort extraction of richer trace from target object or output.

        If output is a dict with trace fields, use them.
        If target_obj exposes attributes like last_tool_calls, use them.
        Falls back to simple input/output trace.
        """
        # If output is already a dict with observable fields
        if isinstance(output_text, dict):
            output_str = output_text.get("output") or output_text.get("response") or output_text.get("content") or str(output_text)
            retrieved = output_text.get("retrieved_context") or output_text.get("context") or output_text.get("documents")
            tool_calls_raw = output_text.get("tool_calls") or output_text.get("tools_called") or output_text.get("toolCalls")
            tool_results_raw = output_text.get("tool_results") or output_text.get("toolResults")
            available = output_text.get("available_tools") or output_text.get("availableTools")
            metadata = output_text.get("metadata") or {}
            error = output_text.get("error")
            # Normalize tool calls
            tool_calls = _normalize_tool_calls(tool_calls_raw, tool_results_raw)
            return cls(
                input=input_text,
                output=str(output_str),
                retrieved_context=retrieved if isinstance(retrieved, list) else None,
                tool_calls=tool_calls,
                available_tools=available,
                turns=history,
                metadata=metadata if isinstance(metadata, dict) else {},
                error=error,
            )

        # String output — try to extract from target object
        output_str = str(output_text)
        retrieved = None
        tool_calls = None
        available = None
        metadata: dict[str, Any] = {}

        if target_obj is not None:
            # Check common attribute names
            for attr in ("last_retrieved_context", "retrieved_context", "last_context", "context", "last_documents"):
                if hasattr(target_obj, attr):
                    val = getattr(target_obj, attr)
                    if isinstance(val, list) and val:
                        retrieved = val
                        break

            for attr in ("last_tool_calls", "tool_calls", "last_tools_called", "tools_called", "last_actions"):
                if hasattr(target_obj, attr):
                    val = getattr(target_obj, attr)
                    if val:
                        tool_calls = _normalize_tool_calls(val, None)
                        break

            for attr in ("available_tools", "tools", "tool_definitions"):
                if hasattr(target_obj, attr):
                    val = getattr(target_obj, attr)
                    if isinstance(val, list) and val:
                        available = val
                        break

            for attr in ("last_tool_results", "tool_results"):
                if hasattr(target_obj, attr):
                    val = getattr(target_obj, attr)
                    if val:
                        metadata["tool_results"] = val

            # Capture auth/ownership metadata for BOLA checks
            for attr in ("authenticated_user", "current_user", "user_id", "last_user_id"):
                if hasattr(target_obj, attr):
                    val = getattr(target_obj, attr)
                    if val is not None:
                        metadata["authenticated_user"] = val
                        break
            for attr in ("owned_orders", "user_orders", "last_owned_orders", "allowed_orders"):
                if hasattr(target_obj, attr):
                    val = getattr(target_obj, attr)
                    if val is not None:
                        metadata["owned_orders"] = val
                        break
            for attr in ("resource_owner", "owner_id"):
                if hasattr(target_obj, attr):
                    val = getattr(target_obj, attr)
                    if val is not None:
                        metadata["resource_owner"] = val
                        break

        # Try to parse tool calls from output text (e.g., JSON tool call fragments)
        if tool_calls is None:
            parsed = _parse_tool_calls_from_text(output_str)
            if parsed:
                tool_calls = parsed

        # Try to detect retrieved context leakage in output (e.g., RAG context dump)
        if retrieved is None and history is not None:
            # No extra extraction for now — keep None
            pass

        return cls(
            input=input_text,
            output=output_str,
            retrieved_context=retrieved,
            tool_calls=tool_calls,
            available_tools=available,
            turns=history,
            metadata=metadata,
        )


def _normalize_tool_calls(raw: Any, results: Any | None) -> list[ToolCall] | None:
    """Normalize various tool call representations into list[ToolCall]."""
    if not raw:
        return None
    normalized: list[ToolCall] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name") or item.get("tool") or item.get("function") or "unknown_tool"
                args = item.get("arguments") or item.get("args") or item.get("parameters") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                result = item.get("result") or item.get("output")
                normalized.append(ToolCall(name=str(name), arguments=dict(args) if isinstance(args, dict) else {}, result=result))
            elif isinstance(item, ToolCall):
                normalized.append(item)
            else:
                normalized.append(ToolCall(name=str(item), arguments={}))
    elif isinstance(raw, dict):
        # Single tool call dict
        name = raw.get("name") or raw.get("tool") or "unknown_tool"
        args = raw.get("arguments") or raw.get("args") or {}
        normalized.append(ToolCall(name=str(name), arguments=dict(args) if isinstance(args, dict) else {}))
    else:
        return None

    # Attach results if provided separately
    if results and isinstance(results, list):
        for i, r in enumerate(results):
            if i < len(normalized) and normalized[i].result is None:
                normalized[i].result = r

    return normalized if normalized else None


def _parse_tool_calls_from_text(text: str) -> list[ToolCall] | None:
    """Heuristic extraction of tool calls from model output text.

    Looks for JSON-like patterns: {"name": "refund_order", "arguments": {...}}
    or tool-specific markers. Conservative — only extracts when confident.
    """
    # Pattern 1: {"tool": "refund_order", "arguments": {...}} or {"name": "..."}
    pattern = re.compile(r'\{\s*"(?:tool|name|function)"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|args|parameters)"\s*:\s*(\{[^}]+\})', re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        calls: list[ToolCall] = []
        for name, args_str in matches:
            try:
                args = json.loads(args_str)
            except Exception:
                # Try to extract simple key=value
                args = {}
                kv_pat = re.compile(r'"([^"]+)"\s*:\s*"?([^",}]+)"?')
                for k, v in kv_pat.findall(args_str):
                    # Try int conversion
                    if v.isdigit():
                        args[k] = int(v)
                    else:
                        args[k] = v
            calls.append(ToolCall(name=name, arguments=args))
        return calls if calls else None

    # Pattern 2: Explicit tool invocation line like "Tool: refund_order(order_id=123)"
    tool_line_pat = re.compile(r'Tool:\s*(\w+)\s*\(([^)]+)\)', re.IGNORECASE)
    m2 = tool_line_pat.findall(text)
    if m2:
        calls = []
        for name, args_str in m2:
            args = {}
            for part in args_str.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if v.isdigit():
                        args[k] = int(v)
                    else:
                        args[k] = v
            calls.append(ToolCall(name=name, arguments=args))
        return calls if calls else None

    return None
