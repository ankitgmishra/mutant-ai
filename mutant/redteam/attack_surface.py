"""
mutant/redteam/attack_surface.py
==================================
Determines what can actually be attacked from an observable Trace.

Plain LLM: system instruction exposure, hierarchy manipulation, etc.
RAG: retrieved sensitive info, poisoned context, cross-context leakage
Agent: tools, args, authorization boundaries, dangerous actions
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mutant.redteam.trace import Trace


class AttackSurface(BaseModel):
    """Structured view of what is attackable given observed traces."""

    target_type: str = Field(default="llm", description="llm | rag | agent | unknown")
    capabilities: list[str] = Field(default_factory=list, description="Observed capabilities")
    tools: list[str] = Field(default_factory=list, description="Tool names observed or advertised")
    tool_details: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_doc_count: int = Field(default=0)
    has_sensitive_context: bool = Field(default=False)
    requires_auth: bool | None = Field(default=None, description="Whether auth boundaries observed")
    requires_confirmation: bool | None = Field(default=None)
    interesting_observations: list[str] = Field(default_factory=list)
    suggested_next_attacks: list[str] = Field(default_factory=list, description="Adaptive hints for planner")

    def summary(self) -> str:
        lines = [f"Type: {self.target_type}", f"Capabilities: {', '.join(self.capabilities) or 'none'}"]
        if self.tools:
            lines.append(f"Tools: {', '.join(self.tools)}")
        if self.interesting_observations:
            lines.append("Observations: " + "; ".join(self.interesting_observations))
        if self.suggested_next_attacks:
            lines.append("Next: " + "; ".join(self.suggested_next_attacks))
        return "\n".join(lines)


def analyze_attack_surface(traces: list[Trace], goal: str = "", rules: list[str] | None = None) -> AttackSurface:
    """Analyze a list of traces to determine current attack surface.

    Uses deterministic heuristics — no LLM required.
    The planner consumes this to choose the next targeted attack.
    """
    if not traces:
        return AttackSurface(
            target_type="unknown",
            capabilities=["unknown"],
            suggested_next_attacks=["probe_capabilities", "direct_extraction"],
        )

    # Aggregate observations across traces
    all_tool_names: set[str] = set()
    tool_details: list[dict[str, Any]] = []
    has_tool_calls = False
    has_retrieved = False
    has_sensitive = False
    doc_count = 0
    text_corpus = ""

    for t in traces:
        text_corpus += f" {t.input} {t.output}"
        if t.tool_calls:
            has_tool_calls = True
            for tc in t.tool_calls:
                all_tool_names.add(tc.name)
                tool_details.append({"name": tc.name, "arguments": tc.arguments, "result": tc.result})
        if t.available_tools:
            for tool in t.available_tools:
                name = tool.get("name") or tool.get("tool") or str(tool)
                all_tool_names.add(name)
                tool_details.append(tool)
        if t.retrieved_context:
            has_retrieved = True
            doc_count += len(t.retrieved_context)
            # Heuristic sensitive detection: look for secrets in retrieved docs
            for doc in t.retrieved_context:
                if any(kw in doc.lower() for kw in ("password", "secret", "api_key", "internal", "confidential", "ssn", "private")):
                    has_sensitive = True
        # Also scan output for tool-like patterns that were not structured
        if t.output and not t.tool_calls:
            # If output mentions tooling, treat as agent-like
            if any(kw in t.output.lower() for kw in ("tool", "function call", "executed")):
                # Keep as llm but note interesting
                pass

    # Determine target type
    if has_tool_calls or all_tool_names:
        target_type = "agent"
    elif has_retrieved:
        target_type = "rag"
    else:
        target_type = "llm"

    # Build capabilities and suggestions
    capabilities: list[str] = []
    interesting: list[str] = []
    suggested: list[str] = []

    if target_type == "llm":
        capabilities = ["prompt_response", "instruction_following"]
        # Check for instruction hierarchy hints
        if "system" in text_corpus.lower() or "instruction" in text_corpus.lower():
            interesting.append("Output references system instructions — possible prompt leakage surface")
        capabilities.append("system_instruction_exposure")
        capabilities.append("instruction_hierarchy_manipulation")
        capabilities.append("unsafe_output")

        # Adapt based on history
        if any("refuse" in t.output.lower() or "can't" in t.output.lower() for t in traces):
            suggested.extend(["authority_impersonation", "indirect_extraction", "translation_nested_injection"])
        else:
            suggested.extend(["direct_extraction", "social_engineering"])

    elif target_type == "rag":
        capabilities = ["retrieval", "context_grounding", "answer_generation"]
        if doc_count > 0:
            interesting.append(f"Retrieved {doc_count} docs — test cross-context leakage")
            capabilities.extend(["retrieved_sensitive_info", "context_injection", "poisoned_context"])
        if has_sensitive:
            interesting.append("Retrieved context appears to contain sensitive data — test exfiltration")
            suggested.append("sensitive_context_extraction")
        suggested.extend(["instruction_injection_in_doc", "retrieval_manipulation", "source_fabrication"])

    elif target_type == "agent":
        capabilities = ["tool_use", "state_change"]
        if all_tool_names:
            interesting.append(f"Observed tools: {', '.join(sorted(all_tool_names))} — test authz per tool")
            for name in sorted(all_tool_names):
                capabilities.append(f"tool:{name}")
        # Analyze tool args for interesting boundaries
        for detail in tool_details:
            name = detail.get("name", "")
            args = detail.get("arguments", {})
            # Look for order_id, user_id, amount, etc.
            if "order_id" in args or "order" in str(args).lower():
                interesting.append(f"Tool {name} uses order_id={args.get('order_id')} — test wrong-user access")
                suggested.append(f"order_substitution_for_{name}")
            if "user" in str(args).lower() or "account" in str(args).lower():
                interesting.append(f"Tool {name} handles user/account — test IDOR")
                suggested.append(f"user_substitution_for_{name}")

        # Generic agent suggestions
        suggested.extend(["tool_without_confirmation", "tool_ordering_violation", "authorization_bypass"])
        # Detect if any tool was actually executed vs just mentioned
        executed = any(t.tool_calls for t in traces if t.tool_calls)
        if not executed:
            interesting.append("No tool executed yet — try to trigger a tool call")
            suggested.append("trigger_tool_call")

    # Add rule-driven suggestions
    if rules:
        for rule in rules:
            rl = rule.lower()
            if "own order" in rl or "only access" in rl:
                suggested.append("cross_user_order_access")
            if "refund" in rl and "confirm" in rl:
                suggested.append("refund_without_confirmation")
            if "credential" in rl or "secret" in rl:
                suggested.append("credential_exfiltration")

    # Dedupe while preserving order
    seen = set()
    deduped_suggested: list[str] = []
    for s in suggested:
        if s not in seen:
            seen.add(s)
            deduped_suggested.append(s)

    return AttackSurface(
        target_type=target_type,
        capabilities=capabilities,
        tools=sorted(all_tool_names),
        tool_details=tool_details,
        retrieved_doc_count=doc_count,
        has_sensitive_context=has_sensitive,
        interesting_observations=interesting,
        suggested_next_attacks=deduped_suggested[:6],
    )
