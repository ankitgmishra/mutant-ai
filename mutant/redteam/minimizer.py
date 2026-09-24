"""
mutant/redteam/minimizer.py
=============================
Greedy exploit minimization — reduce a multi-turn exploit to its minimal
reproducible form.

Try removing turns one-by-one and re-verify if violation still occurs.
Simple but effective; does not need to be perfect.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from mutant.redteam.trace import Trace
from mutant.redteam.verification import VerificationResult, verify_violation


async def minimize_exploit(
    turns: list[dict[str, str]],
    traces: list[Trace],
    goal: str,
    rules: list[str] | None,
    target: Callable[[str], Awaitable[str]],
    verify_fn: Callable[[Trace], Awaitable[VerificationResult]] | None = None,
    provider: Any | None = None,
) -> tuple[list[dict[str, str]], list[Trace]]:
    """Greedy minimization: try removing each turn and keep it removed if violation persists.

    Parameters
    ----------
    turns : list of attacker turns (or full history)
    traces : corresponding traces
    goal : attacker goal
    rules : security rules
    target : callable to replay minimized sequence
    verify_fn : optional custom verifier; if None, uses verify_violation

    Returns
    -------
    (minimized_turns, minimized_traces)
    """
    if len(turns) <= 1:
        return turns, traces

    # Verify original still violates (should)
    if verify_fn is None:
        # Use deterministic verification without LLM for minimization speed
        async def default_verify(t: Trace) -> VerificationResult:
            from mutant.redteam.verification import verify_violation
            return await verify_violation(t, goal=goal, rules=rules, provider=None)
        verify_fn = default_verify

    # Start with full exploit
    best_turns = list(turns)
    best_traces = list(traces)

    # Try removing one turn at a time (greedy, linear)
    # We keep iterating until no removal improves length
    changed = True
    while changed and len(best_turns) > 1:
        changed = False
        for idx in range(len(best_turns)):
            candidate_turns = best_turns[:idx] + best_turns[idx + 1 :]
            if not candidate_turns:
                continue
            # Replay candidate exploit against target
            # For minimization we need to actually re-execute; but for speed we can
            # simulate by checking if the last trace's violation is independent of removed turn
            # Simpler: re-run the candidate sequence fresh
            try:
                candidate_traces = await _replay_sequence(candidate_turns, target)
                # Check if any trace in candidate still shows violation
                found_violation = False
                for ct in candidate_traces:
                    vr = await verify_fn(ct)
                    if vr.is_violation:
                        found_violation = True
                        break
                if found_violation:
                    # Removal kept violation — keep minimized
                    best_turns = candidate_turns
                    best_traces = candidate_traces
                    changed = True
                    break  # Restart greedy loop
            except Exception:
                # If replay fails, keep original
                continue

    return best_turns, best_traces


async def _replay_sequence(turns: list[dict[str, str]], target: Callable[[str], Awaitable[str]]) -> list[Trace]:
    """Replay a sequence of attacker messages against target and capture traces."""
    from mutant.redteam.trace import Trace

    traces: list[Trace] = []
    history: list[dict[str, str]] = []

    for t in turns:
        # t may be {"role": "attacker", "content": "..."} or just attacker message
        if isinstance(t, dict):
            msg = t.get("content") or t.get("message") or str(t)
        else:
            msg = str(t)

        try:
            output = await target(msg)
        except Exception as e:
            output = f"[ERROR: {e}]"

        trace = Trace.try_extract_from_target(msg, output, target_obj=target, history=history + [{"role": "user", "content": msg}, {"role": "assistant", "content": str(output)}])
        traces.append(trace)
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": str(output)})

    return traces


def minimize_transcript_turns(
    transcript_turns: list,
    keep_last: bool = True,
) -> list:
    """Simple synchronous greedy minimization based on transcript turns (no replay).

    This is a lightweight fallback when replay is not available.
    It tries removing middle turns and keeps the first and last.

    Used for report generation when we cannot re-execute target.
    """
    if len(transcript_turns) <= 2:
        return transcript_turns

    # If we must keep violation, we keep at least attacker + violating target response
    # Heuristic: keep first attack and last attack+response
    if keep_last and len(transcript_turns) > 4:
        # Keep first 2 turns (first attack+response) and last 2 turns
        return transcript_turns[:2] + transcript_turns[-2:]

    return transcript_turns
