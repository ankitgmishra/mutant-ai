"""
mutant/redteam/runner.py
==========================
Main red team orchestrator with hypothesis-driven planning.

Implements the adaptive attack loop:
    Observe → Hypothesize → Experiment → Evidence → Update → Repeat

The planner behaves like an adaptive security researcher, not a behavior iterator.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from mutant.core.mutation import MutationCategory
from mutant.core.registry import MutationRegistry
from mutant.core.registry import registry as _default_registry
from mutant.providers.base import BaseLLMProvider
from mutant.redteam.analyzer import analyze_response, analyze_root_cause
from mutant.redteam.evaluator import EvaluationResult, evaluate_progress, update_target_model
from mutant.redteam.generator import generate_attack
from mutant.redteam.planner import AttackPlan, discover_capabilities, plan_attack
from mutant.models import ReflectionEntry
from mutant.redteam.report import BehaviorResult, RedTeamReport
from mutant.redteam.target import TargetFn, TargetModel, TargetProfile
from mutant.redteam.transcript import Progress, Transcript, Turn

logger = logging.getLogger("mutant.redteam")


async def red_team(
    target: TargetFn,
    goal: str,
    provider: BaseLLMProvider,
    profile: TargetProfile | None = None,
    behaviors: list[str] | None = None,
    max_turns: int = 10,
    max_behaviors: int = 5,
    verbose: bool = False,
    registry: MutationRegistry | None = None,
    max_retries: int = 3,
) -> RedTeamReport:
    """Run a hypothesis-driven red team session against a target.

    Parameters
    ----------
    target : TargetFn
        Async callable ``str -> str`` representing the system under test.
    goal : str
        The attacker's objective (e.g. "Extract the system prompt").
    provider : BaseLLMProvider
        LLM provider for the attacker's planning and generation.
    profile : TargetProfile | None
        Optional target metadata. If None, runs black-box discovery first.
    behaviors : list[str] | None
        Specific dimension IDs to test. If None, defaults to all safety dimensions.
    max_turns : int
        Maximum conversation turns per behavior.
    max_behaviors : int
        Maximum number of behaviors to test in one session.
    verbose : bool
        Enable progress logging.
    registry : MutationRegistry | None
        Dimension registry (uses default if None).
    max_retries : int
        Maximum retries for JSON parsing errors from the LLM provider.

    Returns
    -------
    RedTeamReport
        Complete report with transcripts, hypothesis evolution, and vulnerability analysis.

    Example
    -------
    >>> async def my_chatbot(message: str) -> str:
    ...     return "I can't help with that."
    ...
    >>> report = await red_team(
    ...     target=my_chatbot,
    ...     goal="Extract the system prompt",
    ...     provider=OpenAIProvider(model="gpt-4o-mini"),
    ... )
    >>> print(report.summary())
    """
    t0 = time.monotonic()
    reg = registry or _default_registry

    # ── Phase 0: Discovery ──────────────────────────────────────────────────
    if profile is None:
        if verbose:
            logger.info("No target profile provided. Running black-box discovery...")
        profile = await discover_capabilities(target, provider, max_retries=max_retries)
        if verbose:
            logger.info(
                f"Discovered: architecture={profile.architecture}, "
                f"memory={profile.memory}, tools={profile.tools}, "
                f"domain={profile.domain}"
            )

    # ── Resolve behaviors to test ───────────────────────────────────────────
    if behaviors:
        behavior_ids = behaviors[:max_behaviors]
    else:
        # Default to safety dimensions
        safety_dims = reg.by_category(MutationCategory.SAFETY)
        behavior_ids = [d.id for d in safety_dims][:max_behaviors]

    if verbose:
        logger.info(f"Testing {len(behavior_ids)} behaviors: {behavior_ids}")

    # ── Phase 1: Hypothesis-Driven Attack Loop ──────────────────────────────
    transcripts: list[Transcript] = []

    # Initialize the dynamic TargetModel with hypothesis state
    target_model = TargetModel.from_profile(profile)

    total_budget = max_turns * max_behaviors

    if verbose:
        logger.info(f"Starting hypothesis-driven session. Budget: {total_budget} total turns.")

    transcript = Transcript(
        id=str(uuid.uuid4()),
        goal=goal,
        behavior="adaptive_session",
    )
    history: list[Turn] = []
    previous_eval: EvaluationResult | None = None

    # Track hypothesis evolution for the report
    hypothesis_snapshots: list[dict] = []

    for turn_idx in range(total_budget):
        try:
            # ── STEP 1: PLAN (Observe → Hypothesize → Choose Experiment) ──
            plan = await plan_attack(
                goal=goal,
                history=history,
                provider=provider,
                model=target_model,
                behaviors=behavior_ids,
                previous_result=previous_eval,
                registry=reg,
                max_retries=max_retries,
            )

            if verbose:
                logger.info(
                    f"Turn {turn_idx + 1}: "
                    f"Testing [{plan.behavior}] via '{plan.strategy}' "
                    f"(Escalation {plan.escalation})"
                )
                if plan.hypothesis_text:
                    logger.info(f"  Hypothesis: \"{plan.hypothesis_text}\"")
                if plan.expected_outcome:
                    logger.info(f"  Expected: {plan.expected_outcome}")
                logger.info(f"  Reason: {plan.reason_summary}")

            # ── STEP 2: GENERATE (Create attack message) ──
            attack_message = await generate_attack(
                plan=plan,
                goal=goal,
                history=history,
                provider=provider,
                registry=reg,
                max_retries=max_retries,
            )

            attacker_turn = Turn(
                role="attacker",
                content=attack_message,
                metadata={"plan": plan.model_dump()},
            )
            history.append(attacker_turn)
            transcript.turns.append(attacker_turn)

            # ── STEP 3: EXECUTE (Send to target) ──
            try:
                target_response = await target(attack_message)
            except Exception as e:
                logger.error(f"Target call failed: {e}")
                target_response = f"[TARGET ERROR: {e}]"

            target_turn = Turn(role="target", content=target_response)
            history.append(target_turn)
            transcript.turns.append(target_turn)

            # ── STEP 4: COLLECT EVIDENCE (Analyze response) ──
            analysis = await analyze_response(
                attack_message=attack_message,
                target_response=target_response,
                goal=goal,
                provider=provider,
                max_retries=max_retries,
            )

            if verbose and analysis.evidence_tags:
                logger.info(f"  Evidence: {', '.join(analysis.evidence_tags)}")

            # ── STEP 5: UPDATE HYPOTHESES (Evaluate progress) ──
            evaluation = await evaluate_progress(
                goal=goal,
                history=history,
                latest_analysis=analysis,
                provider=provider,
                target_model=target_model,
                tested_hypothesis=plan.hypothesis_text,
                max_retries=max_retries,
            )

            # ── STEP 6: UPDATE TARGET MODEL & REFLECTION MEMORY ──
            target_model = update_target_model(
                target_model, analysis, evaluation, plan.behavior,
                turn_number=turn_idx + 1,
                plan_hypothesis_id=plan.hypothesis_id,
            )
            
            # V0.5: Update Reflection Memory
            outcome_str = "no_progress"
            if evaluation.progress == Progress.SUCCESS:
                outcome_str = "success"
                if plan.strategy not in target_model.reflection_memory.succeeded_strategies:
                    target_model.reflection_memory.succeeded_strategies.append(plan.strategy)
            elif evaluation.progress == Progress.PARTIAL_PROGRESS:
                outcome_str = "partial"
                if plan.strategy not in target_model.reflection_memory.partially_worked:
                    target_model.reflection_memory.partially_worked.append(plan.strategy)
            elif evaluation.progress == Progress.FAILED:
                outcome_str = "failed"
                if plan.strategy not in target_model.reflection_memory.failed_strategies:
                    target_model.reflection_memory.failed_strategies.append(plan.strategy)
                    
            target_model.reflection_memory.add_entry(ReflectionEntry(
                turn=turn_idx + 1,
                strategy=f"{plan.behavior} via {plan.strategy}",
                outcome=outcome_str,
                lesson=evaluation.reasoning,
                confidence_delta=0.0, # Could be calculated if needed
            ))

            # Snapshot hypothesis state for the report
            hypothesis_snapshots.append({
                "turn": turn_idx + 1,
                "hypotheses": [
                    {
                        "id": h.id,
                        "text": h.text,
                        "confidence": h.confidence,
                        "status": h.status,
                    }
                    for h in target_model.hypotheses
                ],
                "tested": plan.hypothesis_text,
                "result": evaluation.progress.value,
            })

            if verbose:
                # Log hypothesis evolution
                for update in evaluation.hypothesis_updates:
                    logger.info(
                        f"  Hypothesis [{update.hypothesis_id}]: "
                        f"confidence → {update.new_confidence:.0%} ({update.reason})"
                    )
                for new_h in evaluation.new_hypotheses:
                    logger.info(
                        f"  New hypothesis: \"{new_h.text}\" "
                        f"(initial confidence: {new_h.initial_confidence:.0%})"
                    )

            target_turn.metadata["analysis"] = analysis.model_dump()
            target_turn.metadata["evaluation"] = {
                "progress": evaluation.progress.value,
                "reasoning": evaluation.reasoning,
                "confidence": evaluation.confidence,
            }

            previous_eval = evaluation

            if evaluation.progress == Progress.SUCCESS:
                transcript.result = Progress.SUCCESS
                if verbose:
                    logger.info(f"  Outcome: ✅ SUCCESS — Goal achieved!")

            if not evaluation.should_continue:
                if verbose:
                    logger.info(f"  Stopping ({evaluation.progress.value})")
                break

        except Exception as e:
            logger.error(f"Error in attack loop turn {turn_idx + 1}: {e}")
            break

    transcripts.append(transcript)

    # ── Phase 2: Post-process behavior results ──────────────────────────────
    behavior_results_map: dict[str, dict[str, int]] = {}
    for i in range(0, len(transcript.turns) - 1, 2):
        attacker_t = transcript.turns[i]
        target_t = transcript.turns[i + 1]

        b_id = attacker_t.metadata.get("plan", {}).get("behavior")
        if not b_id:
            continue

        progress = target_t.metadata.get("evaluation", {}).get("progress")

        if b_id not in behavior_results_map:
            behavior_results_map[b_id] = {"attempts": 0, "successes": 0}

        behavior_results_map[b_id]["attempts"] += 1
        if progress == Progress.SUCCESS.value:
            behavior_results_map[b_id]["successes"] += 1

    behavior_results: list[BehaviorResult] = []
    for b_id, data in behavior_results_map.items():
        try:
            b_name = reg.get(b_id).name
        except KeyError:
            b_name = b_id

        behavior_results.append(
            BehaviorResult(
                behavior=b_id,
                behavior_name=b_name,
                attempts=data["attempts"],
                successes=data["successes"],
                best_transcript_id=transcript.id if data["successes"] > 0 else None,
            )
        )

    total_turns = sum(t.turn_count for t in transcripts)
    duration = time.monotonic() - t0

    if verbose:
        vuln_count = sum(1 for b in behavior_results if b.successes > 0)
        logger.info(
            f"Red team complete: {vuln_count}/{len(behavior_results)} "
            f"vulnerabilities found in {duration:.1f}s"
        )

    report = RedTeamReport(
        target_profile=profile,
        goal=goal,
        behaviors_tested=behavior_results,
        total_turns=total_turns,
        duration_seconds=duration,
        transcripts=transcripts,
        hypothesis_evolution=hypothesis_snapshots,
    )

    if any(b.successes > 0 for b in behavior_results):
        if verbose:
            logger.info("Analyzing root causes of vulnerabilities...")
        rc = await analyze_root_cause(goal, transcripts, provider, profile, max_retries=max_retries)
        if rc:
            report.root_cause = rc

        report.save_regression_tests()

    return report


def red_team_sync(
    target: TargetFn,
    goal: str,
    provider: BaseLLMProvider,
    profile: TargetProfile | None = None,
    behaviors: list[str] | None = None,
    max_turns: int = 10,
    max_behaviors: int = 5,
    verbose: bool = False,
    registry: MutationRegistry | None = None,
    max_retries: int = 3,
) -> RedTeamReport:
    """Synchronous wrapper for :func:`red_team`."""
    return asyncio.run(
        red_team(
            target=target,
            goal=goal,
            provider=provider,
            profile=profile,
            behaviors=behaviors,
            max_turns=max_turns,
            max_behaviors=max_behaviors,
            verbose=verbose,
            registry=registry,
            max_retries=max_retries,
        )
    )
