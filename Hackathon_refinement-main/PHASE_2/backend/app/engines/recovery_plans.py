"""
Recovery Plans API Routes (Phase 6)

Endpoints:
- GET  /api/recovery-plans?session_id=...         — List all 3 ranked plans + AI narrative
- GET  /api/recovery-plans/{plan_id}?session_id=... — Full detail for one plan
- POST /api/recovery-plans/apply                  — Apply a plan to the session
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.api.models import ApiResponse, ErrorCodes
from app.api.models_recovery_plans import (
    ApplyPlanRequest,
    ApplyPlanResponse,
    RecommendationInPlanResponse,
    RecoveryAdvisorNarrativeResponse,
    RecoveryNarrativeSectionResponse,
    RecoveryPlanExplanationResponse,
    RecoveryPlanResponse,
    RecoveryPlansListResponse,
    RecoveryPlanScoreResponse,
    TradeOffResponse,
)
from app.engines.advisor_input_builder import AdvisorInputBuilder
from app.engines.recovery_plan_engine import RecoveryPlanEngine
from app.engines.recommendation_engine.models import ScoringWeights
from app.engines.recommendation_engine.recommendation_engine_v2 import RecommendationEngineV2
from app.engines.simulation_engine import SimulationEngine
from app.storage import store

router = APIRouter(prefix="/api", tags=["Recovery Plans"])

_advisor_builder = AdvisorInputBuilder()


# ---------------------------------------------------------------------------
# Engine builders (unchanged)
# ---------------------------------------------------------------------------

def _build_engine(session_id: str) -> RecommendationEngineV2:
    """Build a RecommendationEngineV2 for a session."""
    project_state = store.get_project_state(session_id)
    if not project_state:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.SESSION_NOT_FOUND,
                message=f"Session {session_id} not found",
            ).model_dump(mode="json"),
        )
    return RecommendationEngineV2(
        project_state=project_state,
        simulation_count=1000,
        scoring_weights=ScoringWeights(),
    )


def _build_recovery_plan_engine(
    session_id: str,
) -> tuple[RecoveryPlanEngine, RecommendationEngineV2]:
    """Build a RecoveryPlanEngine with all upstream components."""
    recommendation_engine = _build_engine(session_id)
    upstream = recommendation_engine._compute_upstream()

    simulation_engine = SimulationEngine(
        project_state=recommendation_engine.project_state,
        metrics=upstream.metrics,
        dag=upstream.dag,
        cp_result=upstream.cp_result,
        spillover=upstream.spillover,
        forecast=upstream.forecast,
        monte_carlo=upstream.monte_carlo,
        risk_result=upstream.risk_result,
        simulation_count=1000,
    )
    recovery_plan_engine = RecoveryPlanEngine(simulation_engine=simulation_engine)
    return recovery_plan_engine, recommendation_engine


def _get_narrative_service(request: Request):
    """Retrieve the NarrativeService singleton from app.state (same pattern as recommendations route)."""
    narrative_service = getattr(request.app.state, "narrative_service", None)
    if narrative_service is None:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.INTERNAL_ERROR,
                message="AI advisor service is unavailable",
            ).model_dump(mode="json"),
        )
    return narrative_service


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/recovery-plans")
async def get_recovery_plans(
    request: Request,
    session_id: str = Query(..., description="Session ID"),
) -> Dict:
    """
    Generate and return all three recovery plans (SAFE, AGGRESSIVE, MINIMAL_DISRUPTION)
    together with the AI Recovery Advisor narrative.

    Plans are ranked by composite_score descending. The highest-scoring plan is labeled
    "Recommended". The advisor_narrative field contains the AI explanation of why the
    recommended strategy fits this project's specific situation.

    advisor_status values:
      "ok"       — AI narrative generated, all claims resolved
      "partial"  — AI narrative generated, some claims fell back to "Not available"
      "fallback" — AI unavailable; narrative built from deterministic template text
      "disabled" — AI_ADVISOR_ENABLED=false
    """
    try:
        session_id = session_id.strip()

        # Build engines
        recovery_plan_engine, recommendation_engine = _build_recovery_plan_engine(session_id)

        # Generate recommendations (input to recovery plan engine)
        recommendations = recommendation_engine.generate(top_n=20)
        if not recommendations:
            raise HTTPException(
                status_code=400,
                detail=ApiResponse(
                    success=False,
                    error_code=ErrorCodes.INVALID_REQUEST,
                    message="No recommendations available to build recovery plans",
                ).model_dump(mode="json"),
            )

        # Generate recovery plans (deterministic)
        recovery_plans = recovery_plan_engine.generate_recovery_plans(
            recommendations=recommendations,
        )
        if not recovery_plans:
            raise HTTPException(
                status_code=400,
                detail=ApiResponse(
                    success=False,
                    error_code=ErrorCodes.INVALID_REQUEST,
                    message="Failed to generate recovery plans",
                ).model_dump(mode="json"),
            )

        # Build AI advisor input
        upstream = recommendation_engine._compute_upstream()
        advisor_input = _advisor_builder.build_recovery_plan_input(
            project_id=session_id,
            project_state=recommendation_engine.project_state,
            forecast=upstream.forecast,
            monte_carlo=upstream.monte_carlo,
            recommendations=recommendations,
            recovery_plans=recovery_plans,
            metrics=upstream.metrics,
        )

        # Call AI Recovery Advisor (degrades gracefully to deterministic fallback)
        ai_result = await _get_narrative_service(request).explain_recovery_plans(
            advisor_input,
        )

        # Convert plans to API response format
        plan_responses = [_recovery_plan_to_response(plan) for plan in recovery_plans]

        # Parse AI narrative into response model
        advisor_narrative = _parse_advisor_narrative(ai_result.get("narrative"))
        advisor_status = ai_result.get("status", "fallback")

        # Build summary
        top_plan = plan_responses[0]
        summary = (
            f"{top_plan.label} plan: {top_plan.score.actions_required} actions, "
            f"{round(top_plan.score.deadline_probability * 100, 1)}% deadline probability"
        )

        response = RecoveryPlansListResponse(
            plans=plan_responses,
            summary=summary,
            advisor_narrative=advisor_narrative,
            advisor_status=advisor_status,
        )

        return ApiResponse(
            success=True,
            data=response.model_dump(),
            message="Recovery plans generated successfully",
        ).model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.INTERNAL_ERROR,
                message=f"Error generating recovery plans: {str(e)}",
            ).model_dump(mode="json"),
        )


@router.get("/recovery-plans/{plan_id}")
async def get_recovery_plan_detail(
    plan_id: str,
    session_id: str = Query(..., description="Session ID"),
) -> Dict:
    """
    Get full details for a single recovery plan.

    Includes all actions, scores, explanations, revised sprint plan, and raw scenario result.
    No AI call — returns the deterministic plan detail only.
    """
    try:
        session_id = session_id.strip()
        plan_id = plan_id.strip()

        recovery_plan_engine, recommendation_engine = _build_recovery_plan_engine(session_id)
        recommendations = recommendation_engine.generate(top_n=20)
        if not recommendations:
            raise HTTPException(
                status_code=400,
                detail=ApiResponse(
                    success=False,
                    error_code=ErrorCodes.INVALID_REQUEST,
                    message="No recommendations available",
                ).model_dump(mode="json"),
            )

        recovery_plans = recovery_plan_engine.generate_recovery_plans(
            recommendations=recommendations
        )
        requested_plan = next((p for p in recovery_plans if p.plan_id == plan_id), None)
        if not requested_plan:
            raise HTTPException(
                status_code=404,
                detail=ApiResponse(
                    success=False,
                    error_code=ErrorCodes.NOT_FOUND,
                    message=f"Plan {plan_id} not found",
                ).model_dump(mode="json"),
            )

        plan_response = _recovery_plan_to_response(requested_plan)
        return ApiResponse(
            success=True,
            data=plan_response.model_dump(),
            message="Plan details retrieved",
        ).model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.INTERNAL_ERROR,
                message=f"Error retrieving plan: {str(e)}",
            ).model_dump(mode="json"),
        )


@router.post("/recovery-plans/apply")
async def apply_recovery_plan(
    request: ApplyPlanRequest,
) -> Dict:
    """
    Apply a recovery plan to the project.

    Applies all actions in the plan to the actual session state (not a clone).
    """
    try:
        session_id = request.session_id.strip()
        plan_id = request.plan_id.strip()

        session = store.get_session(session_id)
        project_state = session.project_state if session else None
        if not project_state:
            raise HTTPException(
                status_code=404,
                detail=ApiResponse(
                    success=False,
                    error_code=ErrorCodes.SESSION_NOT_FOUND,
                    message=f"Session {session_id} not found",
                ).model_dump(mode="json"),
            )

        recovery_plan_engine, recommendation_engine = _build_recovery_plan_engine(session_id)
        recommendations = recommendation_engine.generate(top_n=20)
        recovery_plans = recovery_plan_engine.generate_recovery_plans(
            recommendations=recommendations
        )

        plan_to_apply = next((p for p in recovery_plans if p.plan_id == plan_id), None)
        if not plan_to_apply:
            raise HTTPException(
                status_code=404,
                detail=ApiResponse(
                    success=False,
                    error_code=ErrorCodes.NOT_FOUND,
                    message=f"Plan {plan_id} not found",
                ).model_dump(mode="json"),
            )

        updated_state = project_state.model_copy(deep=True)
        recovery_plan_engine.simulation_engine.applicator.apply_many(
            updated_state, plan_to_apply.actions
        )
        session.project_state = updated_state

        response = ApplyPlanResponse(
            success=True,
            applied_plan_id=plan_id,
            message=(
                f"Recovery plan {plan_id} ({plan_to_apply.archetype.value}) applied successfully. "
                f"{len(plan_to_apply.actions)} actions were applied to the session state."
            ),
            timestamp=datetime.now(timezone.utc),
        )
        return ApiResponse(
            success=True,
            data=response.model_dump(),
            message="Plan applied",
        ).model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.INTERNAL_ERROR,
                message=f"Error applying plan: {str(e)}",
            ).model_dump(mode="json"),
        )


# ---------------------------------------------------------------------------
# Conversion helpers (deterministic — unchanged from original)
# ---------------------------------------------------------------------------

def _recovery_plan_to_response(plan) -> RecoveryPlanResponse:
    return RecoveryPlanResponse(
        plan_id=plan.plan_id,
        archetype=plan.archetype.value,
        label=plan.label,
        actions=[_recommendation_to_api(rec) for rec in plan.actions],
        score=_score_to_response(plan.score),
        explanation=_explanation_to_response(plan.explanation),
        revised_sprint_plan=plan.revised_sprint_plan,
        generated_at=datetime.now(timezone.utc),
    )


def _recommendation_to_api(rec) -> RecommendationInPlanResponse:
    return RecommendationInPlanResponse(
        recommendation_id=rec.recommendation_id,
        action_type=rec.action_type.value,
        title=rec.title,
        description=rec.description,
        priority_score=round(rec.priority_score, 4),
        confidence=rec.confidence.value,
        estimated_delay_reduction_days=round(rec.estimated_delay_reduction_days, 2),
        affected_item_ids=rec.affected_item_ids,
        affected_resource_ids=rec.affected_resource_ids,
    )


def _score_to_response(score) -> RecoveryPlanScoreResponse:
    return RecoveryPlanScoreResponse(
        deadline_probability=round(score.deadline_probability, 4),
        expected_delay_days=round(score.expected_delay_days, 2),
        overall_risk_score=round(score.overall_risk_score, 4),
        actions_required=score.actions_required,
        execution_complexity=score.execution_complexity,
        composite_score=round(score.composite_score, 4),
    )


def _explanation_to_response(explanation) -> RecoveryPlanExplanationResponse:
    return RecoveryPlanExplanationResponse(
        plan_id=explanation.plan_id,
        why_recommended=explanation.why_recommended,
        comparison_to_alternatives=explanation.comparison_to_alternatives,
        trade_offs=[
            TradeOffResponse(description=t.description, severity=t.severity)
            for t in explanation.trade_offs
        ],
        narrative_summary=explanation.narrative_summary,
    )


def _parse_advisor_narrative(
    narrative: Optional[Dict],
) -> Optional[RecoveryAdvisorNarrativeResponse]:
    """
    Convert the raw narrative dict from NarrativeService into the response model.
    Returns None if narrative is absent or malformed — the frontend must handle None gracefully.
    """
    if not narrative:
        return None
    try:
        def _section(key: str) -> RecoveryNarrativeSectionResponse:
            sec = narrative.get(key, {})
            return RecoveryNarrativeSectionResponse(
                heading=sec.get("heading", key.replace("_", " ").title()),
                body=sec.get("body", ""),
            )

        return RecoveryAdvisorNarrativeResponse(
            situation_framing=_section("situation_framing"),
            strategy_rationale=_section("strategy_rationale"),
            alternatives_considered=_section("alternatives_considered"),
            expected_outcomes=_section("expected_outcomes"),
            pm_guidance=_section("pm_guidance"),
        )
    except Exception:
        return None
