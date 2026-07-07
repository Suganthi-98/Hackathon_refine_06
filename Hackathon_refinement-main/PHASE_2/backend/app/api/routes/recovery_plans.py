"""
Recovery Plans API Routes (Phase 6)

Endpoints:
- GET  /api/recovery-plans?session_id=...  — List all 3 ranked plans
- GET  /api/recovery-plans/{plan_id}?session_id=...  — Full detail for one plan  
- POST /api/recovery-plans/apply  — Apply a plan to the session
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.models import ApiResponse, ErrorCodes
from app.api.models_recovery_plans import (
    ApplyPlanRequest,
    ApplyPlanResponse,
    RecommendationInPlanResponse,
    RecoveryPlanExplanationResponse,
    RecoveryPlanResponse,
    RecoveryPlansListResponse,
    RecoveryPlanScoreResponse,
    TradeOffResponse,
)
from app.engines.recovery_plan_engine import RecoveryPlanEngine
from app.engines.recommendation_engine.models import ScoringWeights
from app.engines.recommendation_engine.recommendation_engine_v2 import RecommendationEngineV2
from app.engines.simulation_engine import SimulationEngine
from app.storage import store

router = APIRouter(prefix="/api", tags=["Recovery Plans"])


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
    return RecommendationEngineV2(project_state=project_state, simulation_count=1000, scoring_weights=ScoringWeights())


def _build_recovery_plan_engine(session_id: str) -> tuple[RecoveryPlanEngine, RecommendationEngineV2]:
    """Build a RecoveryPlanEngine with all upstream components."""
    recommendation_engine = _build_engine(session_id)
    
    # Get upstream engine outputs (metrics, dag, critical path, etc.)
    upstream = recommendation_engine._compute_upstream()
    
    # Build SimulationEngine with upstream outputs
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
    
    # Build RecoveryPlanEngine
    recovery_plan_engine = RecoveryPlanEngine(simulation_engine=simulation_engine)
    
    return recovery_plan_engine, recommendation_engine


@router.get("/recovery-plans")
async def get_recovery_plans(
    session_id: str = Query(..., description="Session ID"),
) -> Dict:
    """
    Generate and return all three recovery plans (SAFE, AGGRESSIVE, MINIMAL_DISRUPTION).
    
    Plans are ranked by composite_score descending. The highest-scoring plan is labeled
    "Recommended". All plans include their scores, explanations, and revised sprint plans.
    """
    try:
        session_id = session_id.strip()
        
        # Build engines
        recovery_plan_engine, recommendation_engine = _build_recovery_plan_engine(session_id)
        
        # Generate recommendations (input for recovery plan generation)
        recommendations = recommendation_engine.generate(top_n=20)  # Get more candidates to work with
        if not recommendations:
            raise HTTPException(
                status_code=400,
                detail=ApiResponse(
                    success=False,
                    error_code=ErrorCodes.INVALID_REQUEST,
                    message="No recommendations available to build recovery plans",
                ).model_dump(mode="json"),
            )
        
        # Generate recovery plans
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
        
        # Convert to API response format
        plan_responses = [_recovery_plan_to_response(plan) for plan in recovery_plans]
        
        # Build summary
        top_plan = plan_responses[0]
        summary = f"{top_plan.label} plan: {top_plan.score.actions_required} actions, {round(top_plan.score.deadline_probability * 100, 1)}% deadline probability"
        
        response = RecoveryPlansListResponse(
            plans=plan_responses,
            summary=summary,
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
    """
    try:
        session_id = session_id.strip()
        plan_id = plan_id.strip()
        
        # Build engines
        recovery_plan_engine, recommendation_engine = _build_recovery_plan_engine(session_id)
        
        # Generate recommendations
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
        
        # Generate all recovery plans
        recovery_plans = recovery_plan_engine.generate_recovery_plans(recommendations=recommendations)
        
        # Find the requested plan
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
        
        # Convert to response format
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
    
    This applies all actions in the plan to the actual session state (not a clone).
    Internally, this calls ActionApplicator.apply_many() on the real ProjectState.
    """
    try:
        session_id = request.session_id.strip()
        plan_id = request.plan_id.strip()
        
        # Get project state
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
        
        # Build engines to find the plan
        recovery_plan_engine, recommendation_engine = _build_recovery_plan_engine(session_id)
        recommendations = recommendation_engine.generate(top_n=20)
        recovery_plans = recovery_plan_engine.generate_recovery_plans(recommendations=recommendations)
        
        # Find the plan to apply
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
        
        # Apply the plan to a deep-cloned project state so the simulation engine remains deterministic
        updated_state = project_state.model_copy(deep=True)
        recovery_plan_engine.simulation_engine.applicator.apply_many(updated_state, plan_to_apply.actions)
        session.project_state = updated_state
        # session.project_state was just replaced wholesale — any ProjectAnalysis
        # cached against the old state is now describing a project that no
        # longer exists. Invalidate so the next get_analysis() call (from this
        # or any other route: /forecast, /risk, /recommendations, ...)
        # rebuilds against the post-plan state instead of silently returning
        # pre-plan numbers.
        store.invalidate_analysis(session_id)
        
        response = ApplyPlanResponse(
            success=True,
            applied_plan_id=plan_id,
            message=f"Recovery plan {plan_id} ({plan_to_apply.archetype.value}) applied successfully. "
                    f"{len(plan_to_apply.actions)} actions were applied to the session state.",
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


# Helper functions to convert internal models to API response models


def _recovery_plan_to_response(plan) -> RecoveryPlanResponse:
    """Convert RecoveryPlan to API response format."""
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
    """Convert Recommendation to simplified API format for plan view."""
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
    """Convert RecoveryPlanScore to API response format."""
    return RecoveryPlanScoreResponse(
        deadline_probability=round(score.deadline_probability, 4),
        expected_delay_days=round(score.expected_delay_days, 2),
        overall_risk_score=round(score.overall_risk_score, 4),
        actions_required=score.actions_required,
        execution_complexity=score.execution_complexity,
        composite_score=round(score.composite_score, 4),
    )


def _explanation_to_response(explanation) -> RecoveryPlanExplanationResponse:
    """Convert RecoveryPlanExplanation to API response format."""
    return RecoveryPlanExplanationResponse(
        plan_id=explanation.plan_id,
        why_recommended=explanation.why_recommended,
        comparison_to_alternatives=explanation.comparison_to_alternatives,
        trade_offs=[TradeOffResponse(description=t.description, severity=t.severity) for t in explanation.trade_offs],
        narrative_summary=explanation.narrative_summary,
    )
