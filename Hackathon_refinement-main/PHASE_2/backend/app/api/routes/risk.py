"""
Phase 3.3 Risk Engine API Route

GET /api/risk - comprehensive project risk analysis

Reads from the session-level ProjectAnalysis (computed once per session)
so risk numbers are always consistent with forecast and recommendations.
"""
from fastapi import APIRouter, HTTPException, Query
from app.storage import store
from app.api.models import ApiResponse, ErrorCodes
from app.api.models_phase3 import RiskResponse

router = APIRouter(prefix="/api", tags=["Phase3.3"])


@router.get("/risk")
async def get_risk_analysis(
    session_id: str = Query(..., description="Session ID"),
):
    """
    Return comprehensive risk analysis for the project.

    This endpoint answers: Why is this project at risk?

    The Risk Engine:
    - Analyzes project data from all previous engines
    - Calculates 4 sub-risk scores (schedule, dependency, resource, scope)
    - Computes overall risk using weighted aggregation
    - Identifies top risk drivers with explanations
    - Provides sprint-level risk breakdown
    - Is deterministic (no randomness beyond Monte Carlo)

    Args:
        session_id: Session ID

    Returns:
        RiskResponse with:
        - overall_risk_score: 0-100 weighted risk score
        - risk_level: LOW, MODERATE, HIGH, VERY_HIGH, or CRITICAL
        - schedule_risk, dependency_risk, resource_risk, scope_risk: Sub-scores
        - top_risk_drivers: Top 10 drivers, ranked by impact
        - sprint_risks: Per-sprint risk analysis
    """
    try:
        analysis = store.get_analysis(session_id)
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=ApiResponse(
                    success=False,
                    error_code=ErrorCodes.SESSION_NOT_FOUND,
                    message=f"Session {session_id} not found",
                ).model_dump(),
            )

        response = RiskResponse(
            session_id=session_id,
            project_name=analysis.project_state.project_info.project_name,
            risk_analysis=analysis.risk_result,
        )

        return ApiResponse(
            success=True,
            data=response.model_dump(),
            message="Risk analysis complete",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.INTERNAL_ERROR,
                message=f"Error calculating risk analysis: {str(e)}",
            ).model_dump(),
        )
