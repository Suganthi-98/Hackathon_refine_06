"""
Phase 3 Forecast API Route

GET /forecast - deterministic single-point forecast

Reads from the session-level ProjectAnalysis (computed once per session)
so the forecast numbers are always consistent with recommendations,
risk, and Monte Carlo — they all share the same engine outputs.
"""
from fastapi import APIRouter, HTTPException, Query
from app.storage import store
from app.api.models import ApiResponse, ErrorCodes
from app.api.models_phase3 import ForecastResponse

router = APIRouter(prefix="/api", tags=["Phase3"])


@router.get("/forecast")
async def get_forecast(session_id: str = Query(..., description="Session ID")):
    """Return the deterministic forecast for the session."""
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

        response = ForecastResponse(
            session_id=session_id,
            project_name=analysis.project_state.project_info.project_name,
            forecast=analysis.forecast,
        )

        return ApiResponse(
            success=True,
            data=response.model_dump(),
            message="Forecast generated",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.PROCESSING_ERROR,
                message=f"Error calculating forecast: {str(e)}",
            ).model_dump(),
        )
