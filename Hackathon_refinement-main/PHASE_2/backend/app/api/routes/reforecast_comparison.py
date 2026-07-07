"""
Reforecast Comparison API Route  ← THE MONEY SHOT

GET /api/reforecast-comparison

Returns a side-by-side snapshot of three scenarios:
  baseline   – numbers from the shared session ProjectAnalysis (the single truth)
  current    – same as baseline (ProjectState has not been mutated)
  after_rec  – result of the last simulate-recommendation call (stored on session)

Using the shared ProjectAnalysis means these numbers always agree with what
/forecast, /risk, and /recommendations return — previously this endpoint ran
its own independent pipeline so values could diverge.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any

from app.api.models import ApiResponse, ErrorCodes
from app.storage import store

router = APIRouter(prefix="/api", tags=["Reforecast"])


def _snapshot_from_analysis(analysis) -> Dict[str, Any]:
    """Extract the standard comparison snapshot from a ProjectAnalysis."""
    mc = analysis.monte_carlo
    forecast = analysis.forecast
    risk = analysis.risk_result

    p50 = mc.most_likely_finish_date.isoformat() if mc.most_likely_finish_date else None
    p80 = mc.p80_finish_date.isoformat() if mc.p80_finish_date else None
    p95 = mc.p95_finish_date.isoformat() if mc.p95_finish_date else None
    target = mc.target_end_date.isoformat() if mc.target_end_date else None

    return {
        "on_time_probability": round(mc.on_time_probability * 100, 1),
        "on_time_risk_level": (
            mc.on_time_risk_level.value
            if hasattr(mc.on_time_risk_level, "value")
            else str(mc.on_time_risk_level)
        ),
        "expected_delay_days": round(forecast.expected_delay_days, 1),
        "overall_risk_score": round(risk.overall_risk_score, 1),
        "p50_date": p50,
        "p80_date": p80,
        "p95_date": p95,
        "target_end_date": target,
    }


@router.get("/reforecast-comparison")
async def get_reforecast_comparison(
    session_id: str = Query(..., description="Session ID"),
):
    """Return side-by-side baseline / current / post-recommendation snapshots."""
    try:
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=ApiResponse(
                    success=False,
                    error_code=ErrorCodes.SESSION_NOT_FOUND,
                    message=f"Session {session_id} not found",
                ).model_dump(),
            )

        # Both baseline and current come from the shared session analysis —
        # no independent engine runs, so numbers are guaranteed consistent.
        analysis = store.get_analysis(session_id)
        baseline = _snapshot_from_analysis(analysis)
        current = baseline.copy()

        after_rec_raw = getattr(session, "last_simulation_result", None)

        if after_rec_raw:
            after_rec = {
                "on_time_probability": round(
                    float(
                        after_rec_raw.get(
                            "after_probability",
                            after_rec_raw.get("baseline_probability", 0),
                        )
                    ) * 100,
                    1,
                ),
                "on_time_risk_level": "IMPROVED",
                "expected_delay_days": round(
                    float(
                        after_rec_raw.get(
                            "after_delay_days",
                            after_rec_raw.get("baseline_delay_days", 0),
                        )
                    ),
                    1,
                ),
                "overall_risk_score": round(
                    float(
                        after_rec_raw.get(
                            "after_risk_score",
                            after_rec_raw.get("baseline_risk_score", 0),
                        )
                    ),
                    1,
                ),
                "p50_date": baseline.get("p50_date"),
                "p80_date": baseline.get("p80_date"),
                "p95_date": baseline.get("p95_date"),
                "target_end_date": baseline.get("target_end_date"),
                "recommendation_id": after_rec_raw.get("recommendation_id"),
                "summary": after_rec_raw.get("summary", ""),
            }
        else:
            after_rec = {**baseline, "on_time_risk_level": "NO_SIMULATION_YET"}

        prob_delta = round(after_rec["on_time_probability"] - baseline["on_time_probability"], 1)
        delay_delta = round(baseline["expected_delay_days"] - after_rec["expected_delay_days"], 1)
        risk_delta = round(baseline["overall_risk_score"] - after_rec["overall_risk_score"], 1)

        data = {
            "session_id": session_id,
            "project_name": session.project_state.project_info.project_name,
            "baseline": baseline,
            "current": current,
            "after_recommendation": after_rec,
            "deltas": {
                "probability_gain_pct": prob_delta,
                "days_saved": delay_delta,
                "risk_score_reduction": risk_delta,
                "has_improvement": prob_delta > 0 or delay_delta > 0,
            },
        }

        return ApiResponse(success=True, data=data, message="Reforecast comparison generated")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.PROCESSING_ERROR,
                message=f"Error generating reforecast comparison: {str(e)}",
            ).model_dump(),
        )
