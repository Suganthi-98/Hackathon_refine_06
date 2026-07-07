"""
Phase 3.2 Monte Carlo API Route

GET /monte-carlo - probabilistic forecast with confidence intervals

Default (no query params): returns the cached session-level Monte Carlo result
so numbers agree exactly with /forecast, /risk, and /recommendations.

With explicit simulations/seed params: re-runs a fresh simulation for custom
what-if exploration (does NOT invalidate the session cache).
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.storage import store
from app.api.models import ApiResponse, ErrorCodes
from app.api.models_phase3 import MonteCarloResponse

router = APIRouter(prefix="/api", tags=["Phase3.2"])


@router.get("/monte-carlo")
async def get_monte_carlo(
    session_id: str = Query(..., description="Session ID"),
    simulations: Optional[int] = Query(
        None,
        description=(
            "Number of simulations for a custom run (100–100000). "
            "Omit to use the cached session result (consistent with all other endpoints)."
        ),
        ge=100,
        le=100_000,
    ),
    seed: Optional[int] = Query(
        None,
        description="Random seed for a custom run (optional). Ignored when simulations is omitted.",
    ),
):
    """Return a probabilistic forecast using Monte Carlo simulation.

    Default behaviour (simulations omitted)
    ----------------------------------------
    Returns the same Monte Carlo result that was used to compute /forecast,
    /risk, and /recommendations — i.e. the numbers are guaranteed consistent.

    Custom run (simulations provided)
    -----------------------------------
    Runs a fresh simulation with the requested count and optional seed.
    Useful for what-if exploration; does NOT affect the session cache.

    Key principle: Target end date is NEVER modified.  It is a fixed business
    commitment used only for probability calculation.
    """
    try:
        if simulations is None:
            # ── Default: return the cached session result ──────────────────
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

            response = MonteCarloResponse(
                session_id=session_id,
                project_name=analysis.project_state.project_info.project_name,
                monte_carlo=analysis.monte_carlo,
            )
            return ApiResponse(
                success=True,
                data=response.model_dump(),
                message="Monte Carlo analysis (cached session result)",
            )

        else:
            # ── Custom run: fresh simulation, does not update cache ────────
            from app.engines.metrics_engine import MetricsEngine
            from app.engines.dependency_engine import DependencyGraphEngine
            from app.engines.critical_path_engine import CriticalPathEngine
            from app.engines.spillover_engine import SpilloverAnalysisEngine
            from app.engines.monte_carlo_engine import MonteCarloEngine

            project_state = store.get_project_state(session_id)
            if not project_state:
                raise HTTPException(
                    status_code=404,
                    detail=ApiResponse(
                        success=False,
                        error_code=ErrorCodes.SESSION_NOT_FOUND,
                        message=f"Session {session_id} not found",
                    ).model_dump(),
                )

            # Reuse cached upstream where possible to avoid redundant work.
            analysis = store.get_analysis(session_id)
            if analysis:
                metrics = analysis.metrics
                cp_result = analysis.cp_result
                spillover = analysis.spillover
            else:
                metrics = MetricsEngine(project_state).calculate()
                dag = DependencyGraphEngine(project_state).build_dag()
                cp_result = CriticalPathEngine(project_state, dag).analyze()
                spillover = SpilloverAnalysisEngine(
                    project_state, metrics.average_item_effort
                ).analyze()

            monte_carlo_result = MonteCarloEngine(
                project_state=project_state,
                metrics=metrics,
                cp_result=cp_result,
                spillover=spillover,
                simulation_count=simulations,
                seed=seed,
            ).calculate()

            response = MonteCarloResponse(
                session_id=session_id,
                project_name=project_state.project_info.project_name,
                monte_carlo=monte_carlo_result,
            )
            return ApiResponse(
                success=True,
                data=response.model_dump(),
                message=f"Monte Carlo analysis ({simulations} simulations, custom run)",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.PROCESSING_ERROR,
                message=f"Error calculating Monte Carlo: {str(e)}",
            ).model_dump(),
        )
