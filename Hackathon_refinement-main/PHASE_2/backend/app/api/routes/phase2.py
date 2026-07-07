"""
Phase 2 API Routes

GET /metrics     - Project metrics
GET /dependencies - Dependency graph analysis
GET /spillover   - Spillover analysis

All three endpoints read from the session-level ProjectAnalysis (computed
once per session) so numbers are always consistent across the whole API.
"""

from fastapi import APIRouter, HTTPException, Query
from app.storage import store
from app.api.models import ApiResponse, ErrorCodes
from app.api.models_phase2 import MetricsResponse, DependenciesResponse, SpilloverResponse

router = APIRouter(prefix="/api", tags=["Phase2"])


def _get_analysis(session_id: str):
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
    return analysis


@router.get("/metrics")
async def get_metrics(session_id: str = Query(..., description="Session ID")):
    """
    Get project metrics for a session.

    Returns aggregated metrics including completion %, velocity, team utilization, and risk indicators.
    """
    try:
        analysis = _get_analysis(session_id)
        metrics = analysis.metrics
        project_state = analysis.project_state

        response = MetricsResponse(
            session_id=session_id,
            project_name=project_state.project_info.project_name,
            total_items=metrics.total_items,
            completed_items=metrics.completed_items,
            in_progress_items=metrics.in_progress_items,
            blocked_items=metrics.blocked_items,
            completion_pct=metrics.completion_pct,
            total_effort_hours=metrics.total_effort_hours,
            remaining_effort_hours=metrics.remaining_effort_hours,
            completed_effort_hours=metrics.completed_effort_hours,
            planned_total_velocity=metrics.planned_total_velocity,
            actual_avg_velocity=metrics.actual_avg_velocity,
            velocity_variance=metrics.velocity_variance,
            team_size=metrics.team_size,
            avg_allocation_pct=metrics.avg_allocation_pct,
            avg_availability_pct=metrics.avg_availability_pct,
            underutilized_count=metrics.underutilized_resource_count,
            active_blocker_count=metrics.active_blocker_count,
            blocker_velocity_impact=metrics.estimated_blocker_velocity_impact,
            current_sprint_number=metrics.current_sprint_number,
            completed_sprints=metrics.completed_sprints,
            dependency_count=metrics.dependency_count,
            expected_spillover_items=int(metrics.expected_spillover_items),
        )

        return ApiResponse(success=True, data=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.PROCESSING_ERROR,
                message=f"Error calculating metrics: {str(e)}",
            ).model_dump(),
        )


@router.get("/dependencies")
async def get_dependencies(session_id: str = Query(..., description="Session ID")):
    """
    Get dependency graph analysis and critical path.

    Returns DAG structure, critical path, risk items, and blocker impacts.
    """
    try:
        analysis = _get_analysis(session_id)
        project_state = analysis.project_state
        dag = analysis.dag
        cp_result = analysis.cp_result
        risk_scores = analysis.impact_scores

        # Identify blocked items
        blocked_items = set()
        for blocker in project_state.blockers:
            if not blocker.actual_resolution_date:
                blocked_items.update(blocker.impacted_item_ids)

        # Build detailed critical path payload
        work_items_by_id = {wi.item_id: wi for wi in project_state.work_items}
        critical_path_details = []
        for item_id in cp_result.critical_path:
            work_item = work_items_by_id.get(item_id)
            if work_item is None:
                raise ValueError(
                    f"Critical path item '{item_id}' exists in DAG but was not found in project work items."
                )
            critical_path_details.append({
                "item_id": item_id,
                "name": work_item.title,
                "effort_hours": work_item.current_estimate_hrs,
                "float_hours": cp_result.item_slack_map.get(item_id, 0.0),
                "sprint_id": work_item.assigned_sprint,
            })

        response = DependenciesResponse(
            session_id=session_id,
            project_name=project_state.project_info.project_name,
            total_items=len(dag.all_nodes),
            total_dependencies=len(project_state.dependencies),
            has_cycles=dag.has_cycles,
            critical_path=cp_result.critical_path,
            critical_path_items=cp_result.critical_path_items,
            critical_path_details=critical_path_details,
            critical_path_duration_hours=cp_result.critical_path_duration_hours,
            critical_path_duration_hours_original=cp_result.critical_path_duration_hours_original,
            critical_path_growth_hours=cp_result.critical_path_growth_hours,
            critical_path_growth_percent=cp_result.critical_path_growth_percent,
            critical_path_duration_days=cp_result.critical_path_duration_days,
            critical_path_item_count=len(cp_result.critical_path),
            total_work_items=len(project_state.work_items),
            total_float_hours=sum(cp_result.item_slack_map.values()),
            high_risk_items=risk_scores.high_risk_items,
            medium_risk_items=risk_scores.medium_risk_items,
            low_risk_items=risk_scores.low_risk_items,
            active_blockers=[b.blocker_id for b in project_state.blockers if not b.actual_resolution_date],
            items_blocked=list(blocked_items),
            zero_slack_items=cp_result.items_on_critical_path,
        )

        return ApiResponse(success=True, data=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.PROCESSING_ERROR,
                message=f"Error analyzing dependencies: {str(e)}",
            ).model_dump(),
        )


@router.get("/spillover")
async def get_spillover(session_id: str = Query(..., description="Session ID")):
    """
    Get spillover analysis and capacity predictions.

    Returns spillover probabilities per item, predicted carryover by sprint, and capacity utilization.
    """
    try:
        analysis = _get_analysis(session_id)
        project_state = analysis.project_state
        spillover_analysis = analysis.spillover

        # Determine overall risk level
        high_risk_count = len(spillover_analysis.high_spillover_risk_items)
        total_expected = sum(spillover_analysis.predicted_spillover_by_sprint.values())

        if high_risk_count >= 5 or total_expected >= 10:
            risk_level = "High"
        elif high_risk_count >= 2 or total_expected >= 5:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        confidence_dict = {
            str(k): (v[0], v[1])
            for k, v in spillover_analysis.spillover_confidence_intervals.items()
        }

        response = SpilloverResponse(
            session_id=session_id,
            project_name=project_state.project_info.project_name,
            high_spillover_risk_items=spillover_analysis.high_spillover_risk_items,
            high_risk_count=high_risk_count,
            predicted_spillover_by_sprint={
                int(k): v for k, v in spillover_analysis.predicted_spillover_by_sprint.items()
            },
            confidence_intervals=confidence_dict,
            sprint_utilization_pct={
                int(k): v for k, v in spillover_analysis.sprint_utilization_pct.items()
            },
            historical_carryover_rate=spillover_analysis.historical_carryover_rate,
            historical_carryover_std_dev=spillover_analysis.historical_carryover_std_dev,
            total_expected_spillover=total_expected,
            risk_level=risk_level,
        )

        return ApiResponse(success=True, data=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.PROCESSING_ERROR,
                message=f"Error analyzing spillover: {str(e)}",
            ).model_dump(),
        )
