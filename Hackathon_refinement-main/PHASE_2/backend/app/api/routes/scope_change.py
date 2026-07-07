"""
Scope Change API Routes

POST /api/scope-change - apply a scope reduction to one or more work items.

CONFIRMATION WORKFLOW (for UI implementation):
1. User selects items to remove from scope
2. Call: POST /api/scope-change?session_id=X&dry_run=true&item_ids=[...]
   → Returns: ScopeChangeResponse with dry_run=true (NO session mutation)
   → UI shows preview: new forecast, risk, effort changes
3. User confirms action
4. Call: POST /api/scope-change?session_id=X&dry_run=false&item_ids=[...]
   → Returns: ScopeChangeResponse with dry_run=false (session PERSISTED)
   → UI shows confirmation: changes applied

This prevents accidental scope reduction without user confirmation.
"""

from fastapi import APIRouter, HTTPException, Query

from app.api.models import ApiResponse, ErrorCodes
from app.api.models_phase3 import ScopeChangeRequest, ScopeChangeResponse
from app.domain.models import WorkItemStatus
from app.engines.project_analysis import ProjectAnalysis
from app.storage import store


router = APIRouter(prefix="/api", tags=["Scope"])


@router.post("/scope-change")
async def apply_scope_change(
    session_id: str = Query(..., description="Session ID"),
    dry_run: bool = Query(False, description="Preview changes without persisting (default: False)"),
    request: ScopeChangeRequest = ..., 
):
    """
    Apply a scope reduction and return updated forecast/risk analysis.
    
    Args:
        session_id: Session identifier
        dry_run: If True, preview changes without persisting to session
        request: ScopeChangeRequest with item_ids and optional reason
        
    Returns:
        ScopeChangeResponse with updated metrics (no changes if dry_run=True)
    """

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

        project_state = session.project_state
        work_item_map = {item.item_id: item for item in project_state.work_items}
        descoped_item_ids = []

        # Validate all items exist before making any changes
        for item_id in request.item_ids:
            if item_id not in work_item_map:
                raise HTTPException(
                    status_code=404,
                    detail=ApiResponse(
                        success=False,
                        error_code=ErrorCodes.VALIDATION_ERROR,
                        message=f"Work item {item_id} not found in session {session_id}",
                    ).model_dump(),
                )

        # For dry_run, clone the project state so nothing here ever touches
        # the session's real ProjectState or its cached ProjectAnalysis.
        if dry_run:
            from copy import deepcopy
            project_state = deepcopy(project_state)
            work_item_map = {item.item_id: item for item in project_state.work_items}

        for item_id in request.item_ids:
            work_item = work_item_map.get(item_id)
            if work_item:
                work_item.status = WorkItemStatus.COMPLETED
                work_item.is_scope_changed = True
                work_item.scope_change_reason = request.reason or "Scope reduced via API"
                work_item.remaining_effort_hrs = 0.0
                work_item.current_estimate_hrs = 0.0
                work_item.actual_effort_hrs = min(work_item.actual_effort_hrs, work_item.estimated_effort_hrs)
                work_item.progress_pct = 1.0
                descoped_item_ids.append(item_id)

        if dry_run:
            # Preview only: compute a fresh, one-off ProjectAnalysis on the
            # clone. Never touches store, never touches the cached analysis.
            analysis = ProjectAnalysis.build(project_state, simulation_count=1000)
        else:
            # Confirmed: project_state here IS session.project_state (same
            # object, mutated in place above), so the session's cached
            # ProjectAnalysis is now stale. Invalidate before rebuilding so
            # every other route (forecast, risk, recommendations, ...) picks
            # up these changes on its next read instead of serving old
            # numbers computed before the scope change.
            session.descoped_item_ids.update(descoped_item_ids)
            store.invalidate_analysis(session_id)
            analysis = store.get_analysis(session_id, simulation_count=1000)

        response = ScopeChangeResponse(
            session_id=session_id,
            project_name=project_state.project_info.project_name,
            dry_run=dry_run,
            descoped_item_ids=descoped_item_ids,
            changed_item_count=len(descoped_item_ids),
            updated_remaining_effort_hours=analysis.metrics.remaining_effort_hours,
            forecast=analysis.forecast,
            risk_analysis=analysis.risk_result,
        )

        mode_msg = "(preview)" if dry_run else "(confirmed)"
        return ApiResponse(
            success=True,
            data=response.model_dump(),
            message=f"Scope change applied {mode_msg}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ApiResponse(
                success=False,
                error_code=ErrorCodes.INTERNAL_ERROR,
                message=f"Error applying scope change: {str(e)}",
            ).model_dump(),
        )