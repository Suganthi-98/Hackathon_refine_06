from typing import Dict, List, Tuple, Any

from app.engines.recommendation_engine.models import RecommendationAction


# Map each RecommendationAction to the forecast-facing fields it intends to change.
# Paths are high-level and interpreted by the applicator check (see helper below).
FORECAST_LEVER_MAP: Dict[RecommendationAction, List[str]] = {
    RecommendationAction.RESOLVE_BLOCKER: [
        "blocker.status",
        "work_item.remaining_effort_hrs",
    ],
    RecommendationAction.CROSS_TRAIN_BACKUP: [
        "resource.primary_skill",
        "sprint.planned_velocity_hrs",
    ],
    RecommendationAction.SPLIT_ITEM: [
        "work_item.remaining_effort_hrs",
        "work_item.current_estimate_hrs",
    ],
    RecommendationAction.SWARM_ITEM: [
        "sprint.planned_velocity_hrs",
        "work_item.remaining_effort_hrs",
    ],
    RecommendationAction.REMOVE_DEPENDENCY_BOTTLENECK: [
        "dependency.lag_days",
    ],
    RecommendationAction.REASSIGN_ITEM: [
        "work_item.assigned_resource",
        "resource.allocation_pct",
    ],
    RecommendationAction.PARALLELIZE_ITEMS: [
        "dependency.lag_days",
        "work_item.current_estimate_hrs",
    ],
    RecommendationAction.REBALANCE_SPRINT_LOAD: [
        "work_item.assigned_sprint",
        "resource.allocation_pct",
    ],
    RecommendationAction.ADD_RESOURCE_SKILL: [
        "resource.primary_skill",
        "resource.availability_pct",
    ],
    RecommendationAction.REBASELINE_ESTIMATE: [
        "work_item.current_estimate_hrs",
    ],
    RecommendationAction.PAIR_REVIEWER: [
        "work_item.remaining_effort_hrs",
    ],
    RecommendationAction.ESCALATE_BLOCKER_EARLY: [
        "blocker.target_resolution_date",
        "work_item.remaining_effort_hrs",
    ],
    RecommendationAction.PULL_FORWARD_ITEM: [
        "work_item.assigned_sprint",
    ],
    RecommendationAction.INSERT_REVIEW_GATE: [
        "work_item.remaining_effort_hrs",
    ],
    RecommendationAction.APPLY_RAMP_UP_DISCOUNT: [
        "sprint.planned_velocity_hrs",
    ],
    RecommendationAction.RESEQUENCE_NON_CRITICAL_ITEM: [
        "dependency.lag_days",
    ],
    RecommendationAction.SPLIT_AND_PAIR: [
        "work_item.current_estimate_hrs",
    ],
    RecommendationAction.ASSIGN_AS_SECOND_REVIEWER: [
        "work_item.remaining_effort_hrs",
    ],
    RecommendationAction.FREEZE_SCOPE_REQUEST: [
        "work_item.current_estimate_hrs",
    ],
}


def _path_is_list_root(path: str) -> str:
    # return the root collection name (work_item, resource, sprint, blocker, dependency)
    return path.split(".")[0] if path and "." in path else path


def sample_lever_values(state_obj: Any, rec: Any, path: str):
    """Return a tuple of values for the given lever path scoped to the recommendation.

    For collection-rooted paths like "work_item.remaining_effort_hrs" we only sample
    the items referenced by the recommendation (e.g. affected_item_ids). For
    scalar paths (if present) we attempt a direct attribute access.
    """
    root = _path_is_list_root(path)
    attr = path.split(".", 1)[1] if "." in path else None

    try:
        if root == "work_item" and attr:
            ids = getattr(rec, "affected_item_ids", []) or []
            if not ids:
                return tuple()
            values = []
            for wi in getattr(state_obj, "work_items", []):
                if wi.item_id not in ids:
                    continue
                values.append(getattr(wi, attr, None))
            return tuple(values)

        if root == "resource" and attr:
            ids = getattr(rec, "affected_resource_ids", []) or []
            if not ids:
                return tuple()
            values = []
            for r in getattr(state_obj, "team", []):
                if r.resource_id not in ids:
                    continue
                values.append(getattr(r, attr, None))
            return tuple(values)

        if root == "sprint" and attr:
            ids = getattr(rec, "affected_sprint_ids", []) or []
            if not ids:
                return tuple()
            values = []
            for s in getattr(state_obj, "sprints", []):
                if s.sprint_name not in ids:
                    continue
                values.append(getattr(s, attr, None))
            return tuple(values)

        if root == "blocker" and attr:
            ids = getattr(rec, "affected_blocker_ids", []) or []
            if not ids:
                return tuple()
            values = []
            for b in getattr(state_obj, "blockers", []):
                if b.blocker_id not in ids:
                    continue
                values.append(getattr(b, attr, None))
            return tuple(values)

        if root == "dependency" and attr:
            # sample dependencies that touch any affected item
            ids = set(getattr(rec, "affected_item_ids", []) or [])
            if not ids:
                return tuple()
            values = []
            for d in getattr(state_obj, "dependencies", []):
                if not (d.predecessor_item_id in ids or d.successor_item_id in ids):
                    continue
                values.append(getattr(d, attr, None))
            return tuple(values)

        # Fallback: attempt to get attribute directly
        if hasattr(state_obj, path):
            return (getattr(state_obj, path),)
    except Exception:
        return tuple()

    return tuple()
