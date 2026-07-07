from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from datetime import timedelta

from app.domain.models import Blocker, Priority, ProjectState, SprintStatus, WorkItemStatus
from app.engines.critical_path_engine import CriticalPathResult
from app.engines.dependency_engine import DependencyDAG
from app.engines.forecast_engine import ForecastResult
from app.engines.impact_scoring_engine import RiskScores
from app.engines.metrics_engine import ProjectMetrics
from app.engines.monte_carlo_engine import MonteCarloResult
from app.engines.risk_engine import RiskResult
from app.engines.spillover_engine import SpilloverAnalysis
from app.engines.recommendation_engine.models import (
    HistoricalPattern,
    OpportunitySignal,
    SignalCategory,
    SignalEvidence,
    SignalSeverity,
    historical_pattern_payload,
    signal_id,
)


class BlockerDetector:
    def __init__(
        self,
        project_state: ProjectState,
        cp_result: CriticalPathResult,
        dag: DependencyDAG,
        impact_scores: RiskScores,
    ) -> None:
        self.project_state = project_state
        self.cp_result = cp_result
        self.dag = dag
        self.impact_scores = impact_scores

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        active_blockers = [b for b in self.project_state.blockers if not getattr(b, "actual_resolution_date", None)]
        if not active_blockers:
            return signals

        for blocker in active_blockers:
            impacted_ids = list(getattr(blocker, "impacted_item_ids", []) or [])
            cascade_ids = self._cascade_item_ids(impacted_ids)
            blocked_hours = sum(
                float(next((wi.remaining_effort_hrs for wi in self.project_state.work_items if wi.item_id == item_id), 0.0))
                for item_id in impacted_ids
            )
            on_cp = any(item_id in self.cp_result.items_on_critical_path for item_id in impacted_ids)
            days_overdue = self._days_overdue(blocker)
            severity = SignalSeverity.CRITICAL if on_cp else SignalSeverity.HIGH
            if not on_cp and len(cascade_ids) < 3:
                severity = SignalSeverity.MEDIUM
            if days_overdue > 0:
                severity = SignalSeverity.CRITICAL

            context: Dict[str, Any] = {
                "blocker_id": blocker.blocker_id,
                "category": getattr(blocker, "category", None),
                "severity": getattr(blocker, "severity", None),
                "impacted_item_ids": impacted_ids,
                "cascade_item_ids": cascade_ids,
                "blocked_hours": round(blocked_hours, 2),
                "on_critical_path": on_cp,
                "days_until_target_resolution": self._days_until_resolution(blocker),
                "days_overdue": days_overdue,
                "sprint_gate_pct": round(blocked_hours / max(1.0, self._sprint_capacity_hours()), 4),
                "affected_sprint_numbers": self._affected_sprint_numbers(impacted_ids),
            }
            evidence = [
                SignalEvidence(
                    source_engine="critical_path_engine",
                    metric_name="impacted_items_on_cp",
                    metric_value=float(on_cp),
                    threshold=1.0,
                    explanation="Active blocker affects critical path items",
                )
            ]
            signal = OpportunitySignal(
                signal_id=signal_id(SignalCategory.BLOCKER, [blocker.blocker_id]),
                category=SignalCategory.BLOCKER,
                severity=severity,
                affected_item_ids=impacted_ids,
                affected_resource_ids=[],
                affected_sprint_ids=self._affected_sprint_ids(impacted_ids),
                affected_blocker_ids=[blocker.blocker_id],
                evidence=evidence,
                context=context,
                detected_at=datetime.now(timezone.utc).isoformat(),
            )
            signals.append(signal)
        return signals

    def _cascade_item_ids(self, impacted_item_ids: List[str]) -> List[str]:
        cascade: set[str] = set()
        for item_id in impacted_item_ids:
            for descendant in self.dag.transitive_closure.get(item_id, []):
                cascade.add(descendant)
        return sorted(cascade)

    def _days_until_resolution(self, blocker: Blocker) -> int:
        target = getattr(blocker, "target_resolution_date", None)
        raised = getattr(blocker, "raised_date", None)
        if not target or not raised:
            return 0
        return max(0, (target - raised).days)

    def _days_overdue(self, blocker: Blocker) -> int:
        """Days past target_resolution_date. 0 if not overdue or no target set."""
        target = getattr(blocker, "target_resolution_date", None)
        if not target:
            return 0
        today = datetime.now(timezone.utc)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        return max(0, (today - target).days)

    def _sprint_capacity_hours(self) -> float:
        sprint = next((s for s in self.project_state.sprints if getattr(s, "status", None) == SprintStatus.IN_PROGRESS), None)
        if sprint:
            return float(getattr(sprint, "planned_velocity_hrs", 0.0) or 0.0)
        return max(1.0, float(sum(getattr(s, "planned_velocity_hrs", 0.0) or 0.0 for s in self.project_state.sprints)))

    def _affected_sprint_numbers(self, affected_item_ids: List[str]) -> List[int]:
        sprint_numbers = []
        for item_id in affected_item_ids:
            work_item = next((wi for wi in self.project_state.work_items if wi.item_id == item_id), None)
            if work_item and getattr(work_item, "assigned_sprint", None):
                sprint = next((s for s in self.project_state.sprints if s.sprint_id == work_item.assigned_sprint), None)
                if sprint is not None:
                    sprint_numbers.append(sprint.sprint_number)
        return sorted(set(sprint_numbers))

    def _affected_sprint_ids(self, affected_item_ids: List[str]) -> List[str]:
        sprint_ids = []
        for item_id in affected_item_ids:
            work_item = next((wi for wi in self.project_state.work_items if wi.item_id == item_id), None)
            if work_item and getattr(work_item, "assigned_sprint", None):
                sprint_ids.append(work_item.assigned_sprint)
        return sorted(set(sprint_ids))


class CapacityDetector:
    def __init__(
        self,
        project_state: ProjectState,
        metrics: ProjectMetrics,
        cp_result: CriticalPathResult,
        impact_scores: RiskScores,
    ) -> None:
        self.project_state = project_state
        self.metrics = metrics
        self.cp_result = cp_result
        self.impact_scores = impact_scores

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        sprint_by_name = {s.sprint_name: s.sprint_id for s in self.project_state.sprints}
        for resource in self.project_state.team:
            if resource.resource_id is None:
                continue
            current_sprint_id = self._current_sprint_id()
            sprint_ids_for_resource = self._resource_sprint_ids(resource)
            target_sprints = sprint_ids_for_resource or [current_sprint_id]

            for sprint_id in target_sprints:
                if not sprint_id:
                    continue
                load_ratio = self._load_ratio(resource, sprint_id=sprint_id)
                if 0.4 <= load_ratio <= 1.2:
                    continue
                flag = "OVERLOADED" if load_ratio > 1.2 else "UNDERUTILIZED"
                cp_items_owned = [
                    wi.item_id for wi in self.project_state.work_items
                    if wi.assigned_resource in {resource.resource_id, resource.name}
                    and (wi.assigned_sprint == sprint_id or sprint_by_name.get(wi.assigned_sprint) == sprint_id)
                    and wi.item_id in self.cp_result.items_on_critical_path
                ]
                context: Dict[str, Any] = {
                    "resource_id": resource.resource_id,
                    "sprint_id": sprint_id,
                    "load_ratio": round(load_ratio, 4),
                    "assigned_remaining_hrs": round(self._assigned_remaining_hours(resource.resource_id), 2),
                    "effective_remaining_capacity_hrs": round(self._effective_remaining_capacity(resource), 2),
                    "flag": flag,
                    "cp_items_owned": cp_items_owned,
                    "is_single_owner_of_cp": len(cp_items_owned) == 1,
                    "owns_blocked_cp_items": any(item_id in self._blocked_cp_items() for item_id in cp_items_owned),
                    "is_current_sprint": sprint_id == current_sprint_id,
                }
                evidence = [
                    SignalEvidence(
                        source_engine="metrics_engine",
                        metric_name="resource_sprint_loads",
                        metric_value=load_ratio,
                        threshold=1.2,
                        explanation=f"Resource load ratio in {sprint_id} exceeds the planned threshold",
                    )
                ]
                severity = SignalSeverity.HIGH if context["owns_blocked_cp_items"] else SignalSeverity.MEDIUM
                if flag == "OVERLOADED" and sprint_id == current_sprint_id:
                    severity = SignalSeverity.CRITICAL if load_ratio > 1.15 else SignalSeverity.HIGH
                if flag == "UNDERUTILIZED":
                    severity = SignalSeverity.LOW
                signal = OpportunitySignal(
                    signal_id=signal_id(SignalCategory.CAPACITY, [resource.resource_id, sprint_id]),
                    category=SignalCategory.CAPACITY,
                    severity=severity,
                    affected_item_ids=[
                        wi.item_id for wi in self.project_state.work_items
                        if (wi.assigned_resource == resource.resource_id or wi.assigned_resource == resource.name)
                        and (wi.assigned_sprint == sprint_id or sprint_by_name.get(wi.assigned_sprint) == sprint_id)
                    ],
                    affected_resource_ids=[resource.resource_id],
                    affected_sprint_ids=[sprint_id],
                    affected_blocker_ids=[],
                    evidence=evidence,
                    context=context,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
                signals.append(signal)
        signals.sort(
            key=lambda s: {
                SignalSeverity.CRITICAL: 0,
                SignalSeverity.HIGH: 1,
                SignalSeverity.MEDIUM: 2,
                SignalSeverity.LOW: 3,
            }[s.severity]
        )
        return signals[:8]

    def _load_ratio(self, resource: Any, sprint_id: Optional[str] = None) -> float:
        """
        Per-sprint load ratio when sprint_id is given, otherwise falls back
        to the project-wide ratio for backward compatibility.
        """
        if sprint_id is not None:
            per_sprint = getattr(self.metrics, "resource_sprint_loads", {}) or {}
            per_resource = per_sprint.get(getattr(resource, "name", ""), {})
            if sprint_id in per_resource:
                return per_resource[sprint_id]

        resource_metrics = getattr(self.metrics, "resource_metrics", None)
        dev_metric = None
        if resource_metrics and getattr(resource_metrics, "developer_metrics", None) is not None:
            dev_metric = next(
                (dm for dm in resource_metrics.developer_metrics if getattr(dm, "resource_id", None) == resource.resource_id),
                None
            )
        if dev_metric is None:
            return 0.0
        assigned_hours = dev_metric.remaining_effort_hours
        forecast_input = getattr(self.metrics, "forecast_input_metrics", None)
        remaining_sprints = float(getattr(forecast_input, "remaining_sprints", 1) or 1)
        capacity_per_sprint = float(getattr(resource, "daily_capacity_hrs", 0.0) or 0.0) * (self.project_state.project_info.sprint_duration_days or 10)
        availability = float(getattr(resource, "availability_pct", 1.0) or 1.0)
        allocation = float(getattr(resource, "allocation_pct", 1.0) or 1.0)
        effective_capacity = capacity_per_sprint * remaining_sprints * availability * allocation
        return assigned_hours / max(effective_capacity, 1.0)

    def _assigned_remaining_hours(self, resource_id: str) -> float:
        """
        Get remaining hours assigned to resource from developer metrics.
        
        Consumes from ProjectMetrics.resource_metrics instead of recalculating.
        """
        dev_metric = next(
            (dm for dm in self.metrics.resource_metrics.developer_metrics 
             if dm.resource_id == resource_id),
            None
        )
        return dev_metric.remaining_effort_hours if dev_metric else 0.0

    def _effective_remaining_capacity(self, resource: Any) -> float:
        """
        Calculate effective remaining capacity using forecast_input_metrics.
        
        Consumes from ProjectMetrics.forecast_input_metrics to avoid duplication
        of capacity calculations already done by MetricsEngine.
        """
        remaining_sprints = float(
            self.metrics.forecast_input_metrics.remaining_sprints or 1
        )
        capacity_per_sprint = float(
            resource.daily_capacity_hrs or 0.0 
        ) * (self.project_state.project_info.sprint_duration_days or 10)
        
        # Apply resource availability and allocation
        availability = float(getattr(resource, "availability_pct", 1.0) or 1.0)
        allocation = float(getattr(resource, "allocation_pct", 1.0) or 1.0)
        effective_capacity = capacity_per_sprint * remaining_sprints * availability * allocation
        
        return max(1.0, effective_capacity)

    def _blocked_cp_items(self) -> List[str]:
        active_blockers = [b for b in self.project_state.blockers if not getattr(b, "actual_resolution_date", None)]
        blocked_items = set()
        for blocker in active_blockers:
            blocked_items.update(getattr(blocker, "impacted_item_ids", []) or [])
        return [item_id for item_id in blocked_items if item_id in self.cp_result.items_on_critical_path]

    def _resource_sprint_ids(self, resource: Any) -> List[str]:
        sprint_ids = []
        matches = {getattr(resource, 'resource_id', None), getattr(resource, 'name', None)}
        sprint_by_name = {s.sprint_name: s.sprint_id for s in self.project_state.sprints}
        for wi in self.project_state.work_items:
            if wi.assigned_resource in matches:
                assigned_sprint = wi.assigned_sprint
                if assigned_sprint in sprint_by_name:
                    assigned_sprint = sprint_by_name[assigned_sprint]
                sprint_ids.append(assigned_sprint)
        return sorted(set(sprint_ids))

    def _current_sprint_id(self) -> Optional[str]:
        sprint = next((s for s in self.project_state.sprints if getattr(s, "status", None) == SprintStatus.IN_PROGRESS), None)
        if sprint:
            return sprint.sprint_id
        not_started = next((s for s in self.project_state.sprints if getattr(s, "status", None) == SprintStatus.NOT_STARTED), None)
        return not_started.sprint_id if not_started else None


class EstimationReliabilityDetector:
    def __init__(self, project_state: ProjectState) -> None:
        self.project_state = project_state

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        resources = {resource.resource_id: resource for resource in getattr(self.project_state, "team", []) if getattr(resource, "resource_id", None)}
        for resource_id, resource in resources.items():
            completed_items = [
                wi for wi in self.project_state.work_items
                if getattr(wi, "assigned_resource", None) == resource_id
                and getattr(wi, "status", None) in {WorkItemStatus.DONE, WorkItemStatus.COMPLETED}
            ]
            if len(completed_items) < 2:
                continue
            estimate_hours = [float(getattr(wi, "current_estimate_hrs", 0.0) or getattr(wi, "estimated_effort_hrs", 0.0) or 0.0) for wi in completed_items]
            actual_hours = [float(getattr(wi, "actual_effort_hrs", 0.0) or 0.0) for wi in completed_items]
            ratios = [a / e if e else 0.0 for a, e in zip(actual_hours, estimate_hours) if e > 0]
            if not ratios:
                continue
            ratio = sum(ratios) / len(ratios)
            remaining_items = [
                wi.item_id for wi in self.project_state.work_items
                if getattr(wi, "assigned_resource", None) == resource_id
                and getattr(wi, "status", None) in {WorkItemStatus.NOT_STARTED, WorkItemStatus.IN_PROGRESS, WorkItemStatus.BLOCKED}
            ]
            if not remaining_items:
                continue
            if ratio >= 1.3:
                severity = SignalSeverity.HIGH
                action_label = "overrun"
            elif ratio <= 0.7:
                severity = SignalSeverity.LOW
                action_label = "underbill"
            else:
                continue
            pattern = HistoricalPattern(
                pattern_type="EstimationReliabilityDetector",
                resource_id=resource_id,
                blocker_category=None,
                sample_size=len(completed_items),
                metric_name="actual_to_estimate_ratio",
                metric_value=round(ratio, 3),
                historical_occurrences=[wi.item_id for wi in completed_items],
                confidence="HIGH" if len(completed_items) >= 3 else "MEDIUM",
            )
            context: Dict[str, Any] = {
                "resource_id": resource_id,
                "ratio": round(ratio, 3),
                "sample_size": len(completed_items),
                "item_ids_used": [wi.item_id for wi in completed_items],
                "remaining_item_ids": remaining_items,
                "historical_pattern": historical_pattern_payload(pattern),
                "action_label": action_label,
            }
            signals.append(
                OpportunitySignal(
                    signal_id=signal_id(SignalCategory.ESTIMATION_RELIABILITY, [resource_id]),
                    category=SignalCategory.ESTIMATION_RELIABILITY,
                    severity=severity,
                    affected_item_ids=remaining_items,
                    affected_resource_ids=[resource_id],
                    affected_sprint_ids=sorted({wi.assigned_sprint for wi in self.project_state.work_items if wi.item_id in remaining_items}),
                    affected_blocker_ids=[],
                    evidence=[
                        SignalEvidence(
                            source_engine="domain_models",
                            metric_name="actual_to_estimate_ratio",
                            metric_value=ratio,
                            threshold=1.3 if action_label == "overrun" else 0.7,
                            explanation=f"{resource.name} has a repeatable estimate {action_label} pattern", 
                        )
                    ],
                    context=context,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        return signals


class SpilloverRootCauseDetector:
    def __init__(self, project_state: ProjectState, spillover=None) -> None:
        self.project_state = project_state
        self.spillover = spillover

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        historical_spillover_sprints = [
            actual.sprint_number for actual in self.project_state.actuals if getattr(actual, "carryover_count", 0) > 0
        ]
        if not historical_spillover_sprints:
            return signals
        current_sprint = next((s for s in self.project_state.sprints if getattr(s, "status", None) == SprintStatus.IN_PROGRESS), None)
        # WorkItem.assigned_sprint holds the sprint NAME (see workbook_parser.py), never the sprint_id.
        # Compare against sprint_name only -- do not reintroduce an id-based or "accept either" check here.
        relevant_sprint_names = {s.sprint_name for s in self.project_state.sprints if getattr(s, "status", None) in {SprintStatus.IN_PROGRESS, SprintStatus.NOT_STARTED}}
        if current_sprint is not None:
            relevant_sprint_names.add(current_sprint.sprint_name)
        target_items = [
            wi for wi in self.project_state.work_items
            if getattr(wi, "status", None) in {WorkItemStatus.NOT_STARTED, WorkItemStatus.IN_PROGRESS, WorkItemStatus.BLOCKED}
            and getattr(wi, "assigned_sprint", None) in relevant_sprint_names
        ]
        for item in target_items:
            cause = self._classify_cause(item)
            pattern = HistoricalPattern(
                pattern_type="SpilloverRootCauseDetector",
                resource_id=getattr(item, "assigned_resource", None),
                blocker_category=None,
                sample_size=len(historical_spillover_sprints),
                metric_name="historical_carryover_count",
                metric_value=float(sum(getattr(actual, "carryover_count", 0) for actual in self.project_state.actuals if getattr(actual, "sprint_number", 0) in historical_spillover_sprints)),
                historical_occurrences=[f"SPR-{s}" for s in historical_spillover_sprints],
                confidence="HIGH" if len(historical_spillover_sprints) >= 2 else "MEDIUM",
            )
            context: Dict[str, Any] = {
                "cause": cause,
                "historical_signature": {"cause": cause, "resource_id": getattr(item, "assigned_resource", None)},
                "historical_sprints": historical_spillover_sprints,
                "historical_pattern": historical_pattern_payload(pattern),
            }
            signals.append(
                OpportunitySignal(
                    signal_id=signal_id(SignalCategory.SPILLOVER, [item.item_id]),
                    category=SignalCategory.SPILLOVER,
                    severity=SignalSeverity.MEDIUM,
                    affected_item_ids=[item.item_id],
                    affected_resource_ids=[getattr(item, "assigned_resource", None)] if getattr(item, "assigned_resource", None) else [],
                    affected_sprint_ids=[item.assigned_sprint],
                    affected_blocker_ids=[],
                    evidence=[
                        SignalEvidence(
                            source_engine="spillover_engine",
                            metric_name="historical_carryover_count",
                            metric_value=float(len(historical_spillover_sprints)),
                            threshold=1.0,
                            explanation="Historical sprint carryover suggests a repeatable spillover signature",
                        )
                    ],
                    context=context,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        return signals

    def _classify_cause(self, item: Any) -> str:
        active_blockers = [b for b in self.project_state.blockers if not getattr(b, "actual_resolution_date", None)]
        related_blockers = [b for b in active_blockers if item.item_id in getattr(b, "impacted_item_ids", [])]
        if related_blockers:
            return "dependency_blocked"
        if getattr(item, "assigned_resource", None) and len([wi for wi in self.project_state.work_items if getattr(wi, "assigned_resource", None) == item.assigned_resource and getattr(wi, "status", None) in {WorkItemStatus.NOT_STARTED, WorkItemStatus.IN_PROGRESS, WorkItemStatus.BLOCKED}]) > 1:
            return "resource_unavailable"
        if getattr(item, "is_scope_changed", False):
            return "scope_growth"
        if float(getattr(item, "actual_effort_hrs", 0.0) or 0.0) > float(getattr(item, "current_estimate_hrs", 0.0) or 0.0):
            return "estimate_wrong"
        return "toolchain_friction"


class SkillMismatchDetector:
    """Flags work items where the assigned resource's skills don't match the
    item's required_skill at all (neither primary nor secondary), while a
    better-matched resource with real availability exists on the team.
    Distinct from CapacityDetector's OVERLOADED reason -- this fires purely
    on skill fit, independent of whether the current owner is overloaded."""

    def __init__(self, project_state: ProjectState) -> None:
        self.project_state = project_state

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        team_by_id = {r.resource_id: r for r in self.project_state.team}
        team_by_name = {r.name: r for r in self.project_state.team}

        for wi in self.project_state.work_items:
            if wi.status not in {WorkItemStatus.NOT_STARTED, WorkItemStatus.IN_PROGRESS}:
                continue
            required_skill = getattr(wi, "required_skill", None)
            owner_ref = getattr(wi, "assigned_resource", None)
            if not required_skill or not owner_ref:
                continue
            owner = team_by_id.get(owner_ref) or team_by_name.get(owner_ref)
            if owner is None:
                continue
            if required_skill in {owner.primary_skill, owner.secondary_skill}:
                continue  # owner has at least a secondary-level match -- not a mismatch

            candidates = [
                r for r in self.project_state.team
                if r.resource_id != owner.resource_id
                and required_skill in {r.primary_skill, r.secondary_skill}
                and float(getattr(r, "availability_pct", 0.0) or 0.0) > 0.1
            ]
            if not candidates:
                continue
            # prefer a primary-skill match over a secondary one if both exist, then by availability
            better = max(
                candidates,
                key=lambda r: (r.primary_skill == required_skill, float(getattr(r, "availability_pct", 0.0) or 0.0)),
            )

            context = {
                "flag": "SKILL_MISMATCH",
                "current_owner_id": owner.resource_id,
                "current_owner_primary_skill": owner.primary_skill,
                "current_owner_secondary_skill": owner.secondary_skill,
                "better_resource_id": better.resource_id,
                "better_resource_primary_skill": better.primary_skill,
                "better_resource_availability_pct": float(getattr(better, "availability_pct", 0.0) or 0.0),
                "required_skill": required_skill,
            }
            evidence = [
                SignalEvidence(
                    source_engine="signal_detectors",
                    metric_name="skill_match",
                    metric_value=0.0,
                    threshold=1.0,
                    explanation=f"{owner.resource_id} has neither primary nor secondary skill '{required_skill}' required by {wi.item_id}",
                )
            ]
            signals.append(OpportunitySignal(
                signal_id=signal_id(SignalCategory.CAPACITY, [wi.item_id, "skill_mismatch"]),
                category=SignalCategory.CAPACITY,
                severity=SignalSeverity.MEDIUM,
                affected_item_ids=[wi.item_id],
                affected_resource_ids=[owner.resource_id, better.resource_id],
                affected_sprint_ids=[wi.assigned_sprint] if getattr(wi, "assigned_sprint", None) else [],
                affected_blocker_ids=[],
                evidence=evidence,
                context=context,
                detected_at=datetime.now(timezone.utc).isoformat(),
            ))
        return signals[:8]


class LowVelocityDetector:
    """Flags work items whose assigned resource has a correctly-matched skill
    but a meaningfully lower historical completion velocity than another
    available, equally-skilled teammate. Velocity is derived (not stored)
    from completed work items, and requires at least 3 completed items
    before a resource's velocity is trusted -- thin-sample resources are
    never used as the 'better' comparison target, to avoid flagging a
    reassignment based on a lucky handful of fast completions."""

    MIN_SAMPLE_SIZE = 3
    MIN_RELATIVE_IMPROVEMENT = 0.3  # only flag if candidate is >=30% faster

    def __init__(self, project_state: ProjectState) -> None:
        self.project_state = project_state

    def _resource_velocity(self, resource_id: str, resource_name: str) -> tuple[float, int]:
        completed = [
            wi for wi in self.project_state.work_items
            if wi.status == WorkItemStatus.COMPLETED
            and getattr(wi, "assigned_resource", None) in {resource_id, resource_name}
        ]
        if not completed:
            return 0.0, 0
        distinct_sprints = {wi.assigned_sprint for wi in completed if getattr(wi, "assigned_sprint", None)}
        sprint_count = max(1, len(distinct_sprints))
        total_hours = sum(float(getattr(wi, "estimated_effort_hrs", 0.0) or 0.0) for wi in completed)
        return total_hours / sprint_count, len(completed)

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        velocities = {r.resource_id: self._resource_velocity(r.resource_id, r.name) for r in self.project_state.team}
        team_by_id = {r.resource_id: r for r in self.project_state.team}
        team_by_name = {r.name: r for r in self.project_state.team}

        for wi in self.project_state.work_items:
            if wi.status not in {WorkItemStatus.NOT_STARTED, WorkItemStatus.IN_PROGRESS}:
                continue
            required_skill = getattr(wi, "required_skill", None)
            owner_ref = getattr(wi, "assigned_resource", None)
            if not required_skill or not owner_ref:
                continue
            owner = team_by_id.get(owner_ref) or team_by_name.get(owner_ref)
            if owner is None or required_skill not in {owner.primary_skill, owner.secondary_skill}:
                continue  # skill mismatch is SkillMismatchDetector's job, not this one

            owner_velocity, owner_samples = velocities.get(owner.resource_id, (0.0, 0))
            if owner_samples < self.MIN_SAMPLE_SIZE:
                continue  # not enough history to say the owner is genuinely slow

            candidates = []
            for r in self.project_state.team:
                if r.resource_id == owner.resource_id:
                    continue
                if required_skill not in {r.primary_skill, r.secondary_skill}:
                    continue
                if float(getattr(r, "availability_pct", 0.0) or 0.0) <= 0.1:
                    continue
                cand_velocity, cand_samples = velocities.get(r.resource_id, (0.0, 0))
                if cand_samples < self.MIN_SAMPLE_SIZE:
                    continue
                if owner_velocity <= 0 or (cand_velocity - owner_velocity) / owner_velocity < self.MIN_RELATIVE_IMPROVEMENT:
                    continue
                candidates.append((r, cand_velocity, cand_samples))
            if not candidates:
                continue
            better, better_velocity, better_samples = max(candidates, key=lambda c: c[1])

            context = {
                "flag": "LOW_VELOCITY",
                "current_owner_id": owner.resource_id,
                "current_owner_velocity": round(owner_velocity, 2),
                "current_owner_sample_size": owner_samples,
                "better_resource_id": better.resource_id,
                "better_resource_velocity": round(better_velocity, 2),
                "better_resource_sample_size": better_samples,
            }
            evidence = [
                SignalEvidence(
                    source_engine="signal_detectors",
                    metric_name="resource_velocity",
                    metric_value=round(owner_velocity, 2),
                    threshold=round(better_velocity, 2),
                    explanation=f"{owner.resource_id} completes ~{round(owner_velocity,1)}hrs/sprint (n={owner_samples}) vs {better.resource_id}'s ~{round(better_velocity,1)}hrs/sprint (n={better_samples})",
                )
            ]
            signals.append(OpportunitySignal(
                signal_id=signal_id(SignalCategory.CAPACITY, [wi.item_id, "low_velocity"]),
                category=SignalCategory.CAPACITY,
                severity=SignalSeverity.MEDIUM,
                affected_item_ids=[wi.item_id],
                affected_resource_ids=[owner.resource_id, better.resource_id],
                affected_sprint_ids=[wi.assigned_sprint] if getattr(wi, "assigned_sprint", None) else [],
                affected_blocker_ids=[],
                evidence=evidence,
                context=context,
                detected_at=datetime.now(timezone.utc).isoformat(),
            ))
        return signals[:8]


class SPOFDetector:
    def __init__(self, project_state: ProjectState, cp_result: Optional[CriticalPathResult] = None) -> None:
        self.project_state = project_state
        self.cp_result = cp_result

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        skills_to_items: Dict[str, List[Any]] = {}
        for work_item in self.project_state.work_items:
            skill = getattr(work_item, "required_skill", None)
            if not skill:
                continue
            if self._is_critical_priority(work_item) or (self.cp_result is not None and work_item.item_id in getattr(self.cp_result, "items_on_critical_path", [])):
                skills_to_items.setdefault(skill, []).append(work_item)
        for skill, critical_items in skills_to_items.items():
            assigned_resource_ids = {getattr(wi, "assigned_resource", None) for wi in critical_items if getattr(wi, "assigned_resource", None)}
            if len(assigned_resource_ids) != 1:
                continue
            sole_resource_id = next(iter(assigned_resource_ids))
            backup_resource = next(
                (
                    resource for resource in self.project_state.team
                    if resource.resource_id != sole_resource_id and self._has_slack(resource)
                ),
                None,
            )
            if backup_resource is None:
                continue
            pattern = HistoricalPattern(
                pattern_type="SPOFDetector",
                resource_id=sole_resource_id,
                blocker_category=None,
                sample_size=1,
                metric_name="single_resource_skill_coverage",
                metric_value=1.0,
                historical_occurrences=[item.item_id for item in critical_items],
                confidence="LOW",
            )
            signals.append(
                OpportunitySignal(
                    signal_id=signal_id(SignalCategory.SPOF, [sole_resource_id]),
                    category=SignalCategory.SPOF,
                    severity=SignalSeverity.CRITICAL,
                    affected_item_ids=[item.item_id for item in critical_items[:1]],
                    affected_resource_ids=[sole_resource_id, backup_resource.resource_id],
                    affected_sprint_ids=[item.assigned_sprint for item in critical_items[:1]],
                    affected_blocker_ids=[],
                    evidence=[
                        SignalEvidence(
                            source_engine="domain_models",
                            metric_name="single_resource_skill_coverage",
                            metric_value=1.0,
                            threshold=1.0,
                            explanation="A single resource carries the critical skill for a critical item",
                        )
                    ],
                    context={
                        "skill_name": skill,
                        "sole_resource_id": sole_resource_id,
                        "backup_resource_id": backup_resource.resource_id,
                        "backup_slack_hours": round(self._slack_hours(backup_resource), 2),
                        "historical_pattern": historical_pattern_payload(pattern),
                    },
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        return signals

    def _has_slack(self, resource: Any) -> bool:
        return self._slack_hours(resource) > 8.0

    def _is_critical_priority(self, work_item: Any) -> bool:
        priority = getattr(work_item, "priority", None)
        if priority is None:
            return False
        if isinstance(priority, Priority):
            return priority == Priority.CRITICAL
        if isinstance(priority, str):
            return priority.lower() == "critical"
        return getattr(priority, "value", None) == "Critical"

    def _slack_hours(self, resource: Any) -> float:
        assigned_remaining = sum(float(getattr(wi, "remaining_effort_hrs", 0.0) or 0.0) for wi in self.project_state.work_items if getattr(wi, "assigned_resource", None) == resource.resource_id)
        capacity = (float(getattr(resource, "daily_capacity_hrs", 0.0) or 0.0) * float(self.project_state.project_info.sprint_duration_days or 10) * float(getattr(resource, "availability_pct", 1.0) or 1.0) * float(getattr(resource, "allocation_pct", 1.0) or 1.0))
        return max(0.0, capacity - assigned_remaining)


class RecurringBlockerDetector:
    def __init__(self, project_state: ProjectState) -> None:
        self.project_state = project_state

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        categories: Dict[str, List[Blocker]] = {}
        for blocker in self.project_state.blockers:
            categories.setdefault(getattr(blocker, "category", None).value if getattr(blocker, "category", None) else "Other", []).append(blocker)
        for category, blockers in categories.items():
            if len(blockers) < 2:
                continue
            active_blockers = [b for b in blockers if not getattr(b, "actual_resolution_date", None)]
            if not active_blockers:
                continue
            resolution_days = []
            for blocker in blockers:
                raised = getattr(blocker, "raised_date", None)
                resolved = getattr(blocker, "actual_resolution_date", None) or getattr(blocker, "target_resolution_date", None)
                if raised and resolved:
                    resolution_days.append((resolved - raised).days)
            avg_days = sum(resolution_days) / len(resolution_days) if resolution_days else 5.0
            max_days = max(resolution_days) if resolution_days else 5.0
            for blocker in active_blockers:
                projected_resolution = getattr(blocker, "raised_date", None) + timedelta(days=round(avg_days))
                target = getattr(blocker, "target_resolution_date", None)
                deadline = getattr(self.project_state.project_info, "target_end_date", None)
                severity = SignalSeverity.CRITICAL if (target and projected_resolution > target) or (deadline and projected_resolution > deadline) else SignalSeverity.HIGH
                pattern = HistoricalPattern(
                    pattern_type="RecurringBlockerDetector",
                    resource_id=getattr(blocker, "owner", None),
                    blocker_category=category,
                    sample_size=len(blockers),
                    metric_name="historical_blocker_resolution_days",
                    metric_value=round(avg_days, 2),
                    historical_occurrences=[b.blocker_id for b in blockers],
                    confidence="HIGH" if len(blockers) >= 3 else "MEDIUM",
                )
                signals.append(
                    OpportunitySignal(
                        signal_id=signal_id(SignalCategory.RECURRING_BLOCKER, [blocker.blocker_id]),
                        category=SignalCategory.RECURRING_BLOCKER,
                        severity=severity,
                        affected_item_ids=list(getattr(blocker, "impacted_item_ids", []) or []),
                        affected_resource_ids=[]
                        if getattr(blocker, "owner", None) is None else [getattr(blocker, "owner", None)],
                        affected_sprint_ids=[],
                        affected_blocker_ids=[blocker.blocker_id],
                        evidence=[
                            SignalEvidence(
                                source_engine="domain_models",
                                metric_name="historical_blocker_resolution_days",
                                metric_value=avg_days,
                                threshold=avg_days,
                                explanation=f"{category} has recurred {len(blockers)} times and is now active again",
                            )
                        ],
                        context={
                            "category": category,
                            "owner": getattr(blocker, "owner", None),
                            "occurrence_count": len(blockers),
                            "avg_resolution_days": round(avg_days, 2),
                            "max_resolution_days": round(max_days, 2),
                            "prior_blocker_ids": [b.blocker_id for b in blockers],
                            "historical_pattern": historical_pattern_payload(pattern),
                        },
                        detected_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
        return signals


class ReworkLoopDetector:
    def __init__(self, project_state: ProjectState) -> None:
        self.project_state = project_state

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        rework_incidents = []
        for wi in self.project_state.work_items:
            if getattr(wi, "status", None) in {WorkItemStatus.DONE, WorkItemStatus.COMPLETED} and float(getattr(wi, "actual_effort_hrs", 0.0) or 0.0) > float(getattr(wi, "current_estimate_hrs", 0.0) or 0.0):
                rework_incidents.append(wi)
        if len(rework_incidents) < 2:
            return signals
        categories = sorted({getattr(wi, "work_type", None).value if getattr(wi, "work_type", None) else "Task" for wi in rework_incidents})
        for category in categories:
            upcoming_items = [
                wi
                for wi in self.project_state.work_items
                if getattr(wi, "status", None) in {WorkItemStatus.NOT_STARTED, WorkItemStatus.IN_PROGRESS, WorkItemStatus.BLOCKED}
                and (
                    (getattr(wi, "work_type", None) is not None and getattr(wi, "work_type", None).value == category)
                    or (getattr(wi, "work_type", None) is None and category == "Task")
                )
            ]
            if not upcoming_items:
                continue
            pattern = HistoricalPattern(
                pattern_type="ReworkLoopDetector",
                resource_id=None,
                blocker_category=category,
                sample_size=len(rework_incidents),
                metric_name="rework_hours",
                metric_value=sum(float(getattr(wi, "actual_effort_hrs", 0.0) or 0.0) - float(getattr(wi, "current_estimate_hrs", 0.0) or 0.0) for wi in rework_incidents),
                historical_occurrences=[wi.item_id for wi in rework_incidents],
                confidence="HIGH" if len(rework_incidents) >= 3 else "MEDIUM",
            )
            signals.append(
                OpportunitySignal(
                    signal_id=signal_id(SignalCategory.REWORK_LOOP, [category]),
                    category=SignalCategory.REWORK_LOOP,
                    severity=SignalSeverity.HIGH,
                    affected_item_ids=[wi.item_id for wi in upcoming_items[:2]],
                    affected_resource_ids=[],
                    affected_sprint_ids=[wi.assigned_sprint for wi in upcoming_items[:2]],
                    affected_blocker_ids=[],
                    evidence=[
                        SignalEvidence(
                            source_engine="domain_models",
                            metric_name="rework_hours",
                            metric_value=float(pattern.metric_value),
                            threshold=10.0,
                            explanation="The same work category has repeated rework incidents",
                        )
                    ],
                    context={
                        "category": category,
                        "historical_item_ids": [wi.item_id for wi in rework_incidents],
                        "rework_hours": round(pattern.metric_value, 2),
                        "historical_pattern": historical_pattern_payload(pattern),
                    },
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        return signals


class RampUpDetector:
    def __init__(self, project_state: ProjectState) -> None:
        self.project_state = project_state

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        current_sprint_number = max((s.sprint_number for s in self.project_state.sprints), default=1)
        for resource in self.project_state.team:
            assigned_sprints = [
                next((s.sprint_number for s in self.project_state.sprints if s.sprint_id == wi.assigned_sprint), None)
                for wi in self.project_state.work_items
                if getattr(wi, "assigned_resource", None) == resource.resource_id
            ]
            first_sprint_number = min([s for s in assigned_sprints if s is not None], default=current_sprint_number)
            if current_sprint_number - first_sprint_number > 2:
                continue
            affected_items = [
                wi.item_id for wi in self.project_state.work_items
                if getattr(wi, "assigned_resource", None) == resource.resource_id
                and getattr(wi, "status", None) in {WorkItemStatus.NOT_STARTED, WorkItemStatus.IN_PROGRESS, WorkItemStatus.BLOCKED}
            ]
            if not affected_items:
                continue
            pattern = HistoricalPattern(
                pattern_type="RampUpDetector",
                resource_id=resource.resource_id,
                blocker_category=None,
                sample_size=1,
                metric_name="first_sprint_number",
                metric_value=float(first_sprint_number),
                historical_occurrences=[resource.resource_id],
                confidence="LOW",
            )
            signals.append(
                OpportunitySignal(
                    signal_id=signal_id(SignalCategory.RAMP_UP, [resource.resource_id]),
                    category=SignalCategory.RAMP_UP,
                    severity=SignalSeverity.MEDIUM,
                    affected_item_ids=affected_items,
                    affected_resource_ids=[resource.resource_id],
                    affected_sprint_ids=[wi.assigned_sprint for wi in self.project_state.work_items if wi.item_id in affected_items],
                    affected_blocker_ids=[],
                    evidence=[
                        SignalEvidence(
                            source_engine="domain_models",
                            metric_name="first_sprint_number",
                            metric_value=float(first_sprint_number),
                            threshold=float(max(1, current_sprint_number - 2)),
                            explanation="The resource appears to be in an early ramp-up period",
                        )
                    ],
                    context={
                        "resource_id": resource.resource_id,
                        "joined_sprint_number": first_sprint_number,
                        "affected_item_ids": affected_items,
                        "historical_pattern": historical_pattern_payload(pattern),
                    },
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        return signals


class ResequencingDetector:
    def __init__(self, project_state: ProjectState, dag: Optional[DependencyDAG] = None, cp_result: Optional[CriticalPathResult] = None) -> None:
        self.project_state = project_state
        self.dag = dag
        self.cp_result = cp_result

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        cp_items = [wi for wi in self.project_state.work_items if getattr(wi, "priority", None) == Priority.CRITICAL]
        if not cp_items:
            return signals
        for cp_item in cp_items:
            for other in self.project_state.work_items:
                if other.item_id == cp_item.item_id or getattr(other, "priority", None) == Priority.CRITICAL:
                    continue
                if getattr(other, "assigned_resource", None) != getattr(cp_item, "assigned_resource", None):
                    continue
                if getattr(other, "assigned_sprint", None) not in {cp_item.assigned_sprint, self._adjacent_sprint_id(cp_item.assigned_sprint)}:
                    continue
                if self._has_dependency_edge(cp_item.item_id, other.item_id):
                    continue
                pattern = HistoricalPattern(
                    pattern_type="ResequencingDetector",
                    resource_id=getattr(cp_item, "assigned_resource", None),
                    blocker_category=None,
                    sample_size=1,
                    metric_name="serialized_non_cp_item",
                    metric_value=1.0,
                    historical_occurrences=[cp_item.item_id, other.item_id],
                    confidence="LOW",
                )
                signals.append(
                    OpportunitySignal(
                        signal_id=signal_id(SignalCategory.RESEQUENCING, [cp_item.item_id, other.item_id]),
                        category=SignalCategory.RESEQUENCING,
                        severity=SignalSeverity.MEDIUM,
                        affected_item_ids=[other.item_id],
                        affected_resource_ids=[getattr(cp_item, "assigned_resource", None)] if getattr(cp_item, "assigned_resource", None) else [],
                        affected_sprint_ids=[cp_item.assigned_sprint],
                        affected_blocker_ids=[],
                        evidence=[
                            SignalEvidence(
                                source_engine="dependency_engine",
                                metric_name="serialized_non_cp_item",
                                metric_value=1.0,
                                threshold=1.0,
                                explanation="A non-critical item is serialized onto the same resource as critical-path work with no dependency edge",
                            )
                        ],
                        context={
                            "critical_item_id": cp_item.item_id,
                            "non_critical_item_id": other.item_id,
                            "hours_freed": round(float(getattr(other, "remaining_effort_hrs", 0.0) or 0.0), 2),
                            "historical_pattern": historical_pattern_payload(pattern),
                        },
                        detected_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
                return signals
        return signals

    def _adjacent_sprint_id(self, sprint_id: str) -> Optional[str]:
        sprint_numbers = {s.sprint_id: s.sprint_number for s in self.project_state.sprints}
        current_number = sprint_numbers.get(sprint_id)
        if current_number is None:
            return None
        for sprint in self.project_state.sprints:
            if sprint.sprint_number == current_number + 1:
                return sprint.sprint_id
        return None

    def _has_dependency_edge(self, predecessor_id: str, successor_id: str) -> bool:
        return any(
            getattr(dep, "predecessor_item_id", None) == predecessor_id and getattr(dep, "successor_item_id", None) == successor_id
            for dep in self.project_state.dependencies
        )


class SwarmTradeoffDetector:
    def __init__(self, project_state: ProjectState, cp_result: Optional[CriticalPathResult] = None) -> None:
        self.project_state = project_state
        self.cp_result = cp_result

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        cp_items = [
            wi for wi in self.project_state.work_items
            if self._is_critical_priority(wi) and float(getattr(wi, "remaining_effort_hrs", 0.0) or 0.0) >= 20.0
        ]
        if not cp_items:
            return signals
        bottleneck = max(cp_items, key=lambda wi: float(getattr(wi, "remaining_effort_hrs", 0.0) or 0.0))
        backup_resource = next(
            (
                resource for resource in self.project_state.team
                if resource.resource_id != getattr(bottleneck, "assigned_resource", None)
                and self._slack_hours(resource.resource_id) > 8.0
            ),
            None,
        )
        if backup_resource is None:
            return signals
        other_item = next(
            (
                wi for wi in self.project_state.work_items
                if getattr(wi, "assigned_resource", None) == backup_resource.resource_id
                and getattr(wi, "status", None) in {WorkItemStatus.NOT_STARTED, WorkItemStatus.IN_PROGRESS, WorkItemStatus.BLOCKED}
            ),
            None,
        )
        if other_item is None:
            return signals
        pattern = HistoricalPattern(
            pattern_type="SwarmTradeoffDetector",
            resource_id=backup_resource.resource_id,
            blocker_category=None,
            sample_size=1,
            metric_name="swarm_tradeoff",
            metric_value=1.0,
            historical_occurrences=[bottleneck.item_id, other_item.item_id],
            confidence="LOW",
        )
        signals.append(
            OpportunitySignal(
                signal_id=signal_id(SignalCategory.SWARM_TRADEOFF, [bottleneck.item_id]),
                category=SignalCategory.SWARM_TRADEOFF,
                severity=SignalSeverity.HIGH,
                affected_item_ids=[bottleneck.item_id],
                affected_resource_ids=[backup_resource.resource_id],
                affected_sprint_ids=[bottleneck.assigned_sprint],
                affected_blocker_ids=[],
                evidence=[
                    SignalEvidence(
                        source_engine="critical_path_engine",
                        metric_name="swarm_tradeoff",
                        metric_value=1.0,
                        threshold=1.0,
                        explanation="Swarming the critical-path bottleneck item can save days at the cost of another item's delay",
                    )
                ],
                context={
                    "days_saved_on_critical_path": 1.5,
                    "delay_caused_to_other_item": 0.5,
                    "other_item_id": other_item.item_id,
                    "other_resource_id": backup_resource.resource_id,
                    "historical_pattern": historical_pattern_payload(pattern),
                },
                detected_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        return signals

    def _is_critical_priority(self, work_item: Any) -> bool:
        priority = getattr(work_item, "priority", None)
        if priority is None:
            return False
        if isinstance(priority, Priority):
            return priority == Priority.CRITICAL
        if isinstance(priority, str):
            return priority.lower() == "critical"
        return getattr(priority, "value", None) == "Critical"

    def _slack_hours(self, resource_id: str) -> float:
        resource = next((r for r in self.project_state.team if r.resource_id == resource_id), None)
        if resource is None:
            return 0.0
        assigned_remaining = sum(float(getattr(wi, "remaining_effort_hrs", 0.0) or 0.0) for wi in self.project_state.work_items if getattr(wi, "assigned_resource", None) == resource.resource_id)
        capacity = (float(getattr(resource, "daily_capacity_hrs", 0.0) or 0.0) * float(self.project_state.project_info.sprint_duration_days or 10) * float(getattr(resource, "availability_pct", 1.0) or 1.0) * float(getattr(resource, "allocation_pct", 1.0) or 1.0))
        return max(0.0, capacity - assigned_remaining)


class SprintDetector:
    def __init__(
        self,
        project_state: ProjectState,
        metrics: ProjectMetrics,
        spillover: SpilloverAnalysis,
        forecast: ForecastResult,
    ) -> None:
        self.project_state = project_state
        self.metrics = metrics
        self.spillover = spillover
        self.forecast = forecast

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        
        # Consume sprint metrics from ProjectMetrics instead of recalculating
        for sprint_metric in self.metrics.sprint_metrics:
            sprint = next(
                (s for s in self.project_state.sprints if s.sprint_id == sprint_metric.sprint_id),
                None
            )
            
            if sprint is None or getattr(sprint, "status", None) == SprintStatus.COMPLETED:
                continue
            
            # Use utilization from metrics instead of recalculating
            utilization_ratio = sprint_metric.completion_pct
            planned_hours = sprint_metric.planned_effort_hours
            capacity_hours = sprint_metric.planned_effort_hours
            
            # Detect under/overloaded sprints
            historical_completion_rates = [
                sm.completion_pct for sm in self.metrics.sprint_metrics
                if sm.sprint_number < sprint_metric.sprint_number and sm.completion_pct > 0
            ]
            historical_avg_completion = (
                sum(historical_completion_rates) / len(historical_completion_rates)
                if historical_completion_rates else 0.7
            )
            planned_vs_capacity_ratio = (
                sprint_metric.planned_effort_hours / max(sprint_metric.actual_effort_hours, sprint_metric.planned_effort_hours, 1.0)
            )

            flag = None
            if sprint_metric.sprint_number != self.metrics.current_sprint_number and utilization_ratio < (historical_avg_completion * 0.6):
                flag = "UNDERLOADED"
            elif planned_vs_capacity_ratio > 1.1 or utilization_ratio > 1.1:
                flag = "OVERLOADED"
            else:
                continue
            
            # Get blocked hours from blocker metrics if available
            blocked_hours = 0.0
            blocked_items = [
                wi.item_id for wi in self.project_state.work_items
                if getattr(wi, "assigned_sprint", None) == sprint_metric.sprint_id
                and getattr(wi, "status", None) == WorkItemStatus.BLOCKED
            ]
            for item_id in blocked_items:
                wi = next(
                    (w for w in self.project_state.work_items if w.item_id == item_id),
                    None
                )
                if wi:
                    blocked_hours += float(getattr(wi, "remaining_effort_hrs", 0.0) or 0.0)
            
            # Get spillover probability from spillover analysis
            spillover_prob = 0.0
            if self.spillover:
                spill_by_sprint = getattr(self.spillover, "predicted_spillover_by_sprint", {}) or {}
                if isinstance(spill_by_sprint, dict):
                    spillover_prob = float(spill_by_sprint.get(sprint_metric.sprint_number, 0.0))
            
            context: Dict[str, Any] = {
                "sprint_id": sprint_metric.sprint_id,
                "sprint_number": sprint_metric.sprint_number,
                "flag": flag,
                "utilization_ratio": round(utilization_ratio, 4),
                "planned_hours": round(planned_hours, 2),
                "capacity_hours": round(capacity_hours, 2),
                "actual_effort_hours": round(sprint_metric.actual_effort_hours, 2),
                "blocked_hours": round(blocked_hours, 2),
                "blocked_pct": round(blocked_hours / max(planned_hours, 1.0), 4),
                "spillover_probability": round(spillover_prob, 4),
                "is_cp_sprint": self._is_cp_sprint(sprint_metric.sprint_id),
            }
            
            evidence = [
                SignalEvidence(
                    source_engine="spillover_engine",
                    metric_name="predicted_spillover",
                    metric_value=spillover_prob,
                    threshold=0.6,
                    explanation="Sprint spillover risk is elevated",
                )
            ]
            
            signals.append(
                OpportunitySignal(
                    signal_id=signal_id(SignalCategory.SPRINT, [sprint_metric.sprint_id]),
                    category=SignalCategory.SPRINT,
                    severity=SignalSeverity.MEDIUM,
                    affected_item_ids=self._items_in_sprint(sprint_metric.sprint_id),
                    affected_resource_ids=[],
                    affected_sprint_ids=[sprint_metric.sprint_id],
                    affected_blocker_ids=[],
                    evidence=evidence,
                    context=context,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        
        return signals

    def _items_in_sprint(self, sprint_id: str) -> List[str]:
        sprint_name_by_id = {s.sprint_id: s.sprint_name for s in self.project_state.sprints}
        return [
            wi.item_id for wi in self.project_state.work_items
            if getattr(wi, "assigned_sprint", None) == sprint_id
            or sprint_name_by_id.get(getattr(wi, "assigned_sprint", None)) == sprint_id
        ]

    def _is_cp_sprint(self, sprint_id: str) -> bool:
        """Check if any CP items are assigned to this sprint."""
        sprint_name_by_id = {s.sprint_id: s.sprint_name for s in self.project_state.sprints}
        return any(
            wi.item_id in self.forecast.items_on_critical_path 
            if hasattr(self.forecast, 'items_on_critical_path') else False
            for wi in self.project_state.work_items
            if getattr(wi, "assigned_sprint", None) == sprint_id
            or sprint_name_by_id.get(getattr(wi, "assigned_sprint", None)) == sprint_id
        )


class CriticalPathDetector:
    def __init__(
        self,
        project_state: ProjectState,
        cp_result: CriticalPathResult,
        dag: DependencyDAG,
        impact_scores: RiskScores,
    ) -> None:
        self.project_state = project_state
        self.cp_result = cp_result
        self.dag = dag
        self.impact_scores = impact_scores

    def detect(self) -> List[OpportunitySignal]:
        signals: List[OpportunitySignal] = []
        active_blockers = [b for b in self.project_state.blockers if not getattr(b, "actual_resolution_date", None)]
        blocked_cp_items = []
        for blocker in active_blockers:
            for item_id in getattr(blocker, "impacted_item_ids", []) or []:
                if item_id in self.cp_result.items_on_critical_path:
                    blocked_cp_items.append(item_id)
        if blocked_cp_items:
            signal = OpportunitySignal(
                signal_id=signal_id(SignalCategory.CRITICAL_PATH, sorted(set(blocked_cp_items))),
                category=SignalCategory.CRITICAL_PATH,
                severity=SignalSeverity.CRITICAL,
                affected_item_ids=sorted(set(blocked_cp_items)),
                affected_resource_ids=[],
                affected_sprint_ids=self._affected_sprint_ids(sorted(set(blocked_cp_items))),
                affected_blocker_ids=[b.blocker_id for b in active_blockers if any(item_id in getattr(b, 'impacted_item_ids', []) or [] for item_id in blocked_cp_items)],
                evidence=[
                    SignalEvidence(
                        source_engine="critical_path_engine",
                        metric_name="cp_at_risk",
                        metric_value=float(len(blocked_cp_items)),
                        threshold=1.0,
                        explanation="Critical path items are affected by active blockers",
                    )
                ],
                context={
                    "cp_nodes": sorted(set(blocked_cp_items)),
                    "cp_remaining_hours": round(self._cp_remaining_hours(sorted(set(blocked_cp_items))), 2),
                    "cp_single_owners": self._cp_single_owners(sorted(set(blocked_cp_items))),
                    "cp_blocked_items": sorted(set(blocked_cp_items)),
                    "flag": "CP_AT_RISK",
                },
                detected_at=datetime.now(timezone.utc).isoformat(),
            )
            signals.append(signal)

        near_critical_items = self._near_critical_items()
        if near_critical_items:
            signals.append(OpportunitySignal(
                signal_id=signal_id(SignalCategory.CRITICAL_PATH, sorted(set(near_critical_items))),
                category=SignalCategory.CRITICAL_PATH,
                severity=SignalSeverity.MEDIUM,
                affected_item_ids=sorted(set(near_critical_items)),
                affected_resource_ids=[],
                affected_sprint_ids=self._affected_sprint_ids(sorted(set(near_critical_items))),
                affected_blocker_ids=[],
                evidence=[
                    SignalEvidence(
                        source_engine="critical_path_engine",
                        metric_name="near_critical_items",
                        metric_value=float(len(near_critical_items)),
                        threshold=1.0,
                        explanation="Items are close to the critical path and may benefit from parallelization",
                    )
                ],
                context={
                    "near_critical_items": sorted(set(near_critical_items)),
                    "flag": "NEAR_CRITICAL",
                },
                detected_at=datetime.now(timezone.utc).isoformat(),
            ))

        dependency_bottleneck_item_ids = self._dependency_bottlenecks()
        if dependency_bottleneck_item_ids:
            signals.append(OpportunitySignal(
                signal_id=signal_id(SignalCategory.CRITICAL_PATH, sorted(set(dependency_bottleneck_item_ids))),
                category=SignalCategory.CRITICAL_PATH,
                severity=SignalSeverity.HIGH,
                affected_item_ids=sorted(set(dependency_bottleneck_item_ids)),
                affected_resource_ids=[],
                affected_sprint_ids=self._affected_sprint_ids(sorted(set(dependency_bottleneck_item_ids))),
                affected_blocker_ids=[],
                evidence=[
                    SignalEvidence(
                        source_engine="critical_path_engine",
                        metric_name="dependency_bottlenecks",
                        metric_value=float(len(dependency_bottleneck_item_ids)),
                        threshold=1.0,
                        explanation="Critical path items have multiple incoming dependencies and may benefit from dependency removal",
                    )
                ],
                context={
                    "dependency_bottleneck_item_ids": sorted(set(dependency_bottleneck_item_ids)),
                    "flag": "DEPENDENCY_BOTTLENECK",
                },
                detected_at=datetime.now(timezone.utc).isoformat(),
            ))

        return signals

    def _affected_sprint_ids(self, item_ids: List[str]) -> List[str]:
        sprint_ids = []
        for item_id in item_ids:
            work_item = next((wi for wi in self.project_state.work_items if wi.item_id == item_id), None)
            if work_item and getattr(work_item, "assigned_sprint", None):
                sprint_ids.append(work_item.assigned_sprint)
        return sorted(set(sprint_ids))

    def _cp_remaining_hours(self, item_ids: List[str]) -> float:
        return sum(float(next((wi.remaining_effort_hrs for wi in self.project_state.work_items if wi.item_id == item_id), 0.0)) for item_id in item_ids)

    def _cp_single_owners(self, item_ids: List[str]) -> List[str]:
        owners = []
        for item_id in item_ids:
            work_item = next((wi for wi in self.project_state.work_items if wi.item_id == item_id), None)
            if work_item and getattr(work_item, "assigned_resource", None):
                owners.append(work_item.assigned_resource)
        return sorted(set(owners))

    def _near_critical_items(self) -> List[str]:
        sprint_duration_hours = self.project_state.project_info.sprint_duration_days * 24.0
        threshold = 0.25 * sprint_duration_hours
        near = []
        slack_map = getattr(self.cp_result, "item_slack_map", {}) or {}
        for each in slack_map:
            if slack_map[each] <= threshold:
                near.append(each)
        return sorted(near)

    def _dependency_bottlenecks(self) -> List[str]:
        reverse_counts: Dict[str, int] = {}
        for node, successors in self.dag.graph.items():
            for successor in successors:
                if successor in self.cp_result.items_on_critical_path:
                    reverse_counts[successor] = reverse_counts.get(successor, 0) + 1
        return [item_id for item_id, count in sorted(reverse_counts.items()) if count >= 3]


class ScheduleDetector:
    """
    Detects schedule-related signals by consuming ForecastResult directly.
    
    This detector avoids recalculating schedule gaps, velocity trends, or other
    forecast-related metrics. Instead, it consumes the output from ForecastEngine,
    which is the single source of truth for schedule forecasts.
    """
    
    def __init__(
        self,
        project_state: ProjectState,
        forecast: ForecastResult,
        monte_carlo: MonteCarloResult,
        risk_result: RiskResult,
        metrics: ProjectMetrics,
    ) -> None:
        self.project_state = project_state
        self.forecast = forecast
        self.monte_carlo = monte_carlo
        self.risk_result = risk_result
        self.metrics = metrics

    def _schedule_gap_hours(self) -> float:
        """
        Extract schedule gap from ForecastResult breakdown.
        
        Consumes from ForecastResult.delay_breakdown instead of recalculating.
        """
        if hasattr(self.forecast, "delay_breakdown") and self.forecast.delay_breakdown:
            return float(self.forecast.delay_breakdown.expected_delay_days * 8.0)
        
        # Fallback: use expected_delay_days directly
        expected_delay = float(getattr(self.forecast, "expected_delay_days", 0.0) or 0.0)
        return max(0.0, expected_delay * 8.0)

    def _velocity_trend(self) -> Optional[float]:
        """
        Extract velocity trend from velocity_metrics in ProjectMetrics.
        
        Consumes from ProjectMetrics.velocity_metrics instead of recalculating.
        """
        return float(getattr(self.metrics.velocity_metrics, "velocity_trend_pct", None) or 0.0)

    def _highest_effort_not_started_items(self, limit: int = 3) -> List[str]:
        """Find highest-effort not-started items to populate as affected items."""
        not_started_items = []
        for wi in self.project_state.work_items:
            status = getattr(wi, "status", None)
            # Skip if already started or completed
            if status in (WorkItemStatus.IN_PROGRESS, WorkItemStatus.COMPLETED, 
                         WorkItemStatus.DONE, WorkItemStatus.BLOCKED, WorkItemStatus.SPILLOVER):
                continue
            effort = float(getattr(wi, "current_estimate_hrs", 0.0) or 0.0)
            if effort == 0.0:
                effort = float(wi.remaining_effort_hrs or 0.0)
            if effort > 0.0:
                not_started_items.append((wi.item_id, effort))
        
        # Sort by effort descending and return top N
        not_started_items.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in not_started_items[:limit]]

    def detect(self) -> List[OpportunitySignal]:
        """
        Detect schedule-related signals from ForecastResult.
        
        Key signals:
        - SCHEDULE_GAP: when expected delay > 0
        - VELOCITY_CONCERN: when velocity is degrading or uncertain
        - SCOPE_CREEP: when scope inflation is detected
        """
        signals: List[OpportunitySignal] = []
        
        # Extract schedule gap from forecast breakdown
        schedule_gap_hours = self._schedule_gap_hours()
        velocity_trend = self._velocity_trend()
        scope_growth_hours = float(getattr(self.forecast, "scope_growth_hours", 0.0) or 0.0)
        
        # Get affected items
        affected_items = self._highest_effort_not_started_items(limit=3)
        
        # Signal 1: Schedule gap (primary signal)
        if self.forecast.expected_delay_days > 0:
            delay_breakdown = self.forecast.delay_breakdown if hasattr(self.forecast, "delay_breakdown") else None
            remaining_days_base = float(delay_breakdown.remaining_days_base_work) if delay_breakdown else 0.0
            remaining_days_blocker = float(delay_breakdown.remaining_days_blocker_loss) if delay_breakdown else 0.0
            remaining_days_spillover = float(delay_breakdown.remaining_days_spillover) if delay_breakdown else 0.0
            
            context: Dict[str, Any] = {
                "schedule_gap_hours": schedule_gap_hours,
                "expected_delay_days": round(self.forecast.expected_delay_days, 2),
                "on_track": self.forecast.on_track,
                "flag": "SCHEDULE_AT_RISK",
                "delay_breakdown": {
                    "base_work_days": round(remaining_days_base, 2),
                    "blocker_loss_days": round(remaining_days_blocker, 2),
                    "spillover_days": round(remaining_days_spillover, 2),
                },
                "scope_growth_hours": round(scope_growth_hours, 2),
                "velocity_trend_pct": round(velocity_trend, 2) if velocity_trend else 0.0,
            }
            
            evidence = [
                SignalEvidence(
                    source_engine="forecast_engine",
                    metric_name="expected_delay_days",
                    metric_value=self.forecast.expected_delay_days,
                    threshold=0.0,
                    explanation="Forecast indicates schedule delay",
                )
            ]
            
            if velocity_trend and velocity_trend < 0:
                evidence.append(SignalEvidence(
                    source_engine="metrics_engine",
                    metric_name="velocity_trend_pct",
                    metric_value=velocity_trend,
                    threshold=0.0,
                    explanation="Velocity is degrading over time",
                ))
                context["velocity_degrading"] = True
            
            signals.append(
                OpportunitySignal(
                    signal_id=signal_id(SignalCategory.SCHEDULE, ["schedule_gap"]),
                    category=SignalCategory.SCHEDULE,
                    severity=SignalSeverity.HIGH,
                    affected_item_ids=affected_items,
                    affected_resource_ids=[],
                    affected_sprint_ids=[],
                    affected_blocker_ids=[],
                    evidence=evidence,
                    context=context,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        
        # Signal 2: Scope creep (secondary signal if scope is growing)
        if scope_growth_hours > 0:
            context: Dict[str, Any] = {
                "scope_inflation_hours": round(scope_growth_hours, 2),
                "scope_inflation_pct": round(
                    (scope_growth_hours / self.forecast.raw_remaining_effort_hours * 100.0)
                    if self.forecast.raw_remaining_effort_hours > 0 else 0.0,
                    2
                ),
                "flag": "SCOPE_CREEP",
            }
            
            signals.append(
                OpportunitySignal(
                    signal_id=signal_id(SignalCategory.SCHEDULE, ["scope_creep"]),
                    category=SignalCategory.SCHEDULE,
                    severity=SignalSeverity.MEDIUM,
                    affected_item_ids=affected_items[:1],  # Top affected item
                    affected_resource_ids=[],
                    affected_sprint_ids=[],
                    affected_blocker_ids=[],
                    evidence=[SignalEvidence(
                        source_engine="forecast_engine",
                        metric_name="scope_growth_hours",
                        metric_value=scope_growth_hours,
                        threshold=0.0,
                        explanation="Scope growth is impacting the schedule",
                    )],
                    context=context,
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        # Signal 3: Scope inflation concentrated in historically risky categories
        high_risk_reasons = {"Customer Request", "Technical Debt"}
        inflation_by_reason = getattr(self.metrics, "scope_inflation_by_reason", {}) or {}
        risky_inflation_hours = sum(
            hours for reason, hours in inflation_by_reason.items()
            if reason in high_risk_reasons
        )
        if risky_inflation_hours > 10.0:
            signals.append(
                OpportunitySignal(
                    signal_id=signal_id(SignalCategory.SCHEDULE, ["scope_inflation_risk"]),
                    category=SignalCategory.SCHEDULE,
                    severity=SignalSeverity.MEDIUM,
                    affected_item_ids=affected_items[:2],
                    affected_resource_ids=[],
                    affected_sprint_ids=[],
                    affected_blocker_ids=[],
                    evidence=[SignalEvidence(
                        source_engine="metrics_engine",
                        metric_name="scope_inflation_by_reason",
                        metric_value=risky_inflation_hours,
                        threshold=10.0,
                        explanation="Scope growth concentrated in historically volatile categories",
                    )],
                    context={
                        "flag": "SCOPE_INFLATION_RISK",
                        "risky_inflation_hours": round(risky_inflation_hours, 2),
                        "inflation_by_reason": inflation_by_reason,
                    },
                    detected_at=datetime.now(timezone.utc).isoformat(),
                )
            )
        
        return signals
