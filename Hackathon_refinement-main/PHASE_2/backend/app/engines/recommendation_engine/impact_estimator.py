from __future__ import annotations

from typing import List

from app.domain.models import ProjectState
from app.engines.recommendation_engine.models import (
    ConfidenceLevel,
    ImpactEstimate,
    RecommendationAction,
    RecommendationCandidate,
    SignalEvidence,
    UpstreamEngineOutputs,
)


# Severity mapping reused for recommendation impact estimation
SEVERITY_SCORES = {
    "Critical": 40.0,
    "High": 20.0,
    "Medium": 10.0,
    "Low": 5.0,
}


class ImpactEstimator:
    def __init__(self, project_state: ProjectState, upstream: UpstreamEngineOutputs) -> None:
        self.project_state = project_state
        self.upstream = upstream

    def estimate(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        """
        Estimate impact of a recommendation candidate.
        
        This method consumes from upstream engines (ProjectMetrics, ForecastResult, RiskResult)
        rather than performing its own calculations. This ensures consistency with the
        single source of truth from upstream engines.
        """
        dispatch = {
            RecommendationAction.RESOLVE_BLOCKER: self._estimate_resolve_blocker,
            RecommendationAction.REASSIGN_ITEM: self._estimate_reassign_item,
            RecommendationAction.SPLIT_ITEM: self._estimate_split_item,
            RecommendationAction.ADVANCE_ITEM_TO_EARLIER_SPRINT: self._estimate_advance_item,
            RecommendationAction.PARALLELIZE_ITEMS: self._estimate_parallelize_items,
            RecommendationAction.REBALANCE_SPRINT_LOAD: self._estimate_rebalance_sprint_load,
            RecommendationAction.REMOVE_DEPENDENCY_BOTTLENECK: self._estimate_remove_dependency_bottleneck,
            RecommendationAction.ADD_RESOURCE_SKILL: self._estimate_add_resource_skill,
            RecommendationAction.REBASELINE_ESTIMATE: self._estimate_rebaseline_estimate,
            RecommendationAction.PAIR_REVIEWER: self._estimate_pair_reviewer,
            RecommendationAction.ESCALATE_BLOCKER_EARLY: self._estimate_escalate_blocker_early,
            RecommendationAction.FREEZE_SCOPE_REQUEST: self._estimate_freeze_scope_request,
            RecommendationAction.PULL_FORWARD_ITEM: self._estimate_pull_forward_item,
            RecommendationAction.SPLIT_AND_PAIR: self._estimate_split_and_pair,
            RecommendationAction.ASSIGN_AS_SECOND_REVIEWER: self._estimate_assign_second_reviewer,
            RecommendationAction.CROSS_TRAIN_BACKUP: self._estimate_cross_train_backup,
            RecommendationAction.INSERT_REVIEW_GATE: self._estimate_insert_review_gate,
            RecommendationAction.APPLY_RAMP_UP_DISCOUNT: self._estimate_apply_ramp_up_discount,
            RecommendationAction.RESEQUENCE_NON_CRITICAL_ITEM: self._estimate_resequence_non_critical_item,
            RecommendationAction.SWARM_ITEM: self._estimate_swarm_item,
        }
        estimator = dispatch.get(candidate.action_type)
        if estimator is None:
            return self._default_estimate(candidate)
        return estimator(candidate)

    def _estimate_resolve_blocker(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        """
        Estimate impact of resolving THIS SPECIFIC blocker, not a pro-rata
        share of all active blockers. Severity and overdue days come from
        the specific blocker this candidate targets.
        """
        blocker_id = (candidate.affected_blocker_ids or [None])[0]
        blocker = next((b for b in self.project_state.blockers if b.blocker_id == blocker_id), None)

        total_blocker_loss_days = 0.0
        if hasattr(self.upstream.forecast, "delay_breakdown") and self.upstream.forecast.delay_breakdown:
            total_blocker_loss_days = float(self.upstream.forecast.delay_breakdown.remaining_days_blocker_loss or 0.0)

        active_blockers = [b for b in self.project_state.blockers if not b.actual_resolution_date]

        severity_weight_map = {"Critical": 0.40, "High": 0.20, "Medium": 0.10, "Low": 0.05}
        this_blocker_weight = severity_weight_map.get(
            getattr(blocker, "severity", None).value if blocker and hasattr(getattr(blocker, "severity", None), "value") else "Medium",
            0.10,
        )
        total_weight = sum(
            severity_weight_map.get(b.severity.value if hasattr(b.severity, "value") else "Medium", 0.10)
            for b in active_blockers
        ) or 1.0
        this_blocker_share = this_blocker_weight / total_weight

        blocker_delay_days = total_blocker_loss_days * this_blocker_share

        overdue_days = 0
        if blocker and getattr(blocker, "target_resolution_date", None):
            from datetime import datetime, timezone
            target = blocker.target_resolution_date
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            overdue_days = max(0, (datetime.now(timezone.utc) - target).days)
            if overdue_days > 0:
                blocker_delay_days *= 1.0 + min(0.3, overdue_days * 0.05)

        impacted_count = len(getattr(blocker, "impacted_item_ids", []) or []) if blocker else 0
        blocked_hours = sum(
            next((wi.remaining_effort_hrs for wi in self.project_state.work_items if wi.item_id == iid), 0.0)
            for iid in (getattr(blocker, "impacted_item_ids", []) or [])
        ) if blocker else 0.0

        severity_label = blocker.severity.value if blocker and hasattr(blocker.severity, "value") else "Medium"
        notes = (
            f"Resolving {blocker_id or 'this blocker'} ({severity_label} severity, blocking {impacted_count} item(s), "
            f"{round(blocked_hours, 0)}h of work)"
            + (f", {overdue_days} day(s) overdue" if overdue_days > 0 else "")
            + f" recovers an estimated {round(blocker_delay_days, 1)} days of the {round(total_blocker_loss_days, 1)} "
            f"total blocker-attributable delay."
        )

        return self._build_estimate(
            candidate,
            hours_recovered=min(blocked_hours, self.upstream.forecast.remaining_effort_hours),
            delay_days=blocker_delay_days,
            risk_reduction=min(0.15 + this_blocker_share * 0.4, 0.45),
            confidence=ConfidenceLevel.HIGH,
            evidence=[self._evidence(
                "ForecastEngine",
                "delay_breakdown.blocker_loss",
                blocker_delay_days,
                0.0,
                f"This blocker accounts for {round(blocker_delay_days, 1)} of {round(total_blocker_loss_days, 1)} total blocker-attributable delay days",
            )],
            notes=notes,
        )

    def _estimate_reassign_item(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        """
        Use the actual remaining hours of the specific item(s) being
        reassigned, and compute real before/after load for both the source
        and receiving resource.
        """
        item_hours = sum(
            next((wi.remaining_effort_hrs for wi in self.project_state.work_items if wi.item_id == iid), 0.0)
            for iid in candidate.affected_item_ids
        )
        hours_recovered = min(item_hours, self.upstream.forecast.remaining_effort_hours)

        source_id = candidate.affected_resource_ids[0] if candidate.affected_resource_ids else None
        receiver_id = candidate.affected_resource_ids[1] if len(candidate.affected_resource_ids) > 1 else None

        source_dev = next((dm for dm in self.upstream.metrics.resource_metrics.developer_metrics if dm.resource_id == source_id), None) if source_id else None
        receiver_dev = next((dm for dm in self.upstream.metrics.resource_metrics.developer_metrics if dm.resource_id == receiver_id), None) if receiver_id else None

        avg_daily_velocity = max(1.0, self.upstream.metrics.actual_avg_velocity / 8.0)
        delay_days = min(item_hours / avg_daily_velocity, self.upstream.forecast.expected_delay_days)

        notes_parts = [f"Moving {round(item_hours, 0)}h of work"]
        if source_dev:
            notes_parts.append(f"reduces source resource's remaining load from {round(source_dev.remaining_effort_hours, 0)}h")
        if receiver_dev:
            notes_parts.append(f"to a receiving resource currently at {round(receiver_dev.remaining_effort_hours, 0)}h")
        notes = ", ".join(notes_parts) + "."

        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=delay_days,
            risk_reduction=min(0.05 + (item_hours / max(1.0, self.upstream.forecast.remaining_effort_hours)) * 0.2, 0.25),
            confidence=ConfidenceLevel.HIGH if (source_dev and receiver_dev) else ConfidenceLevel.MEDIUM,
            evidence=[self._evidence(
                "MetricsEngine",
                "resource_sprint_loads",
                item_hours,
                0.0,
                "Reassigning this specific item's hours reduces source resource contention"
            )],
            notes=notes,
        )

    def _sum_item_remaining_effort(self, item_ids: List[str]) -> float:
        return sum(
            next((wi.remaining_effort_hrs for wi in self.project_state.work_items if wi.item_id == iid), 0.0)
            for iid in item_ids
        )

    def _resource_metric(self, resource_id: str | None):
        if not resource_id:
            return None
        return next(
            (dm for dm in self.upstream.metrics.resource_metrics.developer_metrics if dm.resource_id == resource_id),
            None,
        )

    def _is_on_critical_path(self, item_ids: List[str]) -> bool:
        cp_items = set(self.upstream.cp_result.items_on_critical_path or [])
        return any(item_id in cp_items for item_id in item_ids)

    def _estimate_split_item(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        """
        Estimate impact of splitting a work item.
        
        Consumes:
        - average_item_effort from ProjectMetrics
        - remaining_effort_hours from ForecastResult
        """
        hours_recovered = min(
            self.upstream.metrics.average_item_effort * 0.5,
            self.upstream.forecast.remaining_effort_hours
        )
        
        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=0.0,
            risk_reduction=0.04,
            confidence=ConfidenceLevel.MEDIUM,
            evidence=[self._evidence(
                "MetricsEngine",
                "average_item_effort",
                self.upstream.metrics.average_item_effort,
                0.0,
                "Splitting large items reduces batch size and improves flow"
            )],
            notes="Splitting an item reduces batch size and can improve execution predictability",
        )

    def _estimate_advance_item(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        """
        Estimate impact of advancing an item to an earlier sprint.
        
        Consumes:
        - expected_delay_days from ForecastResult
        - remaining_effort_hours from ForecastResult
        - critical_path information from upstream
        """
        is_on_cp = any(
            item_id in self.upstream.cp_result.items_on_critical_path
            for item_id in candidate.affected_item_ids
        )

        spillover_days = 0.0
        if hasattr(self.upstream.forecast, "delay_breakdown") and self.upstream.forecast.delay_breakdown:
            spillover_days = float(self.upstream.forecast.delay_breakdown.remaining_days_spillover or 0.0)

        item_hours = sum(
            next((wi.remaining_effort_hrs for wi in self.project_state.work_items if wi.item_id == iid), 0.0)
            for iid in candidate.affected_item_ids
        )

        if is_on_cp:
            cap = min(spillover_days * 0.6, self.upstream.forecast.expected_delay_days * 0.5)
        else:
            cap = min(spillover_days * 0.3, self.upstream.forecast.expected_delay_days * 0.25)

        remaining_effort = float(getattr(self.upstream.forecast, "remaining_effort_hours", 0.0) or 0.0)
        item_fraction = item_hours / max(remaining_effort, 1.0)
        delay_reduction = cap * item_fraction

        hours_recovered = min(item_hours, self.upstream.forecast.remaining_effort_hours)

        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=delay_reduction,
            risk_reduction=0.08 if is_on_cp else 0.06,
            confidence=ConfidenceLevel.MEDIUM if is_on_cp else ConfidenceLevel.LOW,
            evidence=[self._evidence(
                "ForecastEngine",
                "expected_delay_days",
                self.upstream.forecast.expected_delay_days,
                0.0,
                f"Advancing item {'on critical path' if is_on_cp else 'reduces schedule pressure'}"
            )],
            notes=(
                f"Advancing an item {'on the critical path' if is_on_cp else ''} "
                f"can reduce downstream schedule pressure"
            ),
        )

    def _estimate_parallelize_items(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        """
        Estimate impact of parallelizing work items.
        
        Consumes:
        - critical_path sequence length from CriticalPathResult
        - dependency_count from ProjectMetrics
        """
        cp_length = float(len(self.upstream.cp_result.critical_path or self.upstream.cp_result.critical_path_items or []))
        dependency_count = float(getattr(self.upstream.metrics, 'dependency_count', 0.0) or 0.0)
        
        # Impact depends on current dependency pressure
        dependency_pressure = min(1.0, dependency_count / max(len(self.project_state.work_items), 1))
        
        hours_recovered = min(
            self.upstream.forecast.remaining_effort_hours * 0.12 * dependency_pressure,
            self.upstream.forecast.remaining_effort_hours
        )
        
        delay_reduction = min(
            self.upstream.forecast.expected_delay_days * 0.2 * dependency_pressure,
            1.5
        )
        
        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=delay_reduction,
            risk_reduction=0.07 if dependency_pressure > 0.5 else 0.04,
            confidence=ConfidenceLevel.LOW,
            evidence=[self._evidence(
                "CriticalPathEngine",
                "critical_path_items",
                float(len(self.upstream.cp_result.critical_path_items)),
                0.0,
                "Parallelizing independent items can reduce serial dependency drag"
            )],
            notes="Parallelizing items has impact proportional to dependency pressure",
        )

    def _estimate_rebalance_sprint_load(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        """
        Estimate impact of rebalancing sprint load.
        
        Consumes:
        - sprint_metrics from ProjectMetrics for current utilization
        """
        # Find affected sprint metrics to assess current load imbalance
        underutilized_sprints = sum(
            1 for sm in self.upstream.metrics.sprint_metrics
            if sm.completion_pct < 0.5
        )
        overutilized_sprints = sum(
            1 for sm in self.upstream.metrics.sprint_metrics
            if sm.completion_pct > 1.0
        )
        
        imbalance = (underutilized_sprints + overutilized_sprints) / max(
            len(self.upstream.metrics.sprint_metrics), 1
        )
        
        hours_recovered = min(
            self.upstream.metrics.average_item_effort * 0.25 * imbalance,
            self.upstream.forecast.remaining_effort_hours
        )
        
        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=0.0,
            risk_reduction=0.03 * imbalance,
            confidence=ConfidenceLevel.LOW,
            evidence=[self._evidence(
                "MetricsEngine",
                "sprint_metrics",
                imbalance,
                0.0,
                f"Sprint load imbalance detected in {round(imbalance * 100, 1)}% of sprints"
            )],
            notes="Sprint rebalancing has limited schedule leverage without slack",
        )

    def _estimate_remove_dependency_bottleneck(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        """
        Estimate impact of removing a dependency bottleneck.
        
        Consumes:
        - critical_path information from upstream
        - dependency metrics from ProjectMetrics
        """
        # Find bottleneck items: those with high in-degree on the critical path
        cp_items = self.upstream.cp_result.items_on_critical_path or []
        
        hours_recovered = min(
            self.upstream.forecast.remaining_effort_hours * 0.15,
            self.upstream.forecast.remaining_effort_hours
        )
        
        # Impact if bottleneck is on critical path
        is_cp_bottleneck = any(
            item_id in cp_items
            for item_id in candidate.affected_item_ids
        )
        
        delay_reduction = min(
            self.upstream.forecast.expected_delay_days * (0.35 if is_cp_bottleneck else 0.15),
            2.5
        )
        
        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=delay_reduction,
            risk_reduction=0.10 if is_cp_bottleneck else 0.06,
            confidence=ConfidenceLevel.MEDIUM if is_cp_bottleneck else ConfidenceLevel.LOW,
            evidence=[self._evidence(
                "DependencyGraphEngine",
                "dependency_count",
                float(self.upstream.metrics.dependency_count or 0.0),
                0.0,
                "Removing a dependency bottleneck eases critical path pressure"
            )],
            notes="Impact depends on whether bottleneck is on the critical path",
        )

    def _estimate_rebaseline_estimate(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        item_hours = sum(next((wi.remaining_effort_hrs for wi in self.project_state.work_items if wi.item_id == iid), 0.0) for iid in candidate.affected_item_ids)
        return self._build_estimate(
            candidate,
            hours_recovered=min(item_hours * 0.2, self.upstream.forecast.remaining_effort_hours),
            delay_days=min(0.5, self.upstream.forecast.expected_delay_days * 0.2),
            risk_reduction=0.08,
            confidence=ConfidenceLevel.MEDIUM,
            evidence=[self._evidence("ForecastEngine", "remaining_effort_hours", item_hours, 0.0, "Historical overrun pattern justifies rebaselining the estimate")],
            notes="Rebaselining estimates based on prior overrun history reduces planning error for the affected work.",
        )

    def _estimate_pair_reviewer(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        return self._build_estimate(
            candidate,
            hours_recovered=10.0,
            delay_days=0.2,
            risk_reduction=0.05,
            confidence=ConfidenceLevel.MEDIUM,
            evidence=[self._evidence("MetricsEngine", "review_pairing", 1.0, 0.0, "Pairing guidance reduces rework risk on shared work")],
            notes="Pairing a reviewer with the work reduces rework and review churn.",
        )

    def _estimate_escalate_blocker_early(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        return self._build_estimate(
            candidate,
            hours_recovered=min(20.0, self.upstream.forecast.remaining_effort_hours),
            delay_days=min(1.0, self.upstream.forecast.expected_delay_days * 0.25),
            risk_reduction=0.1,
            confidence=ConfidenceLevel.HIGH,
            evidence=[self._evidence("ForecastEngine", "expected_delay_days", self.upstream.forecast.expected_delay_days, 0.0, "Early escalation prevents repeated blocker loss")],
            notes="Escalating the blocker earlier protects the schedule from repeated delay.",
        )

    def _estimate_freeze_scope_request(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        return self._build_estimate(
            candidate,
            hours_recovered=15.0,
            delay_days=0.5,
            risk_reduction=0.07,
            confidence=ConfidenceLevel.MEDIUM,
            evidence=[self._evidence("ForecastEngine", "scope_growth_hours", self.upstream.forecast.scope_growth_hours, 0.0, "Scope growth is the driver behind the request")],
            notes="Freezing scope limits surprise work and preserves the planned schedule.",
        )

    def _estimate_pull_forward_item(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        return self._build_estimate(
            candidate,
            hours_recovered=0.0,
            delay_days=0.3,
            risk_reduction=0.04,
            confidence=ConfidenceLevel.LOW,
            evidence=[self._evidence("ForecastEngine", "expected_delay_days", self.upstream.forecast.expected_delay_days, 0.0, "Pulling work forward reduces sequencing pressure")],
            notes="Pulling the item forward helps protect the critical path when sequencing pressure is present.",
        )

    def _estimate_split_and_pair(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        return self._build_estimate(
            candidate,
            hours_recovered=15.0,
            delay_days=0.4,
            risk_reduction=0.06,
            confidence=ConfidenceLevel.MEDIUM,
            evidence=[self._evidence("MetricsEngine", "average_item_effort", self.upstream.metrics.average_item_effort, 0.0, "Splitting the work keeps review handoff manageable")],
            notes="Splitting the work and pairing it with a second reviewer reduces review contention.",
        )

    def _estimate_assign_second_reviewer(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        return self._build_estimate(
            candidate,
            hours_recovered=8.0,
            delay_days=0.2,
            risk_reduction=0.05,
            confidence=ConfidenceLevel.MEDIUM,
            evidence=[self._evidence("MetricsEngine", "review_pairing", 1.0, 0.0, "A second reviewer improves review throughput")],
            notes="Assigning a second reviewer reduces review turnaround delays.",
        )

    def _estimate_cross_train_backup(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        target_resource_id = candidate.affected_resource_ids[0] if candidate.affected_resource_ids else None
        resource_risk_score = float(
            getattr(self.upstream.risk_result, "resource_risk", {}).score
            if hasattr(self.upstream.risk_result, "resource_risk") else 0.0
        )
        resource_metric = self._resource_metric(target_resource_id)
        risk_factor = min(0.08 + resource_risk_score * 0.2, 0.25)
        if resource_metric and resource_metric.allocation_pct * resource_metric.availability_pct > 0.8:
            risk_factor = min(risk_factor + 0.05, 0.30)

        notes = (
            "No current-window delay reduction is estimated for cross-training; "
            "the benefit is future sprint resilience from backup coverage."
        )
        if target_resource_id:
            notes += f" Target resource: {target_resource_id}."

        return self._build_estimate(
            candidate,
            hours_recovered=0.0,
            delay_days=0.0,
            risk_reduction=risk_factor,
            confidence=ConfidenceLevel.MEDIUM if resource_risk_score > 0.1 else ConfidenceLevel.LOW,
            evidence=[self._evidence(
                "RiskEngine",
                "resource_risk",
                resource_risk_score,
                0.0,
                "Cross-training backup coverage reduces single-resource dependency risk",
            )],
            notes=notes,
        )

    def _estimate_insert_review_gate(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        item_hours = self._sum_item_remaining_effort(candidate.affected_item_ids)
        is_on_cp = self._is_on_critical_path(candidate.affected_item_ids)
        hours_recovered = min(item_hours * 0.06, self.upstream.forecast.remaining_effort_hours)
        delay_days = min(self.upstream.forecast.expected_delay_days * (0.18 if is_on_cp else 0.1), 1.0)
        risk_reduction = min(0.04 + (item_hours / max(1.0, self.upstream.forecast.remaining_effort_hours)) * 0.1, 0.12)
        confidence = ConfidenceLevel.HIGH if is_on_cp else ConfidenceLevel.MEDIUM
        notes = (
            "A review gate reduces rework by tightening quality feedback loops and "
            f"is estimated to recover {round(hours_recovered,1)}h of effort from the affected item(s)."
        )

        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=delay_days,
            risk_reduction=risk_reduction,
            confidence=confidence,
            evidence=[self._evidence(
                "ForecastEngine",
                "expected_delay_days",
                self.upstream.forecast.expected_delay_days,
                0.0,
                "A review gate can reduce schedule pressure by lowering rework-driven delay",
            )],
            notes=notes,
        )

    def _estimate_apply_ramp_up_discount(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        target_resource_id = candidate.affected_resource_ids[0] if candidate.affected_resource_ids else None
        resource_metric = self._resource_metric(target_resource_id)
        load_factor = 0.0
        if resource_metric is not None:
            load_factor = 1.0 - (resource_metric.allocation_pct * resource_metric.availability_pct)
        risk_reduction = min(0.04 + max(0.0, load_factor) * 0.12, 0.12)
        confidence = ConfidenceLevel.MEDIUM if load_factor > 0.2 else ConfidenceLevel.LOW
        notes = (
            "Applying a ramp-up discount improves forecast realism for a newly ramped resource "
            "without claiming immediate current-window delay reduction."
        )
        if target_resource_id:
            notes += f" Target resource: {target_resource_id}."

        return self._build_estimate(
            candidate,
            hours_recovered=0.0,
            delay_days=0.0,
            risk_reduction=risk_reduction,
            confidence=confidence,
            evidence=[self._evidence(
                "ForecastEngine",
                "remaining_effort_hours",
                self.upstream.forecast.remaining_effort_hours,
                0.0,
                "A ramp-up discount reduces forecast error risk for new resources",
            )],
            notes=notes,
        )

    def _estimate_resequence_non_critical_item(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        item_hours = self._sum_item_remaining_effort(candidate.affected_item_ids)
        item_fraction = item_hours / max(self.upstream.forecast.remaining_effort_hours, 1.0)
        is_on_cp = self._is_on_critical_path(candidate.affected_item_ids)
        hours_recovered = min(item_hours * 0.03, self.upstream.forecast.remaining_effort_hours)
        delay_days = min(self.upstream.forecast.expected_delay_days * (0.14 if is_on_cp else 0.08) * item_fraction, 1.2)
        risk_reduction = 0.06 if is_on_cp else 0.05

        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=delay_days,
            risk_reduction=risk_reduction,
            confidence=ConfidenceLevel.MEDIUM,
            evidence=[self._evidence(
                "CriticalPathEngine",
                "critical_path_duration_hours",
                float(self.upstream.cp_result.critical_path_duration_hours or 0.0),
                0.0,
                "Resequencing non-critical work protects critical-path throughput",
            )],
            notes=(
                "Resequencing non-critical work reduces shared queue pressure for affected downstream work "
                f"and is estimated to recover {round(hours_recovered,1)}h of effort." 
            ),
        )

    def _estimate_swarm_item(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        item_hours = self._sum_item_remaining_effort(candidate.affected_item_ids)
        is_on_cp = self._is_on_critical_path(candidate.affected_item_ids)
        hours_recovered = min(item_hours * 0.15, self.upstream.forecast.remaining_effort_hours)
        delay_days = min(self.upstream.forecast.expected_delay_days * (0.35 if is_on_cp else 0.18), 2.0)
        risk_reduction = 0.08 if is_on_cp else 0.05

        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=delay_days,
            risk_reduction=risk_reduction,
            confidence=ConfidenceLevel.MEDIUM,
            evidence=[self._evidence(
                "CriticalPathEngine",
                "critical_path_duration_hours",
                float(self.upstream.cp_result.critical_path_duration_hours or 0.0),
                0.0,
                "Swarming a bottleneck item shortens critical-path duration in the current forecast window",
            )],
            notes=(
                "Swarming the bottleneck item reduces the item's remaining work and critical-path pressure "
                f"by an estimated {round(hours_recovered,1)}h."
            ),
        )

    def _estimate_add_resource_skill(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        """
        Estimate impact of adding resource skill coverage.
        
        Consumes:
        - risk_result from RiskEngine for resource risk
        - resource_metrics from ProjectMetrics
        """
        # Skill coverage helps when resource risk is high
        resource_risk_score = float(
            getattr(self.upstream.risk_result, "resource_risk", {}).score
            if hasattr(self.upstream.risk_result, "resource_risk") else 0.0
        )
        
        hours_recovered = min(
            self.upstream.metrics.average_item_effort * 0.3 * min(1.0, resource_risk_score),
            self.upstream.forecast.remaining_effort_hours
        )
        
        return self._build_estimate(
            candidate,
            hours_recovered=hours_recovered,
            delay_days=0.0,
            risk_reduction=0.08 if resource_risk_score > 0.5 else 0.04,
            confidence=ConfidenceLevel.MEDIUM if resource_risk_score > 0.5 else ConfidenceLevel.LOW,
            evidence=[self._evidence(
                "RiskEngine",
                "resource_risk_score",
                resource_risk_score,
                0.0,
                "Skill coverage improves capacity resilience"
            )],
            notes="Impact depends on current resource risk level",
        )

    def _default_estimate(self, candidate: RecommendationCandidate) -> ImpactEstimate:
        return self._build_estimate(
            candidate,
            hours_recovered=0.0,
            delay_days=0.0,
            risk_reduction=0.0,
            confidence=ConfidenceLevel.LOW,
            evidence=[self._evidence("ForecastEngine", "remaining_effort_hours", self.upstream.forecast.remaining_effort_hours, 0.0, "No direct impact estimate available")],
            notes="Fell back to a neutral estimate",
        )

    def _build_estimate(
        self,
        candidate: RecommendationCandidate,
        *,
        hours_recovered: float,
        delay_days: float,
        risk_reduction: float,
        confidence: ConfidenceLevel,
        evidence: List[SignalEvidence],
        notes: str,
    ) -> ImpactEstimate:
        cap = max(0.0, self.upstream.forecast.remaining_effort_hours)
        return ImpactEstimate(
            estimated_hours_recovered=float(min(max(hours_recovered, 0.0), cap)),
            estimated_delay_reduction_days=float(max(delay_days, 0.0)),
            estimated_risk_reduction=float(max(risk_reduction, 0.0)),
            confidence=confidence,
            evidence=evidence,
            calculation_notes=notes,
        )

    def _evidence(self, source_engine: str, metric_name: str, metric_value: float, threshold: float, explanation: str) -> SignalEvidence:
        return SignalEvidence(
            source_engine=source_engine,
            metric_name=metric_name,
            metric_value=float(metric_value),
            threshold=float(threshold),
            explanation=explanation,
        )
