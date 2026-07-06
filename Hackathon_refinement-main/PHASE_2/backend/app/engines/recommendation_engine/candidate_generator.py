from __future__ import annotations

from typing import Any, Dict, List

from app.domain.models import ProjectState
from app.engines.recommendation_engine.models import (
    HistoricalPattern,
    OpportunitySignal,
    RecommendationAction,
    RecommendationCandidate,
    SignalCategory,
    SignalSeverity,
    UpstreamEngineOutputs,
    historical_pattern_payload,
    stable_id,
)


class CandidateGenerator:
    def __init__(self, project_state: ProjectState, upstream: UpstreamEngineOutputs) -> None:
        self.project_state = project_state
        self.upstream = upstream
        self._active_signal: OpportunitySignal | None = None

    def generate(self, signals: List[OpportunitySignal]) -> List[RecommendationCandidate]:
        emitted: Dict[str, RecommendationCandidate] = {}
        for signal in signals:
            self._active_signal = signal
            try:
                if signal.category == SignalCategory.BLOCKER:
                    for candidate in self._from_blocker_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.CAPACITY:
                    for candidate in self._from_capacity_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.SPRINT:
                    for candidate in self._from_sprint_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.CRITICAL_PATH:
                    for candidate in self._from_critical_path_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.SCHEDULE:
                    for candidate in self._from_schedule_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.ESTIMATION_RELIABILITY:
                    for candidate in self._from_estimation_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.SPILLOVER:
                    for candidate in self._from_spillover_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.SPOF:
                    for candidate in self._from_spof_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.RECURRING_BLOCKER:
                    for candidate in self._from_recurring_blocker_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.REWORK_LOOP:
                    for candidate in self._from_rework_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.RAMP_UP:
                    for candidate in self._from_ramp_up_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.RESEQUENCING:
                    for candidate in self._from_resequencing_signal(signal):
                        self._deduplicate(emitted, candidate)
                elif signal.category == SignalCategory.SWARM_TRADEOFF:
                    for candidate in self._from_swarm_signal(signal):
                        self._deduplicate(emitted, candidate)
            finally:
                self._active_signal = None

        return [candidate for candidate in emitted.values() if self._check_feasibility(candidate)]

    def _from_blocker_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        blocker_ids = signal.affected_blocker_ids or []
        if blocker_ids:
            blocker_id = blocker_ids[0]

            # Look up actual blocker from project state for rich, actionable context
            blocker = next(
                (b for b in self.project_state.blockers if b.blocker_id == blocker_id),
                None,
            )
            category = blocker.category.value if blocker else "Unknown"
            owner = (blocker.owner or "Unassigned") if blocker else "Unassigned"
            impacted = signal.affected_item_ids
            impacted_summary = ", ".join(impacted[:3]) + (f" (+{len(impacted) - 3} more)" if len(impacted) > 3 else "")

            # Title: short, actionable, shown on the card
            title = f"Resolve {category} blocker — {blocker_id} (Owner: {owner})"

            # Description: truncated escalation notes + impacted items
            raw_notes = (blocker.description or "") if blocker else ""
            short_notes = (raw_notes[:200] + "…") if len(raw_notes) > 200 else raw_notes
            description = (
                f"{short_notes} | Blocking {len(impacted)} item(s): {impacted_summary}"
                if short_notes
                else f"{blocker_id} is blocking {len(impacted)} item(s): {impacted_summary}"
            )

            candidates.append(self._build_candidate(
                action_type=RecommendationAction.RESOLVE_BLOCKER,
                title=title,
                description=description,
                affected_item_ids=signal.affected_item_ids,
                affected_resource_ids=[],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=[blocker_id],
                root_signal_id=signal.signal_id,
                simulation_params={"target_blocker_id": blocker_id},
                feasibility_checks={"blocker_active": True},
            ))
        for item_id in signal.affected_item_ids[:1]:
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.ADVANCE_ITEM_TO_EARLIER_SPRINT,
                title=f"Advance item ({item_id})",
                description=f"Advance work item {item_id} to an earlier sprint",
                affected_item_ids=[item_id],
                affected_resource_ids=[],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_item_id": item_id},
                feasibility_checks={"has_capacity": True},
            ))
        return candidates

    def _from_capacity_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        if not signal.affected_resource_ids:
            return candidates

        resource_id = signal.affected_resource_ids[0]
        item_id = signal.affected_item_ids[0] if signal.affected_item_ids else ""
        flag = signal.context.get("flag", "") if signal.context else ""
        is_cp_owner = bool(signal.context.get("is_single_owner_of_cp", False)) if signal.context else False
        load_ratio = float(signal.context.get("load_ratio", 0.0)) if signal.context else 0.0

        if flag == "UNDERUTILIZED":
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.REBALANCE_SPRINT_LOAD,
                title=f"Rebalance sprint load ({resource_id})",
                description=f"Redistribute work away from underutilized resource {resource_id}",
                affected_item_ids=signal.affected_item_ids[:2],
                affected_resource_ids=[resource_id],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_resource_id": resource_id, "load_ratio": load_ratio},
                feasibility_checks={"resource_exists": True, "has_capacity": True},
            ))
            return candidates

        if is_cp_owner and flag == "OVERLOADED":
            item = next((wi for wi in self.project_state.work_items if wi.item_id == item_id), None) if item_id else None
            item_hours = float(getattr(item, "remaining_effort_hrs", 0.0) or 0.0) if item else 0.0
            SPLIT_THRESHOLD_HRS = 24.0  # items larger than this are worth splitting; smaller ones are cheaper to just reassign
            if item_id and item_hours > SPLIT_THRESHOLD_HRS:
                candidates.append(self._build_candidate(
                    action_type=RecommendationAction.SPLIT_ITEM,
                    title=f"Split item to relieve CP owner ({item_id})",
                    description=f"Split work item {item_id} to reduce critical path ownership pressure on {resource_id}",
                    affected_item_ids=[item_id],
                    affected_resource_ids=[resource_id],
                    affected_sprint_ids=signal.affected_sprint_ids,
                    affected_blocker_ids=signal.affected_blocker_ids,
                    root_signal_id=signal.signal_id,
                    simulation_params={"target_item_id": item_id, "target_resource_id": resource_id},
                    feasibility_checks={"item_large_enough": True},
                ))
            else:
                candidates.append(self._build_candidate(
                    action_type=RecommendationAction.REASSIGN_ITEM,
                    title=f"Reassign work ({item_id or resource_id})",
                    description=f"Move work away from overloaded resource {resource_id}",
                    affected_item_ids=signal.affected_item_ids,
                    affected_resource_ids=[resource_id],
                    affected_sprint_ids=signal.affected_sprint_ids,
                    affected_blocker_ids=signal.affected_blocker_ids,
                    root_signal_id=signal.signal_id,
                    simulation_params={"target_resource_id": resource_id},
                    feasibility_checks={"resource_exists": True, "has_capacity": True},
                ))
            return candidates

        if flag == "OVERLOADED" and (len(signal.affected_sprint_ids) > 1 or load_ratio > 1.3 or signal.severity == SignalSeverity.HIGH):
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.ADD_RESOURCE_SKILL,
                title=f"Add resource skill ({resource_id})",
                description=f"Add capacity or skill support for overloaded resource {resource_id}",
                affected_item_ids=signal.affected_item_ids[:1],
                affected_resource_ids=[resource_id],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_resource_id": resource_id, "load_ratio": load_ratio},
            ))
            return candidates

        if flag == "OVERLOADED":
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.REASSIGN_ITEM,
                title=f"Reassign item ({item_id or resource_id})",
                description=f"Reassign work to ease overloaded resource {resource_id}",
                affected_item_ids=signal.affected_item_ids,
                affected_resource_ids=[resource_id],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_resource_id": resource_id, "target_item_id": item_id},
                feasibility_checks={"resource_exists": True, "has_capacity": True},
            ))
            return candidates

        return candidates

    def _from_sprint_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        if signal.affected_item_ids:
            item_id = signal.affected_item_ids[0]
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.ADVANCE_ITEM_TO_EARLIER_SPRINT,
                title=f"Advance item ({item_id})",
                description=f"Advance sprint-bound item {item_id}",
                affected_item_ids=[item_id],
                affected_resource_ids=[],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_item_id": item_id},
                feasibility_checks={"has_capacity": True},
            ))
        return candidates

    def _from_critical_path_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        flag = signal.context.get("flag", "") if signal.context else ""

        if flag == "NEAR_CRITICAL":
            for item_id in signal.affected_item_ids[:2]:
                candidates.append(self._build_candidate(
                    action_type=RecommendationAction.PARALLELIZE_ITEMS,
                    title=f"Parallelize item ({item_id})",
                    description=f"Reduce sequential dependency risk by parallelizing work around {item_id}",
                    affected_item_ids=[item_id],
                    affected_resource_ids=[],
                    affected_sprint_ids=signal.affected_sprint_ids,
                    affected_blocker_ids=signal.affected_blocker_ids,
                    root_signal_id=signal.signal_id,
                    simulation_params={"target_item_id": item_id},
                    feasibility_checks={"has_capacity": True},
                ))
            return candidates

        if flag == "DEPENDENCY_BOTTLENECK":
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.REMOVE_DEPENDENCY_BOTTLENECK,
                title="Remove dependency bottleneck",
                description="Reduce critical path dependency fan-in by removing or decoupling dependency bottlenecks.",
                affected_item_ids=signal.affected_item_ids,
                affected_resource_ids=[],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"dependency_items": signal.affected_item_ids},
                feasibility_checks={"dependencies_editable": True},
            ))
            return candidates

        for item_id in signal.affected_item_ids[:2]:
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.ADVANCE_ITEM_TO_EARLIER_SPRINT,
                title=f"Advance item ({item_id})",
                description=f"Protect critical path item {item_id}",
                affected_item_ids=[item_id],
                affected_resource_ids=[],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_item_id": item_id},
                feasibility_checks={"has_capacity": True},
            ))
        return candidates

    def _from_schedule_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        if not signal.affected_item_ids:
            return candidates
        
        # Get the context flag to understand which type of schedule issue this is
        flag = signal.context.get("flag", "SCHEDULE_GAP") if signal.context else "SCHEDULE_GAP"
        schedule_gap_hours = float(signal.context.get("schedule_gap_hours", 0.0)) if signal.context else 0.0
        
        # Generate SPLIT_ITEM candidate for the first item
        item_id = signal.affected_item_ids[0]
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.SPLIT_ITEM,
            title=f"Split item ({item_id})",
            description=f"Split work item {item_id} to reduce schedule pressure",
            affected_item_ids=[item_id],
            affected_resource_ids=[],
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"target_item_id": item_id},
            feasibility_checks={"item_large_enough": True},
        ))
        
        # Generate REBALANCE_SPRINT_LOAD candidate
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.REBALANCE_SPRINT_LOAD,
            title="Rebalance sprint load",
            description=f"Rebalance work items across sprints to address schedule pressure (gap: {schedule_gap_hours:.1f}h)",
            affected_item_ids=signal.affected_item_ids[:2],  # Top 2 items
            affected_resource_ids=[],
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"gap_hours": schedule_gap_hours},
            feasibility_checks={"has_future_sprints": True},
        ))
        
        # Generate ADD_RESOURCE_SKILL candidate when gap is large
        if schedule_gap_hours > 20.0:
            # Try to infer required skill from the top affected item for a richer candidate
            required_skill = "General"
            if signal.affected_item_ids:
                item_id = signal.affected_item_ids[0]
                item = next((wi for wi in self.project_state.work_items if wi.item_id == item_id), None)
                if item and getattr(item, "required_skill", None):
                    required_skill = item.required_skill

            candidates.append(self._build_candidate(
                action_type=RecommendationAction.ADD_RESOURCE_SKILL,
                title="Add resource capacity",
                description=f"Add resources or increase capacity to close schedule gap ({schedule_gap_hours:.1f}h)",
                affected_item_ids=signal.affected_item_ids[:1],
                affected_resource_ids=[],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"gap_hours": schedule_gap_hours, "required_skill": required_skill},
                feasibility_checks={"budget_available": True},
            ))
        
        return candidates

    def _from_estimation_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        resource_id = (signal.context.get("resource_id") or signal.affected_resource_ids[0] if signal.affected_resource_ids else None)
        if not resource_id:
            return candidates
        item_ids = signal.affected_item_ids[:1]
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.REBASELINE_ESTIMATE,
            title=f"Rebaseline estimates ({resource_id})",
            description="Adjust estimates using the historical overrun pattern to improve forecast quality.",
            affected_item_ids=item_ids,
            affected_resource_ids=[resource_id],
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"target_resource_id": resource_id},
            feasibility_checks={"resource_exists": True},
        ))
        return candidates

    def _from_spillover_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        cause = (signal.context.get("cause") or "dependency_blocked").lower()
        action = RecommendationAction.ESCALATE_BLOCKER_EARLY
        title = "Escalate blocker early"
        if cause == "resource_unavailable":
            action = RecommendationAction.REBALANCE_SPRINT_LOAD
            title = "Rebalance sprint load"
        elif cause == "estimate_wrong":
            action = RecommendationAction.REBASELINE_ESTIMATE
            title = "Rebaseline estimate"
        elif cause == "scope_growth":
            action = RecommendationAction.FREEZE_SCOPE_REQUEST
            title = "Freeze scope request"
        elif cause == "toolchain_friction":
            action = RecommendationAction.INSERT_REVIEW_GATE
            title = "Insert review gate"
        item_ids = signal.affected_item_ids[:1]
        candidates.append(self._build_candidate(
            action_type=action,
            title=title,
            description="Address the recurring spillover pattern before it causes a late sprint carryover.",
            affected_item_ids=item_ids,
            affected_resource_ids=signal.affected_resource_ids,
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"cause": cause},
            feasibility_checks={"has_capacity": True},
        ))
        return candidates

    def _from_spof_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        resource_id = signal.affected_resource_ids[0] if signal.affected_resource_ids else None
        item_ids = signal.affected_item_ids[:1]
        if not resource_id:
            return candidates
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.CROSS_TRAIN_BACKUP,
            title=f"Cross-train backup ({resource_id})",
            description="Create backup coverage for the single point of failure before it becomes a delivery issue.",
            affected_item_ids=item_ids,
            affected_resource_ids=[resource_id],
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"target_resource_id": resource_id},
            feasibility_checks={"resource_exists": True},
        ))
        return candidates

    def _from_recurring_blocker_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        blocker_ids = signal.affected_blocker_ids[:1]
        if not blocker_ids:
            return candidates
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.ESCALATE_BLOCKER_EARLY,
            title="Escalate recurring blocker early",
            description="Escalate the recurring blocker category earlier to avoid repeated delay.",
            affected_item_ids=signal.affected_item_ids,
            affected_resource_ids=signal.affected_resource_ids,
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"blocker_category": signal.context.get("category")},
            feasibility_checks={"blocker_active": True},
        ))
        return candidates

    def _from_rework_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        item_ids = signal.affected_item_ids[:1]
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.INSERT_REVIEW_GATE,
            title="Insert review gate",
            description="Add a review or QA gate to interrupt the rework loop before it repeats.",
            affected_item_ids=item_ids,
            affected_resource_ids=signal.affected_resource_ids,
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"category": signal.context.get("category")},
            feasibility_checks={"has_capacity": True},
        ))
        return candidates

    def _from_ramp_up_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        resource_id = signal.affected_resource_ids[0] if signal.affected_resource_ids else None
        item_ids = signal.affected_item_ids[:1]
        if not resource_id:
            return candidates
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.APPLY_RAMP_UP_DISCOUNT,
            title=f"Apply ramp-up discount ({resource_id})",
            description="Use a temporary forecast discount for a newly ramped resource to improve estimate realism.",
            affected_item_ids=item_ids,
            affected_resource_ids=[resource_id],
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"target_resource_id": resource_id},
            feasibility_checks={"resource_exists": True},
        ))
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.PAIR_REVIEWER,
            title=f"Pair reviewer ({resource_id})",
            description="Pair a reviewer with the new joiner on critical path work to reduce rework risk.",
            affected_item_ids=item_ids,
            affected_resource_ids=[resource_id],
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"target_resource_id": resource_id},
            feasibility_checks={"resource_exists": True},
        ))
        return candidates

    def _from_resequencing_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        item_ids = signal.affected_item_ids[:1]
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.RESEQUENCE_NON_CRITICAL_ITEM,
            title="Resequence non-critical item",
            description="Move the non-critical item off the shared resource's plate to protect critical path work.",
            affected_item_ids=item_ids,
            affected_resource_ids=signal.affected_resource_ids,
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"critical_item": signal.context.get("critical_item_id")},
            feasibility_checks={"has_capacity": True},
        ))
        return candidates

    def _from_swarm_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        item_ids = signal.affected_item_ids[:1]
        candidates.append(self._build_candidate(
            action_type=RecommendationAction.SWARM_ITEM,
            title="Swarm the critical-path item",
            description="Add a second resource to swarm the bottleneck item with explicit trade-off handling.",
            affected_item_ids=item_ids,
            affected_resource_ids=signal.affected_resource_ids,
            affected_sprint_ids=signal.affected_sprint_ids,
            affected_blocker_ids=signal.affected_blocker_ids,
            root_signal_id=signal.signal_id,
            simulation_params={"days_saved": signal.context.get("days_saved_on_critical_path")},
            feasibility_checks={"resource_exists": True},
        ))
        return candidates

    def _deduplicate(self, existing: Dict[str, RecommendationCandidate], new: RecommendationCandidate) -> None:
        existing_candidate = existing.get(new.recommendation_id)
        if existing_candidate is None:
            existing[new.recommendation_id] = new
            return
        merged_ids = sorted(set(existing_candidate.supporting_signal_ids) | set(new.supporting_signal_ids))
        existing[existing_candidate.recommendation_id] = RecommendationCandidate(
            recommendation_id=existing_candidate.recommendation_id,
            action_type=existing_candidate.action_type,
            title=existing_candidate.title,
            description=existing_candidate.description,
            affected_item_ids=existing_candidate.affected_item_ids,
            affected_resource_ids=existing_candidate.affected_resource_ids,
            affected_sprint_ids=existing_candidate.affected_sprint_ids,
            affected_blocker_ids=existing_candidate.affected_blocker_ids,
            root_cause_signal_id=existing_candidate.root_cause_signal_id,
            supporting_signal_ids=merged_ids,
            simulation_params=existing_candidate.simulation_params,
            feasibility_checks=existing_candidate.feasibility_checks,
        )

    def _check_feasibility(self, candidate: RecommendationCandidate) -> bool:
        return all(candidate.feasibility_checks.values()) if candidate.feasibility_checks else True

    def _build_candidate(
        self,
        *,
        action_type: RecommendationAction,
        title: str,
        description: str,
        affected_item_ids: List[str],
        affected_resource_ids: List[str],
        affected_sprint_ids: List[str],
        affected_blocker_ids: List[str],
        root_signal_id: str,
        simulation_params: Dict[str, Any],
        feasibility_checks: Dict[str, bool],
    ) -> RecommendationCandidate:
        target_ids = list(affected_item_ids) + list(affected_resource_ids) + list(affected_sprint_ids) + list(affected_blocker_ids)
        merged_params = dict(simulation_params)
        if self._active_signal is not None:
            historical_pattern = self._active_signal.context.get("historical_pattern")
            if historical_pattern is not None:
                merged_params.setdefault("historical_pattern", historical_pattern)
            if "signal_category" not in merged_params:
                merged_params["signal_category"] = self._active_signal.category.value
        if self._active_signal is not None and "historical_pattern" not in merged_params:
            merged_params.setdefault("historical_pattern", self._build_fallback_historical_pattern(self._active_signal))
        return RecommendationCandidate(
            recommendation_id=stable_id(action_type.value, target_ids),
            action_type=action_type,
            title=title,
            description=description,
            affected_item_ids=affected_item_ids,
            affected_resource_ids=affected_resource_ids,
            affected_sprint_ids=affected_sprint_ids,
            affected_blocker_ids=affected_blocker_ids,
            root_cause_signal_id=root_signal_id,
            supporting_signal_ids=[root_signal_id],
            simulation_params=merged_params,
            feasibility_checks=feasibility_checks,
        )

    def _build_fallback_historical_pattern(self, signal: OpportunitySignal) -> Dict[str, Any] | None:
        resource_id = signal.affected_resource_ids[0] if signal.affected_resource_ids else None
        occurrences = signal.affected_item_ids or signal.affected_blocker_ids or signal.affected_resource_ids or ["fallback"]
        pattern = HistoricalPattern(
            pattern_type=f"Fallback{signal.category.value}",
            resource_id=resource_id,
            blocker_category=None,
            sample_size=max(1, len(occurrences)),
            metric_name=signal.category.value,
            metric_value=1.0,
            historical_occurrences=occurrences,
            confidence="MEDIUM",
        )
        return historical_pattern_payload(pattern)
