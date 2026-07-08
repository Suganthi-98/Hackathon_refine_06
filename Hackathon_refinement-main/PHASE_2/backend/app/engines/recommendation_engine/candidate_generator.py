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
        # Fix #4: only emit ADVANCE if an eligible earlier sprint actually exists.
        for item_id in signal.affected_item_ids[:1]:
            if not self._can_advance_item(item_id):
                continue
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
                feasibility_checks={"has_capacity": True, "earlier_sprint_exists": True},
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

        if flag == "SKILL_MISMATCH":
            current_owner_id = signal.context.get("current_owner_id", resource_id)
            better_resource_id = signal.context.get("better_resource_id")
            required_skill = signal.context.get("required_skill")
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.REASSIGN_ITEM,
                title=f"Reassign due to skill mismatch ({item_id})",
                description=(
                    f"{current_owner_id} lacks '{required_skill}' as a primary or secondary skill for {item_id}; "
                    f"{better_resource_id} has it as a primary skill with "
                    f"{signal.context.get('better_resource_availability_pct', 0.0):.0%} availability."
                ),
                affected_item_ids=[item_id],
                affected_resource_ids=[current_owner_id, better_resource_id] if better_resource_id else [current_owner_id],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_resource_id": better_resource_id, "target_item_id": item_id, "reason": "skill_mismatch"},
                feasibility_checks={"resource_exists": True, "skill_match_confirmed": True},
            ))
            return candidates

        if flag == "LOW_VELOCITY":
            current_owner_id = signal.context.get("current_owner_id", resource_id)
            better_resource_id = signal.context.get("better_resource_id")
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.REASSIGN_ITEM,
                title=f"Reassign due to low relative velocity ({item_id})",
                description=(
                    f"{current_owner_id} completes ~{signal.context.get('current_owner_velocity', 0.0)}hrs/sprint "
                    f"(n={signal.context.get('current_owner_sample_size', 0)} completed items) on this skill, vs "
                    f"{better_resource_id}'s ~{signal.context.get('better_resource_velocity', 0.0)}hrs/sprint "
                    f"(n={signal.context.get('better_resource_sample_size', 0)}). Both have the required skill."
                ),
                affected_item_ids=[item_id],
                affected_resource_ids=[current_owner_id, better_resource_id] if better_resource_id else [current_owner_id],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_resource_id": better_resource_id, "target_item_id": item_id, "reason": "low_velocity"},
                feasibility_checks={"resource_exists": True, "velocity_sample_sufficient": True},
            ))
            return candidates

        if flag == "UNDERUTILIZED":
            # Bug #6 fix: the underutilized resource is the *receiver*, not the source.
            # The applicator does `item.assigned_resource = resource_id` — meaning the
            # affected_item_ids must come from a *different*, overloaded resource, not from
            # the underutilized one's own backlog (which would be a guaranteed self-reassignment
            # no-op). Find the most overloaded peer who shares at least one sprint with this
            # underutilized resource, then take items from *that* peer.
            sprint_ids = set(signal.affected_sprint_ids)
            overloaded_peers = [
                r for r in self.project_state.team
                if r.resource_id != resource_id
            ]
            # Pick the peer with the most remaining effort in the shared sprint(s)
            def _peer_remaining_hrs(r: Any) -> float:
                return sum(
                    float(getattr(wi, "remaining_effort_hrs", 0.0) or 0.0)
                    for wi in self.project_state.work_items
                    if getattr(wi, "assigned_resource", None) in {r.resource_id, r.name}
                    and getattr(wi, "assigned_sprint", None) in sprint_ids
                )
            source_peer = max(overloaded_peers, key=_peer_remaining_hrs, default=None) if overloaded_peers else None
            if source_peer is None:
                # No peer to pull from — don't emit a candidate that can't mutate state
                return candidates
            source_items = [
                wi.item_id for wi in self.project_state.work_items
                if getattr(wi, "assigned_resource", None) in {source_peer.resource_id, source_peer.name}
                and getattr(wi, "assigned_sprint", None) in sprint_ids
            ]
            if not source_items:
                return candidates
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.REBALANCE_SPRINT_LOAD,
                title=f"Rebalance sprint load → {resource_id}",
                description=(
                    f"{resource_id} has spare capacity (load ratio {load_ratio:.0%}); "
                    f"move work from {source_peer.resource_id} to absorb {len(source_items)} item(s)."
                ),
                affected_item_ids=source_items[:2],
                affected_resource_ids=[resource_id],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_resource_id": resource_id, "load_ratio": load_ratio, "source_resource_id": source_peer.resource_id},
                feasibility_checks={"resource_exists": True, "has_capacity": True, "source_identified": True},
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
                feasibility_checks={},
            ))
            return candidates

        if flag == "OVERLOADED":
            # Find a real receiver instead of self-targeting the overloaded resource --
            # this was the original "Reassign work" gap flagged from the start: no
            # candidate-resource search existed at all. Score by skill match + availability,
            # same approach used by SkillMismatchDetector, since we don't have a
            # per-resource confidence-weighted velocity signal reliably available for
            # every overloaded case (LowVelocityDetector handles that distinctly).
            item = next((wi for wi in self.project_state.work_items if wi.item_id == item_id), None) if item_id else None
            required_skill = getattr(item, "required_skill", None) if item else None
            better = None
            if required_skill:
                candidates_pool = [
                    r for r in self.project_state.team
                    if r.resource_id != resource_id
                    and required_skill in {r.primary_skill, r.secondary_skill}
                    and float(getattr(r, "availability_pct", 0.0) or 0.0) > 0.1
                ]
                if candidates_pool:
                    better = max(
                        candidates_pool,
                        key=lambda r: (
                            r.primary_skill == required_skill,
                            float(getattr(r, "availability_pct", 0.0) or 0.0) * (1 - float(getattr(r, "allocation_pct", 0.0) or 0.0)),
                        ),
                    )
            if better is None:
                # No genuine receiver found -- do not emit a self-targeting candidate that
                # would silently no-op when simulated. Reject rather than fake a reassignment.
                return candidates
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.REASSIGN_ITEM,
                title=f"Reassign item ({item_id or resource_id})",
                description=(
                    f"Move {item_id} from overloaded {resource_id} (load ratio {load_ratio:.2f}) to "
                    f"{better.resource_id}, who has the required skill and "
                    f"{float(getattr(better, 'availability_pct', 0.0) or 0.0):.0%} availability."
                ),
                affected_item_ids=signal.affected_item_ids,
                affected_resource_ids=[resource_id, better.resource_id],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"target_resource_id": better.resource_id, "target_item_id": item_id},
                feasibility_checks={"resource_exists": True, "has_capacity": True, "receiver_identified": True},
            ))
            return candidates

        return candidates

    def _from_sprint_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        if signal.affected_item_ids:
            item_id = signal.affected_item_ids[0]
            # Fix #4: skip if already in the earliest eligible sprint
            if self._can_advance_item(item_id):
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
                    feasibility_checks={"has_capacity": True, "earlier_sprint_exists": True},
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

        # Fix #4: only emit if an earlier eligible sprint exists for this item
        for item_id in signal.affected_item_ids[:2]:
            if not self._can_advance_item(item_id):
                continue
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
                feasibility_checks={"has_capacity": True, "earlier_sprint_exists": True},
            ))
        return candidates

    def _from_schedule_signal(self, signal: OpportunitySignal) -> List[RecommendationCandidate]:
        candidates: List[RecommendationCandidate] = []
        if not signal.affected_item_ids:
            return candidates

        flag = signal.context.get("flag", "SCHEDULE_GAP") if signal.context else "SCHEDULE_GAP"
        schedule_gap_hours = float(signal.context.get("schedule_gap_hours", 0.0)) if signal.context else 0.0

        # SPLIT_ITEM — no resource needed; applicator works directly on the item
        item_id = signal.affected_item_ids[0]
        item = next((wi for wi in self.project_state.work_items if wi.item_id == item_id), None)
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

        # Fix #7: REBALANCE_SPRINT_LOAD requires a real target resource (the receiver).
        # The applicator does `item.assigned_resource = resource_id` — if resource_id is
        # empty the guard returns immediately, guaranteed no-op. Find an underutilized
        # resource to absorb the overloaded items before emitting this candidate.
        sprint_ids = set(signal.affected_sprint_ids)
        def _resource_remaining_hrs(r: Any) -> float:
            return sum(
                float(getattr(wi, "remaining_effort_hrs", 0.0) or 0.0)
                for wi in self.project_state.work_items
                if getattr(wi, "assigned_resource", None) in {r.resource_id, r.name}
                and getattr(wi, "assigned_sprint", None) in sprint_ids
            )
        def _resource_capacity_hrs(r: Any) -> float:
            daily = float(getattr(r, "daily_capacity_hrs", 8.0) or 8.0)
            avail = float(getattr(r, "availability_pct", 1.0) or 1.0)
            return daily * avail * 10  # approximate sprint capacity (10 working days)

        receiver = None
        for r in sorted(self.project_state.team, key=lambda r: _resource_remaining_hrs(r)):
            cap = _resource_capacity_hrs(r)
            if cap > 0 and _resource_remaining_hrs(r) < cap * 0.8:
                receiver = r
                break

        if receiver is not None:
            candidates.append(self._build_candidate(
                action_type=RecommendationAction.REBALANCE_SPRINT_LOAD,
                title=f"Rebalance sprint load → {receiver.resource_id}",
                description=(
                    f"Rebalance work to {receiver.resource_id} to close schedule gap ({schedule_gap_hours:.1f}h). "
                    f"Items in scope: {', '.join(signal.affected_item_ids[:2])}"
                ),
                affected_item_ids=signal.affected_item_ids[:2],
                affected_resource_ids=[receiver.resource_id],
                affected_sprint_ids=signal.affected_sprint_ids,
                affected_blocker_ids=signal.affected_blocker_ids,
                root_signal_id=signal.signal_id,
                simulation_params={"gap_hours": schedule_gap_hours, "target_resource_id": receiver.resource_id},
                feasibility_checks={"has_future_sprints": True, "receiver_identified": True},
            ))
        # If no underutilized receiver exists, suppress the REBALANCE candidate — emitting
        # it with an empty resource_id would be a guaranteed no-op.

        # Fix #10: ADD_RESOURCE_SKILL requires a real resource ID in affected_resource_ids.
        # The applicator's guard is `if resource_id is None: return`. Find the most
        # overloaded resource with the required skill gap before emitting this candidate.
        if schedule_gap_hours > 20.0:
            required_skill = "General"
            if item and getattr(item, "required_skill", None):
                required_skill = item.required_skill

            # Find the most loaded resource who would benefit from a skill/capacity boost.
            # Prefer someone already assigned to affected items; fall back to most loaded overall.
            skill_target = None
            for wi_id in signal.affected_item_ids[:3]:
                wi = next((w for w in self.project_state.work_items if w.item_id == wi_id), None)
                if wi and getattr(wi, "assigned_resource", None):
                    skill_target = wi.assigned_resource
                    break
            if skill_target is None and self.project_state.team:
                # Fallback: resource with highest remaining effort in affected sprints
                skill_target = max(
                    (r.resource_id for r in self.project_state.team),
                    key=lambda rid: sum(
                        float(getattr(w, "remaining_effort_hrs", 0.0) or 0.0)
                        for w in self.project_state.work_items
                        if getattr(w, "assigned_resource", None) == rid
                        and getattr(w, "assigned_sprint", None) in sprint_ids
                    ),
                    default=None,
                )
            if skill_target is not None:
                candidates.append(self._build_candidate(
                    action_type=RecommendationAction.ADD_RESOURCE_SKILL,
                    title=f"Add resource capacity ({skill_target})",
                    description=(
                        f"Boost capacity or add '{required_skill}' skill for {skill_target} "
                        f"to close schedule gap ({schedule_gap_hours:.1f}h)."
                    ),
                    affected_item_ids=signal.affected_item_ids[:1],
                    affected_resource_ids=[skill_target],
                    affected_sprint_ids=signal.affected_sprint_ids,
                    affected_blocker_ids=signal.affected_blocker_ids,
                    root_signal_id=signal.signal_id,
                    simulation_params={"gap_hours": schedule_gap_hours, "required_skill": required_skill, "target_resource_id": skill_target},
                    feasibility_checks={"budget_available": True, "resource_identified": True},
                ))
            # If no resource can be identified, suppress — empty resource_id is a guaranteed no-op.

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

    # ------------------------------------------------------------------ #
    # Shared guard: emit ADVANCE_ITEM_TO_EARLIER_SPRINT only when a real  #
    # earlier, non-completed sprint exists for the item.                  #
    # ------------------------------------------------------------------ #
    def _can_advance_item(self, item_id: str) -> bool:
        """Return True iff there is at least one earlier, eligible sprint to advance this item into."""
        item = next((wi for wi in self.project_state.work_items if wi.item_id == item_id), None)
        if item is None:
            return False
        sprint_by_name = {s.sprint_name: s for s in self.project_state.sprints}
        current_sprint = sprint_by_name.get(item.assigned_sprint)
        if current_sprint is None:
            return False
        # An eligible earlier sprint must have a lower sprint_number and not be Completed.
        from app.domain.models import SprintStatus  # local import to avoid circular at module level
        earlier = [
            s for s in self.project_state.sprints
            if s.sprint_number < current_sprint.sprint_number
            and s.status != SprintStatus.COMPLETED
        ]
        return bool(earlier)

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
        feasibility_checks: Dict[str, bool] | None = None,
    ) -> RecommendationCandidate:
        target_ids = list(affected_item_ids) + list(affected_resource_ids) + list(affected_sprint_ids) + list(affected_blocker_ids)
        merged_params = dict(simulation_params)
        # Backwards-compatibility: accept omitted feasibility_checks and normalize to empty dict
        feasibility_checks = feasibility_checks or {}
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
