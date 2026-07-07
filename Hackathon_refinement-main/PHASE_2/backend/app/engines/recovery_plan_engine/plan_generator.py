"""
Recovery Plan Generator

Builds three candidate recovery plans using constrained combinatorial search
with greedy construction. Each plan follows a distinct archetype strategy:
- SAFE: Highest impact, lowest risk
- AGGRESSIVE: Maximum delay recovery
- MINIMAL_DISRUPTION: Minimum blast radius
"""

import hashlib
from typing import List, Optional, Set, Dict

from app.engines.recommendation_engine.models import ConfidenceLevel, Recommendation, RecommendationAction
from app.engines.recovery_plan_engine.safe_plan_builder import SafePlanBuilder
from app.engines.recovery_plan_engine.conflict_detector import ConflictDetector
from app.engines.recovery_plan_engine.models import RecoveryPlanArchetype, RecoveryPlanCandidate


class RecoveryPlanGenerator:
    """
    Generates three candidate recovery plans from a ranked list of recommendations.
    
    Uses deterministic, greedy construction:
    1. Sort recommendations by archetype-specific key
    2. Iterate and greedily add recommendations that don't conflict
    3. Stop when max_actions reached or quality threshold crossed
    
    This keeps plan generation fast (<1 sec) and deterministic (reproducible results).
    """

    def __init__(self, max_actions_per_plan: int = 5):
        """
        Args:
            max_actions_per_plan: Maximum number of actions per plan. Default 5.
        """
        self.max_actions_per_plan = max_actions_per_plan

    def generate_all_archetypes(
        self,
        ranked_recommendations: List[Recommendation],
        critical_path_item_ids: Optional[Set[str]] = None,
        resource_loads: Optional[dict] = None,
        simulation_results: Optional[Dict[str, object]] = None,
    ) -> List[RecoveryPlanCandidate]:
        """
        Generate three candidate plans: SAFE, AGGRESSIVE, MINIMAL_DISRUPTION.
        
        Args:
            ranked_recommendations: List of recommendations, already ranked by priority.
            critical_path_item_ids: Set of item IDs on the critical path (if available).
            resource_loads: Dict of resource_id -> load_percentage (if available).
        
        Returns:
            List of three RecoveryPlanCandidate objects, one per archetype.
        """
        safe_plan = self.build_safe_plan(ranked_recommendations, max_actions_override=3, simulation_results=simulation_results)
        aggressive_plan = self.build_aggressive_plan(ranked_recommendations, max_actions_override=8)
        minimal_disruption_plan = self.build_minimal_disruption_plan(
            ranked_recommendations,
            critical_path_item_ids,
            resource_loads,
            max_actions_override=2,
        )

        return [safe_plan, aggressive_plan, minimal_disruption_plan]

    def build_safe_plan(
        self,
        ranked_recommendations: List[Recommendation],
        max_actions_override: Optional[int] = None,
        simulation_results: Optional[Dict[str, object]] = None,
    ) -> RecoveryPlanCandidate:
        """
        Build the SAFE plan: highest impact, lowest risk.
        
        Strategy:
        - Sort by priority_score descending (already sorted in input)
        - Greedily add non-conflicting recommendations
        - Stop if adding recommendation would drop confidence below HIGH/MEDIUM
        - Enforce max_actions cap
        
        This produces the "safe, defensible" plan that a PM would actually recommend.
        """
        plan_cap = self.max_actions_per_plan if max_actions_override is None else max_actions_override

        # Prefer simulation-driven selection when available
        if simulation_results:
            selected = SafePlanBuilder.build(ranked_recommendations, simulation_results, max_actions=plan_cap)
            plan = list(selected)
        else:
            plan = []
            used_item_ids: Set[str] = set()
            used_resource_ids: Set[str] = set()
            action_type_counts: dict = {}
            candidates: List[Recommendation] = []

            # Build an initial candidate list by filtering down to safe recs.
            for rec in ranked_recommendations:
                # Check confidence threshold: reject LOW confidence items
                if rec.confidence == ConfidenceLevel.LOW:
                    continue

                # Check for conflicts with existing plan actions or shared items is deferred until selection.
                candidates.append(rec)

            # Phase 1: select diverse action types first.
            for rec in candidates:
                if len(plan) >= plan_cap:
                    break

                if self._detect_conflict_in_plan(rec, plan):
                    continue

                if set(rec.affected_item_ids) & used_item_ids:
                    continue

                action_type = rec.action_type.value
                if action_type_counts.get(action_type, 0) >= 2:
                    continue

                plan.append(rec)
                action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
                used_item_ids.update(rec.affected_item_ids)
                used_resource_ids.update(rec.affected_resource_ids)

            # Phase 2: if the plan is not full, allow additional recommendations even if
            # they repeat an action type, as long as they are safe and non-conflicting.
            if len(plan) < plan_cap:
                for rec in candidates:
                    if len(plan) >= plan_cap:
                        break

                    if rec in plan:
                        continue

                    if self._detect_conflict_in_plan(rec, plan):
                        continue

                    if set(rec.affected_item_ids) & used_item_ids:
                        continue

                    plan.append(rec)
                    action_type_counts[rec.action_type.value] = action_type_counts.get(rec.action_type.value, 0) + 1
                    used_item_ids.update(rec.affected_item_ids)
                    used_resource_ids.update(rec.affected_resource_ids)

        plan_id = self._generate_plan_id(RecoveryPlanArchetype.SAFE.value)
        return RecoveryPlanCandidate(
            plan_id=plan_id,
            archetype=RecoveryPlanArchetype.SAFE,
            actions=plan,
        )

    def build_aggressive_plan(
        self,
        ranked_recommendations: List[Recommendation],
        max_actions_override: Optional[int] = None,
    ) -> RecoveryPlanCandidate:
        """
        Build the AGGRESSIVE plan: maximum delay recovery.
        
        Strategy:
        - Sort by estimated_delay_reduction_days descending (highest impact first)
        - Greedily add non-conflicting recommendations
        - Allow MEDIUM confidence items (not just HIGH)
        - Enforce max_actions cap (typically 5)
        - Stop if marginal delay recovery per action drops below threshold
        
        This produces the "aggressive, fast recovery" plan at the cost of higher complexity.
        """
        plan_cap = self.max_actions_per_plan if max_actions_override is None else max_actions_override
        # Sort by delay reduction descending
        sorted_by_delay = sorted(
            ranked_recommendations,
            key=lambda r: r.estimated_delay_reduction_days,
            reverse=True,
        )
        
        plan = []
        used_item_ids: Set[str] = set()
        used_resource_ids: Set[str] = set()
        accumulated_delay_reduction = 0.0
        
        for rec in sorted_by_delay:
            if len(plan) >= plan_cap:
                break
            
            # Reject only LOW confidence items in aggressive plan (allow MEDIUM)
            if rec.confidence == ConfidenceLevel.LOW:
                continue
            
            # Check for conflicts with existing plan actions
            if self._detect_conflict_in_plan(rec, plan):
                continue
            
            # Aggressive plans may include multiple actions on the same item
            # if they are not explicitly incompatible. Do not block actions
            # solely because they share affected_item_ids.
            
            # Add to plan
            plan.append(rec)
            used_item_ids.update(rec.affected_item_ids)
            used_resource_ids.update(rec.affected_resource_ids)
            accumulated_delay_reduction += rec.estimated_delay_reduction_days
        
        plan_id = self._generate_plan_id(RecoveryPlanArchetype.AGGRESSIVE.value)
        return RecoveryPlanCandidate(
            plan_id=plan_id,
            archetype=RecoveryPlanArchetype.AGGRESSIVE,
            actions=plan,
        )

    def build_minimal_disruption_plan(
        self,
        ranked_recommendations: List[Recommendation],
        critical_path_item_ids: Optional[Set[str]] = None,
        resource_loads: Optional[dict] = None,
        max_actions_override: Optional[int] = None,
    ) -> RecoveryPlanCandidate:
        """
        Build the MINIMAL_DISRUPTION plan: smallest blast radius.
        
        Strategy:
        - Filter to only recommendations that:
          a) Do NOT touch critical path items
          b) Do NOT reassign items to resources at >90% load (even after change)
          c) Prefer reassignments, scope splits, and underutilized-resource absorption
        - Greedily add filtered recommendations by priority_score
        - Enforce max_actions cap
        
        This produces the "safest changes, smallest blast radius" plan that minimizes disruption.
        """
        if critical_path_item_ids is None:
            critical_path_item_ids = set()
        if resource_loads is None:
            resource_loads = {}
        plan_cap = self.max_actions_per_plan if max_actions_override is None else max_actions_override
        # Filter to safe recommendations (not on critical path, won't overload resources)
        safe_recs = []
        for rec in ranked_recommendations:

            # Reject if touches critical path
            if set(rec.affected_item_ids) & critical_path_item_ids:
                continue
            
            # Reject if would overload a resource
            would_overload = False
            for resource_id in rec.affected_resource_ids:
                current_load = resource_loads.get(resource_id, 0.0)
                # Conservative: reject if already at or above 90%
                if current_load >= 90.0:
                    would_overload = True
                    break
            if would_overload:
                continue
            
            # Prefer certain action types (less disruptive)
            if rec.action_type in [
                RecommendationAction.REASSIGN_ITEM,
                RecommendationAction.SPLIT_ITEM,
                RecommendationAction.REBALANCE_SPRINT_LOAD,
            ]:
                safe_recs.append(rec)
        
        # Now greedily build from filtered list
        plan = []
        used_item_ids: Set[str] = set()
        used_resource_ids: Set[str] = set()
        
        for rec in safe_recs:
            if len(plan) >= self.max_actions_per_plan:
                break
            
            # Check for conflicts with existing plan actions
            if self._detect_conflict_in_plan(rec, plan):
                continue
            
            # Check if recommendation already partially in plan
            if set(rec.affected_item_ids) & used_item_ids:
                continue
            
            # Add to plan
            plan.append(rec)
            used_item_ids.update(rec.affected_item_ids)
            used_resource_ids.update(rec.affected_resource_ids)
        
        plan_id = self._generate_plan_id(RecoveryPlanArchetype.MINIMAL_DISRUPTION.value)
        return RecoveryPlanCandidate(
            plan_id=plan_id,
            archetype=RecoveryPlanArchetype.MINIMAL_DISRUPTION,
            actions=plan,
        )

    @staticmethod
    def _generate_plan_id(plan_type: str = "") -> str:
        """Generate a deterministic unique plan ID based on plan type."""
        digest = hashlib.sha1(plan_type.encode("utf-8")).hexdigest()[:10]
        return f"plan_{digest}"

    @staticmethod
    def _detect_conflict_in_plan(rec: Recommendation, plan: List[Recommendation]) -> bool:
        """
        Check if a recommendation conflicts with any item in the existing plan.
        Uses ConflictDetector for consistency.
        """
        for existing_rec in plan:
            if ConflictDetector.detect_conflict(rec, existing_rec):
                return True
        return False
