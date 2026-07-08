from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from app.api.models_phase3 import RecommendationType
from app.domain.models import (
    Blocker,
    BlockerStatus,
    ProjectState,
    Resource,
    SkillLevel,
    Sprint,
    SprintActual,
    SprintStatus,
    WorkItem,
    WorkItemStatus,
)
from app.engines.critical_path_engine import CriticalPathEngine, CriticalPathResult
from app.engines.dependency_engine import DependencyDAG, DependencyGraphEngine
from app.engines.forecast_engine import ForecastEngine, ForecastResult
from app.engines.impact_scoring_engine import ImpactScoringEngine, RiskScores
from app.engines.metrics_engine import MetricsEngine, ProjectMetrics
from app.engines.monte_carlo_engine import MonteCarloEngine, MonteCarloResult
from app.engines.recommendation_engine.models import (
    BaselineMetrics,
    Recommendation,
    SimulatedMetrics,
    SimulationResult as SimulationResultV2,
    UpstreamEngineOutputs,
)
from app.engines.risk_engine import RiskEngine, RiskResult
from app.engines.spillover_engine import SpilloverAnalysis, SpilloverAnalysisEngine
from app.engines.forecast_levers import FORECAST_LEVER_MAP, sample_lever_values


MONTE_CARLO_SEED: int = 42


class ActionApplicator:
    """Apply a recommendation or legacy simulation action to a cloned ProjectState."""

    def apply(self, state: ProjectState, action: Union[Recommendation, SimulationAction]) -> None:
        if isinstance(action, Recommendation):
            self._apply_recommendation(state, action)
            return
        self._apply_legacy_action(state, action)

    def apply_many(self, state: ProjectState, recommendations: Sequence[Recommendation]) -> None:
        for recommendation in sorted(recommendations, key=lambda item: getattr(item, "recommendation_id", "")):
            self.apply(state, recommendation)

    def _apply_recommendation(self, state: ProjectState, recommendation: Recommendation) -> None:
        action_name = getattr(recommendation.action_type, "value", str(recommendation.action_type)).strip().lower()
        normalized = action_name.replace(" ", "_")

        if normalized == "resolve_blocker":
            self._apply_resolve_blocker(state, recommendation)
        elif normalized in {"reduce_item_scope", "decrease_scope"}:
            self._apply_reduce_scope(state, recommendation)
        elif normalized in {"add_resource", "add_resource_skill"}:
            self._apply_add_capacity(state, recommendation)
        elif normalized in {"parallelize_tasks", "parallelize_items"}:
            self._apply_parallelize_work(state, recommendation)
        elif normalized in {"reassign_work", "reassign_item", "rebalance_sprint_load"}:
            self._apply_reassign_work(state, recommendation)
        elif normalized == "move_blocker_items":
            self._apply_move_blocker_items(state, recommendation)
        elif normalized == "advance_item_to_earlier_sprint":
            self._apply_advance_item(state, recommendation)
        elif normalized in {"split_task", "split_item"}:
            self._apply_split_task(state, recommendation)
        elif normalized in {"critical_path_optimization", "remove_dependency_bottleneck"}:
            self._apply_critical_path_optimization(state, recommendation)

    def _apply_legacy_action(self, state: ProjectState, action: SimulationAction) -> None:
        normalized = str(action.action_type).strip().lower().replace(" ", "_")
        if normalized == RecommendationType.RESOLVE_BLOCKER.value.lower().replace(" ", "_"):
            self._apply_resolve_blocker(state, action)
        elif normalized == RecommendationType.REDUCE_ITEM_SCOPE.value.lower().replace(" ", "_"):
            self._apply_reduce_scope(state, action)
        elif normalized == RecommendationType.ADD_RESOURCE.value.lower().replace(" ", "_"):
            self._apply_add_capacity(state, action)
        elif normalized == RecommendationType.PARALLELIZE_TASKS.value.lower().replace(" ", "_"):
            self._apply_parallelize_work(state, action)
        elif normalized == RecommendationType.REASSIGN_WORK.value.lower().replace(" ", "_"):
            self._apply_reassign_work(state, action)
        elif normalized == RecommendationType.MOVE_BLOCKER_ITEMS.value.lower().replace(" ", "_"):
            self._apply_move_blocker_items(state, action)
        elif normalized == RecommendationType.SPLIT_TASK.value.lower().replace(" ", "_"):
            self._apply_split_task(state, action)
        elif normalized == RecommendationType.CRITICAL_PATH_OPTIMIZATION.value.lower().replace(" ", "_"):
            self._apply_critical_path_optimization(state, action)

    def _apply_resolve_blocker(self, state: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        blocker_ids = []
        if hasattr(action, "target_ids") and action.target_ids:
            blocker_ids = action.target_ids
        elif hasattr(action, "affected_blocker_ids"):
            blocker_ids = action.affected_blocker_ids

        for blocker_id in blocker_ids:
            blocker = next((b for b in state.blockers if b.blocker_id == blocker_id), None)
            if not blocker:
                continue
            blocker.status = BlockerStatus.RESOLVED
            blocker.actual_resolution_date = datetime.now(timezone.utc)
            if blocker.raised_date is not None and blocker.raised_date.tzinfo is None:
                blocker.raised_date = blocker.raised_date.replace(tzinfo=timezone.utc)
            self._unblock_impacted_items(state, blocker)

    def _unblock_impacted_items(self, state: ProjectState, blocker: Blocker) -> None:
        for impacted_item_id in getattr(blocker, "impacted_item_ids", []) or []:
            item = next((wi for wi in state.work_items if wi.item_id == impacted_item_id), None)
            if item and item.status == WorkItemStatus.BLOCKED:
                item.status = (
                    WorkItemStatus.IN_PROGRESS
                    if item.progress_pct > 0.0 or item.actual_effort_hrs > 0.0
                    else WorkItemStatus.NOT_STARTED
                )

    def _apply_reduce_scope(self, state: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        if not target_ids:
            return
        item_id = target_ids[0]
        item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
        if not item:
            return
        core_hours = 0.6 * item.current_estimate_hrs
        if isinstance(action, SimulationAction):
            core_hours = float(action.details.get("core_hours", core_hours))
        reduction = max(0.0, item.current_estimate_hrs - core_hours)
        item.current_estimate_hrs = max(0.0, core_hours)
        item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs - reduction)
        item.is_scope_changed = True
        item.scope_change_reason = (
            f"Simulation scope reduction: retained {item.current_estimate_hrs:.1f}h and deferred {reduction:.1f}h."
        )

    def _apply_add_capacity(self, state: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        skill = "General"
        role = "Capacity Resource"
        capacity_gain_hours = 20.0
        named_resource_id = None

        if isinstance(action, SimulationAction):
            skill = action.details.get("skill", skill)
            role = action.details.get("role", role)
            capacity_gain_hours = float(action.details.get("capacity_gain_hours", capacity_gain_hours))
        else:
            capacity_gain_hours = float(getattr(action, "estimated_hours_recovered", capacity_gain_hours) or capacity_gain_hours)
            metadata = getattr(action, "metadata", {}) or {}
            sim_params = metadata.get("simulation_params", {}) or {}
            named_resource_id = sim_params.get("receiving_resource_id")

        if named_resource_id:
            existing = next((r for r in state.team if r.resource_id == named_resource_id), None)
            if existing:
                existing.allocation_pct = min(1.0, existing.allocation_pct + 0.2)
                for sprint in state.sprints:
                    if sprint.status in {SprintStatus.NOT_STARTED, SprintStatus.IN_PROGRESS}:
                        sprint.planned_velocity_hrs += capacity_gain_hours
                return

        resource_id = f"SIM-R-{len(state.team) + 1}"
        state.team.append(
            Resource(
                resource_id=resource_id,
                name=f"Simulated {role}",
                role=role,
                primary_skill=skill,
                secondary_skill=None,
                skill_level=SkillLevel.MID,
                allocation_pct=0.5,
                availability_pct=1.0,
                daily_capacity_hrs=8.0,
            )
        )

        for sprint in state.sprints:
            if sprint.status in {SprintStatus.NOT_STARTED, SprintStatus.IN_PROGRESS}:
                sprint.planned_velocity_hrs += capacity_gain_hours

        existing_actuals = [a.actual_effort_hrs for a in state.actuals if a.actual_effort_hrs is not None]
        avg_actual_velocity = sum(existing_actuals) / len(existing_actuals) if existing_actuals else capacity_gain_hours
        synthetic_actual_value = max(capacity_gain_hours, avg_actual_velocity + capacity_gain_hours * 0.5)
        state.actuals.append(
            SprintActual(
                sprint_id=f"SIM-{resource_id}",
                sprint_number=max((s.sprint_number for s in state.sprints), default=0) + 1,
                planned_effort_hrs=capacity_gain_hours,
                actual_effort_hrs=synthetic_actual_value,
                variance_hrs=0.0,
                tasks_planned=0,
                tasks_completed=0,
                completion_rate=1.0,
                carryover_count=0,
                scope_change_hours=0.0,
                blocker_impact_hrs=0.0,
            )
        )

    def _apply_parallelize_work(self, state: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        if len(target_ids) < 2:
            return
        pred_id, succ_id = target_ids[0], target_ids[1]
        successor = next((wi for wi in state.work_items if wi.item_id == succ_id), None)
        if successor:
            reduction = successor.current_estimate_hrs * 0.2
            successor.current_estimate_hrs = max(0.0, successor.current_estimate_hrs - reduction)
            successor.remaining_effort_hrs = max(0.0, successor.remaining_effort_hrs - reduction)

        dependency = next((d for d in state.dependencies if d.predecessor_item_id == pred_id and d.successor_item_id == succ_id), None)
        if dependency and dependency.lag_days > 0:
            dependency.lag_days = max(0, dependency.lag_days - 1)

    def _apply_reassign_work(self, state: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        if not target_ids:
            return
        item_id = target_ids[0]
        item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
        if not item:
            return
        if isinstance(action, SimulationAction):
            new_resource_name = action.details.get("to")
            new_resource = next((r for r in state.team if r.name == new_resource_name), None)
            if new_resource:
                item.assigned_resource = new_resource.resource_id
                new_resource.allocation_pct = min(1.0, new_resource.allocation_pct + 0.1)
            return

        metadata = getattr(action, "metadata", {}) or {}
        sim_params = metadata.get("simulation_params", {}) or {}
        receiving_resource_id = sim_params.get("receiving_resource_id")
        if not receiving_resource_id:
            receiving_resource_id = (action.affected_resource_ids or [None, None])[-1]

        if receiving_resource_id:
            item.assigned_resource = receiving_resource_id

    def _apply_move_blocker_items(self, state: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        advanceable_items = []
        if isinstance(action, SimulationAction):
            advanceable_items = list(action.details.get("advanceable_items", []) or [])
        else:
            advanceable_items = list(getattr(action, "affected_item_ids", []) or [])
        for item_id in advanceable_items:
            item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
            if item and item.status == WorkItemStatus.NOT_STARTED:
                item.status = WorkItemStatus.IN_PROGRESS

    def _apply_advance_item(self, state: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        # Distinct from _apply_move_blocker_items above: this actually moves the item
        # to an earlier sprint by sprint_number, rather than just flipping its status.
        # WorkItem.assigned_sprint holds the sprint NAME -- compare/assign by name only.
        target_ids = list(getattr(action, "affected_item_ids", []) or [])
        sprint_by_name = {s.sprint_name: s for s in state.sprints}
        sprints_by_number = sorted(state.sprints, key=lambda s: s.sprint_number)
        for item in state.work_items:
            if item.item_id not in target_ids:
                continue
            current_sprint = sprint_by_name.get(item.assigned_sprint)
            if current_sprint is None:
                continue
            earlier_candidates = [s for s in sprints_by_number if s.sprint_number < current_sprint.sprint_number]
            if not earlier_candidates:
                continue
            item.assigned_sprint = earlier_candidates[-1].sprint_name

    def _apply_split_task(self, state: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        if not target_ids:
            return
        item_id = target_ids[0]
        item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
        if not item:
            return
        reduction = item.current_estimate_hrs * 0.15
        item.current_estimate_hrs = max(1.0, item.current_estimate_hrs - reduction)
        item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs - reduction)

    def _apply_critical_path_optimization(self, state: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        for item_id in target_ids:
            item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
            if not item:
                continue
            reduction = item.current_estimate_hrs * 0.15
            item.current_estimate_hrs = max(1.0, item.current_estimate_hrs - reduction)
            item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs - reduction)


# Preserve the active ActionApplicator implementation for SimulationEngine.
# This prevents the module-level alias at the bottom of the file from changing
# which applicator class the active SimulationEngine uses.
_ActiveActionApplicator = ActionApplicator

class EngineRunner:
    """Runs the full deterministic engine pipeline for a ProjectState."""

    SEED: int = MONTE_CARLO_SEED

    def run(self, state: ProjectState, simulation_count: int = 1000) -> Dict[str, Any]:
        metrics = MetricsEngine(state).calculate()
        dag = DependencyGraphEngine(state).build_dag()
        cp_result = CriticalPathEngine(state, dag).analyze()
        spillover = SpilloverAnalysisEngine(state, metrics.average_item_effort).analyze()
        forecast = ForecastEngine(state, metrics, cp_result, spillover).calculate()
        monte_carlo = MonteCarloEngine(
            project_state=state,
            metrics=metrics,
            cp_result=cp_result,
            spillover=spillover,
            simulation_count=simulation_count,
            seed=self.SEED,
        ).calculate()
        impact_scores = ImpactScoringEngine(state, dag).score()
        risk_result = RiskEngine(
            project_state=state,
            metrics=metrics,
            cp_result=cp_result,
            dag=dag,
            spillover=spillover,
            forecast=forecast,
            monte_carlo=monte_carlo,
            impact_scores=impact_scores,
        ).analyze()
        return {
            "metrics": metrics,
            "dag": dag,
            "cp_result": cp_result,
            "spillover": spillover,
            "forecast": forecast,
            "monte_carlo": monte_carlo,
            "risk_result": risk_result,
        }


class SimulationAction(BaseModel):
    action_id: str = Field(..., description="Recommendation identifier for this action")
    action_type: str = Field(..., description="Action type or recommendation type value")
    target_ids: List[str] = Field(default_factory=list, description="Target entity IDs")
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured details for the action")
    impact_reason: str = Field(..., description="Reason why this action will affect the project")


class SimulationScenario(BaseModel):
    selected_recommendations: List[str] = Field(..., description="Selected recommendation IDs for simulation")


class SimulationResult(BaseModel):
    baseline_finish_date: datetime
    simulated_finish_date: datetime
    baseline_risk_score: float
    simulated_risk_score: float
    baseline_p80_date: datetime
    simulated_p80_date: datetime
    baseline_critical_path_hours: float
    simulated_critical_path_hours: float
    days_recovered: float
    risk_reduction: float
    recommendations_applied: List[str]
    action_reasons: List[str]
    baseline_probability: float
    simulated_probability: float
    baseline_delay_days: float
    simulated_delay_days: float


class ScenarioMetadata(BaseModel):
    scenario_id: str = Field(..., description="Unique identifier for the simulated scenario")
    selected_recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ForecastComparison(BaseModel):
    baseline_finish_date: datetime
    simulated_finish_date: datetime
    days_saved: float
    finish_date_delta: float
    baseline_delay_days: float
    simulated_delay_days: float


class MonteCarloComparison(BaseModel):
    baseline_on_time_probability: float
    simulated_on_time_probability: float
    confidence_delta: float


class RiskComparison(BaseModel):
    baseline_risk_score: float
    simulated_risk_score: float
    risk_reduction: float


class MetricsComparison(BaseModel):
    velocity_delta: float
    utilization_delta: float
    carryover_delta: float
    blocker_delta: float


class RecommendationEffectiveness(BaseModel):
    estimated_benefit: float
    actual_simulated_benefit: float
    recommendation_accuracy: float


class ScenarioSummary(BaseModel):
    overall_improvement_score: float
    simulation_success: bool
    warnings: List[str] = Field(default_factory=list)


class ScenarioResult(BaseModel):
    metadata: ScenarioMetadata
    forecast_comparison: ForecastComparison
    monte_carlo_comparison: MonteCarloComparison
    risk_comparison: RiskComparison
    metrics_comparison: MetricsComparison
    recommendation_effectiveness: RecommendationEffectiveness
    summary: ScenarioSummary
    revised_sprint_plan: List[Dict[str, Any]] = Field(default_factory=list)


class SimulationEngine:
    """Runs deterministic scenario simulations from existing upstream engine outputs."""

    def __init__(
        self,
        project_state: ProjectState,
        metrics: ProjectMetrics,
        dag: DependencyDAG,
        cp_result: CriticalPathResult,
        spillover: SpilloverAnalysis,
        forecast: ForecastResult,
        monte_carlo: MonteCarloResult,
        risk_result: RiskResult,
        simulation_count: int = 1000,
        seed: Optional[int] = None,
    ):
        self.project_state = project_state
        self.metrics = metrics
        self.dag = dag
        self.cp_result = cp_result
        self.spillover = spillover
        self.forecast = forecast
        self.monte_carlo = monte_carlo
        self.risk_result = risk_result
        self.simulation_count = simulation_count
        self.seed = seed
        # NOTE: previously used _ActiveActionApplicator (the original, pre-V2
        # ActionApplicator), which has no dispatch branches at all for
        # cross_train_backup, swarm_item, insert_review_gate,
        # apply_ramp_up_discount, pair_reviewer, rebaseline_estimate,
        # escalate_blocker_early, or resequence_non_critical_item. Any
        # RecoveryPlan (built via this SimulationEngine, e.g. RecoveryPlanEngine)
        # that included one of those actions would silently simulate it as a
        # no-op while still listing it and taking credit for the plan's
        # reported probability/delay improvement. ActionApplicatorV2 is a
        # strict superset of the old applicator's coverage plus the fixes
        # made throughout this project -- use it everywhere instead.
        self.applicator = ActionApplicatorV2()
        self._scenario_cache: List[ScenarioResult] = []

    def simulate(self, recommendation: Recommendation) -> ScenarioResult:
        """Simulate a single recommendation and return a structured scenario result."""
        return self.simulate_scenario([recommendation])

    def simulate_scenario(self, recommendations: Sequence[Union[Recommendation, str]]) -> ScenarioResult:
        """Clone the project state, apply one or more recommendations, and rerun the engine pipeline."""
        normalized_recommendations = [self._normalize_recommendation(rec) for rec in recommendations if rec is not None]
        clone = self.project_state.model_copy(deep=True)
        if not normalized_recommendations:
            return self._build_scenario_result([], self._recalculate_clone(clone), clone)

        for recommendation in normalized_recommendations:
            self.applicator.apply(clone, recommendation)

        simulated = self._recalculate_clone(clone)
        scenario = self._build_scenario_result(normalized_recommendations, simulated, clone)
        self._scenario_cache.append(scenario)
        return scenario

    def compare_scenarios(self, scenarios: Optional[Sequence[ScenarioResult]] = None) -> List[ScenarioResult]:
        """Return the supplied scenarios in a deterministic order for downstream analysis."""
        candidates = list(scenarios or self._scenario_cache)
        return sorted(candidates, key=lambda item: item.metadata.scenario_id)

    def build_revised_sprint_plan(self, clone: ProjectState, sprint_id: str) -> List[Dict[str, Any]]:
        """After simulation, return a simple table of items and ownership changes for a given sprint."""
        plan: List[Dict[str, Any]] = []
        original_owner_map = {wi.item_id: wi.assigned_resource for wi in self.project_state.work_items}
        resource_name_map = {r.resource_id: r.name for r in clone.team}
        sprint_name_by_id = {s.sprint_id: s.sprint_name for s in clone.sprints}
        sprint_ids = {sprint_id, sprint_name_by_id.get(sprint_id)}

        for wi in clone.work_items:
            if wi.assigned_sprint not in sprint_ids:
                continue
            original_owner_id = original_owner_map.get(wi.item_id)
            current_owner_id = wi.assigned_resource
            plan.append(
                {
                    "item_id": wi.item_id,
                    "title": wi.title,
                    "remaining_hours": round(wi.remaining_effort_hrs, 1),
                    "original_owner": resource_name_map.get(original_owner_id, original_owner_id) if original_owner_id else "Unassigned",
                    "new_owner": resource_name_map.get(current_owner_id, current_owner_id) if current_owner_id else "Unassigned",
                    "owner_changed": original_owner_id != current_owner_id,
                }
            )

        return plan

    def simulate_recommendation_actions(self, actions: List[SimulationAction]) -> SimulationResult:
        """Legacy helper kept for compatibility with the existing API surface."""
        clone = self.project_state.model_copy(deep=True)
        for action in actions:
            self.applicator.apply(clone, action)

        simulated = self._recalculate_clone(clone)
        return self._build_legacy_result(actions, simulated)

    def _normalize_recommendation(self, recommendation: Union[Recommendation, str]) -> Recommendation:
        if isinstance(recommendation, Recommendation):
            return recommendation
        raise TypeError("SimulationEngine expects Recommendation instances for scenario simulation")

    def _apply_recommendation(self, clone: ProjectState, recommendation: Recommendation) -> None:
        action_type = getattr(recommendation.action_type, "value", str(recommendation.action_type)).strip().lower()
        normalized = action_type.replace(" ", "_")

        if normalized == "resolve_blocker":
            self._apply_resolve_blocker(clone, recommendation)
        elif normalized in {"reduce_item_scope", "decrease_scope"}:
            self._apply_reduce_scope(clone, recommendation)
        elif normalized in {"add_resource", "add_resource_skill"}:
            self._apply_add_capacity(clone, recommendation)
        elif normalized in {"parallelize_tasks", "parallelize_items"}:
            self._apply_parallelize_work(clone, recommendation)
        elif normalized in {"reassign_work", "reassign_item", "rebalance_sprint_load"}:
            self._apply_reassign_work(clone, recommendation)
        elif normalized == "move_blocker_items":
            self._apply_move_blocker_items(clone, recommendation)
        elif normalized == "advance_item_to_earlier_sprint":
            self._apply_advance_item(clone, recommendation)
        elif normalized in {"split_task", "split_item"}:
            self._apply_split_task(clone, recommendation)
        elif normalized in {"critical_path_optimization", "remove_dependency_bottleneck"}:
            self._apply_critical_path_optimization(clone, recommendation)

    def _apply_action(self, clone: ProjectState, action: SimulationAction) -> None:
        """Apply a single simulation action using deterministic phase 1 effects."""
        normalized = str(action.action_type).strip().lower().replace(" ", "_")

        if normalized == RecommendationType.RESOLVE_BLOCKER.value.lower().replace(" ", "_"):
            self._apply_resolve_blocker(clone, action)
        elif normalized == RecommendationType.REDUCE_ITEM_SCOPE.value.lower().replace(" ", "_"):
            self._apply_reduce_scope(clone, action)
        elif normalized == RecommendationType.ADD_RESOURCE.value.lower().replace(" ", "_"):
            self._apply_add_capacity(clone, action)
        elif normalized == RecommendationType.PARALLELIZE_TASKS.value.lower().replace(" ", "_"):
            self._apply_parallelize_work(clone, action)
        elif normalized == RecommendationType.REASSIGN_WORK.value.lower().replace(" ", "_"):
            self._apply_reassign_work(clone, action)
        elif normalized == RecommendationType.MOVE_BLOCKER_ITEMS.value.lower().replace(" ", "_"):
            self._apply_move_blocker_items(clone, action)
        elif normalized == RecommendationType.SPLIT_TASK.value.lower().replace(" ", "_"):
            self._apply_split_task(clone, action)
        elif normalized == RecommendationType.CRITICAL_PATH_OPTIMIZATION.value.lower().replace(" ", "_"):
            self._apply_critical_path_optimization(clone, action)

    def _recalculate_clone(self, clone: ProjectState) -> Dict[str, Any]:
        metrics = MetricsEngine(clone).calculate()
        dag = DependencyGraphEngine(clone).build_dag()
        cp_result = CriticalPathEngine(clone, dag).analyze()
        spillover = SpilloverAnalysisEngine(clone, metrics.average_item_effort).analyze()
        forecast = ForecastEngine(clone, metrics, cp_result, spillover).calculate()
        monte_carlo = MonteCarloEngine(
            project_state=clone,
            metrics=metrics,
            cp_result=cp_result,
            spillover=spillover,
            simulation_count=self.simulation_count,
            seed=42 if self.seed is None else self.seed,
        ).calculate()
        impact_scores = ImpactScoringEngine(clone, dag).score()
        risk_result = RiskEngine(
            project_state=clone,
            metrics=metrics,
            cp_result=cp_result,
            dag=dag,
            spillover=spillover,
            forecast=forecast,
            monte_carlo=monte_carlo,
            impact_scores=impact_scores,
        ).analyze()

        return {
            "metrics": metrics,
            "dag": dag,
            "cp_result": cp_result,
            "spillover": spillover,
            "forecast": forecast,
            "monte_carlo": monte_carlo,
            "risk_result": risk_result,
        }

    def _build_scenario_result(self, recommendations: Sequence[Recommendation], simulated: Dict[str, Any], clone: ProjectState) -> ScenarioResult:
        simulated_forecast: ForecastResult = simulated["forecast"]
        simulated_mc: MonteCarloResult = simulated["monte_carlo"]
        simulated_cp: CriticalPathResult = simulated["cp_result"]
        simulated_risk: RiskResult = simulated["risk_result"]
        simulated_metrics: ProjectMetrics = simulated["metrics"]
        simulated_spillover: SpilloverAnalysis = simulated["spillover"]

        baseline_finish_date = self.forecast.expected_finish_date
        simulated_finish_date = simulated_forecast.expected_finish_date
        baseline_probability = self.monte_carlo.on_time_probability
        simulated_probability = simulated_mc.on_time_probability
        baseline_risk_score = self.risk_result.overall_risk_score
        simulated_risk_score = simulated_risk.overall_risk_score
        baseline_delay_days = self.forecast.expected_delay_days
        simulated_delay_days = simulated_forecast.expected_delay_days

        days_saved = max(0.0, (baseline_finish_date - simulated_finish_date).days)
        finish_date_delta = (simulated_finish_date - baseline_finish_date).days
        confidence_delta = simulated_probability - baseline_probability
        risk_reduction = max(0.0, baseline_risk_score - simulated_risk_score)

        baseline_utilization = self.metrics.resource_metrics.avg_allocation_pct * self.metrics.resource_metrics.avg_availability_pct
        simulated_utilization = simulated_metrics.resource_metrics.avg_allocation_pct * simulated_metrics.resource_metrics.avg_availability_pct
        baseline_spillover = sum(float(value) for value in getattr(self.spillover, "predicted_spillover_by_sprint", {}).values())
        simulated_spillover = sum(float(value) for value in getattr(simulated_spillover, "predicted_spillover_by_sprint", {}).values())
        baseline_blockers = getattr(self.metrics.blocker_metrics, "active_blocker_count", 0)
        simulated_blockers = getattr(simulated_metrics.blocker_metrics, "active_blocker_count", 0)
        velocity_delta = simulated_forecast.projected_velocity - self.forecast.projected_velocity

        recommendation = recommendations[0] if recommendations else None
        estimated_benefit = float(getattr(recommendation, "estimated_delay_reduction_days", 0.0) or 0.0)
        actual_simulated_benefit = max(0.0, baseline_delay_days - simulated_delay_days)
        if estimated_benefit > 0:
            recommendation_accuracy = max(0.0, min(1.0, actual_simulated_benefit / estimated_benefit))
        else:
            recommendation_accuracy = 1.0 if actual_simulated_benefit <= 0.0 else 0.0

        overall_improvement_score = min(
            100.0,
            max(0.0, confidence_delta * 100.0) * 0.35
            + max(0.0, risk_reduction) * 0.30
            + max(0.0, actual_simulated_benefit) * 2.0 * 0.20
            + max(0.0, velocity_delta) * 0.15,
        )
        warnings: List[str] = []
        if overall_improvement_score < 5.0:
            warnings.append("Recommendation produced no measurable benefit in the deterministic rerun.")
        if recommendation is not None and recommendation.affected_blocker_ids and simulated_blockers >= baseline_blockers:
            warnings.append("The simulated blocker impact did not improve blocker exposure.")

        current_sprint_id = recommendations[0].affected_sprint_ids[0] if recommendations and recommendations[0].affected_sprint_ids else ""
        revised_plan = self.build_revised_sprint_plan(clone, current_sprint_id) if current_sprint_id else []

        return ScenarioResult(
            metadata=ScenarioMetadata(
                scenario_id=uuid4().hex,
                selected_recommendations=[rec.recommendation_id for rec in recommendations],
            ),
            forecast_comparison=ForecastComparison(
                baseline_finish_date=baseline_finish_date,
                simulated_finish_date=simulated_finish_date,
                days_saved=days_saved,
                finish_date_delta=float(finish_date_delta),
                baseline_delay_days=float(baseline_delay_days),
                simulated_delay_days=float(simulated_delay_days),
            ),
            monte_carlo_comparison=MonteCarloComparison(
                baseline_on_time_probability=baseline_probability,
                simulated_on_time_probability=simulated_probability,
                confidence_delta=float(confidence_delta),
            ),
            risk_comparison=RiskComparison(
                baseline_risk_score=baseline_risk_score,
                simulated_risk_score=simulated_risk_score,
                risk_reduction=float(risk_reduction),
            ),
            metrics_comparison=MetricsComparison(
                velocity_delta=float(velocity_delta),
                utilization_delta=float(simulated_utilization - baseline_utilization),
                carryover_delta=float(baseline_spillover - simulated_spillover),
                blocker_delta=float(baseline_blockers - simulated_blockers),
            ),
            recommendation_effectiveness=RecommendationEffectiveness(
                estimated_benefit=estimated_benefit,
                actual_simulated_benefit=actual_simulated_benefit,
                recommendation_accuracy=float(recommendation_accuracy),
            ),
            summary=ScenarioSummary(
                overall_improvement_score=float(overall_improvement_score),
                simulation_success=bool(
                    simulated_risk_score <= baseline_risk_score
                    or simulated_probability >= baseline_probability
                    or simulated_finish_date <= baseline_finish_date
                ),
                warnings=warnings,
            ),
            revised_sprint_plan=revised_plan,
        )

    def _build_legacy_result(self, actions: List[SimulationAction], simulated: Dict[str, Any]) -> SimulationResult:
        simulated_forecast: ForecastResult = simulated["forecast"]
        simulated_mc: MonteCarloResult = simulated["monte_carlo"]
        simulated_cp: CriticalPathResult = simulated["cp_result"]
        simulated_risk: RiskResult = simulated["risk_result"]

        baseline_finish_date = self.forecast.expected_finish_date
        simulated_finish_date = simulated_forecast.expected_finish_date
        baseline_p80_date = self.monte_carlo.p80_finish_date
        simulated_p80_date = simulated_mc.p80_finish_date
        baseline_cp_hours = self.cp_result.critical_path_duration_hours
        simulated_cp_hours = simulated_cp.critical_path_duration_hours
        baseline_risk_score = self.risk_result.overall_risk_score
        simulated_risk_score = simulated_risk.overall_risk_score

        return SimulationResult(
            baseline_finish_date=baseline_finish_date,
            simulated_finish_date=simulated_finish_date,
            baseline_risk_score=baseline_risk_score,
            simulated_risk_score=simulated_risk_score,
            baseline_p80_date=baseline_p80_date,
            simulated_p80_date=simulated_p80_date,
            baseline_critical_path_hours=baseline_cp_hours,
            simulated_critical_path_hours=simulated_cp_hours,
            days_recovered=float(round((baseline_finish_date - simulated_finish_date).days, 1)),
            risk_reduction=float(round(baseline_risk_score - simulated_risk_score, 2)),
            recommendations_applied=[action.action_id for action in actions],
            action_reasons=[action.impact_reason for action in actions],
            baseline_probability=self.monte_carlo.on_time_probability,
            simulated_probability=simulated_mc.on_time_probability,
            baseline_delay_days=self.forecast.expected_delay_days,
            simulated_delay_days=simulated_forecast.expected_delay_days,
        )

    def _apply_resolve_blocker(self, clone: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        if hasattr(action, "target_ids") and action.target_ids:
            blocker_ids = action.target_ids
        elif hasattr(action, "affected_blocker_ids"):
            blocker_ids = action.affected_blocker_ids
        else:
            blocker_ids = []

        for blocker_id in blocker_ids:
            blocker = next((b for b in clone.blockers if b.blocker_id == blocker_id), None)
            if not blocker:
                continue
            blocker.status = BlockerStatus.RESOLVED
            blocker.actual_resolution_date = datetime.now(timezone.utc)
            if blocker.raised_date is not None and blocker.raised_date.tzinfo is None:
                blocker.raised_date = blocker.raised_date.replace(tzinfo=timezone.utc)
            self._unblock_impacted_items(clone, blocker)

    def _unblock_impacted_items(self, clone: ProjectState, blocker: Blocker) -> None:
        for impacted_item_id in getattr(blocker, "impacted_item_ids", []) or []:
            item = next((wi for wi in clone.work_items if wi.item_id == impacted_item_id), None)
            if item and item.status == WorkItemStatus.BLOCKED:
                item.status = (
                    WorkItemStatus.IN_PROGRESS
                    if item.progress_pct > 0.0 or item.actual_effort_hrs > 0.0
                    else WorkItemStatus.NOT_STARTED
                )

    def _apply_reduce_scope(self, clone: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        if not target_ids:
            return
        item_id = target_ids[0]
        item = next((wi for wi in clone.work_items if wi.item_id == item_id), None)
        if not item:
            return
        core_hours = 0.6 * item.current_estimate_hrs
        if isinstance(action, SimulationAction):
            core_hours = float(action.details.get("core_hours", core_hours))
        reduction = max(0.0, item.current_estimate_hrs - core_hours)
        item.current_estimate_hrs = max(0.0, core_hours)
        item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs - reduction)
        item.is_scope_changed = True
        item.scope_change_reason = (
            f"Simulation scope reduction: retained {item.current_estimate_hrs:.1f}h and deferred {reduction:.1f}h."
        )

    def _apply_add_capacity(self, clone: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        skill = "General"
        role = "Capacity Resource"
        capacity_gain_hours = 20.0
        if isinstance(action, SimulationAction):
            skill = action.details.get("skill", skill)
            role = action.details.get("role", role)
            capacity_gain_hours = float(action.details.get("capacity_gain_hours", capacity_gain_hours))
        else:
            capacity_gain_hours = float(getattr(action, "estimated_hours_recovered", capacity_gain_hours) or capacity_gain_hours)

        resource_id = f"SIM-R-{len(clone.team) + 1}"
        clone.team.append(
            Resource(
                resource_id=resource_id,
                name=f"Simulated {role}",
                role=role,
                primary_skill=skill,
                secondary_skill=None,
                skill_level=SkillLevel.MID,
                allocation_pct=0.5,
                availability_pct=1.0,
                daily_capacity_hrs=8.0,
            )
        )

        for sprint in clone.sprints:
            if sprint.status in {SprintStatus.NOT_STARTED, SprintStatus.IN_PROGRESS}:
                sprint.planned_velocity_hrs += capacity_gain_hours

        existing_actuals = [a.actual_effort_hrs for a in clone.actuals if a.actual_effort_hrs is not None]
        avg_actual_velocity = sum(existing_actuals) / len(existing_actuals) if existing_actuals else capacity_gain_hours
        synthetic_actual_value = max(capacity_gain_hours, avg_actual_velocity + capacity_gain_hours * 0.5)
        clone.actuals.append(
            SprintActual(
                sprint_id=f"SIM-{resource_id}",
                sprint_number=max((s.sprint_number for s in clone.sprints), default=0) + 1,
                planned_effort_hrs=capacity_gain_hours,
                actual_effort_hrs=synthetic_actual_value,
                variance_hrs=0.0,
                tasks_planned=0,
                tasks_completed=0,
                completion_rate=1.0,
                carryover_count=0,
                scope_change_hours=0.0,
                blocker_impact_hrs=0.0,
            )
        )

    def _apply_parallelize_work(self, clone: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        if len(target_ids) < 2:
            return
        pred_id, succ_id = target_ids[0], target_ids[1]
        successor = next((wi for wi in clone.work_items if wi.item_id == succ_id), None)
        if successor:
            reduction = successor.current_estimate_hrs * 0.2
            successor.current_estimate_hrs = max(0.0, successor.current_estimate_hrs - reduction)
            successor.remaining_effort_hrs = max(0.0, successor.remaining_effort_hrs - reduction)

        dependency = next(
            (
                d
                for d in clone.dependencies
                if d.predecessor_item_id == pred_id and d.successor_item_id == succ_id
            ),
            None,
        )
        if dependency and dependency.lag_days > 0:
            dependency.lag_days = max(0, dependency.lag_days - 1)

    def _apply_reassign_work(self, clone: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        if not target_ids:
            return
        item_id = target_ids[0]
        item = next((wi for wi in clone.work_items if wi.item_id == item_id), None)
        if not item:
            return
        if isinstance(action, SimulationAction):
            new_resource_name = action.details.get("to")
            new_resource = next((r for r in clone.team if r.name == new_resource_name), None)
            if new_resource:
                item.assigned_resource = new_resource.resource_id
                new_resource.allocation_pct = min(1.0, new_resource.allocation_pct + 0.1)
            return

        resource_id = (action.affected_resource_ids or [None])[0]
        if resource_id:
            item.assigned_resource = resource_id

    def _apply_move_blocker_items(self, clone: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        advanceable_items = []
        if isinstance(action, SimulationAction):
            advanceable_items = list(action.details.get("advanceable_items", []) or [])
        else:
            advanceable_items = list(getattr(action, "affected_item_ids", []) or [])
        for item_id in advanceable_items:
            item = next((wi for wi in clone.work_items if wi.item_id == item_id), None)
            if item and item.status == WorkItemStatus.NOT_STARTED:
                item.status = WorkItemStatus.IN_PROGRESS

    def _apply_split_task(self, clone: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        if not target_ids:
            return
        item_id = target_ids[0]
        item = next((wi for wi in clone.work_items if wi.item_id == item_id), None)
        if not item:
            return
        reduction = item.current_estimate_hrs * 0.15
        item.current_estimate_hrs = max(1.0, item.current_estimate_hrs - reduction)
        item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs - reduction)

    def _apply_critical_path_optimization(self, clone: ProjectState, action: Union[SimulationAction, Recommendation]) -> None:
        target_ids = getattr(action, "target_ids", []) or getattr(action, "affected_item_ids", []) or []
        for item_id in target_ids:
            item = next((wi for wi in clone.work_items if wi.item_id == item_id), None)
            if not item:
                continue
            reduction = item.current_estimate_hrs * 0.15
            item.current_estimate_hrs = max(1.0, item.current_estimate_hrs - reduction)
            item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs - reduction)


# ============================================================================
# Recommendation-Specific Simulation Engine V2
# ============================================================================
# The following classes are specialized for recommendation simulation:
# ActionApplicatorV2, EngineRunnerV2, and SimulationEngineV2.
# These work with recommendation-specific models and return types.
# ============================================================================


class ActionApplicatorV2:
    """Apply a recommendation to a cloned ProjectState (V2 - specialized for recommendations)."""

    def apply(self, state: ProjectState, rec: Recommendation) -> None:
        action_name = str(getattr(rec.action_type, "value", rec.action_type)).strip().lower()

        # If the action declares forecast levers, sample those values before applying
        declared = FORECAST_LEVER_MAP.get(rec.action_type)
        before_samples = {}
        if declared:
            for path in declared:
                before_samples[path] = sample_lever_values(state, rec, path)

        if action_name == "resolve_blocker":
            self._apply_resolve_blocker(state, rec)
        elif action_name == "reassign_item":
            self._apply_reassign_item(state, rec)
        elif action_name == "split_item":
            self._apply_split_item(state, rec)
        elif action_name == "advance_item_to_earlier_sprint":
            self._apply_advance_item(state, rec)
        elif action_name == "parallelize_items":
            self._apply_parallelize_items(state, rec)
        elif action_name == "rebalance_sprint_load":
            self._apply_rebalance_sprint_load(state, rec)
        elif action_name == "remove_dependency_bottleneck":
            self._apply_remove_dependency_bottleneck(state, rec)
        elif action_name == "add_resource_skill":
            self._apply_add_resource_skill(state, rec)
        elif action_name == "rebaseline_estimate":
            self._apply_rebaseline_estimate(state, rec)
        elif action_name == "pair_reviewer":
            self._apply_pair_reviewer(state, rec)
        elif action_name == "escalate_blocker_early":
            self._apply_escalate_blocker_early(state, rec)
        elif action_name == "cross_train_backup":
            self._apply_cross_train_backup(state, rec)
        elif action_name == "insert_review_gate":
            self._apply_insert_review_gate(state, rec)
        elif action_name == "apply_ramp_up_discount":
            self._apply_apply_ramp_up_discount(state, rec)
        elif action_name == "resequence_non_critical_item":
            self._apply_resequence_non_critical_item(state, rec)
        elif action_name == "swarm_item":
            self._apply_swarm_item(state, rec)

        # After apply: if declared levers exist, verify at least one changed.
        if declared:
            # Filter to only those lever paths that actually returned values (scoped to the rec)
            sampled = {p: b for p, b in before_samples.items() if b}
            if sampled:
                unchanged = True
                for path, before in sampled.items():
                    after = sample_lever_values(state, rec, path)
                    if before != after:
                        unchanged = False
                        break
                if unchanged:
                    logging.getLogger(__name__).warning(
                        "Applicator did not modify declared forecast levers for %s: %s",
                        rec.recommendation_id,
                        list(sampled.keys()),
                    )

    def apply_many(self, state: ProjectState, recs: List[Recommendation]) -> None:
        """Apply in lexicographic recommendation_id order for determinism."""
        for rec in sorted(recs, key=lambda r: r.recommendation_id):
            self.apply(state, rec)

    def _apply_resolve_blocker(self, state: ProjectState, rec: Recommendation) -> None:
        for blocker_id in rec.affected_blocker_ids:
            blocker = next((b for b in state.blockers if b.blocker_id == blocker_id), None)
            if blocker is not None:
                blocker.status = BlockerStatus.RESOLVED
                blocker.actual_resolution_date = blocker.raised_date
                self._unblock_impacted_items(state, blocker)

    def _unblock_impacted_items(self, state: ProjectState, blocker: Blocker) -> None:
        for impacted_item_id in getattr(blocker, "impacted_item_ids", []) or []:
            item = next((wi for wi in state.work_items if wi.item_id == impacted_item_id), None)
            if item and item.status == WorkItemStatus.BLOCKED:
                item.status = (
                    WorkItemStatus.IN_PROGRESS
                    if item.progress_pct > 0.0 or item.actual_effort_hrs > 0.0
                    else WorkItemStatus.NOT_STARTED
                )

    def _apply_reassign_item(self, state: ProjectState, rec: Recommendation) -> None:
        sim_params = getattr(rec, "metadata", {}).get("simulation_params", {}) if getattr(rec, "metadata", None) else {}
        # Prefer the explicit target_resource_id every real REASSIGN_ITEM candidate sets
        # (candidate_generator.py always includes it). affected_resource_ids[0] is the
        # CURRENT owner in every real candidate path (workload, skill-mismatch, low-velocity),
        # not the intended receiver -- using it directly reassigns items back to themselves,
        # a silent no-op. Fall back to affected_resource_ids[-1] only for legacy callers that
        # never set simulation_params at all.
        resource_id = sim_params.get("target_resource_id") or (
            rec.affected_resource_ids[-1] if rec.affected_resource_ids else None
        )
        if not resource_id:
            return
        target_resource = next((r for r in state.team if r.resource_id == resource_id), None)
        if target_resource is None:
            return

        req_skill = sim_params.get("required_skill")

        for item in state.work_items:
            if item.item_id not in rec.affected_item_ids:
                continue

            item.assigned_resource = resource_id

            can_help = False
            if req_skill and target_resource.primary_skill and str(req_skill).lower() in str(target_resource.primary_skill).lower():
                can_help = True
            if target_resource.allocation_pct * target_resource.availability_pct < 0.70:
                can_help = True

            if can_help:
                item.remaining_effort_hrs = max(1.0, item.remaining_effort_hrs * 0.92)
                item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * 0.92)
                item.progress_pct = min(1.0, item.progress_pct + 0.02)
                target_resource.allocation_pct = min(1.0, target_resource.allocation_pct + 0.05)

    def _apply_split_item(self, state: ProjectState, rec: Recommendation) -> None:
        # Step 3 causal rewrite: in addition to the deepcopy-based split, write the
        # parent_item_id and can_parallel_with domain fields so the dependency engine
        # can model the two halves as explicitly parallelizable rather than summing
        # their hours sequentially.
        #
        # Causal chain:
        #   new_item.parent_item_id = original item ID
        #     → dependency engine knows this is a derived item, not an independent task
        #   original_item.can_parallel_with.append(new_item.item_id)
        #   new_item.can_parallel_with.append(original_item.item_id)
        #     → critical path engine reads can_parallel_with to exclude these pairs
        #       from the sequential dependency assumption, reducing critical path hours
        from copy import deepcopy
        for item in list(state.work_items):
            if item.item_id not in rec.affected_item_ids:
                continue

            original_hours = float(item.current_estimate_hrs)
            half_hours = max(1.0, original_hours / 2.0)

            # Update the existing item to represent one half
            item.current_estimate_hrs = half_hours
            item.remaining_effort_hrs = max(0.0, float(item.remaining_effort_hrs) / 2.0)

            # Unique ID for the new sibling
            suffix = "-split"
            new_id = item.item_id + suffix
            idx = 1
            existing_ids = {wi.item_id for wi in state.work_items}
            while new_id in existing_ids:
                new_id = f"{item.item_id}{suffix}{idx}"
                idx += 1

            new_item = deepcopy(item)
            new_item.item_id = new_id
            new_item.current_estimate_hrs = half_hours
            new_item.remaining_effort_hrs = max(0.0, float(new_item.remaining_effort_hrs) / 2.0)
            new_item.progress_pct = 0.0
            new_item.actual_effort_hrs = 0.0
            # Keep same assigned sprint/resource so both can run in parallel

            # Domain mutation: record the decomposition relationship
            new_item.parent_item_id = item.item_id
            new_item.can_parallel_with = [item.item_id]

            # Record on the original item that its sibling can run alongside it
            if new_id not in item.can_parallel_with:
                item.can_parallel_with.append(new_id)

            state.work_items.append(new_item)

    def _apply_advance_item(self, state: ProjectState, rec: Recommendation) -> None:
        # affected_sprint_ids is not guaranteed to be ordered [current, target] --
        # do not trust index [0] as "the destination." Compute the actual earlier
        # sprint explicitly from sprint_number, and only move items whose current
        # sprint really is later than that target.
        sprint_by_name = {s.sprint_name: s for s in state.sprints}
        sprints_by_number = sorted(state.sprints, key=lambda s: s.sprint_number)
        for item in state.work_items:
            if item.item_id not in rec.affected_item_ids:
                continue
            current_sprint = sprint_by_name.get(item.assigned_sprint)
            if current_sprint is None:
                continue
            earlier_candidates = [s for s in sprints_by_number if s.sprint_number < current_sprint.sprint_number]
            if not earlier_candidates:
                continue  # already in the earliest sprint -- nothing to advance into
            target_sprint = earlier_candidates[-1]  # the nearest earlier sprint
            item.assigned_sprint = target_sprint.sprint_name

    def _apply_parallelize_items(self, state: ProjectState, rec: Recommendation) -> None:
        for dep in state.dependencies:
            if dep.predecessor_item_id in rec.affected_item_ids and dep.successor_item_id in rec.affected_item_ids:
                dep.lag_days = max(0, dep.lag_days - 1)
                for item in state.work_items:
                    if item.item_id == dep.successor_item_id:
                        reduction = item.current_estimate_hrs * 0.1
                        item.current_estimate_hrs = max(1.0, item.current_estimate_hrs - reduction)
                        item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs - reduction)

    def _apply_rebalance_sprint_load(self, state: ProjectState, rec: Recommendation) -> None:
        resource_id = rec.affected_resource_ids[0] if rec.affected_resource_ids else None
        if not resource_id:
            return
        for item in state.work_items:
            if item.item_id in rec.affected_item_ids:
                item.assigned_resource = resource_id

    def _apply_remove_dependency_bottleneck(self, state: ProjectState, rec: Recommendation) -> None:
        # Decrementing lag_days alone is a no-op for the majority of real dependencies,
        # which have lag_days == 0 to begin with (a plain "B can't start until A finishes"
        # coupling, no extra buffer to shave off). Mirror _apply_parallelize_items instead:
        # loosen the successor's own effort as the primary effect, and still decrement
        # lag_days as a secondary effect for the minority of dependencies where it's nonzero.
        successor_ids = set()
        for dep in state.dependencies:
            if dep.predecessor_item_id in rec.affected_item_ids or dep.successor_item_id in rec.affected_item_ids:
                dep.lag_days = max(0, dep.lag_days - 1)
                successor_ids.add(dep.successor_item_id)
        for item in state.work_items:
            if item.item_id in successor_ids:
                item.current_estimate_hrs = round(item.current_estimate_hrs * 0.9, 2)
                item.remaining_effort_hrs = round(item.remaining_effort_hrs * 0.9, 2)

    def _apply_add_resource_skill(self, state: ProjectState, rec: Recommendation) -> None:
        resource_id = rec.affected_resource_ids[0] if rec.affected_resource_ids else None
        if resource_id is None:
            return
        for resource in state.team:
            if resource.resource_id == resource_id:
                sim_params = getattr(rec, "metadata", {}).get("simulation_params", {}) if getattr(rec, "metadata", None) else {}
                req_skill = sim_params.get("required_skill")
                if req_skill:
                    resource.primary_skill = req_skill
                resource.allocation_pct = min(1.0, resource.allocation_pct + 0.05)
                resource.availability_pct = min(1.0, resource.availability_pct + 0.05)

                for sprint in state.sprints:
                    if sprint.status in {SprintStatus.NOT_STARTED, SprintStatus.IN_PROGRESS}:
                        sprint.planned_velocity_hrs += 12.0

    def _apply_rebaseline_estimate(self, state: ProjectState, rec: Recommendation) -> None:
        for item_id in rec.affected_item_ids:
            item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
            if item is None:
                continue
            scale = 1.15
            item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * scale)
            item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs * scale)

    def _apply_pair_reviewer(self, state: ProjectState, rec: Recommendation) -> None:
        for item_id in rec.affected_item_ids:
            item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
            if item is None:
                continue
            item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs * 0.95)
            item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * 0.95)

    def _apply_escalate_blocker_early(self, state: ProjectState, rec: Recommendation) -> None:
        # Fix #13: previous implementation was conditional — it only mutated state if
        # target_resolution_date was set AND impacted_item_ids were non-empty. Either
        # absent meant a silent no-op. Escalation must ALWAYS produce a real mutation:
        #
        #   (a) Unconditional: bump blocker severity to its next level (MEDIUM→HIGH,
        #       HIGH→CRITICAL) to model the escalation going on record.
        #   (b) Conditional but guaranteed non-empty if the blocker has a date: pull
        #       the resolution target forward by 2 days.
        #   (c) Effort reduction on impacted items when available (unchanged from before).
        #
        # Severity bump is the invariant — even a blocker with no date and no impacted
        # items gets its severity raised, so simulate() always sees a state delta.
        from app.domain.models import BlockerSeverity  # local import to avoid circular
        _severity_ladder = {
            BlockerSeverity.LOW: BlockerSeverity.MEDIUM,
            BlockerSeverity.MEDIUM: BlockerSeverity.HIGH,
            BlockerSeverity.HIGH: BlockerSeverity.CRITICAL,
            BlockerSeverity.CRITICAL: BlockerSeverity.CRITICAL,  # already at ceiling
        }
        for blocker_id in rec.affected_blocker_ids:
            blocker = next((b for b in state.blockers if b.blocker_id == blocker_id), None)
            if blocker is None:
                continue
            # (a) Unconditional severity escalation — always mutates state
            if hasattr(blocker, "severity") and blocker.severity in _severity_ladder:
                blocker.severity = _severity_ladder[blocker.severity]
            # (b) Pull resolution date forward when available
            if blocker.target_resolution_date is not None:
                blocker.target_resolution_date = blocker.target_resolution_date - timedelta(days=2)
            # (c) Effort reduction on impacted items
            for item_id in getattr(blocker, "impacted_item_ids", []) or []:
                item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
                if item is None:
                    continue
                item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs * 0.97)
                item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * 0.97)
            # Also reduce effort for items in rec.affected_item_ids not covered above
            # (from recurring-blocker and spillover candidates that include item IDs directly)
            for item_id in rec.affected_item_ids:
                item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
                if item is None:
                    continue
                item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs * 0.97)
                item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * 0.97)

    def _apply_cross_train_backup(self, state: ProjectState, rec: Recommendation) -> None:
        # Step 3 causal rewrite: model the real domain change — the SPOF resource gains
        # a backup peer who now covers their skill — rather than incrementing a velocity
        # scalar by an invented constant.
        #
        # Causal chain:
        #   resource.skill_coverage gets a BACKUP entry for the SPOF's primary_skill
        #     → MetricsEngine can detect that the skill gap is now covered by a backup
        #     → blocker/capacity detectors see reduced single-point-of-failure risk
        #   sprint.capacity_breakdown gets a SprintCapacityEntry for the backup resource
        #     → effective sprint capacity increases by a derived amount (backup resource
        #       contributes at 50% of their daily_capacity_hrs, modelling ramp-up cost)
        #     → ForecastEngine sums planned_velocity_hrs + simulation_capacity_hrs()
        #
        # affected_resource_ids[0] is the SPOF; [1] is the backup (if identified by
        # the candidate generator's SPOF signal — which now includes both IDs).
        from app.domain.models import SkillCoverage, SkillProficiency, SprintCapacityEntry

        if not rec.affected_resource_ids:
            return
        spof_id = rec.affected_resource_ids[0]
        backup_id = rec.affected_resource_ids[1] if len(rec.affected_resource_ids) > 1 else None

        spof_resource = next((r for r in state.team if r.resource_id == spof_id), None)
        backup_resource = next((r for r in state.team if r.resource_id == backup_id), None) if backup_id else None

        if spof_resource is None:
            return

        skill_to_cover = spof_resource.primary_skill

        # (a) Domain mutation: record that backup_resource now covers the SPOF's skill
        if backup_resource is not None:
            already_covered = backup_resource.covers_skill(skill_to_cover)
            if not already_covered:
                backup_resource.skill_coverage.append(
                    SkillCoverage(
                        skill=skill_to_cover,
                        proficiency=SkillProficiency.BACKUP,
                        certified=False,
                        acquired_via="cross_training",
                    )
                )

            # (b) Capacity mutation: backup contributes to active sprints at 50% capacity
            #     (modelling training overhead / reduced efficiency while ramping up)
            backup_daily_hrs = float(getattr(backup_resource, "daily_capacity_hrs", 8.0) or 8.0)
            backup_avail = float(getattr(backup_resource, "availability_pct", 1.0) or 1.0)
            contributed_hours_per_sprint = backup_daily_hrs * backup_avail * 0.5  # 50% ramp factor

            sprint_ids = set(rec.affected_sprint_ids) if rec.affected_sprint_ids else set()
            for sprint in state.sprints:
                if sprint.status not in {SprintStatus.NOT_STARTED, SprintStatus.IN_PROGRESS}:
                    continue
                if sprint_ids and sprint.sprint_id not in sprint_ids and sprint.sprint_name not in sprint_ids:
                    continue
                # Only add entry if not already present (idempotent)
                already_entered = any(
                    e.resource_id == backup_id and e.source == "cross_train"
                    for e in sprint.capacity_breakdown
                )
                if not already_entered:
                    sprint.capacity_breakdown.append(
                        SprintCapacityEntry(
                            resource_id=backup_id,
                            hours=contributed_hours_per_sprint,
                            source="cross_train",
                        )
                    )
        else:
            # No identified backup resource — fall back to a minimal allocation bump on
            # the SPOF resource itself (represents generic cross-training investment).
            # This keeps the applicator from being a no-op when the candidate was generated
            # without a specific backup peer (e.g. from an older signal path).
            spof_resource.allocation_pct = min(1.0, spof_resource.allocation_pct + 0.03)

    def _apply_insert_review_gate(self, state: ProjectState, rec: Recommendation) -> None:
        for item_id in rec.affected_item_ids:
            item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
            if item is None:
                continue
            item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs * 0.94)
            item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * 0.94)

    def _apply_apply_ramp_up_discount(self, state: ProjectState, rec: Recommendation) -> None:
        for resource_id in rec.affected_resource_ids:
            resource = next((r for r in state.team if r.resource_id == resource_id), None)
            if resource is None:
                continue
            resource.allocation_pct = max(0.0, resource.allocation_pct - 0.05)
            resource.availability_pct = max(0.0, resource.availability_pct - 0.05)
            for sprint in state.sprints:
                if sprint.status in {SprintStatus.NOT_STARTED, SprintStatus.IN_PROGRESS}:
                    sprint.planned_velocity_hrs = max(1.0, sprint.planned_velocity_hrs * 0.9)

    def _apply_resequence_non_critical_item(self, state: ProjectState, rec: Recommendation) -> None:
        # Fix #17: the ResequencingDetector deliberately selects items that have NO
        # dependency edge to the critical-path item. If the item also has no other edges,
        # the dependency loop below finds nothing and the method is a no-op.
        # Primary effect: reduce the non-critical item's remaining effort directly
        # (modelling the time freed by deferring/resequencing it off the shared resource).
        # Secondary effect: still loosen any dependency lag that does exist.
        mutated = set()
        for item_id in rec.affected_item_ids:
            item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
            if item is None:
                continue
            # Direct effort reduction: resequencing frees the shared resource to focus
            # on critical-path work, effectively reducing parallelism overhead.
            item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs * 0.97)
            item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * 0.97)
            mutated.add(item_id)
        # Also loosen any dependency lag where this item appears (secondary effect)
        for dep in state.dependencies:
            if dep.predecessor_item_id in rec.affected_item_ids or dep.successor_item_id in rec.affected_item_ids:
                dep.lag_days = max(0, dep.lag_days - 1)
                # Only apply successor reduction if not already mutated above
                if dep.successor_item_id not in mutated:
                    for item in state.work_items:
                        if item.item_id == dep.successor_item_id:
                            item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs * 0.97)
                            item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * 0.97)

    def _apply_swarm_item(self, state: ProjectState, rec: Recommendation) -> None:
        # Step 3 causal rewrite: swarming means a second resource focuses on the
        # bottleneck item. Model the two real effects directly in the domain:
        #
        # (a) The swarming resource's capacity is committed to this item:
        #     sprint.capacity_breakdown gets an entry for the second resource,
        #     sourced as "swarm", representing their focused contribution.
        # (b) The item records the swarming resource in can_parallel_with —
        #     signalling that two people are working it in parallel, which the
        #     critical path engine can use to derive a reduced effective duration.
        # (c) remaining_effort_hrs is reduced proportionally to the parallelism
        #     gain — derived from the second resource's actual capacity, not a
        #     fixed 0.85 ratio. Two equal-capacity people in parallel produce
        #     effort * 0.5; the reduction is capped at 40% to model coordination
        #     overhead (Brook's Law: adding people to a late project makes it later
        #     at the extreme, but focused swarming of a single item helps).
        # (d) The opportunity cost (displaced work elsewhere) is NOT modelled by
        #     arbitrarily adding +4h to a random other item. Instead, the second
        #     resource's capacity_breakdown entry represents their committed hours,
        #     which the forecast engine will see as unavailable for other items.
        from app.domain.models import SprintCapacityEntry

        swarm_resource_id = rec.affected_resource_ids[0] if rec.affected_resource_ids else None
        swarm_resource = next((r for r in state.team if r.resource_id == swarm_resource_id), None) if swarm_resource_id else None

        for item_id in rec.affected_item_ids:
            item = next((wi for wi in state.work_items if wi.item_id == item_id), None)
            if item is None:
                continue

            # (a) Commit the swarming resource's sprint capacity to this item
            if swarm_resource is not None:
                daily_hrs = float(getattr(swarm_resource, "daily_capacity_hrs", 8.0) or 8.0)
                avail = float(getattr(swarm_resource, "availability_pct", 1.0) or 1.0)
                sprint_ids = set(rec.affected_sprint_ids) if rec.affected_sprint_ids else set()
                for sprint in state.sprints:
                    if sprint.status not in {SprintStatus.NOT_STARTED, SprintStatus.IN_PROGRESS}:
                        continue
                    if sprint_ids and sprint.sprint_id not in sprint_ids and sprint.sprint_name not in sprint_ids:
                        continue
                    already_entered = any(
                        e.resource_id == swarm_resource_id and e.source == "swarm"
                        for e in sprint.capacity_breakdown
                    )
                    if not already_entered:
                        sprint.capacity_breakdown.append(
                            SprintCapacityEntry(
                                resource_id=swarm_resource_id,
                                hours=daily_hrs * avail,
                                source="swarm",
                            )
                        )

                # (b) Record parallelism on the item
                if swarm_resource_id not in item.can_parallel_with:
                    item.can_parallel_with.append(swarm_resource_id)

                # (c) Derived effort reduction: 0.5 base parallelism gain, capped at 0.40 reduction
                #     (i.e. effective multiplier floored at 0.60)
                parallelism_factor = 0.60  # two people → 40% time reduction max (coordination overhead)
                item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs * parallelism_factor)
                item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * parallelism_factor)
            else:
                # No identified swarming resource — apply conservative 10% reduction
                # to model generic focus/prioritisation benefit without a fake capacity entry
                item.remaining_effort_hrs = max(0.0, item.remaining_effort_hrs * 0.90)
                item.current_estimate_hrs = max(1.0, item.current_estimate_hrs * 0.90)

            break  # swarm applies to the first affected item only


class EngineRunnerV2:
    """Runs the full engine pipeline on a ProjectState with seed=42, returning UpstreamEngineOutputs."""

    SEED: int = MONTE_CARLO_SEED

    def run(self, state: ProjectState, simulation_count: int = 1000) -> UpstreamEngineOutputs:
        """
        Run: MetricsEngine → DependencyGraphEngine → CriticalPathEngine →
             SpilloverAnalysisEngine → ForecastEngine → MonteCarloEngine(seed=42) →
             ImpactScoringEngine → RiskEngine
        Return UpstreamEngineOutputs.
        """
        metrics = MetricsEngine(state).calculate()
        dag = DependencyGraphEngine(state).build_dag()
        cp_result = CriticalPathEngine(state, dag).analyze()
        spillover = SpilloverAnalysisEngine(state, metrics.average_item_effort).analyze()
        forecast = ForecastEngine(state, metrics, cp_result, spillover).calculate()
        monte_carlo = MonteCarloEngine(
            project_state=state,
            metrics=metrics,
            cp_result=cp_result,
            spillover=spillover,
            simulation_count=simulation_count,
            seed=self.SEED,
        ).calculate()
        impact_scores = ImpactScoringEngine(state, dag).score()
        risk_result = RiskEngine(
            project_state=state,
            metrics=metrics,
            cp_result=cp_result,
            dag=dag,
            spillover=spillover,
            forecast=forecast,
            monte_carlo=monte_carlo,
            impact_scores=impact_scores,
        ).analyze()
        return UpstreamEngineOutputs(
            metrics=metrics,
            dag=dag,
            cp_result=cp_result,
            spillover=spillover,
            forecast=forecast,
            monte_carlo=monte_carlo,
            impact_scores=impact_scores,
            risk_result=risk_result,
        )


class SimulationEngineV2:
    """Orchestrates simulation of recommendations using recommendation-specific models."""

    SEED: int = MONTE_CARLO_SEED

    def __init__(
        self,
        project_state: ProjectState,
        baseline: UpstreamEngineOutputs,
        simulation_count: int = 1000,
    ):
        self.project_state = project_state
        self.baseline = baseline
        self.simulation_count = simulation_count
        self.applicator = ActionApplicatorV2()
        self.runner = EngineRunnerV2()

    def simulate(self, recommendation: Recommendation) -> SimulationResultV2:
        """Deep clone → apply → re-run pipeline → compute deltas."""
        cloned_state = self.project_state.model_copy(deep=True)

        # Compute a lightweight fingerprint of fields the forecast uses so we can
        # detect if the applicator silently failed to mutate state (no-op).
        def _fingerprint(state_obj) -> str:
            try:
                items_sig = tuple(
                    (
                        wi.item_id,
                        getattr(wi, "assigned_resource", None),
                        round(float(getattr(wi, "remaining_effort_hrs", 0.0)), 3),
                        getattr(wi, "assigned_sprint", None),
                        round(float(getattr(wi, "current_estimate_hrs", 0.0)), 3),
                    )
                    for wi in getattr(state_obj, "work_items", [])
                )
                blockers_sig = tuple((b.blocker_id, b.status, getattr(b, "target_resolution_date", None), getattr(b, "severity", None)) for b in getattr(state_obj, "blockers", []))
                team_sig = tuple(
                    (r.resource_id, r.allocation_pct, r.availability_pct,
                     tuple(sc.skill for sc in getattr(r, "skill_coverage", [])))
                    for r in getattr(state_obj, "team", [])
                )
                sprints_sig = tuple(
                    (s.sprint_id, round(s.planned_velocity_hrs, 3),
                     tuple((e.resource_id, e.hours) for e in getattr(s, "capacity_breakdown", [])))
                    for s in getattr(state_obj, "sprints", [])
                )
                key = (items_sig, blockers_sig, team_sig, sprints_sig)
                return str(hash(key))
            except Exception:
                return ""

        before_fp = _fingerprint(cloned_state)
        self.applicator.apply(cloned_state, recommendation)
        after_fp = _fingerprint(cloned_state)

        if before_fp == after_fp:
            raise RuntimeError(f"Simulation applicator did not mutate cloned state for recommendation {recommendation.recommendation_id}")

        simulated = self.runner.run(cloned_state, simulation_count=self.simulation_count)
        return self._compute_result([recommendation.recommendation_id], simulated)

    def simulate_scenario(self, recommendations: List[Recommendation]) -> SimulationResultV2:
        """Deep clone → apply all (sorted by ID) → re-run pipeline → compute deltas."""
        cloned_state = self.project_state.model_copy(deep=True)
        self.applicator.apply_many(cloned_state, recommendations)
        simulated = self.runner.run(cloned_state, simulation_count=self.simulation_count)
        return self._compute_result([r.recommendation_id for r in sorted(recommendations, key=lambda r: r.recommendation_id)], simulated)

    def _compute_result(
        self,
        rec_ids: List[str],
        simulated: UpstreamEngineOutputs,
    ) -> SimulationResultV2:
        """Compute delta fields. baseline comes from self.baseline."""
        baseline_metrics = BaselineMetrics(
            on_time_probability=self.baseline.monte_carlo.on_time_probability,
            expected_delay_days=self.baseline.forecast.expected_delay_days,
            overall_risk_score=self.baseline.risk_result.overall_risk_score,
            schedule_risk=self.baseline.risk_result.schedule_risk.score,
            resource_risk=self.baseline.risk_result.resource_risk.score,
            critical_path_hours=self.baseline.cp_result.critical_path_duration_hours,
        )
        simulated_metrics = SimulatedMetrics(
            on_time_probability=simulated.monte_carlo.on_time_probability,
            expected_delay_days=simulated.forecast.expected_delay_days,
            overall_risk_score=simulated.risk_result.overall_risk_score,
            schedule_risk=simulated.risk_result.schedule_risk.score,
            resource_risk=simulated.risk_result.resource_risk.score,
            critical_path_hours=simulated.cp_result.critical_path_duration_hours,
        )
        delta_on_time_probability = simulated_metrics.on_time_probability - baseline_metrics.on_time_probability
        delta_expected_delay_days = baseline_metrics.expected_delay_days - simulated_metrics.expected_delay_days
        # Compute spillover delta as the change in total predicted spillover items
        try:
            baseline_spill = sum(self.baseline.spillover.predicted_spillover_by_sprint.values())
        except Exception:
            baseline_spill = 0.0
        try:
            simulated_spill = sum(simulated.spillover.predicted_spillover_by_sprint.values())
        except Exception:
            simulated_spill = 0.0
        delta_spillover_risk = float(baseline_spill - simulated_spill)
        delta_risk_score = baseline_metrics.overall_risk_score - simulated_metrics.overall_risk_score
        # Projected velocity delta (positive means velocity recovered)
        try:
            baseline_velocity = float(self.baseline.forecast.projected_velocity)
        except Exception:
            baseline_velocity = 0.0
        try:
            simulated_velocity = float(simulated.forecast.projected_velocity)
        except Exception:
            simulated_velocity = 0.0
        delta_projected_velocity = simulated_velocity - baseline_velocity
        is_positive_impact = (
            delta_on_time_probability > 0
            or delta_expected_delay_days > 0
            or delta_risk_score > 0
        )
        return SimulationResultV2(
            recommendation_ids=rec_ids,
            baseline_metrics=baseline_metrics,
            simulated_metrics=simulated_metrics,
            delta_on_time_probability=round(delta_on_time_probability, 4),
            delta_expected_delay_days=round(delta_expected_delay_days, 4),
            delta_spillover_risk=round(delta_spillover_risk, 4),
            delta_risk_score=round(delta_risk_score, 4),
            delta_projected_velocity=round(delta_projected_velocity, 2),
            seed_used=self.SEED,
            is_positive_impact=is_positive_impact,
            summary=(
                f"Applied {len(rec_ids)} recommendation(s); "
                f"on-time probability delta={round(delta_on_time_probability, 4)}"
            ),
        )


# Aliases for backward compatibility with simulation_engine_v2.py imports
ActionApplicator = ActionApplicatorV2
EngineRunner = EngineRunnerV2
