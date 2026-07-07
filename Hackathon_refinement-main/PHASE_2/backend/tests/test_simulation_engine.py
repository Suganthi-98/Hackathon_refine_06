import pytest
from datetime import datetime, timedelta

from app.domain.models import (
    ProjectInfo,
    Resource,
    Sprint,
    WorkItem,
    Dependency,
    Blocker,
    SprintActual,
    ProjectState,
    SkillLevel,
    WorkItemType,
    Priority,
    WorkItemStatus,
    SprintStatus,
    BlockerSeverity,
    BlockerStatus,
    BlockerCategory,
    DependencyType,
)
from app.engines.metrics_engine import MetricsEngine
from app.engines.dependency_engine import DependencyGraphEngine
from app.engines.critical_path_engine import CriticalPathEngine
from app.engines.spillover_engine import SpilloverAnalysisEngine
from app.engines.forecast_engine import ForecastEngine
from app.engines.monte_carlo_engine import MonteCarloEngine
from app.engines.impact_scoring_engine import ImpactScoringEngine
from app.engines.risk_engine import RiskEngine
from app.engines.simulation_engine import SimulationEngine, SimulationAction
from app.api.models_phase3 import RecommendationType
from app.engines.recommendation_engine.models import Recommendation, RecommendationAction, ConfidenceLevel
from app.engines.ai_advisor import AIAdvisor


def make_simulation_project_state() -> ProjectState:
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="Simulation Test",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=start_date + timedelta(days=60),
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )

    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="SQL",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.8,
            availability_pct=0.8,
        )
    ]

    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=13),
            working_days=10,
            sprint_goal="Build",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=160.0,
            carryover_count=1,
        ),
        Sprint(
            sprint_id="S2",
            sprint_name="Sprint 2",
            sprint_number=2,
            start_date=start_date + timedelta(days=14),
            end_date=start_date + timedelta(days=27),
            working_days=10,
            sprint_goal="Finish",
            status=SprintStatus.NOT_STARTED,
            planned_velocity_hrs=160.0,
            carryover_count=0,
        ),
    ]

    work_items = [
        WorkItem(
            item_id="WI-01",
            title="API work",
            work_type=WorkItemType.TASK,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            assigned_resource="R1",
            required_skill="Python",
            priority=Priority.HIGH,
            estimated_effort_hrs=80.0,
            current_estimate_hrs=80.0,
            actual_effort_hrs=20.0,
            remaining_effort_hrs=60.0,
            progress_pct=0.25,
            status=WorkItemStatus.IN_PROGRESS,
        ),
        WorkItem(
            item_id="WI-02",
            title="Blocked integration",
            work_type=WorkItemType.TASK,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            assigned_resource="R1",
            required_skill="Python",
            priority=Priority.MEDIUM,
            estimated_effort_hrs=40.0,
            current_estimate_hrs=40.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=40.0,
            progress_pct=0.0,
            status=WorkItemStatus.BLOCKED,
        ),
    ]

    dependencies = [
        Dependency(
            dependency_id="DEP-01",
            predecessor_item_id="WI-02",
            successor_item_id="WI-01",
            dependency_type=DependencyType.FINISH_TO_START,
            is_on_critical_path=True,
            lag_days=0,
        )
    ]

    blockers = [
        Blocker(
            blocker_id="BLK-01",
            related_item_id="WI-02",
            impacted_item_ids=["WI-02", "WI-01"],
            description="Test blocker",
            severity=BlockerSeverity.HIGH,
            status=BlockerStatus.OPEN,
            owner="Ops",
            raised_date=start_date,
            target_resolution_date=start_date + timedelta(days=7),
            category=BlockerCategory.OTHER,
        )
    ]

    actuals = [
        SprintActual(
            sprint_id="S0",
            sprint_number=1,
            planned_effort_hrs=150.0,
            actual_effort_hrs=140.0,
            variance_hrs=10.0,
            tasks_planned=8,
            tasks_completed=7,
            completion_rate=0.875,
            carryover_count=1,
            scope_change_hours=0.0,
            blocker_impact_hrs=5.0,
        )
    ]

    return ProjectState(
        project_id="SIM-TEST",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=actuals,
    )


@pytest.fixture
def simulation_engine():
    state = make_simulation_project_state()
    metrics = MetricsEngine(state).calculate()
    dag = DependencyGraphEngine(state).build_dag()
    cp_result = CriticalPathEngine(state, dag).analyze()
    spill = SpilloverAnalysisEngine(state, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(state, metrics, cp_result, spill).calculate()
    monte_carlo = MonteCarloEngine(
        project_state=state,
        metrics=metrics,
        cp_result=cp_result,
        spillover=spill,
        simulation_count=50,
        seed=42,
    ).calculate()
    impact_scores = ImpactScoringEngine(state, dag).score()
    risk_result = RiskEngine(
        project_state=state,
        metrics=metrics,
        cp_result=cp_result,
        dag=dag,
        spillover=spill,
        forecast=forecast,
        monte_carlo=monte_carlo,
        impact_scores=impact_scores,
    ).analyze()
    return SimulationEngine(
        project_state=state,
        metrics=metrics,
        dag=dag,
        cp_result=cp_result,
        spillover=spill,
        forecast=forecast,
        monte_carlo=monte_carlo,
        risk_result=risk_result,
        simulation_count=50,
        seed=42,
    )


def test_simulation_engine_resolve_blocker(simulation_engine):
    action = SimulationAction(
        action_id="REC-001",
        action_type=RecommendationType.RESOLVE_BLOCKER.value,
        target_ids=["BLK-01"],
        details={"blocker_id": "BLK-01"},
        impact_reason="Resolving BLK-01 removes an active blocker on the critical path.",
    )
    result = simulation_engine.simulate_recommendation_actions([action])

    assert result.recommendations_applied == ["REC-001"]
    assert result.action_reasons == ["Resolving BLK-01 removes an active blocker on the critical path."]
    assert result.simulated_risk_score <= result.baseline_risk_score
    assert result.simulated_probability >= result.baseline_probability
    assert result.simulated_p80_date <= result.baseline_p80_date


def test_simulation_engine_resolve_blocker_updates_work_item_status(simulation_engine):
    action = SimulationAction(
        action_id="REC-001",
        action_type=RecommendationType.RESOLVE_BLOCKER.value,
        target_ids=["BLK-01"],
        details={"blocker_id": "BLK-01"},
        impact_reason="Resolving BLK-01 removes an active blocker on the critical path.",
    )
    clone = simulation_engine.project_state.model_copy(deep=True)
    simulation_engine._apply_resolve_blocker(clone, action)

    assert clone.blockers[0].actual_resolution_date is not None
    assert clone.blockers[0].status == BlockerStatus.RESOLVED
    assert clone.work_items[1].status != WorkItemStatus.BLOCKED

    metrics_after = MetricsEngine(clone).calculate()
    assert metrics_after.blocked_items == 0


def test_simulation_engine_add_capacity(simulation_engine):
    action = SimulationAction(
        action_id="REC-002",
        action_type=RecommendationType.ADD_RESOURCE.value,
        target_ids=[],
        details={"skill": "Python", "role": "Engineer", "capacity_gain_hours": 20.0},
        impact_reason="Adding capacity improves sprint velocity and reduces schedule exposure.",
    )
    result = simulation_engine.simulate_recommendation_actions([action])

    assert result.recommendations_applied == ["REC-002"]
    assert result.simulated_finish_date <= result.baseline_finish_date
    assert result.days_recovered >= 0


def test_simulation_engine_reassign_work_uses_explicit_target_resource_id(simulation_engine):
    # NOTE: previously named "...uses_last_affected_resource_as_receiver" and asserted that
    # behavior directly -- that was locking in a fragile, undocumented convention (guessing
    # the receiver from list position) rather than testing real behavior. Every actual
    # REASSIGN_ITEM candidate produced by candidate_generator.py sets an explicit
    # target_resource_id in simulation_params; this test now matches that real contract.
    clone = simulation_engine.project_state.model_copy(deep=True)
    clone.team.append(
        Resource(
            resource_id="R2",
            name="Bob",
            role="Engineer",
            primary_skill="Python",
            secondary_skill=None,
            skill_level=SkillLevel.MID,
            allocation_pct=0.5,
            availability_pct=1.0,
        )
    )

    recommendation = Recommendation(
        recommendation_id="REC-REASSIGN",
        title="Reassign API work",
        description="Move item WI-01 from R1 to R2.",
        action_type=RecommendationAction.REASSIGN_ITEM,
        priority_score=0.75,
        confidence=ConfidenceLevel.MEDIUM,
        estimated_hours_recovered=0.0,
        estimated_delay_reduction_days=0.0,
        estimated_risk_reduction=0.0,
        affected_item_ids=["WI-01"],
        affected_resource_ids=["R1", "R2"],
        affected_sprint_ids=["S1"],
        affected_blocker_ids=[],
        root_cause_signal_id="SIG-002",
        metadata={"simulation_params": {"target_resource_id": "R2"}},
    )

    simulation_engine.applicator.apply(clone, recommendation)

    reassigned_item = next(wi for wi in clone.work_items if wi.item_id == "WI-01")
    assert reassigned_item.assigned_resource == "R2"


def test_simulation_engine_advance_item_moves_to_destination_sprint_name(simulation_engine):
    clone = simulation_engine.project_state.model_copy(deep=True)
    item = next(wi for wi in clone.work_items if wi.item_id == "WI-01")
    item.assigned_sprint = "Sprint 2"

    recommendation = Recommendation(
        recommendation_id="REC-ADVANCE",
        title="Advance work item earlier",
        description="Move work item earlier to the previous sprint.",
        action_type=RecommendationAction.ADVANCE_ITEM_TO_EARLIER_SPRINT,
        priority_score=0.9,
        confidence=ConfidenceLevel.HIGH,
        estimated_hours_recovered=4.0,
        estimated_delay_reduction_days=1.0,
        estimated_risk_reduction=0.1,
        affected_item_ids=["WI-01"],
        affected_resource_ids=[],
        affected_sprint_ids=["S2", "S1"],
        affected_blocker_ids=[],
        root_cause_signal_id="SIG-ADVANCE",
    )

    simulation_engine.applicator.apply(clone, recommendation)

    updated_item = next(wi for wi in clone.work_items if wi.item_id == "WI-01")
    assert updated_item.assigned_sprint == "Sprint 1"


def test_simulation_engine_supports_scenario_api(simulation_engine):
    recommendation = Recommendation(
        recommendation_id="REC-NEW",
        title="Resolve blocker",
        description="Resolve blocker on the critical path",
        action_type=RecommendationAction.RESOLVE_BLOCKER,
        priority_score=0.95,
        confidence=ConfidenceLevel.HIGH,
        estimated_hours_recovered=10.0,
        estimated_delay_reduction_days=3.0,
        estimated_risk_reduction=0.2,
        affected_item_ids=["WI-02"],
        affected_resource_ids=["R1"],
        affected_sprint_ids=["S1"],
        affected_blocker_ids=["BLK-01"],
        root_cause_signal_id="SIG-001",
    )

    scenario = simulation_engine.simulate(recommendation)
    assert scenario.metadata.selected_recommendations == [recommendation.recommendation_id]
    assert scenario.forecast_comparison.simulated_finish_date <= scenario.forecast_comparison.baseline_finish_date
    assert scenario.summary.simulation_success is True

    comparisons = simulation_engine.compare_scenarios([scenario])
    assert len(comparisons) == 1
    assert comparisons[0].metadata.scenario_id == scenario.metadata.scenario_id


def test_ai_advisor_turns_scenario_results_into_guidance(simulation_engine):
    recommendation = Recommendation(
        recommendation_id="REC-NEW",
        title="Resolve blocker",
        description="Resolve blocker on the critical path",
        action_type=RecommendationAction.RESOLVE_BLOCKER,
        priority_score=0.95,
        confidence=ConfidenceLevel.HIGH,
        estimated_hours_recovered=10.0,
        estimated_delay_reduction_days=3.0,
        estimated_risk_reduction=0.2,
        affected_item_ids=["WI-02"],
        affected_resource_ids=["R1"],
        affected_sprint_ids=["S1"],
        affected_blocker_ids=["BLK-01"],
        root_cause_signal_id="SIG-001",
    )

    advisor = AIAdvisor(simulation_engine)
    advice = advisor.advise(recommendation)

    assert advice.recommendation_ids == [recommendation.recommendation_id]
    assert advice.priority in {"high", "medium", "low"}
    assert advice.summary
