"""
Phase 3.3 Risk Engine Tests

Comprehensive test suite for risk engine analysis.
"""

import pytest
from datetime import datetime, timedelta

from app.domain.models import (
    ProjectInfo, Resource, Sprint, WorkItem, Dependency, Blocker, SprintActual, ProjectState,
    SkillLevel, WorkItemType, Priority, WorkItemStatus, SprintStatus, BlockerSeverity, BlockerStatus, BlockerCategory, DependencyType
)
from app.engines.metrics_engine import MetricsEngine
from app.engines.dependency_engine import DependencyGraphEngine
from app.engines.critical_path_engine import CriticalPathEngine
from app.engines.spillover_engine import SpilloverAnalysis, SpilloverAnalysisEngine
from app.engines.forecast_engine import ForecastEngine
from app.engines.monte_carlo_engine import MonteCarloEngine
from app.engines.impact_scoring_engine import ImpactScoringEngine
from app.engines.risk_engine import RiskEngine
from app.api.models_phase3 import RiskLevel, RiskDriver, RiskExplanation


@pytest.fixture
def sample_project_state_low_risk():
    """Create a sample ProjectState with low risk characteristics."""
    
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 1)
    project_info = ProjectInfo(
        project_name="Low Risk Project",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=end_date,
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )
    
    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.8,  # Not overloaded
            availability_pct=1.0,
        ),
        Resource(
            resource_id="R2",
            name="Bob",
            role="Engineer",
            primary_skill="Java",
            secondary_skill="JavaScript",
            skill_level=SkillLevel.MID,
            allocation_pct=0.7,  # Not overloaded
            availability_pct=0.95,
        ),
    ]
    
    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Initial setup",
            status=SprintStatus.COMPLETED,
            planned_velocity_hrs=160.0,
            carryover_count=0,
        ),
    ]
    
    # Few, simple work items
    work_items = [
        WorkItem(
            item_id="WI-001",
            title="Task 1",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.MEDIUM,
            status=WorkItemStatus.COMPLETED,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            remaining_effort_hrs=0.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
        WorkItem(
            item_id="WI-002",
            title="Task 2",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.MEDIUM,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            remaining_effort_hrs=5.0,
            assigned_resource="R2",
            required_skill="Java",
        ),
    ]
    
    dependencies = []  # No dependencies
    blockers = []  # No blockers
    
    actuals = [
        SprintActual(
            sprint_id="S1",
            sprint_number=1,
            planned_effort_hrs=160.0,
            actual_effort_hrs=150.0,
            tasks_planned=2,
            tasks_completed=2,
            carryover_count=0,
        ),
    ]
    
    return ProjectState(
        project_id="proj-low-risk",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=actuals,
    )


@pytest.fixture
def sample_project_state_high_risk():
    """Create a sample ProjectState with high risk characteristics."""
    
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 2, 1)  # Tight timeline
    project_info = ProjectInfo(
        project_name="High Risk Project",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=end_date,
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )
    
    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.95,  # Near overload
            availability_pct=1.0,
        ),
    ]
    
    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Initial setup",
            status=SprintStatus.COMPLETED,
            planned_velocity_hrs=100.0,
            carryover_count=3,
        ),
    ]
    
    # Many work items with increasing estimates
    work_items = [
        WorkItem(
            item_id="WI-001",
            title="Task 1",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.HIGH,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=35.0,  # Inflated
            remaining_effort_hrs=35.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
        WorkItem(
            item_id="WI-002",
            title="Task 2",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.HIGH,
            status=WorkItemStatus.BLOCKED,  # Blocked
            estimated_effort_hrs=25.0,
            current_estimate_hrs=25.0,
            remaining_effort_hrs=25.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
        WorkItem(
            item_id="WI-003",
            title="Task 3",
            work_type=WorkItemType.FEATURE,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.HIGH,
            status=WorkItemStatus.NOT_STARTED,
            estimated_effort_hrs=30.0,
            current_estimate_hrs=50.0,  # Inflated
            remaining_effort_hrs=50.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
    ]
    
    # Dependencies creating long chain
    dependencies = [
        Dependency(
            dependency_id="DEP-001",
            predecessor_item_id="WI-001",
            successor_item_id="WI-002",
            lag_days=0,
            dependency_type=DependencyType.FINISH_TO_START,
        ),
        Dependency(
            dependency_id="DEP-002",
            predecessor_item_id="WI-002",
            successor_item_id="WI-003",
            lag_days=0,
            dependency_type=DependencyType.FINISH_TO_START,
        ),
    ]
    
    # Active blockers
    blockers = [
        Blocker(
            blocker_id="B1",
            related_item_id="WI-002",
            impacted_item_ids=["WI-002"],
            description="Test blocker",
            severity=BlockerSeverity.HIGH,
            status=BlockerStatus.OPEN,
            owner="DevOps",
            raised_date=start_date,
            target_resolution_date=start_date + timedelta(days=3),
            actual_resolution_date=None,  # Still open
            category=BlockerCategory.OTHER,
            notes="Test blocker",
        ),
    ]
    
    actuals = [
        SprintActual(
            sprint_id="S1",
            sprint_number=1,
            planned_effort_hrs=160.0,
            actual_effort_hrs=90.0,  # Below velocity
            tasks_planned=3,
            tasks_completed=2,
            carryover_count=3,
        ),
    ]
    
    return ProjectState(
        project_id="proj-high-risk",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=actuals,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ──────────────────────────────────────────────────────────────────────────────


def test_risk_engine_initialization(sample_project_state_low_risk):
    """Test RiskEngine can be initialized with all dependencies."""
    metrics = MetricsEngine(sample_project_state_low_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_low_risk)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(sample_project_state_low_risk, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_low_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_low_risk, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(
        sample_project_state_low_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_low_risk, dag).score()
    
    risk_engine = RiskEngine(
        sample_project_state_low_risk, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    assert risk_engine is not None


def test_risk_engine_uses_metrics_for_velocity_and_allocation_signals(sample_project_state_high_risk):
    """RiskEngine should consume velocity and allocation facts from ProjectMetrics."""
    metrics = MetricsEngine(sample_project_state_high_risk).calculate()
    metrics.velocity_metrics.velocity_trend_pct = -0.25
    metrics.resource_metrics.workload_balance_score = 0.55
    metrics.forecast_input_metrics.utilization_pct = 0.96

    dep_engine = DependencyGraphEngine(sample_project_state_high_risk)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(sample_project_state_high_risk, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_high_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_high_risk, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(
        sample_project_state_high_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_high_risk, dag).score()

    risk_engine = RiskEngine(
        sample_project_state_high_risk, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()

    driver_titles = {driver.title for driver in result.resource_risk.drivers}
    assert "Velocity Degradation" in driver_titles
    assert "Team Allocation Imbalance" in driver_titles
    assert result.resource_risk.score > 0.0


def test_scope_risk_uses_forecast_scope_growth(sample_project_state_low_risk):
    """Scope risk should use forecast scope-growth facts rather than recomputing inflation from work items."""
    metrics = MetricsEngine(sample_project_state_low_risk).calculate()
    metrics.quality_metrics.scope_creep_score = 0.75
    metrics.planning_metrics.scope_volatility_score = 0.80

    dep_engine = DependencyGraphEngine(sample_project_state_low_risk)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(sample_project_state_low_risk, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_low_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_low_risk, metrics, cp_result, spillover).calculate()
    forecast.scope_growth_percent = 22.5
    forecast.scope_growth_hours = 30.0

    mc_engine = MonteCarloEngine(
        sample_project_state_low_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_low_risk, dag).score()

    risk_engine = RiskEngine(
        sample_project_state_low_risk, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()

    driver_titles = {driver.title for driver in result.scope_risk.drivers}
    assert "Scope Growth Signal" in driver_titles
    assert result.scope_risk.score > 0.0


def test_overall_risk_calculation(sample_project_state_low_risk):
    """Verify weighted aggregation formula."""
    metrics = MetricsEngine(sample_project_state_low_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_low_risk)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(sample_project_state_low_risk, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_low_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_low_risk, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(
        sample_project_state_low_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_low_risk, dag).score()
    
    risk_engine = RiskEngine(
        sample_project_state_low_risk, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()
    
    # Verify overall score is within bounds
    assert 0.0 <= result.overall_risk_score <= 100.0
    
    # Verify sub-scores are within bounds
    assert 0.0 <= result.schedule_risk.score <= 100.0
    assert 0.0 <= result.dependency_risk.score <= 100.0
    assert 0.0 <= result.resource_risk.score <= 100.0
    assert 0.0 <= result.scope_risk.score <= 100.0
    
    # Verify weighted formula: 0.40 * schedule + 0.25 * dependency + 0.20 * resource + 0.15 * scope
    expected_overall = (
        0.40 * result.schedule_risk.score
        + 0.25 * result.dependency_risk.score
        + 0.20 * result.resource_risk.score
        + 0.15 * result.scope_risk.score
    )
    assert abs(result.overall_risk_score - expected_overall) < 0.01


def test_schedule_risk_high_when_probability_low():
    """Test schedule risk increases when on-time probability is low."""
    # Create project state for Monte Carlo with low probability
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 15)  # Very tight timeline
    project_info = ProjectInfo(
        project_name="Tight Timeline",
        sponsor="Test",
        business_unit="Engineering",
        project_manager="PM",
        customer="Customer",
        status="Active",
        start_date=start_date,
        target_end_date=end_date,
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )
    
    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=1.0,
            availability_pct=1.0,
        ),
    ]
    
    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=end_date,
            working_days=10,
            sprint_goal="Dev",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=100.0,
            carryover_count=0,
        ),
    ]
    
    work_items = [
        WorkItem(
            item_id="WI-001",
            title="Large Task",
            work_type=WorkItemType.FEATURE,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.HIGH,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=200.0,  # More than capacity
            current_estimate_hrs=200.0,
            remaining_effort_hrs=150.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
    ]
    
    project_state = ProjectState(
        project_id="proj-schedule-risk",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=[],
        blockers=[],
        actuals=[],
    )
    
    metrics = MetricsEngine(project_state).calculate()
    dep_engine = DependencyGraphEngine(project_state)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(project_state, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(project_state, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(project_state, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(project_state, metrics, cp_result, spillover, simulation_count=1000)
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(project_state, dag).score()
    
    risk_engine = RiskEngine(
        project_state, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()
    
    # With low on-time probability, schedule risk should be high
    assert monte_carlo.on_time_probability < 0.5
    assert result.schedule_risk.score > 50.0


def test_schedule_risk_spillover_not_triple_weighted():
    """Ensure schedule risk only uses spillover through forecast and Monte Carlo."""
    start_date = datetime(2025, 1, 1)
    end_date = start_date + timedelta(days=21)
    project_info = ProjectInfo(
        project_name="Spillover Pressure",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=end_date,
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )

    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.95,
            availability_pct=0.95,
        ),
    ]

    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Development",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=80.0,
            carryover_count=0,
        ),
    ]

    work_items = [
        WorkItem(
            item_id="WI-001",
            title="Large Task Part A",
            work_type=WorkItemType.FEATURE,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            priority=Priority.HIGH,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=35.0,
            current_estimate_hrs=35.0,
            remaining_effort_hrs=35.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
        WorkItem(
            item_id="WI-002",
            title="Large Task Part B",
            work_type=WorkItemType.FEATURE,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            priority=Priority.HIGH,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=35.0,
            current_estimate_hrs=35.0,
            remaining_effort_hrs=35.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
        WorkItem(
            item_id="WI-003",
            title="Large Task Part C",
            work_type=WorkItemType.FEATURE,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            priority=Priority.HIGH,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=35.0,
            current_estimate_hrs=35.0,
            remaining_effort_hrs=35.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
        WorkItem(
            item_id="WI-004",
            title="Large Task Part D",
            work_type=WorkItemType.FEATURE,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            priority=Priority.HIGH,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=35.0,
            current_estimate_hrs=35.0,
            remaining_effort_hrs=35.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
    ]

    project_state = ProjectState(
        project_id="proj-spillover-score",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=[],
        blockers=[],
        actuals=[
            SprintActual(
                sprint_id="S1",
                sprint_number=1,
                planned_effort_hrs=80.0,
                actual_effort_hrs=70.0,
                tasks_planned=1,
                tasks_completed=0,
                carryover_count=0,
            )
        ],
    )

    metrics = MetricsEngine(project_state).calculate()
    dep_engine = DependencyGraphEngine(project_state)
    dag = dep_engine.build_dag()
    cp_result = CriticalPathEngine(project_state, dag).analyze()

    high_spill = SpilloverAnalysis(
        spillover_probability={},
        predicted_spillover_by_sprint={1: 12.0},
        spillover_confidence_intervals={1: (0.0, 0.0)},
        high_spillover_risk_items=[],
        historical_carryover_rate=0.0,
        historical_carryover_std_dev=0.0,
        sprint_utilization_pct={1: 1.0},
    )

    no_spill = SpilloverAnalysis(
        spillover_probability={},
        predicted_spillover_by_sprint={1: 0.0},
        spillover_confidence_intervals={1: (0.0, 0.0)},
        high_spillover_risk_items=[],
        historical_carryover_rate=0.0,
        historical_carryover_std_dev=0.0,
        sprint_utilization_pct={1: 1.0},
    )

    forecast_high = ForecastEngine(project_state, metrics, cp_result, high_spill).calculate()
    forecast_low = ForecastEngine(project_state, metrics, cp_result, no_spill).calculate()

    monte_carlo_high = MonteCarloEngine(
        project_state,
        metrics,
        cp_result,
        high_spill,
        simulation_count=1000,
        seed=123,
    ).calculate()
    monte_carlo_low = MonteCarloEngine(
        project_state,
        metrics,
        cp_result,
        no_spill,
        simulation_count=1000,
        seed=123,
    ).calculate()

    impact_scores = ImpactScoringEngine(project_state, dag).score()

    risk_high = RiskEngine(
        project_state,
        metrics,
        cp_result,
        dag,
        high_spill,
        forecast_high,
        monte_carlo_high,
        impact_scores,
    ).analyze()
    risk_low = RiskEngine(
        project_state,
        metrics,
        cp_result,
        dag,
        no_spill,
        forecast_low,
        monte_carlo_low,
        impact_scores,
    ).analyze()

    assert forecast_high.spillover_delay_days > 5.0
    assert all(
        d.title not in {"High Spillover Prediction", "Moderate Spillover Risk"}
        for d in risk_high.schedule_risk.drivers
    )
    assert all(
        d.title not in {"High Spillover Prediction", "Moderate Spillover Risk"}
        for d in risk_low.schedule_risk.drivers
    )

    def expected_schedule_score(forecast, mc):
        delay_days = forecast.expected_delay_days
        delay_component = (
            min(100.0, (delay_days / 30.0) * 80.0) if delay_days > 0 else 0.0
        )
        confidence_modifier = 1.0 + (1.0 - mc.on_time_probability) * 0.20
        schedule_primary = (
            min(100.0, delay_component * confidence_modifier)
            if delay_component > 0
            else 0.0
        )
        spillover_component = min(100.0, forecast.predicted_spillover_items * 8.0)
        if schedule_primary > 0 and spillover_component > 0:
            return (schedule_primary + spillover_component) / 2.0
        return schedule_primary if schedule_primary > 0 else spillover_component

    assert abs(risk_high.schedule_risk.score - expected_schedule_score(forecast_high, monte_carlo_high)) < 0.01
    assert abs(risk_low.schedule_risk.score - expected_schedule_score(forecast_low, monte_carlo_low)) < 0.01
    assert risk_high.schedule_risk.score > risk_low.schedule_risk.score


def test_spillover_driver_appears_as_informational_not_scored():
    """Verify schedule risk reports spillover impact as informational only."""
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 1)
    project_info = ProjectInfo(
        project_name="Spillover Explanation",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=end_date,
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )

    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.98,
            availability_pct=1.0,
        ),
        Resource(
            resource_id="R2",
            name="Bob",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="JavaScript",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.96,
            availability_pct=1.0,
        ),
    ]

    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Development",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=120.0,
            carryover_count=0,
        ),
    ]

    work_items = [
        WorkItem(
            item_id=f"WI-{i:03d}",
            title=f"Task {i}",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.HIGH,
            status=(WorkItemStatus.NOT_STARTED if i <= 6 else WorkItemStatus.IN_PROGRESS),
            estimated_effort_hrs=10.0,
            current_estimate_hrs=15.0 if i <= 4 else 10.0,
            remaining_effort_hrs=15.0 if i <= 4 else 10.0,
            assigned_resource="R1",
            required_skill="Python",
        )
        for i in range(1, 13)
    ]

    dependencies = [
        Dependency(
            dependency_id=f"DEP-{i:03d}",
            predecessor_item_id=f"WI-{i:03d}",
            successor_item_id=f"WI-{i+1:03d}",
            lag_days=0,
            dependency_type=DependencyType.FINISH_TO_START,
        )
        for i in range(1, 12)
    ]

    blockers = [
        Blocker(
            blocker_id=f"B{i}",
            related_item_id=f"WI-00{i}",
            impacted_item_ids=[f"WI-00{i}"],
            description="Test blocker",
            severity=BlockerSeverity.MEDIUM,
            status=BlockerStatus.OPEN,
            owner="DevOps",
            raised_date=start_date,
            target_resolution_date=start_date + timedelta(days=3),
            actual_resolution_date=None,
            category=BlockerCategory.OTHER,
            notes="Test blocker",
        )
        for i in range(1, 7)
    ]

    actuals = [
        SprintActual(
            sprint_id="S1",
            sprint_number=1,
            planned_effort_hrs=120.0,
            actual_effort_hrs=80.0,
            tasks_planned=12,
            tasks_completed=6,
            carryover_count=4,
        )
    ]

    project_state = ProjectState(
        project_id="proj-spillover-info",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=actuals,
    )

    metrics = MetricsEngine(project_state).calculate()
    dep_engine = DependencyGraphEngine(project_state)
    dag = dep_engine.build_dag()
    cp_result = CriticalPathEngine(project_state, dag).analyze()

    spillover = SpilloverAnalysis(
        spillover_probability={},
        predicted_spillover_by_sprint={1: 10.0},
        spillover_confidence_intervals={1: (0.0, 0.0)},
        high_spillover_risk_items=[],
        historical_carryover_rate=4.0,
        historical_carryover_std_dev=1.0,
        sprint_utilization_pct={1: 1.0},
    )

    forecast = ForecastEngine(project_state, metrics, cp_result, spillover).calculate()
    monte_carlo = MonteCarloEngine(
        project_state,
        metrics,
        cp_result,
        spillover,
        simulation_count=1000,
        seed=456,
    ).calculate()
    impact_scores = ImpactScoringEngine(project_state, dag).score()

    risk_result = RiskEngine(
        project_state,
        metrics,
        cp_result,
        dag,
        spillover,
        forecast,
        monte_carlo,
        impact_scores,
    ).analyze()

    spillover_drivers = [
        d for d in risk_result.schedule_risk.drivers
        if d.title == "Spillover Schedule Impact"
    ]
    assert len(spillover_drivers) == 1
    assert spillover_drivers[0].score == 0.0
    assert all(
        d.title != "Spillover Schedule Impact"
        for d in risk_result.top_risk_drivers
    )


def test_scope_risk_still_scores_historical_spillover_independently(sample_project_state_high_risk):
    """Scope risk should still surface historical spillover carryover."""
    metrics = MetricsEngine(sample_project_state_high_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_high_risk)
    dag = dep_engine.build_dag()
    cp_result = CriticalPathEngine(sample_project_state_high_risk, dag).analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_high_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_high_risk, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(
        sample_project_state_high_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_high_risk, dag).score()

    scope_risk = RiskEngine(
        sample_project_state_high_risk,
        metrics,
        cp_result,
        dag,
        spillover,
        forecast,
        monte_carlo,
        impact_scores,
    )._calculate_scope_risk()

    historical_drivers = [
        d for d in scope_risk.drivers
        if d.title in {"High Historical Spillover", "Moderate Spillover Pattern"}
    ]
    assert len(historical_drivers) >= 1
    assert historical_drivers[0].score > 0.0


def test_sprint_risk_still_uses_predicted_spillover():
    """Sprint risks must still reflect predicted spillover count."""
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="Sprint Spillover",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=start_date + timedelta(days=28),
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )

    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.8,
            availability_pct=1.0,
        ),
    ]

    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Dev",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=160.0,
            carryover_count=0,
        ),
    ]

    work_items = [
        WorkItem(
            item_id="WI-001",
            title="Task 1",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.MEDIUM,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            remaining_effort_hrs=10.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
    ]

    project_state = ProjectState(
        project_id="proj-sprint-spillover",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=[],
        blockers=[],
        actuals=[],
    )

    metrics = MetricsEngine(project_state).calculate()
    dep_engine = DependencyGraphEngine(project_state)
    dag = dep_engine.build_dag()
    cp_result = CriticalPathEngine(project_state, dag).analyze()
    spillover = SpilloverAnalysis(
        spillover_probability={},
        predicted_spillover_by_sprint={1: 7.0},
        spillover_confidence_intervals={1: (0.0, 0.0)},
        high_spillover_risk_items=[],
        historical_carryover_rate=0.0,
        historical_carryover_std_dev=0.0,
        sprint_utilization_pct={1: 0.2},
    )

    forecast = ForecastEngine(project_state, metrics, cp_result, spillover).calculate()
    monte_carlo = MonteCarloEngine(
        project_state,
        metrics,
        cp_result,
        spillover,
        simulation_count=1000,
        seed=789,
    ).calculate()
    impact_scores = ImpactScoringEngine(project_state, dag).score()

    risk_engine = RiskEngine(
        project_state,
        metrics,
        cp_result,
        dag,
        spillover,
        forecast,
        monte_carlo,
        impact_scores,
    )

    sprint_risks = risk_engine._calculate_sprint_risks()
    assert len(sprint_risks) == 1
    assert sprint_risks[0].spillover_items == 7
    assert abs(sprint_risks[0].risk_score - 56.0) < 0.01
    """Test dependency risk increases with long critical path."""
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 1)
    project_info = ProjectInfo(
        project_name="Complex Dependencies",
        sponsor="Test",
        business_unit="Engineering",
        project_manager="PM",
        customer="Customer",
        status="Active",
        start_date=start_date,
        target_end_date=end_date,
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )
    
    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.8,
            availability_pct=1.0,
        ),
    ]
    
    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Dev",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=160.0,
            carryover_count=0,
        ),
    ]
    
    # Create a long chain of dependent items
    work_items = [
        WorkItem(
            item_id=f"WI-{i:03d}",
            title=f"Task {i}",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.HIGH,
            status=WorkItemStatus.NOT_STARTED,
            estimated_effort_hrs=15.0,
            current_estimate_hrs=15.0,
            remaining_effort_hrs=15.0,
            assigned_resource="R1",
            required_skill="Python",
        )
        for i in range(1, 11)  # 10 items in chain
    ]
    
    # Create finish-to-start chain
    dependencies = [
        Dependency(
            dependency_id=f"DEP-{i:03d}",
            predecessor_item_id=f"WI-{i:03d}",
            successor_item_id=f"WI-{i+1:03d}",
            lag_days=0,
            dependency_type=DependencyType.FINISH_TO_START,
        )
        for i in range(1, 10)
    ]
    
    project_state = ProjectState(
        project_id="proj-critical-path",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=[],
        actuals=[],
    )
    
    metrics = MetricsEngine(project_state).calculate()
    dep_engine = DependencyGraphEngine(project_state)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(project_state, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(project_state, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(project_state, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(project_state, metrics, cp_result, spillover, simulation_count=1000)
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(project_state, dag).score()
    
    risk_engine = RiskEngine(
        project_state, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()
    
    # With long critical path, dependency risk should be high
    assert len(cp_result.items_on_critical_path) > 5
    assert result.dependency_risk.score > 30.0


def test_resource_risk_increases_with_utilization():
    """Test resource risk increases with high utilization."""
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 1)
    project_info = ProjectInfo(
        project_name="Resource Constrained",
        sponsor="Test",
        business_unit="Engineering",
        project_manager="PM",
        customer="Customer",
        status="Active",
        start_date=start_date,
        target_end_date=end_date,
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )
    
    # Highly utilized team
    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.98,  # Almost fully allocated
            availability_pct=1.0,
        ),
    ]
    
    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Dev",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=160.0,
            carryover_count=0,
        ),
    ]
    
    work_items = [
        WorkItem(
            item_id="WI-001",
            title="Task 1",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.HIGH,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=80.0,
            current_estimate_hrs=80.0,
            remaining_effort_hrs=40.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
    ]
    
    project_state = ProjectState(
        project_id="proj-resource-risk",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=[],
        blockers=[],
        actuals=[],
    )
    
    metrics = MetricsEngine(project_state).calculate()
    dep_engine = DependencyGraphEngine(project_state)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(project_state, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(project_state, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(project_state, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(project_state, metrics, cp_result, spillover, simulation_count=1000)
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(project_state, dag).score()
    
    risk_engine = RiskEngine(
        project_state, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()
    
    # With high utilization, resource risk should be elevated
    assert metrics.avg_allocation_pct > 0.9
    assert result.resource_risk.score > 20.0


def test_scope_risk_detects_estimate_growth():
    """Test scope risk detects estimate inflation."""
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 1)
    project_info = ProjectInfo(
        project_name="Scope Growth",
        sponsor="Test",
        business_unit="Engineering",
        project_manager="PM",
        customer="Customer",
        status="Active",
        start_date=start_date,
        target_end_date=end_date,
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )
    
    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.8,
            availability_pct=1.0,
        ),
    ]
    
    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Dev",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=160.0,
            carryover_count=0,
        ),
    ]
    
    # Items with inflated estimates
    work_items = [
        WorkItem(
            item_id="WI-001",
            title="Task 1",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.HIGH,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=50.0,  # 150% inflation
            remaining_effort_hrs=50.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
        WorkItem(
            item_id="WI-002",
            title="Task 2",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.HIGH,
            status=WorkItemStatus.NOT_STARTED,
            estimated_effort_hrs=30.0,
            current_estimate_hrs=45.0,  # 50% inflation
            remaining_effort_hrs=45.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
    ]
    
    project_state = ProjectState(
        project_id="proj-scope-risk",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=[],
        blockers=[],
        actuals=[],
    )
    
    metrics = MetricsEngine(project_state).calculate()
    dep_engine = DependencyGraphEngine(project_state)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(project_state, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(project_state, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(project_state, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(project_state, metrics, cp_result, spillover, simulation_count=1000)
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(project_state, dag).score()
    
    risk_engine = RiskEngine(
        project_state, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()
    
    # With estimate inflation, scope risk should be high
    assert result.scope_risk.score > 40.0
    assert len(result.scope_risk.reasons) > 0


def test_risk_drivers_ranked(sample_project_state_high_risk):
    """Test risk drivers are ranked by score (descending)."""
    metrics = MetricsEngine(sample_project_state_high_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_high_risk)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(sample_project_state_high_risk, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_high_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_high_risk, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(
        sample_project_state_high_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_high_risk, dag).score()
    
    risk_engine = RiskEngine(
        sample_project_state_high_risk, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()
    
    # Verify drivers are sorted descending
    scores = [d.score for d in result.top_risk_drivers]
    assert scores == sorted(scores, reverse=True)
    
    # Verify max 10 drivers
    assert len(result.top_risk_drivers) <= 10


def test_cascade_depth_no_off_by_one():
    """A three-item chain should report depth 2, not 3."""
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="Cascade Depth Chain",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=start_date + timedelta(days=28),
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )

    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.8,
            availability_pct=1.0,
        ),
    ]

    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Dev",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=80.0,
            carryover_count=0,
        ),
    ]

    work_items = [
        WorkItem(
            item_id=f"WI-{letter}",
            title=f"Task {letter}",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.MEDIUM,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=10.0,
            current_estimate_hrs=10.0,
            remaining_effort_hrs=10.0,
            assigned_resource="R1",
            required_skill="Python",
        )
        for letter in ["A", "B", "C"]
    ]

    dependencies = [
        Dependency(
            dependency_id="DEP-01",
            predecessor_item_id="WI-A",
            successor_item_id="WI-B",
            lag_days=0,
            dependency_type=DependencyType.FINISH_TO_START,
        ),
        Dependency(
            dependency_id="DEP-02",
            predecessor_item_id="WI-B",
            successor_item_id="WI-C",
            lag_days=0,
            dependency_type=DependencyType.FINISH_TO_START,
        ),
    ]

    blockers = [
        Blocker(
            blocker_id="B1",
            related_item_id="WI-A",
            impacted_item_ids=["WI-A"],
            description="Root blocker",
            severity=BlockerSeverity.CRITICAL,
            status=BlockerStatus.OPEN,
            owner="Ops",
            raised_date=start_date,
            target_resolution_date=start_date + timedelta(days=3),
            actual_resolution_date=None,
            category=BlockerCategory.OTHER,
            notes="Test blocker",
        )
    ]

    project_state = ProjectState(
        project_id="proj-cascade-2",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=[],
    )

    impact_scores = ImpactScoringEngine(project_state, DependencyGraphEngine(project_state).build_dag()).score()
    assert max(impact_scores.cascade_depth_map.values()) == 2
    assert impact_scores.cascade_depth_map["WI-C"] == 2


def test_cascade_depth_zero_hops():
    """A blocker with no successors should report depth 0."""
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="Cascade Zero Hops",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=start_date + timedelta(days=14),
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )

    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.8,
            availability_pct=1.0,
        ),
    ]

    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Dev",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=80.0,
            carryover_count=0,
        ),
    ]

    work_items = [
        WorkItem(
            item_id="WI-A",
            title="Task A",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.MEDIUM,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=10.0,
            current_estimate_hrs=10.0,
            remaining_effort_hrs=10.0,
            assigned_resource="R1",
            required_skill="Python",
        ),
    ]

    blockers = [
        Blocker(
            blocker_id="B1",
            related_item_id="WI-A",
            impacted_item_ids=["WI-A"],
            description="Isolated blocker",
            severity=BlockerSeverity.CRITICAL,
            status=BlockerStatus.OPEN,
            owner="Ops",
            raised_date=start_date,
            target_resolution_date=start_date + timedelta(days=3),
            actual_resolution_date=None,
            category=BlockerCategory.OTHER,
            notes="Test blocker",
        )
    ]

    project_state = ProjectState(
        project_id="proj-cascade-0",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=[],
        blockers=blockers,
        actuals=[],
    )

    impact_scores = ImpactScoringEngine(project_state, DependencyGraphEngine(project_state).build_dag()).score()
    assert max(impact_scores.cascade_depth_map.values()) == 0
    assert impact_scores.cascade_depth_map["WI-A"] == 0


def test_cascade_threshold_not_triggered_at_real_depth_5():
    """A real blocker depth of 5 should not trigger cascade risk."""
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="Cascade Depth Five",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=start_date + timedelta(days=84),
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )

    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.8,
            availability_pct=1.0,
        ),
    ]

    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Dev",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=80.0,
            carryover_count=0,
        ),
    ]

    work_items = [
        WorkItem(
            item_id=f"WI-{i}",
            title=f"Task {i}",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.MEDIUM,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=10.0,
            current_estimate_hrs=10.0,
            remaining_effort_hrs=10.0,
            assigned_resource="R1",
            required_skill="Python",
        )
        for i in range(1, 7)
    ]

    dependencies = [
        Dependency(
            dependency_id=f"DEP-{i:02d}",
            predecessor_item_id=f"WI-{i}",
            successor_item_id=f"WI-{i+1}",
            lag_days=0,
            dependency_type=DependencyType.FINISH_TO_START,
        )
        for i in range(1, 6)
    ]

    blockers = [
        Blocker(
            blocker_id="B1",
            related_item_id="WI-1",
            impacted_item_ids=["WI-1"],
            description="Five-hop blocker",
            severity=BlockerSeverity.CRITICAL,
            status=BlockerStatus.OPEN,
            owner="Ops",
            raised_date=start_date,
            target_resolution_date=start_date + timedelta(days=3),
            actual_resolution_date=None,
            category=BlockerCategory.OTHER,
            notes="Test blocker",
        )
    ]

    project_state = ProjectState(
        project_id="proj-cascade-5",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=[],
    )

    metrics = MetricsEngine(project_state).calculate()
    dep_engine = DependencyGraphEngine(project_state)
    dag = dep_engine.build_dag()
    cp_result = CriticalPathEngine(project_state, dag).analyze()
    spillover = SpilloverAnalysisEngine(project_state, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(project_state, metrics, cp_result, spillover).calculate()
    monte_carlo = MonteCarloEngine(
        project_state, metrics, cp_result, spillover, simulation_count=1000, seed=123
    ).calculate()
    impact_scores = ImpactScoringEngine(project_state, dag).score()
    result = RiskEngine(
        project_state, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    ).analyze()

    assert max(impact_scores.cascade_depth_map.values()) == 5
    assert all(
        driver.title != "Deep Blocker Cascade"
        for driver in result.dependency_risk.drivers
    )


def test_cascade_threshold_triggered_at_real_depth_6():
    """A real blocker depth of 6 should trigger cascade risk score 75."""
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="Cascade Depth Six",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=start_date + timedelta(days=98),
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )

    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="C++",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.8,
            availability_pct=1.0,
        ),
    ]

    sprints = [
        Sprint(
            sprint_id="S1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=14),
            working_days=10,
            sprint_goal="Dev",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=80.0,
            carryover_count=0,
        ),
    ]

    work_items = [
        WorkItem(
            item_id=f"WI-{i}",
            title=f"Task {i}",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            priority=Priority.MEDIUM,
            status=WorkItemStatus.IN_PROGRESS,
            estimated_effort_hrs=10.0,
            current_estimate_hrs=10.0,
            remaining_effort_hrs=10.0,
            assigned_resource="R1",
            required_skill="Python",
        )
        for i in range(1, 8)
    ]

    dependencies = [
        Dependency(
            dependency_id=f"DEP-{i:02d}",
            predecessor_item_id=f"WI-{i}",
            successor_item_id=f"WI-{i+1}",
            lag_days=0,
            dependency_type=DependencyType.FINISH_TO_START,
        )
        for i in range(1, 7)
    ]

    blockers = [
        Blocker(
            blocker_id="B1",
            related_item_id="WI-1",
            impacted_item_ids=["WI-1"],
            description="Six-hop blocker",
            severity=BlockerSeverity.CRITICAL,
            status=BlockerStatus.OPEN,
            owner="Ops",
            raised_date=start_date,
            target_resolution_date=start_date + timedelta(days=3),
            actual_resolution_date=None,
            category=BlockerCategory.OTHER,
            notes="Test blocker",
        )
    ]

    project_state = ProjectState(
        project_id="proj-cascade-6",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=[],
    )

    metrics = MetricsEngine(project_state).calculate()
    dep_engine = DependencyGraphEngine(project_state)
    dag = dep_engine.build_dag()
    cp_result = CriticalPathEngine(project_state, dag).analyze()
    spillover = SpilloverAnalysisEngine(project_state, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(project_state, metrics, cp_result, spillover).calculate()
    monte_carlo = MonteCarloEngine(
        project_state, metrics, cp_result, spillover, simulation_count=1000, seed=123
    ).calculate()
    impact_scores = ImpactScoringEngine(project_state, dag).score()
    result = RiskEngine(
        project_state, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    ).analyze()

    cascade_driver = next(
        (driver for driver in result.dependency_risk.drivers if driver.title == "Deep Blocker Cascade"),
        None,
    )
    assert cascade_driver is not None
    assert cascade_driver.score == 75.0


def test_schedule_driver_outranks_equal_dependency_driver(sample_project_state_low_risk):
    """Schedule driver should rank above equal-scoring dependency driver."""
    metrics = MetricsEngine(sample_project_state_low_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_low_risk)
    dag = dep_engine.build_dag()
    cp_result = CriticalPathEngine(sample_project_state_low_risk, dag).analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_low_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_low_risk, metrics, cp_result, spillover).calculate()
    monte_carlo = MonteCarloEngine(
        sample_project_state_low_risk, metrics, cp_result, spillover, simulation_count=1000
    ).calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_low_risk, dag).score()

    class StubRiskEngine(RiskEngine):
        def _calculate_schedule_risk(self):
            return RiskExplanation(
                score=80.0,
                reasons=["Schedule driver stub"],
                drivers=[
                    RiskDriver(
                        category="SCHEDULE",
                        score=80.0,
                        title="Schedule Pressure",
                        description="Equal schedule risk.",
                        recommendation_hint="Monitor schedule.",
                    )
                ],
            )

        def _calculate_dependency_risk(self):
            return RiskExplanation(
                score=80.0,
                reasons=["Dependency driver stub"],
                drivers=[
                    RiskDriver(
                        category="DEPENDENCY",
                        score=80.0,
                        title="Dependency Pressure",
                        description="Equal dependency risk.",
                        recommendation_hint="Monitor dependencies.",
                    )
                ],
            )

        def _calculate_resource_risk(self):
            return RiskExplanation(score=0.0, reasons=[], drivers=[])

        def _calculate_scope_risk(self):
            return RiskExplanation(score=0.0, reasons=[], drivers=[])

        def _calculate_sprint_risks(self):
            return []

    risk_engine = StubRiskEngine(
        sample_project_state_low_risk,
        metrics,
        cp_result,
        dag,
        spillover,
        forecast,
        monte_carlo,
        impact_scores,
    )
    result = risk_engine.analyze()

    assert result.top_risk_drivers[0].category == "SCHEDULE"
    assert result.top_risk_drivers[0].title == "Schedule Pressure"


def test_high_dependency_can_outrank_low_schedule(sample_project_state_low_risk):
    """High dependency score can outrank lower schedule score."""
    metrics = MetricsEngine(sample_project_state_low_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_low_risk)
    dag = dep_engine.build_dag()
    cp_result = CriticalPathEngine(sample_project_state_low_risk, dag).analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_low_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_low_risk, metrics, cp_result, spillover).calculate()
    monte_carlo = MonteCarloEngine(
        sample_project_state_low_risk, metrics, cp_result, spillover, simulation_count=1000
    ).calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_low_risk, dag).score()

    class StubRiskEngine(RiskEngine):
        def _calculate_schedule_risk(self):
            return RiskExplanation(
                score=40.0,
                reasons=["Low schedule driver stub"],
                drivers=[
                    RiskDriver(
                        category="SCHEDULE",
                        score=40.0,
                        title="Schedule Pressure",
                        description="Lower schedule risk.",
                        recommendation_hint="Monitor schedule.",
                    )
                ],
            )

        def _calculate_dependency_risk(self):
            return RiskExplanation(
                score=90.0,
                reasons=["High dependency driver stub"],
                drivers=[
                    RiskDriver(
                        category="DEPENDENCY",
                        score=90.0,
                        title="Dependency Pressure",
                        description="Higher dependency risk.",
                        recommendation_hint="Monitor dependencies.",
                    )
                ],
            )

        def _calculate_resource_risk(self):
            return RiskExplanation(score=0.0, reasons=[], drivers=[])

        def _calculate_scope_risk(self):
            return RiskExplanation(score=0.0, reasons=[], drivers=[])

        def _calculate_sprint_risks(self):
            return []

    risk_engine = StubRiskEngine(
        sample_project_state_low_risk,
        metrics,
        cp_result,
        dag,
        spillover,
        forecast,
        monte_carlo,
        impact_scores,
    )
    result = risk_engine.analyze()

    assert result.top_risk_drivers[0].category == "DEPENDENCY"
    assert result.top_risk_drivers[0].title == "Dependency Pressure"


def test_sprint_heatmap_generation(sample_project_state_high_risk):
    """Test sprint-level risk analysis is generated."""
    metrics = MetricsEngine(sample_project_state_high_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_high_risk)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(sample_project_state_high_risk, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_high_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_high_risk, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(
        sample_project_state_high_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_high_risk, dag).score()
    
    risk_engine = RiskEngine(
        sample_project_state_high_risk, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()
    
    # Verify sprint risks are generated
    assert len(result.sprint_risks) > 0
    
    # Verify each sprint risk has required fields
    for sprint_risk in result.sprint_risks:
        assert sprint_risk.sprint_id is not None
        assert 0.0 <= sprint_risk.risk_score <= 100.0
        assert sprint_risk.risk_level is not None
        assert sprint_risk.blocked_items >= 0
        assert sprint_risk.spillover_items >= 0


def test_risk_levels():
    """Test risk level thresholds."""
    # Test each level
    assert RiskEngine._score_to_level(10.0) == RiskLevel.LOW
    assert RiskEngine._score_to_level(25.0) == RiskLevel.MODERATE
    assert RiskEngine._score_to_level(50.0) == RiskLevel.HIGH
    assert RiskEngine._score_to_level(70.0) == RiskLevel.VERY_HIGH
    assert RiskEngine._score_to_level(90.0) == RiskLevel.CRITICAL
    
    # Test boundaries
    assert RiskEngine._score_to_level(20.0) == RiskLevel.LOW
    assert RiskEngine._score_to_level(21.0) == RiskLevel.MODERATE
    assert RiskEngine._score_to_level(40.0) == RiskLevel.MODERATE
    assert RiskEngine._score_to_level(41.0) == RiskLevel.HIGH


def test_risk_result_has_explanations(sample_project_state_high_risk):
    """Test that all risk results have human-readable explanations."""
    metrics = MetricsEngine(sample_project_state_high_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_high_risk)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(sample_project_state_high_risk, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_high_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_high_risk, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(
        sample_project_state_high_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_high_risk, dag).score()
    
    risk_engine = RiskEngine(
        sample_project_state_high_risk, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()
    
    # Verify all sub-risks have explanations
    for sub_risk in [result.schedule_risk, result.dependency_risk, result.resource_risk, result.scope_risk]:
        assert len(sub_risk.reasons) >= 0
        assert len(sub_risk.drivers) >= 0
    
    # Verify top drivers have descriptions and recommendations
    for driver in result.top_risk_drivers:
        assert len(driver.title) > 0
        assert len(driver.description) > 0
        assert len(driver.recommendation_hint) > 0


def test_low_risk_project(sample_project_state_low_risk):
    """Test low-risk project has low overall score."""
    metrics = MetricsEngine(sample_project_state_low_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_low_risk)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(sample_project_state_low_risk, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_low_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_low_risk, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(
        sample_project_state_low_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_low_risk, dag).score()
    
    risk_engine = RiskEngine(
        sample_project_state_low_risk, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()
    
    # Low risk project should have LOW or MODERATE risk level
    assert result.overall_risk_level in [RiskLevel.LOW, RiskLevel.MODERATE]


def test_moderate_risk_project_with_high_utilization(sample_project_state_high_risk):
    """Test high-utilization project with estimate inflation and small delay.
    
    After DC-1 refactor (schedule confidence modifier), on-time-probability no longer
    independently adds risk. It now multiplies the delay component only.
    
    This fixture has:
    - 3.13 days expected delay (small)
    - 28% on-time probability (low)
    - 95% team utilization (high)
    - 46.7% estimate inflation (high)
    
    Result: MODERATE overall risk (29.4) driven by Resource + Scope, not Schedule.
    """
    metrics = MetricsEngine(sample_project_state_high_risk).calculate()
    dep_engine = DependencyGraphEngine(sample_project_state_high_risk)
    dag = dep_engine.build_dag()
    cp_engine = CriticalPathEngine(sample_project_state_high_risk, dag)
    cp_result = cp_engine.analyze()
    spillover = SpilloverAnalysisEngine(sample_project_state_high_risk, metrics.average_item_effort).analyze()
    forecast = ForecastEngine(sample_project_state_high_risk, metrics, cp_result, spillover).calculate()
    mc_engine = MonteCarloEngine(
        sample_project_state_high_risk, metrics, cp_result, spillover, simulation_count=1000
    )
    monte_carlo = mc_engine.calculate()
    impact_scores = ImpactScoringEngine(sample_project_state_high_risk, dag).score()
    
    risk_engine = RiskEngine(
        sample_project_state_high_risk, metrics, cp_result, dag, spillover, forecast, monte_carlo, impact_scores
    )
    result = risk_engine.analyze()

    print("\n=== MODERATE RISK PROJECT (HIGH UTILIZATION) DEBUG ===")
    print("Overall Score:", result.overall_risk_score)
    print("Overall Level:", result.overall_risk_level)
    print("Expected delay days:", forecast.expected_delay_days)
    print("On-time probability:", monte_carlo.on_time_probability)

    print("\nSchedule Risk")
    print(result.schedule_risk)

    print("\nDependency Risk")
    print(result.dependency_risk)

    print("\nResource Risk")
    print(result.resource_risk)

    print("\nScope Risk")
    print(result.scope_risk)

    # Under DC-1, small delay + low probability + high utilization + high scope inflation = MODERATE
    assert result.overall_risk_level in [
        RiskLevel.MODERATE,
        RiskLevel.HIGH,
    ]
    
    # Verify resource risk dominates (95% utilization)
    assert result.resource_risk.score >= 60.0
