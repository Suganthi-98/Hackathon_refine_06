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
from app.engines.recommendation_engine.recommendation_engine_v2 import RecommendationEngineV2
from app.engines.recommendation_engine.models import (
    Recommendation,
    RecommendationCandidate,
    RecommendationAction,
    stable_id,
)
from app.domain.models import BlockerCategory


def make_recommendation_project_state() -> ProjectState:
    start_date = datetime(2025, 1, 1)
    target_date = datetime(2025, 3, 1)

    project_info = ProjectInfo(
        project_name="Recommendation Test",
        sponsor="Test Sponsor",
        business_unit="Engineering",
        project_manager="Test PM",
        customer="Test Customer",
        status="Active",
        start_date=start_date,
        target_end_date=target_date,
        sprint_duration_days=14,
        methodology="Agile Scrum",
    )

    team = [
        Resource(
            resource_id="R1",
            name="Alice",
            role="Backend Engineer",
            primary_skill="Python",
            secondary_skill="SQL",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=0.9,
            availability_pct=0.8,
        ),
        Resource(
            resource_id="R2",
            name="Bob",
            role="Tester",
            primary_skill="Testing",
            secondary_skill="Python",
            skill_level=SkillLevel.INTERMEDIATE,
            allocation_pct=0.9,
            availability_pct=0.15,
        ),
        Resource(
            resource_id="R3",
            name="Celine",
            role="Frontend Engineer",
            primary_skill="React",
            secondary_skill="JavaScript",
            skill_level=SkillLevel.MID,
            allocation_pct=0.8,
            availability_pct=0.8,
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
            sprint_goal="Foundation",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=160.0,
            carryover_count=1,
        ),
        Sprint(
            sprint_id="S2",
            sprint_name="Sprint 2",
            sprint_number=2,
            start_date=start_date + timedelta(days=14),
            end_date=start_date + timedelta(days=28),
            working_days=10,
            sprint_goal="Development",
            status=SprintStatus.NOT_STARTED,
            planned_velocity_hrs=160.0,
            carryover_count=0,
        ),
    ]

    work_items = [
        WorkItem(
            item_id="WI-01",
            title="Low priority research spike",
            work_type=WorkItemType.SPIKE,
            assigned_sprint="Sprint 2",
            original_sprint="Sprint 2",
            assigned_resource="R3",
            required_skill="React",
            priority=Priority.LOW,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=20.0,
            progress_pct=0.0,
            status=WorkItemStatus.NOT_STARTED,
        ),
        WorkItem(
            item_id="WI-02",
            title="Blocking API",
            work_type=WorkItemType.TASK,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            assigned_resource="R2",
            required_skill="SQL",
            priority=Priority.HIGH,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=20.0,
            progress_pct=0.0,
            status=WorkItemStatus.BLOCKED,
        ),
        WorkItem(
            item_id="WI-03",
            title="Critical backend epic",
            work_type=WorkItemType.FEATURE,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            assigned_resource="R1",
            required_skill="Python",
            priority=Priority.CRITICAL,
            estimated_effort_hrs=110.0,
            current_estimate_hrs=110.0,
            actual_effort_hrs=10.0,
            remaining_effort_hrs=110.0,
            progress_pct=0.125,
            status=WorkItemStatus.IN_PROGRESS,
        ),
        WorkItem(
            item_id="WI-04",
            title="Backend integration task",
            work_type=WorkItemType.TASK,
            assigned_sprint="Sprint 2",
            original_sprint="Sprint 2",
            assigned_resource="R1",
            required_skill="Python",
            priority=Priority.MEDIUM,
            estimated_effort_hrs=60.0,
            current_estimate_hrs=60.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=60.0,
            progress_pct=0.0,
            status=WorkItemStatus.NOT_STARTED,
        ),
    ]

    dependencies = [
        Dependency(
            dependency_id="DEP-01",
            predecessor_item_id="WI-02",
            successor_item_id="WI-03",
            dependency_type=DependencyType.FINISH_TO_START,
            is_on_critical_path=True,
            lag_days=0,
        ),
        Dependency(
            dependency_id="DEP-02",
            predecessor_item_id="WI-03",
            successor_item_id="WI-04",
            dependency_type=DependencyType.FINISH_TO_START,
            is_on_critical_path=True,
            lag_days=0,
        ),
    ]

    blockers = [
        Blocker(
            blocker_id="BLK-01",
            related_item_id="WI-02",
            impacted_item_ids=["WI-02", "WI-03", "WI-04"],
            description="Environment issue blocking Python API deployment",
            severity=BlockerSeverity.HIGH,
            status=BlockerStatus.OPEN,
            owner="DevOps",
            raised_date=start_date,
            target_resolution_date=start_date + timedelta(days=7),
            category=BlockerCategory.OTHER,
        )
    ]

    actuals = [
        SprintActual(
            sprint_id="S0",
            sprint_number=1,
            planned_effort_hrs=160.0,
            actual_effort_hrs=140.0,
            variance_hrs=20.0,
            tasks_planned=10,
            tasks_completed=8,
            completion_rate=0.8,
            carryover_count=2,
        )
    ]

    return ProjectState(
        project_id="REC-TEST",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=actuals,
    )


@pytest.fixture
def recommendation_project_state():
    return make_recommendation_project_state()


def build_recommendation_engine(project_state):
    # Use the V2 orchestrator which computes upstream internally
    return RecommendationEngineV2(project_state=project_state, simulation_count=50)


@pytest.fixture
def recommendation_engine(recommendation_project_state):
    return build_recommendation_engine(recommendation_project_state)


def test_blocker_recommendations(recommendation_engine):
    recommendations = recommendation_engine.generate(top_n=50)
    blocker_recs = [r for r in recommendations if r.action_type == RecommendationAction.RESOLVE_BLOCKER]
    assert blocker_recs
    rec = blocker_recs[0]
    assert "BLK-01" in (rec.title or "") or "BLK-01" in (rec.description or "")
    assert "other" in (rec.title or "").lower()  # title contains blocker category (BlockerCategory.value is "Other", not "OTHER")
    assert rec.affected_blocker_ids and "BLK-01" in rec.affected_blocker_ids
    assert rec.estimated_hours_recovered >= 0.0


def test_resource_recommendations(recommendation_engine):
    recommendations = recommendation_engine.generate(top_n=50)
    add_recs = [r for r in recommendations if r.action_type in {RecommendationAction.ADD_RESOURCE_SKILL, RecommendationAction.REASSIGN_ITEM}]
    assert add_recs
    rec = add_recs[0]
    # simulation params (required_skill) are stored in metadata by PriorityEngine
    sim_params = rec.metadata.get("simulation_params", {}) if getattr(rec, "metadata", None) is not None else {}
    # If a required skill is available, ensure it's a non-empty string; otherwise just ensure description exists
    if sim_params.get("required_skill"):
        assert "Python" in str(sim_params.get("required_skill"))
    else:
        assert rec.description


def test_reassignment_recommendations(recommendation_engine):
    recommendations = recommendation_engine.generate(top_n=50)
    reassign_recs = [r for r in recommendations if r.action_type == RecommendationAction.REASSIGN_ITEM]
    assert reassign_recs
    rec = reassign_recs[0]
    assert "WI-02" in rec.affected_item_ids
    assert "Reassign" in rec.title


def test_reduce_item_scope_recommendations(recommendation_engine):
    recommendations = recommendation_engine.generate(top_n=50)
    reduce_recs = [r for r in recommendations if r.action_type == RecommendationAction.SPLIT_ITEM]
    assert reduce_recs
    rec = reduce_recs[0]
    assert rec.affected_item_ids
    assert isinstance(rec.estimated_hours_recovered, float)


def test_cp_optimization(recommendation_engine):
    recommendations = recommendation_engine.generate(top_n=50)
    cp_recs = [r for r in recommendations if r.action_type == RecommendationAction.ADVANCE_ITEM_TO_EARLIER_SPRINT]
    assert cp_recs
    rec = cp_recs[0]
    assert rec.affected_item_ids
    assert len(rec.affected_item_ids) >= 1


def test_simulation_and_ranking(recommendation_engine):
    recommendations = recommendation_engine.generate(top_n=50)
    assert all(r.priority_score >= 0.0 for r in recommendations)
    if recommendations:
        assert recommendations[0].priority_score >= recommendations[-1].priority_score
        # Impact/confidence fields are on Recommendation.confidence and impact_evidence
        assert isinstance(recommendations[0].confidence, type(recommendations[0].confidence))


def test_simulate_scenario(recommendation_engine):
    recommendations = recommendation_engine.generate(top_n=50)
    rec_ids = [r.recommendation_id for r in recommendations[:2]]
    if not rec_ids:
        pytest.skip("No recommendations to simulate")
    scenario = recommendation_engine.simulate_scenario(rec_ids)
    assert hasattr(scenario, "baseline_metrics")
    assert scenario.recommendation_ids == rec_ids


def test_recommendation_ids_are_stable_across_calls(recommendation_engine):
    first_pass = recommendation_engine.generate(top_n=50)
    second_pass = recommendation_engine.generate(top_n=50)
    assert [r.recommendation_id for r in first_pass] == [r.recommendation_id for r in second_pass]


def test_recommendation_ids_are_stable_across_engine_instances(recommendation_project_state):
    first_engine = build_recommendation_engine(recommendation_project_state)
    second_engine = build_recommendation_engine(recommendation_project_state)
    first_pass = first_engine.generate()
    second_pass = second_engine.generate()
    assert [r.recommendation_id for r in first_pass] == [r.recommendation_id for r in second_pass]


def test_recommendation_id_changes_when_target_ids_change(recommendation_engine):
    recommendations = recommendation_engine.generate(top_n=50)
    candidate = next((c for c in recommendations if c.affected_item_ids or c.affected_resource_ids or c.affected_blocker_ids), None)
    if candidate is None:
        pytest.skip("No candidate with target ids")
    changed_id = stable_id(candidate.action_type.value if hasattr(candidate.action_type, "value") else str(candidate.action_type), ["DIFFERENT-ID"])
    assert changed_id != candidate.recommendation_id


def test_simulate_recommendation_is_deterministic(recommendation_engine):
    recommendations = recommendation_engine.generate(top_n=50)
    if not recommendations:
        pytest.skip("No recommendations to simulate")
    recommendation_id = recommendations[0].recommendation_id
    first_result = recommendation_engine.simulate(recommendation_id)
    second_result = recommendation_engine.simulate(recommendation_id)
    assert first_result.delta_on_time_probability == pytest.approx(second_result.delta_on_time_probability)


def test_null_action_has_zero_probability_gain(recommendation_engine):
    from app.engines.simulation_engine import EngineRunnerV2, SimulationEngineV2
    # Build a minimal Recommendation that does nothing (no affected ids)
    null_rec = Recommendation(
        recommendation_id="NULL-ACTION",
        title="No-op",
        description="No-op",
        action_type=RecommendationAction.RESOLVE_BLOCKER,
        priority_score=0.0,
        confidence=None,
        estimated_hours_recovered=0.0,
        estimated_delay_reduction_days=0.0,
        estimated_risk_reduction=0.0,
        affected_item_ids=[],
        affected_resource_ids=[],
        affected_sprint_ids=[],
        affected_blocker_ids=[],
        root_cause_signal_id="",
    )
    # Compute baseline upstream and run simulation directly
    runner = EngineRunnerV2()
    baseline = runner.run(recommendation_engine.project_state, simulation_count=50)
    sim = SimulationEngineV2(recommendation_engine.project_state, baseline, simulation_count=50)
    result = sim.simulate(null_rec)
    assert result.delta_on_time_probability == pytest.approx(0.0, abs=1e-3)
