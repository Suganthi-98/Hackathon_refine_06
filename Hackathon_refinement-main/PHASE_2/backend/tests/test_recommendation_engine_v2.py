from datetime import datetime, timedelta

from app.domain.models import (
    Blocker,
    BlockerCategory,
    BlockerSeverity,
    BlockerStatus,
    Dependency,
    DependencyType,
    ProjectInfo,
    ProjectState,
    Priority,
    Resource,
    SkillLevel,
    Sprint,
    SprintActual,
    SprintStatus,
    WorkItem,
    WorkItemStatus,
    WorkItemType,
)
from app.engines.advisor_input_builder import AdvisorInputBuilder
from app.engines.recommendation_engine.models import RecommendationAction
from app.engines.recommendation_engine.recommendation_engine_v2 import RecommendationEngineV2


def make_project_state() -> ProjectState:
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="V2 Recommendation",
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
        )
    ]

    work_items = [
        WorkItem(
            item_id="WI-01",
            title="API work",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
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
            assigned_sprint="S1",
            original_sprint="S1",
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
            sprint_id="SA-1",
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
        project_id="REC-V2",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=actuals,
    )


def test_recommendation_engine_v2_caches_upstream_once(monkeypatch):
    state = make_project_state()
    calls = []
    original_run = __import__("app.engines.simulation_engine", fromlist=["EngineRunner"]).EngineRunner.run

    def wrapped_run(self, state_arg, simulation_count=1000):
        calls.append(simulation_count)
        return original_run(self, state_arg, simulation_count)

    monkeypatch.setattr("app.engines.recommendation_engine.recommendation_engine_v2.EngineRunner.run", wrapped_run)

    engine = RecommendationEngineV2(state, simulation_count=50)
    engine.generate(top_n=5)
    engine.generate(top_n=3)

    assert len(calls) == 1


def test_recommendation_engine_v2_simulate_without_prior_generate():
    state = make_project_state()
    engine = RecommendationEngineV2(state, simulation_count=50)
    recommendations = engine.generate(top_n=5)
    rec_id = recommendations[0].recommendation_id

    simulation = engine.simulate(rec_id)

    assert simulation.recommendation_ids == [rec_id]
    assert simulation.seed_used == 42


def test_recommendation_engine_v2_generates_actionable_recommendations():
    state = make_project_state()
    engine = RecommendationEngineV2(state, simulation_count=50)
    recommendations = engine.generate(top_n=5)

    assert recommendations
    assert len({rec.recommendation_id for rec in recommendations}) == len(recommendations)
    assert all(
        rec.affected_item_ids or rec.affected_resource_ids or rec.affected_blocker_ids
        for rec in recommendations
    )


def test_recommendation_engine_v2_exposes_validation_cache():
    state = make_project_state()
    engine = RecommendationEngineV2(state, simulation_count=50)
    recommendations = engine.generate(top_n=5)

    assert recommendations
    validation = engine.get_validation(recommendations[0].recommendation_id)
    assert validation is not None
    assert validation.recommendation_id == recommendations[0].recommendation_id


def test_recommendation_input_preserves_historical_pattern_for_narrative():
    state = make_project_state()
    engine = RecommendationEngineV2(state, simulation_count=50)
    recommendations = engine.generate(top_n=5)

    assert recommendations
    rec = next((item for item in recommendations if item.action_type == RecommendationAction.REBASELINE_ESTIMATE), None)
    assert rec is not None

    builder = AdvisorInputBuilder()
    advisor_input = builder.build_recommendation_input(
        project_id="session-1",
        project_state=state,
        forecast=engine._compute_upstream().forecast,
        monte_carlo=engine._compute_upstream().monte_carlo,
        recommendations=recommendations,
        metrics=engine._compute_upstream().metrics,
    )

    recommendation_fact = next(
        (fact for fact in advisor_input.recommendations if fact.recommendation_id == rec.recommendation_id),
        None,
    )
    assert recommendation_fact is not None
    assert recommendation_fact.historical_pattern is not None
    assert recommendation_fact.historical_pattern["pattern_type"]


def test_generated_recommendations_have_historical_evidence_confidence_and_simulation_support():
    state = make_project_state()
    engine = RecommendationEngineV2(state, simulation_count=50)
    recommendations = engine.generate(top_n=10)

    assert recommendations
    assert all(rec.metadata.get("historical_pattern") for rec in recommendations)
    assert all(rec.confidence for rec in recommendations)
    assert all(rec.estimated_hours_recovered >= 0.0 for rec in recommendations)
    assert all(rec.estimated_delay_reduction_days >= 0.0 for rec in recommendations)

    for rec in recommendations:
        simulated = engine.simulate(rec.recommendation_id)
        assert simulated.recommendation_ids == [rec.recommendation_id]
