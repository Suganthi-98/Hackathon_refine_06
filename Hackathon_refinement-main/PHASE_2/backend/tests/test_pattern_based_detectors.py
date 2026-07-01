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
from app.engines.recommendation_engine.signal_detectors import (
    EstimationReliabilityDetector,
    RampUpDetector,
    ReworkLoopDetector,
    RecurringBlockerDetector,
    ResequencingDetector,
    SPOFDetector,
    SpilloverRootCauseDetector,
    SwarmTradeoffDetector,
)


def make_pattern_state() -> ProjectState:
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="Pattern Test",
        sponsor="Sponsor",
        business_unit="Engineering",
        project_manager="PM",
        customer="Customer",
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
            daily_capacity_hrs=8.0,
        ),
        Resource(
            resource_id="R2",
            name="Bob",
            role="Engineer",
            primary_skill="Testing",
            secondary_skill="Python",
            skill_level=SkillLevel.MID,
            allocation_pct=0.8,
            availability_pct=0.8,
            daily_capacity_hrs=8.0,
        ),
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
            carryover_count=0,
        ),
        Sprint(
            sprint_id="S2",
            sprint_name="Sprint 2",
            sprint_number=2,
            start_date=start_date + timedelta(days=14),
            end_date=start_date + timedelta(days=27),
            working_days=10,
            sprint_goal="Build",
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
            assigned_sprint="S1",
            original_sprint="S1",
            assigned_resource="R1",
            required_skill="Python",
            priority=Priority.CRITICAL,
            estimated_effort_hrs=40.0,
            current_estimate_hrs=40.0,
            actual_effort_hrs=60.0,
            remaining_effort_hrs=20.0,
            progress_pct=1.0,
            status=WorkItemStatus.DONE,
        ),
        WorkItem(
            item_id="WI-02",
            title="API work 2",
            work_type=WorkItemType.TASK,
            assigned_sprint="S1",
            original_sprint="S1",
            assigned_resource="R1",
            required_skill="Python",
            priority=Priority.HIGH,
            estimated_effort_hrs=40.0,
            current_estimate_hrs=40.0,
            actual_effort_hrs=55.0,
            remaining_effort_hrs=0.0,
            progress_pct=1.0,
            status=WorkItemStatus.DONE,
        ),
        WorkItem(
            item_id="WI-03",
            title="API work 3",
            work_type=WorkItemType.TASK,
            assigned_sprint="S2",
            original_sprint="S2",
            assigned_resource="R1",
            required_skill="Python",
            priority=Priority.MEDIUM,
            estimated_effort_hrs=40.0,
            current_estimate_hrs=40.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=40.0,
            progress_pct=0.0,
            status=WorkItemStatus.NOT_STARTED,
        ),
        WorkItem(
            item_id="WI-04",
            title="Review work",
            work_type=WorkItemType.TASK,
            assigned_sprint="S2",
            original_sprint="S2",
            assigned_resource="R2",
            required_skill="Testing",
            priority=Priority.MEDIUM,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=20.0,
            progress_pct=0.0,
            status=WorkItemStatus.NOT_STARTED,
        ),
    ]
    dependencies = [
        Dependency(
            dependency_id="DEP-01",
            predecessor_item_id="WI-04",
            successor_item_id="WI-03",
            dependency_type=DependencyType.FINISH_TO_START,
            is_on_critical_path=True,
            lag_days=0,
        )
    ]
    blockers = [
        Blocker(
            blocker_id="BLK-01",
            related_item_id="WI-03",
            impacted_item_ids=["WI-03"],
            description="Test blocker",
            severity=BlockerSeverity.HIGH,
            status=BlockerStatus.OPEN,
            owner="Ops",
            raised_date=start_date,
            target_resolution_date=start_date + timedelta(days=7),
            category=BlockerCategory.OTHER,
        ),
        Blocker(
            blocker_id="BLK-02",
            related_item_id="WI-04",
            impacted_item_ids=["WI-04"],
            description="Repeated blocker",
            severity=BlockerSeverity.HIGH,
            status=BlockerStatus.RESOLVED,
            owner="Ops",
            raised_date=start_date - timedelta(days=30),
            target_resolution_date=start_date - timedelta(days=20),
            actual_resolution_date=start_date - timedelta(days=10),
            category=BlockerCategory.OTHER,
        ),
    ]
    actuals = [SprintActual(sprint_id="S1", sprint_number=1, planned_effort_hrs=160.0, actual_effort_hrs=140.0, variance_hrs=20.0, tasks_planned=4, tasks_completed=3, completion_rate=0.75, carryover_count=1)]
    return ProjectState(
        project_id="P1",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=dependencies,
        blockers=blockers,
        actuals=actuals,
    )


def test_estimation_reliability_detector_emits_signal_for_repeating_overruns():
    state = make_pattern_state()
    detector = EstimationReliabilityDetector(state)
    signals = detector.detect()
    assert signals
    assert signals[0].category.value == "estimation_reliability"
    assert signals[0].context["historical_pattern"]["sample_size"] >= 2


def test_estimation_reliability_detector_skips_single_sample():
    state = make_pattern_state()
    state.work_items = [wi for wi in state.work_items if wi.item_id != "WI-02"]
    detector = EstimationReliabilityDetector(state)
    assert detector.detect() == []


def test_spillover_root_cause_detector_emits_signal_for_repeated_signature():
    state = make_pattern_state()
    detector = SpilloverRootCauseDetector(state)
    signals = detector.detect()
    assert signals
    assert signals[0].context["historical_pattern"]["pattern_type"] == "SpilloverRootCauseDetector"


def test_spof_detector_emits_signal_for_single_point_of_failure():
    state = make_pattern_state()
    detector = SPOFDetector(state)
    signals = detector.detect()
    assert signals
    assert signals[0].category.value == "single_point_of_failure"


def test_recurring_blocker_detector_emits_signal_for_repeated_category():
    state = make_pattern_state()
    detector = RecurringBlockerDetector(state)
    signals = detector.detect()
    assert signals
    assert signals[0].context["historical_pattern"]["sample_size"] >= 2


def test_rework_loop_detector_emits_signal_for_repeated_rework():
    state = make_pattern_state()
    detector = ReworkLoopDetector(state)
    signals = detector.detect()
    assert signals
    assert signals[0].context["historical_pattern"]["sample_size"] >= 2


def test_ramp_up_detector_emits_signal_for_new_joiner():
    state = make_pattern_state()
    detector = RampUpDetector(state)
    signals = detector.detect()
    assert signals
    assert signals[0].context["historical_pattern"]["pattern_type"] == "RampUpDetector"


def test_resequencing_detector_emits_signal_for_serialized_non_cp_item():
    state = make_pattern_state()
    detector = ResequencingDetector(state)
    signals = detector.detect()
    assert signals
    assert signals[0].category.value == "resequencing_opportunity"


def test_swarm_tradeoff_detector_emits_signal_with_tradeoff_context():
    state = make_pattern_state()
    detector = SwarmTradeoffDetector(state)
    signals = detector.detect()
    assert signals
    assert "days_saved_on_critical_path" in signals[0].context
