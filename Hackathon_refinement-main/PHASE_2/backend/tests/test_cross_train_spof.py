import pytest
from datetime import datetime, timedelta

from app.domain.models import (
    ProjectInfo,
    Resource,
    Sprint,
    WorkItem,
    ProjectState,
    SkillLevel,
    WorkItemType,
    Priority,
    WorkItemStatus,
    SprintStatus,
)

from app.engines.recommendation_engine.signal_detectors import SPOFDetector
from app.engines.recommendation_engine.models import Recommendation, RecommendationAction, ConfidenceLevel, SignalCategory
from app.engines.simulation_engine import ActionApplicatorV2


def make_spof_project_state() -> ProjectState:
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="SPOF Test",
        sponsor="Test",
        business_unit="Eng",
        project_manager="PM",
        customer="Cust",
        status="Active",
        start_date=start_date,
        target_end_date=start_date + timedelta(days=60),
        sprint_duration_days=14,
        methodology="Agile",
    )

    # Two team members: one sole owner, one potential backup
    team = [
        Resource(
            resource_id="sandeep_annamalai",
            name="Sandeep Annamalai",
            role="Engineer",
            primary_skill="General",
            secondary_skill=None,
            skill_level=SkillLevel.MID,
            allocation_pct=1.0,
            availability_pct=1.0,
        ),
        Resource(
            resource_id="backup_dev",
            name="Backup Dev",
            role="Engineer",
            primary_skill="General",
            secondary_skill=None,
            skill_level=SkillLevel.MID,
            allocation_pct=0.2,
            availability_pct=1.0,
        ),
    ]

    sprints = [
        Sprint(
            sprint_id="SPR-1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=13),
            working_days=10,
            sprint_goal="Goal",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=100.0,
            carryover_count=0,
        )
    ]

    # Two critical items both assigned to the same display-name owner (workbook-style)
    work_items = [
        WorkItem(
            item_id="WI-1",
            title="Critical A",
            work_type=WorkItemType.TASK,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            assigned_resource="Sandeep Annamalai",
            required_skill="General",
            priority=Priority.CRITICAL,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=20.0,
            progress_pct=0.0,
            status=WorkItemStatus.NOT_STARTED,
        ),
        WorkItem(
            item_id="WI-2",
            title="Critical B",
            work_type=WorkItemType.TASK,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            assigned_resource="Sandeep Annamalai",
            required_skill="General",
            priority=Priority.CRITICAL,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=20.0,
            progress_pct=0.0,
            status=WorkItemStatus.NOT_STARTED,
        ),
    ]

    return ProjectState(
        project_id="SPOF-TEST",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=[],
        blockers=[],
        actuals=[],
    )


def make_spof_multi_skill_project_state() -> ProjectState:
    start_date = datetime(2025, 1, 1)
    project_info = ProjectInfo(
        project_name="SPOF Multi-skill Test",
        sponsor="Test",
        business_unit="Eng",
        project_manager="PM",
        customer="Cust",
        status="Active",
        start_date=start_date,
        target_end_date=start_date + timedelta(days=60),
        sprint_duration_days=14,
        methodology="Agile",
    )

    team = [
        Resource(
            resource_id="owner_dev",
            name="Owner Dev",
            role="Engineer",
            primary_skill="Python",
            secondary_skill="SQL",
            skill_level=SkillLevel.SENIOR,
            allocation_pct=1.0,
            availability_pct=1.0,
        ),
        Resource(
            resource_id="backup_dev",
            name="Backup Dev",
            role="Engineer",
            primary_skill="Python",
            secondary_skill=None,
            skill_level=SkillLevel.MID,
            allocation_pct=0.2,
            availability_pct=1.0,
        ),
    ]

    sprints = [
        Sprint(
            sprint_id="SPR-1",
            sprint_name="Sprint 1",
            sprint_number=1,
            start_date=start_date,
            end_date=start_date + timedelta(days=13),
            working_days=10,
            sprint_goal="Goal",
            status=SprintStatus.IN_PROGRESS,
            planned_velocity_hrs=120.0,
            carryover_count=0,
        )
    ]

    work_items = [
        WorkItem(
            item_id="WI-1",
            title="Critical Python work",
            work_type=WorkItemType.TASK,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            assigned_resource="Owner Dev",
            required_skill="Python",
            priority=Priority.CRITICAL,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=20.0,
            progress_pct=0.0,
            status=WorkItemStatus.NOT_STARTED,
        ),
        WorkItem(
            item_id="WI-2",
            title="Critical SQL work",
            work_type=WorkItemType.TASK,
            assigned_sprint="Sprint 1",
            original_sprint="Sprint 1",
            assigned_resource="Owner Dev",
            required_skill="SQL",
            priority=Priority.CRITICAL,
            estimated_effort_hrs=20.0,
            current_estimate_hrs=20.0,
            actual_effort_hrs=0.0,
            remaining_effort_hrs=20.0,
            progress_pct=0.0,
            status=WorkItemStatus.NOT_STARTED,
        ),
    ]

    return ProjectState(
        project_id="SPOF-MULTI-SKILL-TEST",
        project_info=project_info,
        team=team,
        sprints=sprints,
        work_items=work_items,
        dependencies=[],
        blockers=[],
        actuals=[],
    )


def test_cross_train_simulated_velocity_delta_nonzero():
    state = make_spof_project_state()

    # Detect SPOF; detector should normalize the display-name owner to the resource_id
    detector = SPOFDetector(state)
    signals = detector.detect()
    assert signals, "Expected at least one SPOF signal"

    spof = signals[0]
    # SPOF detector should emit resource IDs (not display names)
    assert any(r == "sandeep_annamalai" for r in spof.affected_resource_ids), "Sole owner id should be normalized"
    assert any(r == "backup_dev" for r in spof.affected_resource_ids), "Backup resource id should be present"

    # Apply a cross-train recommendation targeted at the backup resource and ensure velocity increases
    rec = Recommendation(
        recommendation_id="REC-CT-1",
        title="Cross-train backup",
        description="Test",
        action_type=RecommendationAction.CROSS_TRAIN_BACKUP,
        priority_score=0.0,
        confidence=ConfidenceLevel.MEDIUM,
        estimated_hours_recovered=0.0,
        estimated_delay_reduction_days=0.0,
        estimated_risk_reduction=0.0,
        affected_item_ids=[],
        affected_resource_ids=["backup_dev"],
        affected_sprint_ids=[],
        affected_blocker_ids=[],
        root_cause_signal_id=spof.signal_id,
    )

    clone = state.model_copy(deep=True)
    before = clone.sprints[0].planned_velocity_hrs
    ActionApplicatorV2().apply(clone, rec)
    after = clone.sprints[0].planned_velocity_hrs
    assert after > before, "Planned velocity should increase after cross-train backup is applied"


def test_spof_detector_emits_one_signal_per_resource_for_multiple_skills():
    state = make_spof_multi_skill_project_state()

    detector = SPOFDetector(state)
    signals = detector.detect()

    assert len(signals) == 1, "SPOF detector should emit one signal per sole owner resource"
    spof_signal = signals[0]
    assert spof_signal.category == SignalCategory.SPOF
    assert spof_signal.affected_resource_ids[0] == "owner_dev"
    assert spof_signal.affected_resource_ids[1] == "backup_dev"
    assert set(spof_signal.affected_item_ids) == {"WI-1", "WI-2"}
    assert "Python" in spof_signal.context.get("skill_names", [])
    assert "SQL" in spof_signal.context.get("skill_names", [])
