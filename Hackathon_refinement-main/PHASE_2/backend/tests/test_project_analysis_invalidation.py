"""
Regression test: ProjectAnalysis cache invalidation.

Guards the exact bug class this fix targets — a route mutates
session.project_state (in place, or by wholesale replacement) but the
session's cached ProjectAnalysis is never invalidated, so every subsequent
route silently serves numbers computed from the OLD state.

This test does not care about the correctness of scope-change or
apply-plan business logic — only that after either mutation path, a fresh
store.get_analysis(session_id) call reflects the new state rather than
returning a stale cached object.
"""
import pytest

from app.domain.models import WorkItemStatus
from app.engines.project_analysis import ProjectAnalysis
from app.storage.session_store import SessionStore

# Reuse the existing sample-state builder so this test tracks the same
# domain model the rest of the suite uses.
from tests.test_phase3 import make_sample_project_state


@pytest.fixture
def store():
    s = SessionStore()
    s.clear_all()
    yield s
    s.clear_all()


def test_scope_change_mutation_invalidates_cache(store):
    state = make_sample_project_state()
    session_id = store.create_session(state)

    # First read populates the cache.
    first = store.get_analysis(session_id, simulation_count=100)
    assert first is not None
    original_remaining_hours = first.metrics.remaining_effort_hours

    # Simulate what scope_change.py's confirmed path does: mutate a work
    # item in place on the SAME ProjectState object the session holds.
    session = store.get_session(session_id)
    work_item = session.project_state.work_items[0]
    work_item.status = WorkItemStatus.COMPLETED
    work_item.remaining_effort_hrs = 0.0
    work_item.progress_pct = 1.0

    # The fix under test: the route must call this before returning.
    store.invalidate_analysis(session_id)

    second = store.get_analysis(session_id, simulation_count=100)
    assert second is not first, (
        "get_analysis() returned the same cached object after invalidate_analysis() — "
        "cache was not actually rebuilt."
    )
    assert second.metrics.remaining_effort_hours != original_remaining_hours, (
        "remaining_effort_hours is identical before and after the scope change — "
        "ProjectAnalysis was rebuilt from stale ProjectState, or the mutation "
        "didn't actually change anything measurable."
    )


def test_apply_plan_mutation_invalidates_cache(store):
    state = make_sample_project_state()
    session_id = store.create_session(state)

    first = store.get_analysis(session_id, simulation_count=100)
    assert first is not None

    # Simulate what recovery_plans.py's apply-plan path does: replace
    # session.project_state wholesale with a mutated clone.
    session = store.get_session(session_id)
    updated_state = session.project_state.model_copy(deep=True)
    updated_state.work_items[0].remaining_effort_hrs = 0.0
    updated_state.work_items[0].status = WorkItemStatus.COMPLETED
    session.project_state = updated_state

    store.invalidate_analysis(session_id)

    second = store.get_analysis(session_id, simulation_count=100)
    assert second is not first
    assert second.project_state is updated_state, (
        "Cached ProjectAnalysis is not built from the replaced ProjectState — "
        "invalidate_analysis()/get_analysis() is reading the wrong session state."
    )


def test_forgetting_invalidate_serves_stale_analysis(store):
    """
    Negative control: proves the bug this fix closes actually exists when
    invalidate_analysis() is skipped, so this test file can't pass vacuously.
    """
    state = make_sample_project_state()
    session_id = store.create_session(state)

    first = store.get_analysis(session_id, simulation_count=100)

    session = store.get_session(session_id)
    session.project_state.work_items[0].remaining_effort_hrs = 0.0
    session.project_state.work_items[0].status = WorkItemStatus.COMPLETED
    # Deliberately NOT calling store.invalidate_analysis(session_id) here.

    stale = store.get_analysis(session_id, simulation_count=100)
    assert stale is first, (
        "Expected the (buggy, uninvalidated) path to return the same cached "
        "object — if this fails, get_analysis() is rebuilding on every call, "
        "which means caching isn't working at all."
    )
