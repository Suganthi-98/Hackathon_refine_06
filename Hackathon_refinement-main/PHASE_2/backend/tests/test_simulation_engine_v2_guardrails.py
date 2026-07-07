import logging
from types import SimpleNamespace

import pytest

from app.engines.simulation_engine import SimulationEngineV2
from app.engines.recommendation_engine.recommendation_engine_v2 import RecommendationEngineV2
from app.engines.recommendation_engine.models import (
    RecommendationCandidate,
    RecommendationAction,
    ConfidenceLevel,
)
from tests.test_recommendation_engine_v2 import make_project_state


def make_simple_recommendation(rec_id: str) -> RecommendationCandidate:
    return RecommendationCandidate(
        recommendation_id=rec_id,
        action_type=RecommendationAction.PULL_FORWARD_ITEM,
        title="No-op",
        description="No-op",
        affected_item_ids=["WI-01"],
        affected_resource_ids=[],
        affected_sprint_ids=[],
        affected_blocker_ids=[],
        root_cause_signal_id="",
        supporting_signal_ids=[],
        simulation_params={},
        feasibility_checks={},
    )


def test_simulation_mutation_check_raises_when_applicator_noops(monkeypatch):
    state = make_project_state()
    engine_v2 = RecommendationEngineV2(state, simulation_count=10)
    upstream = engine_v2._compute_upstream()
    sim = SimulationEngineV2(state, upstream, simulation_count=10)

    # Force applicator to be a no-op
    monkeypatch.setattr(sim.applicator, "apply", lambda s, r: None)

    rec = make_simple_recommendation("rec-noop")

    with pytest.raises(RuntimeError):
        sim.simulate(rec)


def test_estimate_simulation_divergence_logs_warning(monkeypatch, caplog):
    state = make_project_state()
    engine = RecommendationEngineV2(state, simulation_count=10)

    # Create two fake recommendations and patch CandidateGenerator and ImpactEstimator
    rec1 = make_simple_recommendation("rec-1")
    rec2 = make_simple_recommendation("rec-2")

    # Monkeypatch candidate generation to return our two recommendations
    from app.engines.recommendation_engine.candidate_generator import CandidateGenerator

    monkeypatch.setattr(CandidateGenerator, "generate", lambda self, signals: [rec1, rec2])

    # Monkeypatch ImpactEstimator.estimate to return a small delay estimate (0.1)
    from app.engines.recommendation_engine.impact_estimator import ImpactEstimator
    from app.engines.recommendation_engine.models import ImpactEstimate

    monkeypatch.setattr(ImpactEstimator, "estimate", lambda self, candidate: ImpactEstimate(
        estimated_hours_recovered=0.0,
        estimated_delay_reduction_days=0.1,
        estimated_risk_reduction=0.0,
        confidence=ConfidenceLevel.HIGH,
        evidence=[],
        calculation_notes="",
    ))

    # Monkeypatch SimulationEngineV2.simulate to return a SimpleNamespace with a large simulated delay
    from app.engines.simulation_engine import SimulationEngineV2 as SimV2Class

    def fake_simulate(self, recommendation):
        return SimpleNamespace(
            recommendation_ids=[recommendation.recommendation_id],
            delta_on_time_probability=0.0,
            delta_expected_delay_days=2.5,
        )

    monkeypatch.setattr(SimV2Class, "simulate", fake_simulate)

    caplog.set_level(logging.WARNING)
    # Run generate which will triage, simulate, and compare estimate vs simulation
    engine.generate(top_n=1)

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Estimate/simulation divergence" in str(w) for w in warnings)
