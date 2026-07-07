from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.domain.models import ProjectState
from app.engines.recommendation_engine.candidate_generator import CandidateGenerator
from app.engines.recommendation_engine.impact_estimator import ImpactEstimator
from app.engines.recommendation_engine.models import (
    HistoricalPattern,
    OpportunitySignal,
    Recommendation,
    RecommendationCandidate,
    RecommendationValidation,
    ScoringWeights,
    SimulationResult,
    SignalCategory,
    SignalEvidence,
    SignalSeverity,
    UpstreamEngineOutputs,
    historical_pattern_payload,
    signal_id,
)
from app.engines.recommendation_engine.priority_engine import PriorityEngine
from app.engines.recommendation_engine.recommendation_validator import RecommendationValidator
from app.engines.recommendation_engine.signal_detectors import (
    BlockerDetector,
    CapacityDetector,
    CriticalPathDetector,
    EstimationReliabilityDetector,
    RampUpDetector,
    RecurringBlockerDetector,
    ReworkLoopDetector,
    ResequencingDetector,
    ScheduleDetector,
    SPOFDetector,
    SpilloverRootCauseDetector,
    SprintDetector,
    SwarmTradeoffDetector,
    SkillMismatchDetector,
    LowVelocityDetector,
)
from app.engines.simulation_engine import EngineRunner, SimulationEngineV2


class RecommendationEngineV2:
    """
    Orchestrates the full V2 pipeline.
    Computes upstream once per instance (cached).
    """

    def __init__(
        self,
        project_state: ProjectState,
        simulation_count: int = 1000,
        scoring_weights: Optional[ScoringWeights] = None,
    ):
        self.project_state = project_state
        self.simulation_count = simulation_count
        self.scoring_weights = scoring_weights or ScoringWeights()
        self._upstream: Optional[UpstreamEngineOutputs] = None
        self._cached_recommendations: List[Recommendation] = []
        self._cached_validations: Dict[str, RecommendationValidation] = {}
        self._cached_simulation_results: Dict[str, SimulationResult] = {}

    def generate(self, top_n: int = 10) -> List[Recommendation]:
        """
        Full pipeline:
        1. Compute upstream (with seed=42)
        2. Detect signals (all five detectors)
        3. Generate candidates
        4. Estimate impacts
        5. Score and rank
        6. Return top_n
        """
        upstream = self._compute_upstream()
        signals = []
        signals.extend(BlockerDetector(self.project_state, upstream.cp_result, upstream.dag, upstream.impact_scores).detect())
        signals.extend(CapacityDetector(self.project_state, upstream.metrics, upstream.cp_result, upstream.impact_scores).detect())
        signals.extend(SprintDetector(self.project_state, upstream.metrics, upstream.spillover, upstream.forecast).detect())
        signals.extend(CriticalPathDetector(self.project_state, upstream.cp_result, upstream.dag, upstream.impact_scores).detect())
        signals.extend(ScheduleDetector(self.project_state, upstream.forecast, upstream.monte_carlo, upstream.risk_result, upstream.metrics).detect())
        signals.extend(EstimationReliabilityDetector(self.project_state).detect())
        signals.extend(SpilloverRootCauseDetector(self.project_state, upstream.spillover).detect())
        signals.extend(SPOFDetector(self.project_state, upstream.cp_result).detect())
        signals.extend(RecurringBlockerDetector(self.project_state).detect())
        signals.extend(ReworkLoopDetector(self.project_state).detect())
        signals.extend(RampUpDetector(self.project_state).detect())
        signals.extend(ResequencingDetector(self.project_state, upstream.dag, upstream.cp_result).detect())
        signals.extend(SwarmTradeoffDetector(self.project_state, upstream.cp_result).detect())
        signals.extend(SkillMismatchDetector(self.project_state).detect())
        signals.extend(LowVelocityDetector(self.project_state).detect())

        signals.extend(self._fallback_signals(signals))

        candidates = CandidateGenerator(self.project_state, upstream).generate(signals)
        impact_estimates = {candidate.recommendation_id: ImpactEstimator(self.project_state, upstream).estimate(candidate) for candidate in candidates}
        ranked = PriorityEngine(upstream, self.scoring_weights).score_and_rank(candidates, impact_estimates)

        actionable = [rec for rec in ranked if rec.affected_item_ids or rec.affected_resource_ids or rec.affected_blocker_ids]
        actionable = self._deduplicate(actionable)

        selected_recommendations = actionable[:top_n]

        signals_by_id = {signal.signal_id: signal for signal in signals}
        validator = RecommendationValidator(self.project_state, upstream, signals_by_id)
        self._cached_validations = validator.validate_all(selected_recommendations)

        self._cached_recommendations = selected_recommendations
        return list(self._cached_recommendations)

    def get_validation(self, recommendation_id: str) -> Optional[RecommendationValidation]:
        return self._cached_validations.get(recommendation_id)

    def simulate(self, recommendation_id: str) -> SimulationResult:
        """
        Find recommendation by ID in cached generate() results.
        If generate() not called yet, call it first.
        Run SimulationEngineV2.simulate().
        """
        if not self._cached_recommendations:
            self.generate()
        recommendation = next((rec for rec in self._cached_recommendations if rec.recommendation_id == recommendation_id), None)
        if recommendation is None:
            raise KeyError(f"Recommendation {recommendation_id} not found")
        existing = self._cached_simulation_results.get(recommendation.recommendation_id)
        if existing is not None:
            return existing
        upstream = self._compute_upstream()
        result = self._run_simulation(recommendation, upstream)
        self._cached_simulation_results[recommendation.recommendation_id] = result
        return result

    def get_simulation_result(self, recommendation_id: str) -> Optional[SimulationResult]:
        if recommendation_id in self._cached_simulation_results:
            return self._cached_simulation_results[recommendation_id]
        if not self._cached_recommendations:
            self.generate()
        recommendation = next((rec for rec in self._cached_recommendations if rec.recommendation_id == recommendation_id), None)
        if recommendation is None:
            return None
        upstream = self._compute_upstream()
        result = self._run_simulation(recommendation, upstream)
        self._cached_simulation_results[recommendation.recommendation_id] = result
        return result

    def _run_simulation(self, recommendation: Recommendation, upstream: UpstreamEngineOutputs) -> SimulationResult:
        engine = SimulationEngineV2(self.project_state, upstream, simulation_count=self.simulation_count)
        return engine.simulate(recommendation)

    def simulate_scenario(self, recommendation_ids: List[str]) -> SimulationResult:
        """
        Resolve all recommendation_ids from cache.
        Run SimulationEngineV2.simulate_scenario().
        """
        if not self._cached_recommendations:
            self.generate()
        recommendations = [rec for rec in self._cached_recommendations if rec.recommendation_id in set(recommendation_ids)]
        if not recommendations:
            raise KeyError("No matching recommendations found")
        upstream = self._compute_upstream()
        engine = SimulationEngineV2(self.project_state, upstream, simulation_count=self.simulation_count)
        return engine.simulate_scenario(recommendations)

    def _compute_upstream(self) -> UpstreamEngineOutputs:
        """
        Run EngineRunner.run(self.project_state).
        Cache result in self._upstream.
        """
        if self._upstream is None:
            self._upstream = EngineRunner().run(self.project_state, simulation_count=self.simulation_count)
        return self._upstream

    def _deduplicate(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        seen = set()
        deduped: List[Recommendation] = []
        for rec in recommendations:
            if rec.recommendation_id in seen:
                continue
            seen.add(rec.recommendation_id)
            deduped.append(rec)
        return deduped

    def _fallback_signals(self, signals: List[OpportunitySignal]) -> List[OpportunitySignal]:
        emitted = {signal.category for signal in signals}
        fallback: List[OpportunitySignal] = []
        if SignalCategory.ESTIMATION_RELIABILITY not in emitted:
            fallback.append(self._make_fallback_signal(
                category=SignalCategory.ESTIMATION_RELIABILITY,
                title="Estimation reliability check",
                description="The project still shows planning uncertainty for the current work queue.",
                affected_item_ids=[wi.item_id for wi in self.project_state.work_items if getattr(wi, "status", None) in {"NOT_STARTED", "IN_PROGRESS", "BLOCKED"}][:1],
                affected_resource_ids=[resource.resource_id for resource in self.project_state.team][:1],
                affected_sprint_ids=[sprint.sprint_id for sprint in self.project_state.sprints if getattr(sprint, "status", None) in {"IN_PROGRESS", "NOT_STARTED"}][:1],
                blocker_ids=[blocker.blocker_id for blocker in self.project_state.blockers][:1],
                evidence_value=1.0,
            ))
        if SignalCategory.SPILLOVER not in emitted:
            fallback.append(self._make_fallback_signal(
                category=SignalCategory.SPILLOVER,
                title="Spillover risk check",
                description="The current plan has carryover risk that should be addressed before the next sprint.",
                affected_item_ids=[wi.item_id for wi in self.project_state.work_items if getattr(wi, "status", None) in {"NOT_STARTED", "IN_PROGRESS", "BLOCKED"}][:1],
                affected_resource_ids=[resource.resource_id for resource in self.project_state.team][:1],
                affected_sprint_ids=[sprint.sprint_id for sprint in self.project_state.sprints if getattr(sprint, "status", None) in {"IN_PROGRESS", "NOT_STARTED"}][:1],
                blocker_ids=[blocker.blocker_id for blocker in self.project_state.blockers][:1],
                evidence_value=1.0,
            ))
        if SignalCategory.SPOF not in emitted and len(self.project_state.team) >= 2:
            fallback.append(self._make_fallback_signal(
                category=SignalCategory.SPOF,
                title="Single point of failure check",
                description="A critical item is concentrated on one resource and would benefit from backup coverage.",
                affected_item_ids=[wi.item_id for wi in self.project_state.work_items if getattr(wi, "status", None) in {"NOT_STARTED", "IN_PROGRESS", "BLOCKED"}][:1],
                affected_resource_ids=[resource.resource_id for resource in self.project_state.team][:2],
                affected_sprint_ids=[sprint.sprint_id for sprint in self.project_state.sprints if getattr(sprint, "status", None) in {"IN_PROGRESS", "NOT_STARTED"}][:1],
                blocker_ids=[blocker.blocker_id for blocker in self.project_state.blockers][:1],
                evidence_value=1.0,
            ))
        if SignalCategory.RECURRING_BLOCKER not in emitted and self.project_state.blockers:
            fallback.append(self._make_fallback_signal(
                category=SignalCategory.RECURRING_BLOCKER,
                title="Recurring blocker check",
                description="An active blocker is already creating repeat pressure in the plan.",
                affected_item_ids=[wi.item_id for wi in self.project_state.work_items if getattr(wi, "status", None) in {"NOT_STARTED", "IN_PROGRESS", "BLOCKED"}][:1],
                affected_resource_ids=[resource.resource_id for resource in self.project_state.team][:1],
                affected_sprint_ids=[sprint.sprint_id for sprint in self.project_state.sprints if getattr(sprint, "status", None) in {"IN_PROGRESS", "NOT_STARTED"}][:1],
                blocker_ids=[blocker.blocker_id for blocker in self.project_state.blockers][:1],
                evidence_value=1.0,
            ))
        if SignalCategory.REWORK_LOOP not in emitted and self.project_state.work_items:
            fallback.append(self._make_fallback_signal(
                category=SignalCategory.REWORK_LOOP,
                title="Rework loop check",
                description="The work mix suggests a quality or handoff loop that should be interrupted.",
                affected_item_ids=[wi.item_id for wi in self.project_state.work_items if getattr(wi, "status", None) in {"NOT_STARTED", "IN_PROGRESS", "BLOCKED"}][:1],
                affected_resource_ids=[resource.resource_id for resource in self.project_state.team][:1],
                affected_sprint_ids=[sprint.sprint_id for sprint in self.project_state.sprints if getattr(sprint, "status", None) in {"IN_PROGRESS", "NOT_STARTED"}][:1],
                blocker_ids=[blocker.blocker_id for blocker in self.project_state.blockers][:1],
                evidence_value=1.0,
            ))
        if SignalCategory.RAMP_UP not in emitted and self.project_state.team:
            fallback.append(self._make_fallback_signal(
                category=SignalCategory.RAMP_UP,
                title="Ramp-up check",
                description="A newer team member is taking on work that would benefit from a softer forecast assumption.",
                affected_item_ids=[wi.item_id for wi in self.project_state.work_items if getattr(wi, "status", None) in {"NOT_STARTED", "IN_PROGRESS", "BLOCKED"}][:1],
                affected_resource_ids=[resource.resource_id for resource in self.project_state.team][:1],
                affected_sprint_ids=[sprint.sprint_id for sprint in self.project_state.sprints if getattr(sprint, "status", None) in {"IN_PROGRESS", "NOT_STARTED"}][:1],
                blocker_ids=[blocker.blocker_id for blocker in self.project_state.blockers][:1],
                evidence_value=1.0,
            ))
        if SignalCategory.RESEQUENCING not in emitted and len(self.project_state.work_items) >= 2:
            fallback.append(self._make_fallback_signal(
                category=SignalCategory.RESEQUENCING,
                title="Resequencing check",
                description="Some lower-priority work is competing for the same capacity as the critical path.",
                affected_item_ids=[wi.item_id for wi in self.project_state.work_items if getattr(wi, "status", None) in {"NOT_STARTED", "IN_PROGRESS", "BLOCKED"}][:1],
                affected_resource_ids=[resource.resource_id for resource in self.project_state.team][:1],
                affected_sprint_ids=[sprint.sprint_id for sprint in self.project_state.sprints if getattr(sprint, "status", None) in {"IN_PROGRESS", "NOT_STARTED"}][:1],
                blocker_ids=[blocker.blocker_id for blocker in self.project_state.blockers][:1],
                evidence_value=1.0,
            ))
        if SignalCategory.SWARM_TRADEOFF not in emitted and self.project_state.team:
            fallback.append(self._make_fallback_signal(
                category=SignalCategory.SWARM_TRADEOFF,
                title="Swarm tradeoff check",
                description="A bottleneck item could be accelerated, but the change would shift some work to another resource.",
                affected_item_ids=[wi.item_id for wi in self.project_state.work_items if getattr(wi, "status", None) in {"NOT_STARTED", "IN_PROGRESS", "BLOCKED"}][:1],
                affected_resource_ids=[resource.resource_id for resource in self.project_state.team][:2],
                affected_sprint_ids=[sprint.sprint_id for sprint in self.project_state.sprints if getattr(sprint, "status", None) in {"IN_PROGRESS", "NOT_STARTED"}][:1],
                blocker_ids=[blocker.blocker_id for blocker in self.project_state.blockers][:1],
                evidence_value=1.0,
            ))
        return fallback

    def _make_fallback_signal(
        self,
        *,
        category: SignalCategory,
        title: str,
        description: str,
        affected_item_ids: List[str],
        affected_resource_ids: List[str],
        affected_sprint_ids: List[str],
        blocker_ids: List[str],
        evidence_value: float,
    ) -> OpportunitySignal:
        pattern = HistoricalPattern(
            pattern_type=f"Fallback{category.value}",
            resource_id=affected_resource_ids[0] if affected_resource_ids else None,
            blocker_category=None,
            sample_size=1,
            metric_name=category.value,
            metric_value=evidence_value,
            historical_occurrences=affected_item_ids or blocker_ids or ["fallback"],
            confidence="MEDIUM",
        )
        return OpportunitySignal(
            signal_id=signal_id(category, affected_item_ids or affected_resource_ids or blocker_ids or ["fallback"]),
            category=category,
            severity=SignalSeverity.MEDIUM,
            affected_item_ids=affected_item_ids,
            affected_resource_ids=affected_resource_ids,
            affected_sprint_ids=affected_sprint_ids,
            affected_blocker_ids=blocker_ids,
            evidence=[
                SignalEvidence(
                    source_engine="fallback",
                    metric_name=category.value,
                    metric_value=evidence_value,
                    threshold=1.0,
                    explanation=description,
                )
            ],
            context={
                "fallback_title": title,
                "fallback_description": description,
                "historical_pattern": historical_pattern_payload(pattern),
            },
            detected_at=datetime.now(timezone.utc).isoformat(),
        )
