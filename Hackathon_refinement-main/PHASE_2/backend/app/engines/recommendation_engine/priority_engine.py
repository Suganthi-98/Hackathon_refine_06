from __future__ import annotations

from typing import Dict, List, Optional

from app.engines.recommendation_engine.models import (
    ImpactEstimate,
    Recommendation,
    RecommendationAction,
    RecommendationCandidate,
    ScoringWeights,
    UpstreamEngineOutputs,
    ConfidenceLevel,
)


class PriorityEngine:
    def __init__(self, upstream: UpstreamEngineOutputs, weights: Optional[ScoringWeights] = None) -> None:
        self.upstream = upstream
        self.weights = weights or ScoringWeights()

    def score_and_rank(
        self,
        candidates: List[RecommendationCandidate],
        impact_estimates: Dict[str, ImpactEstimate],
    ) -> List[Recommendation]:
        ranked: List[Recommendation] = []
        for candidate in candidates:
            impact = impact_estimates.get(candidate.recommendation_id)
            if impact is None:
                continue
            priority_score = self._score(candidate, impact)
            priority_score = max(0.0, min(1.0, priority_score))
            ranked.append(
                Recommendation(
                    recommendation_id=candidate.recommendation_id,
                    title=candidate.title,
                    description=candidate.description,
                    action_type=candidate.action_type,
                    priority_score=priority_score,
                    confidence=impact.confidence,
                    estimated_hours_recovered=impact.estimated_hours_recovered,
                    estimated_delay_reduction_days=impact.estimated_delay_reduction_days,
                    estimated_risk_reduction=impact.estimated_risk_reduction,
                    affected_item_ids=candidate.affected_item_ids,
                    affected_resource_ids=candidate.affected_resource_ids,
                    affected_sprint_ids=candidate.affected_sprint_ids,
                    affected_blocker_ids=candidate.affected_blocker_ids,
                    root_cause_signal_id=candidate.root_cause_signal_id,
                    supporting_signal_ids=candidate.supporting_signal_ids,
                    impact_evidence=impact.evidence,
                    metadata={
                        "simulation_params": candidate.simulation_params,
                        "feasibility_checks": candidate.feasibility_checks,
                        "historical_pattern": self._historical_pattern_from_candidate(candidate),
                    },
                )
            )

        ranked.sort(key=lambda item: (-item.priority_score, item.recommendation_id))
        return ranked

    def _historical_pattern_from_candidate(self, candidate: RecommendationCandidate) -> Dict[str, Any] | None:
        signal_meta = candidate.simulation_params.get("historical_pattern")
        if isinstance(signal_meta, dict):
            return signal_meta
        return None

    def _score(self, candidate: RecommendationCandidate, impact: ImpactEstimate) -> float:
        blocker_factor = 1.0 if RecommendationAction.RESOLVE_BLOCKER == candidate.action_type else 0.0
        schedule_factor = 1.0 if impact.estimated_delay_reduction_days > 0.0 else 0.0
        cp_factor = 1.0 if candidate.action_type in {RecommendationAction.ADVANCE_ITEM_TO_EARLIER_SPRINT, RecommendationAction.PARALLELIZE_ITEMS} else 0.0
        capacity_factor = 1.0 if candidate.action_type in {RecommendationAction.REASSIGN_ITEM, RecommendationAction.ADD_RESOURCE_SKILL} else 0.0
        risk_factor = min(1.0, impact.estimated_risk_reduction)

        hours_factor = min(1.0, impact.estimated_hours_recovered / max(1.0, self.upstream.forecast.remaining_effort_hours))

        overdue_days = candidate.simulation_params.get("overdue_days", 0) if candidate.simulation_params else 0
        sprint_duration = 14
        urgency_multiplier = 1.0 + min(0.5, overdue_days / sprint_duration)

        cascade_count = len(candidate.affected_item_ids) if candidate.action_type in {
            RecommendationAction.RESOLVE_BLOCKER,
            RecommendationAction.ADVANCE_ITEM_TO_EARLIER_SPRINT,
            RecommendationAction.REMOVE_DEPENDENCY_BOTTLENECK,
        } else 0
        cascade_multiplier = 1.0 + min(0.3, cascade_count * 0.05)

        confidence_multiplier = {"HIGH": 1.1, "MEDIUM": 1.0, "LOW": 0.85}.get(
            impact.confidence.value if hasattr(impact.confidence, "value") else str(impact.confidence), 1.0
        )

        base_score = (
            self.weights.w_risk * risk_factor
            + self.weights.w_schedule * schedule_factor
            + self.weights.w_blocker * blocker_factor
            + self.weights.w_cp * cp_factor
            + self.weights.w_capacity * capacity_factor
            + 0.1 * hours_factor
        )

        if candidate.action_type in {
            RecommendationAction.CROSS_TRAIN_BACKUP,
            RecommendationAction.SWARM_ITEM,
        }:
            base_score = max(base_score, 0.65)

        return base_score * urgency_multiplier * cascade_multiplier * confidence_multiplier
