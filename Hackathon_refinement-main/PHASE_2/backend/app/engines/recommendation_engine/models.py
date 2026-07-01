from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from app.engines.critical_path_engine import CriticalPathResult
from app.engines.dependency_engine import DependencyDAG
from app.engines.forecast_engine import ForecastResult
from app.engines.impact_scoring_engine import RiskScores
from app.engines.metrics_engine import ProjectMetrics
from app.engines.monte_carlo_engine import MonteCarloResult
from app.engines.risk_engine import RiskResult
from app.engines.spillover_engine import SpilloverAnalysis


class SignalCategory(str, Enum):
    BLOCKER = "blocker"
    CAPACITY = "capacity"
    SPRINT = "sprint"
    CRITICAL_PATH = "critical_path"
    SCHEDULE = "schedule"
    RISK = "risk"
    SPILLOVER = "spillover"
    DEPENDENCY = "dependency"
    ESTIMATION_RELIABILITY = "estimation_reliability"
    SPOF = "single_point_of_failure"
    RECURRING_BLOCKER = "recurring_blocker_category"
    REWORK_LOOP = "rework_loop"
    RAMP_UP = "ramp_up_discount"
    RESEQUENCING = "resequencing_opportunity"
    SWARM_TRADEOFF = "swarm_tradeoff"


class SignalSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationAction(str, Enum):
    RESOLVE_BLOCKER = "resolve_blocker"
    REASSIGN_ITEM = "reassign_item"
    SPLIT_ITEM = "split_item"
    ADVANCE_ITEM_TO_EARLIER_SPRINT = "advance_item_to_earlier_sprint"
    PARALLELIZE_ITEMS = "parallelize_items"
    REBALANCE_SPRINT_LOAD = "rebalance_sprint_load"
    REMOVE_DEPENDENCY_BOTTLENECK = "remove_dependency_bottleneck"
    ADD_RESOURCE_SKILL = "add_resource_skill"
    REBASELINE_ESTIMATE = "rebaseline_estimate"
    PAIR_REVIEWER = "pair_reviewer"
    ESCALATE_BLOCKER_EARLY = "escalate_blocker_early"
    FREEZE_SCOPE_REQUEST = "freeze_scope_request"
    PULL_FORWARD_ITEM = "pull_forward_item"
    SPLIT_AND_PAIR = "split_and_pair"
    ASSIGN_AS_SECOND_REVIEWER = "assign_as_second_reviewer"
    CROSS_TRAIN_BACKUP = "cross_train_backup"
    INSERT_REVIEW_GATE = "insert_review_gate"
    APPLY_RAMP_UP_DISCOUNT = "apply_ramp_up_discount"
    RESEQUENCE_NON_CRITICAL_ITEM = "resequence_non_critical_item"
    SWARM_ITEM = "swarm_item"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class SignalEvidence:
    source_engine: str
    metric_name: str
    metric_value: float
    threshold: float
    explanation: str


@dataclass(frozen=True)
class HistoricalPattern:
    pattern_type: str
    resource_id: str | None
    blocker_category: str | None
    sample_size: int
    metric_name: str
    metric_value: float
    historical_occurrences: List[str]
    confidence: str


@dataclass(frozen=True)
class OpportunitySignal:
    signal_id: str
    category: SignalCategory
    severity: SignalSeverity
    affected_item_ids: List[str]
    affected_resource_ids: List[str]
    affected_sprint_ids: List[str]
    affected_blocker_ids: List[str]
    evidence: List[SignalEvidence]
    context: Dict[str, Any]
    detected_at: str


@dataclass
class RecommendationCandidate:
    recommendation_id: str
    action_type: RecommendationAction
    title: str
    description: str
    affected_item_ids: List[str]
    affected_resource_ids: List[str]
    affected_sprint_ids: List[str]
    affected_blocker_ids: List[str]
    root_cause_signal_id: str
    supporting_signal_ids: List[str] = field(default_factory=list)
    simulation_params: Dict[str, Any] = field(default_factory=dict)
    feasibility_checks: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ImpactEstimate:
    estimated_hours_recovered: float
    estimated_delay_reduction_days: float
    estimated_risk_reduction: float
    confidence: ConfidenceLevel
    evidence: List[SignalEvidence]
    calculation_notes: str


@dataclass
class Recommendation:
    recommendation_id: str
    title: str
    description: str
    action_type: RecommendationAction
    priority_score: float
    confidence: ConfidenceLevel
    estimated_hours_recovered: float
    estimated_delay_reduction_days: float
    estimated_risk_reduction: float
    affected_item_ids: List[str]
    affected_resource_ids: List[str]
    affected_sprint_ids: List[str]
    affected_blocker_ids: List[str]
    root_cause_signal_id: str
    supporting_signal_ids: List[str] = field(default_factory=list)
    impact_evidence: List[SignalEvidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "action_type": self.action_type.value,
            "title": self.title,
            "description": self.description,
            "affected_item_ids": self.affected_item_ids,
            "affected_resource_ids": self.affected_resource_ids,
            "affected_sprint_ids": self.affected_sprint_ids,
            "affected_blocker_ids": self.affected_blocker_ids,
            "root_cause_signal_id": self.root_cause_signal_id,
            "supporting_signal_ids": self.supporting_signal_ids,
            "priority_score": round(self.priority_score, 4),
            "confidence": self.confidence.value,
            "estimated_hours_recovered": round(self.estimated_hours_recovered, 2),
            "estimated_delay_reduction_days": round(self.estimated_delay_reduction_days, 2),
            "estimated_risk_reduction": round(self.estimated_risk_reduction, 2),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TradeOff:
    description: str
    severity: str


@dataclass(frozen=True)
class RecommendationValidation:
    recommendation_id: str
    why_selected: List[str]
    why_better_than_alternatives: List[str]
    rejected_alternatives: List[str]
    delay_reduction_summary: str
    probability_improvement_summary: str
    confidence_label: ConfidenceLevel
    confidence_reasoning: str
    trade_offs: List[TradeOff]
    one_line_pitch: str


@dataclass(frozen=True)
class BaselineMetrics:
    on_time_probability: float
    expected_delay_days: float
    overall_risk_score: float
    schedule_risk: float
    resource_risk: float
    critical_path_hours: float


@dataclass(frozen=True)
class SimulatedMetrics:
    on_time_probability: float
    expected_delay_days: float
    overall_risk_score: float
    schedule_risk: float
    resource_risk: float
    critical_path_hours: float


@dataclass(frozen=True)
class SimulationResult:
    recommendation_ids: List[str]
    baseline_metrics: BaselineMetrics
    simulated_metrics: SimulatedMetrics
    delta_on_time_probability: float
    delta_expected_delay_days: float
    delta_spillover_risk: float
    delta_risk_score: float
    delta_projected_velocity: float
    seed_used: int
    is_positive_impact: bool
    summary: str


@dataclass
class ScoringWeights:
    w_risk: float = 0.30
    w_schedule: float = 0.25
    w_blocker: float = 0.25
    w_cp: float = 0.15
    w_capacity: float = 0.05

    def __post_init__(self) -> None:
        total = self.w_risk + self.w_schedule + self.w_blocker + self.w_cp + self.w_capacity
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"ScoringWeights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class UpstreamEngineOutputs:
    metrics: ProjectMetrics
    dag: DependencyDAG
    cp_result: CriticalPathResult
    spillover: SpilloverAnalysis
    forecast: ForecastResult
    monte_carlo: MonteCarloResult
    impact_scores: RiskScores
    risk_result: RiskResult


def stable_id(action_type: str, target_ids: List[str]) -> str:
    key = f"{action_type}:{':'.join(sorted(target_ids))}"
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def historical_pattern_payload(pattern: HistoricalPattern | None) -> Dict[str, Any] | None:
    if pattern is None:
        return None
    return {
        "pattern_type": pattern.pattern_type,
        "resource_id": pattern.resource_id,
        "blocker_category": pattern.blocker_category,
        "sample_size": pattern.sample_size,
        "metric_name": pattern.metric_name,
        "metric_value": pattern.metric_value,
        "historical_occurrences": pattern.historical_occurrences,
        "confidence": pattern.confidence,
    }


def signal_id(category: SignalCategory, target_ids: List[str]) -> str:
    key = f"sig:{category.value}:{':'.join(sorted(target_ids))}"
    return hashlib.sha1(key.encode()).hexdigest()[:10]
