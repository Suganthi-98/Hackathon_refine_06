"""
API Models for Recovery Plans (Phase 6)

Pydantic models for serializing recovery plans to/from JSON.
Follows the same pattern as models_phase3.py.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecoveryPlanScoreResponse(BaseModel):
    """Score metrics for a recovery plan."""
    
    deadline_probability: float = Field(..., description="Probability of hitting deadline (0.0-1.0)")
    expected_delay_days: float = Field(..., description="Expected project delay in days")
    overall_risk_score: float = Field(..., description="Overall risk score (0.0-1.0)")
    actions_required: int = Field(..., description="Number of actions in the plan")
    execution_complexity: str = Field(..., description="Complexity level: Low, Medium, or High")
    composite_score: float = Field(..., description="Weighted composite score (0.0-1.0)")


class TradeOffResponse(BaseModel):
    """Trade-off for a recovery plan."""
    
    description: str = Field(..., description="Description of the trade-off")
    severity: str = Field(..., description="Severity level: low, medium, or high")


class RecoveryPlanExplanationResponse(BaseModel):
    """Narrative explanation for a recovery plan."""
    
    plan_id: str = Field(..., description="Unique plan identifier")
    why_recommended: List[str] = Field(..., description="Bullet-point reasons why this plan is strong")
    comparison_to_alternatives: List[str] = Field(..., description="How this plan compares to alternatives")
    trade_offs: List[TradeOffResponse] = Field(..., description="Trade-offs and risks")
    narrative_summary: str = Field(..., description="One-paragraph summary of the plan")


class RecommendationInPlanResponse(BaseModel):
    """Simplified recommendation for display in plan action list."""
    
    recommendation_id: str = Field(..., description="Unique recommendation ID")
    action_type: str = Field(..., description="Type of action (e.g., resolve_blocker, reassign_item)")
    title: str = Field(..., description="Short title of the recommendation")
    description: str = Field(..., description="Detailed description")
    priority_score: float = Field(..., description="Priority score (0.0-1.0)")
    confidence: str = Field(..., description="Confidence level: HIGH, MEDIUM, LOW")
    estimated_delay_reduction_days: float = Field(..., description="Expected delay reduction in days")
    affected_item_ids: List[str] = Field(default_factory=list, description="Work items affected")
    affected_resource_ids: List[str] = Field(default_factory=list, description="Resources affected")


class RecoveryPlanResponse(BaseModel):
    """Complete recovery plan response."""
    
    plan_id: str = Field(..., description="Unique plan identifier")
    archetype: str = Field(..., description="Plan archetype: SAFE, AGGRESSIVE, or MINIMAL_DISRUPTION")
    label: str = Field(..., description="User-facing label: Recommended, Alternative, etc.")
    actions: List[RecommendationInPlanResponse] = Field(..., description="Actions in the plan")
    score: RecoveryPlanScoreResponse = Field(..., description="Scored metrics")
    explanation: RecoveryPlanExplanationResponse = Field(..., description="Narrative explanation")
    revised_sprint_plan: List[Dict[str, Any]] = Field(default_factory=list, description="Updated sprint plan")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="When plan was generated")


class RecoveryPlansListResponse(BaseModel):
    """List of all recovery plans."""
    
    plans: List[RecoveryPlanResponse] = Field(..., description="All generated plans, ranked by composite_score")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="When plans were generated")
    summary: str = Field(..., description="One-line summary of top plan")


class ApplyPlanRequest(BaseModel):
    """Request to apply a recovery plan to the project."""
    
    plan_id: str = Field(..., description="ID of the plan to apply")
    session_id: str = Field(..., description="Session ID for the project")


class ApplyPlanResponse(BaseModel):
    """Response after applying a recovery plan."""
    
    success: bool = Field(..., description="Whether plan was successfully applied")
    applied_plan_id: str = Field(..., description="ID of the applied plan")
    message: str = Field(..., description="Success or error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When plan was applied")


class RecoveryNarrativeSectionResponse(BaseModel):
    """One rendered section of the Recovery Advisor narrative."""
    heading: str = Field(..., description="Section heading")
    body: str = Field(..., description="Rendered narrative text")


class RecoveryAdvisorNarrativeResponse(BaseModel):
    """
    AI Recovery Advisor narrative — five sections explaining the full
    recovery story: how the project got here, why the recommended plan fits,
    what alternatives were considered, expected outcomes, and PM guidance.
    """
    situation_framing: RecoveryNarrativeSectionResponse
    strategy_rationale: RecoveryNarrativeSectionResponse
    alternatives_considered: RecoveryNarrativeSectionResponse
    expected_outcomes: RecoveryNarrativeSectionResponse
    pm_guidance: RecoveryNarrativeSectionResponse


class RecoveryPlansListResponse(BaseModel):  # type: ignore[no-redef]
    """List of all recovery plans, now including the AI advisor narrative."""
    plans: List[RecoveryPlanResponse] = Field(..., description="All generated plans, ranked by composite_score")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    summary: str = Field(..., description="One-line summary of top plan")
    advisor_narrative: Optional[RecoveryAdvisorNarrativeResponse] = Field(
        default=None,
        description="AI Recovery Advisor narrative. None when AI is disabled or in fallback.",
    )
    advisor_status: str = Field(
        default="disabled",
        description="'ok' | 'partial' | 'fallback' | 'disabled'",
    )
