"""
Forecast Engine (deterministic)

Produces a single-point forecast based on remaining effort, current velocity,
critical-path sequencing, spillover, and blocker impacts. No Monte Carlo,
no probabilities.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from app.domain.models import ProjectState, SprintStatus
from app.engines.metrics_engine import ProjectMetrics
from app.engines.critical_path_engine import CriticalPathResult
from app.engines.spillover_engine import SpilloverAnalysis
from app.api.models_phase3 import (
    ForecastResult,
    ForecastDelayBreakdown,
    ForecastScheduleDiagnostics,
    ForecastEffortBreakdown,
    ForecastConfidence,
    ForecastDriver,
    ForecastEvidence,
    ForecastAssumptions,
    ForecastExplanation,
)


class ForecastEngine:
    """Deterministic forecast engine.

    High-level approach:
    - Use remaining effort (sum of remaining_effort_hrs) as the work to schedule.
    - Adjust for dependency sequencing by ensuring remaining work is at least
      the critical path duration (hours) — this captures serialisation delays.
    - Add spillover-induced extra work (predicted_spillover_count * avg_item_effort).
    - Project velocity = historical avg velocity per sprint adjusted for active
      blocker impact (velocity reduction factor). No randomness.
    - Compute remaining_sprints = adjusted_remaining_effort / projected_velocity
      and convert to days using project sprint length.
    - Return a single expected finish date (now + days) and derived fields.
    """

    def __init__(
        self,
        project_state: ProjectState,
        metrics: ProjectMetrics,
        cp_result: CriticalPathResult,
        spillover: Optional[SpilloverAnalysis] = None,
    ):
        self.project_state = project_state
        self.metrics = metrics
        self.cp_result = cp_result
        self.spillover = spillover

    def calculate(self) -> ForecastResult:
        """Calculate deterministic forecast and return ForecastResult."""

        remaining_effort = float(self.metrics.remaining_effort_hours)
        cp_remaining_hours = float(getattr(self.cp_result, "critical_path_remaining_hours", 0.0) or 0.0)
        adjusted_remaining = max(remaining_effort, cp_remaining_hours)

        avg_item_effort = float(getattr(self.metrics, "average_item_effort", 20.0) or 20.0)
        spillover_hours = 0.0
        predicted_spillover_items = 0.0
        if self.spillover:
            try:
                total_spill = sum(self.spillover.predicted_spillover_by_sprint.values())
                predicted_spillover_items = float(total_spill)
                spillover_hours = float(total_spill) * avg_item_effort
            except Exception:
                predicted_spillover_items = 0.0
                spillover_hours = 0.0

        base_velocity = float(
            self.metrics.actual_avg_velocity
            or getattr(self.metrics, "effective_project_velocity", 0.0)
            or self.metrics.planned_total_velocity
            or 1.0
        )
        # NOTE: actual_avg_velocity (empirical, from historical SprintActual records) is
        # preferred over effective_project_velocity (theoretical, capacity-based estimate)
        # so this deterministic forecast agrees with MonteCarloEngine, which uses the same
        # preference order. Do not swap this back without also updating monte_carlo_engine.py --
        # the two engines disagreeing on which velocity to trust is what caused expected_delay_days
        # and on_time_probability to contradict each other for projects where the two velocity
        # figures diverge significantly (e.g. team is not operating near theoretical capacity).
        # effective_project_velocity is still used as a fallback for brand-new projects with no
        # completed sprints yet, where there is no historical data to compute actual_avg_velocity from.
        blocker_impact = float(getattr(self.metrics, "estimated_blocker_velocity_impact", 0.0) or 0.0)

        sprint_days = float(self.project_state.project_info.sprint_duration_days or 14)
        # If future sprints have zero planned velocity (workbook leaves them empty),
        # substitute historical average velocity for those sprints when estimating
        # the effective remaining sprint capacity. Apply only to in-progress and
        # not-started sprints (do not alter completed sprints).
        try:
            remaining_sprints = [
                s for s in self.project_state.sprints
                if s.status in (SprintStatus.IN_PROGRESS, SprintStatus.NOT_STARTED)
            ]
            if remaining_sprints:
                per_sprint_caps = [
                    (s.planned_velocity_hrs if getattr(s, 'planned_velocity_hrs', 0.0) and s.planned_velocity_hrs > 0 else float(self.metrics.actual_avg_velocity or 0.0))
                    for s in remaining_sprints
                ]
                avg_remaining_planned_velocity = sum(per_sprint_caps) / len(per_sprint_caps) if per_sprint_caps else 0.0
                # Prefer a non-zero average remaining planned velocity when computing base velocity
                if avg_remaining_planned_velocity > 0:
                    base_velocity = max(base_velocity, avg_remaining_planned_velocity)
        except Exception:
            pass
        # Compute velocity_without_spillover AFTER the substitution above settles base_velocity --
        # otherwise this and base_schedule_days below are derived from two different base_velocity
        # values, and their difference gets mislabeled as blocker-caused delay when it is really
        # just the sprint-velocity-substitution effect (reproducible even when blocker_impact == 0).
        velocity_without_spillover = max(base_velocity * (1.0 - blocker_impact), base_velocity * 0.25)
        base_schedule_days = (adjusted_remaining / base_velocity) * sprint_days if base_velocity > 0 else 0.0
        days_without_spillover = (
            (adjusted_remaining / velocity_without_spillover) * sprint_days
            if velocity_without_spillover > 0 else 0.0
        )

        spillover_penalty_days = (
            (spillover_hours / velocity_without_spillover) * sprint_days
            if velocity_without_spillover > 0 else 0.0
        )
        spillover_fraction = (
            min(0.4, spillover_penalty_days / max(1.0, days_without_spillover))
            if days_without_spillover > 0 else 0.0
        )
        projected_velocity = max(
            base_velocity * (1.0 - blocker_impact) * (1.0 - spillover_fraction * 0.5),
            base_velocity * 0.25,
        )
        remaining_days_blocker_loss = max(0.0, days_without_spillover - base_schedule_days)
        raw_remaining_days = (adjusted_remaining / projected_velocity) * sprint_days if projected_velocity > 0 else 0.0
        spillover_delay_days = max(0.0, raw_remaining_days - days_without_spillover)
        remaining_days_base_work = base_schedule_days
        remaining_days_total = base_schedule_days + remaining_days_blocker_loss + spillover_delay_days
        velocity_floor = base_velocity * 0.25
        velocity_floor_saturated_by_blockers = bool(velocity_without_spillover <= velocity_floor + 1e-6 and spillover_hours > 0.0)

        cp_remaining_days = 0.0
        if cp_remaining_hours > remaining_effort and base_velocity > 0:
            cp_remaining_days = ((cp_remaining_hours - remaining_effort) / base_velocity) * sprint_days
        spillover_days_diag = spillover_delay_days
        blocker_days_diag = remaining_days_blocker_loss
        diagnostic_total = base_schedule_days + cp_remaining_days + spillover_days_diag + blocker_days_diag

        project_start = self.project_state.project_info.forecast_anchor_date()
        days_elapsed = self._calculate_schedule_elapsed_days(sprint_days)
        expected_finish = project_start + timedelta(days=days_elapsed + remaining_days_total)

        target_end_date = self.project_state.project_info.target_end_date
        planned_window_days = float((target_end_date - project_start).days)
        expected_delay_raw = days_elapsed + remaining_days_total - planned_window_days
        expected_delay_days = float(round(expected_delay_raw, 2))
        on_track = expected_delay_days <= 0

        total_effort = float(getattr(self.metrics, "total_effort_hours", 0.0) or 0.0)
        completion_pct = (
            max(0.0, min(1.0, (total_effort - remaining_effort) / total_effort))
            if total_effort > 0 else 0.0
        )

        scope_growth_hours = float(
            sum(max(0.0, wi.current_estimate_hrs - wi.estimated_effort_hrs) for wi in self.project_state.work_items)
        )
        scope_growth_percent = float(round((scope_growth_hours / total_effort * 100.0) if total_effort > 0 else 0.0, 2))
        projected_velocity_per_day = float(projected_velocity / sprint_days if sprint_days > 0 else 0.0)
        scope_impact_days = float(round(scope_growth_hours / projected_velocity_per_day, 2)) if projected_velocity_per_day > 0 else 0.0

        blocker_penalty_hours_calc = (
            remaining_days_blocker_loss * (velocity_without_spillover / sprint_days)
            if sprint_days > 0 else 0.0
        )
        blocker_penalty_hours_final = min(float(adjusted_remaining), max(0.0, blocker_penalty_hours_calc))

        scope_growth_message = (
            f"Scope growth contributes {scope_impact_days:.1f} days to the forecast."
            if scope_growth_hours > 0 else "Scope growth is not material to the forecast."
        )
        if velocity_floor_saturated_by_blockers:
            spillover_message = (
                f"Spillover is present, but blockers already reduce velocity to the floor level."
            )
        elif spillover_delay_days > 0:
            spillover_message = f"Spillover adds approximately {spillover_delay_days:.1f} days to the forecast."
        else:
            spillover_message = "No material spillover delay is projected."

        delay_breakdown = ForecastDelayBreakdown(
            planned_window_days=float(round(planned_window_days, 2)),
            days_elapsed=float(round(days_elapsed, 2)),
            remaining_days_total=float(round(remaining_days_total, 2)),
            remaining_days_base_work=float(round(remaining_days_base_work, 2)),
            remaining_days_spillover=float(round(spillover_delay_days, 2)),
            remaining_days_blocker_loss=float(round(remaining_days_blocker_loss, 2)),
            expected_delay_days=float(round(days_elapsed + remaining_days_total - planned_window_days, 2)),
        )
        schedule_diagnostics = ForecastScheduleDiagnostics(
            is_additive=False,
            base_schedule_days=float(round(base_schedule_days, 2)),
            spillover_days=float(round(spillover_days_diag, 2)),
            blocker_days=float(round(blocker_days_diag, 2)),
            critical_path_days=float(round(cp_remaining_days, 2)),
            diagnostic_total_days=float(round(diagnostic_total, 2)),
            velocity_floor_saturated_by_blockers=velocity_floor_saturated_by_blockers,
            spillover_message=spillover_message,
        )
        effort_breakdown = ForecastEffortBreakdown(
            raw_remaining_effort_hours=float(round(remaining_effort, 2)),
            critical_path_remaining_hours=float(round(cp_remaining_hours, 2)),
            spillover_penalty_hours=float(round(spillover_hours, 2)),
            blocker_penalty_hours=float(round(blocker_penalty_hours_final, 2)),
            forecast_adjusted_effort_hours=float(round(adjusted_remaining, 2)),
        )

        confidence = self._build_confidence()
        forecast_drivers = self._build_forecast_drivers(
            scope_impact_days=scope_impact_days,
            remaining_days_blocker_loss=remaining_days_blocker_loss,
            cp_remaining_days=cp_remaining_days,
            spillover_delay_days=spillover_delay_days,
            remaining_days_base_work=remaining_days_base_work,
        )
        forecast_evidence = self._build_forecast_evidence()
        assumptions = self._build_assumptions()
        explanation = self._build_explanation(expected_delay_days, confidence, forecast_drivers)

        return ForecastResult(
            target_end_date=target_end_date,
            expected_finish_date=expected_finish,
            expected_delay_days=float(round(expected_delay_days, 2)),
            remaining_effort_hours=adjusted_remaining,
            completion_percentage=completion_pct,
            projected_velocity=projected_velocity,
            on_track=on_track,
            raw_remaining_effort_hours=remaining_effort,
            critical_path_remaining_hours=cp_remaining_hours,
            predicted_spillover_items=predicted_spillover_items,
            spillover_delay_days=float(round(spillover_delay_days, 2)),
            spillover_penalty_hours=spillover_hours,
            blocker_penalty_hours=float(round(blocker_penalty_hours_final, 2)),
            forecast_adjusted_effort_hours=adjusted_remaining,
            scope_growth_hours=float(round(scope_growth_hours, 2)),
            scope_growth_percent=scope_growth_percent,
            scope_impact_days=scope_impact_days,
            scope_growth_message=scope_growth_message,
            delay_breakdown=delay_breakdown,
            schedule_diagnostics=schedule_diagnostics,
            effort_breakdown=effort_breakdown,
            confidence=confidence,
            forecast_drivers=forecast_drivers,
            forecast_evidence=forecast_evidence,
            forecast_assumptions=assumptions,
            forecast_explanation=explanation,
            forecast_vs_montecarlo_note=(
                "The deterministic forecast applies worst-credible-case assumptions: "
                "full blocker velocity reduction and a capped velocity penalty from "
                "predicted spillover (spillover reduces effective throughput rather "
                "than adding a separate block of schedule time). "
                "Monte Carlo samples the full uncertainty range: spillover impact "
                "between 0-100% of predicted and blocker impact between 0% and the "
                "maximum estimated value. "
                "The on-time probability reflects how often optimistic scenarios occur. "
                "The delay figure reflects the pessimistic single-point estimate. "
                "Both are correct — they answer different questions."
            ),
        )

    def _build_confidence(self) -> ForecastConfidence:
        """Derive a deterministic forecast confidence score from measurable indicators."""
        velocity_stability = max(0.0, min(1.0, float(self.metrics.velocity_metrics.velocity_stability_score or 0.0)))
        planning_accuracy = max(0.0, min(1.0, float(self.metrics.planning_metrics.planning_accuracy_score or 0.0)))
        estimation_variance = max(0.0, min(1.0, 1.0 - min(1.0, abs(self.metrics.velocity_variance) / max(self.metrics.actual_avg_velocity, 1.0))))
        carryover_consistency = max(0.0, min(1.0, 1.0 - min(1.0, self.metrics.historical_carryover_rate)))
        blocker_volatility = max(0.0, min(1.0, 1.0 - min(1.0, self.metrics.active_blocker_count / max(self.metrics.total_items, 1))))
        dependency_density = max(0.0, min(1.0, 1.0 - min(1.0, self.metrics.dependency_count / max(self.metrics.total_items, 1))))
        historical_stability = max(0.0, min(1.0, float(self.metrics.velocity_metrics.velocity_stability_score or 0.0)))

        confidence_score = (
            0.25 * velocity_stability
            + 0.2 * planning_accuracy
            + 0.15 * estimation_variance
            + 0.15 * carryover_consistency
            + 0.1 * blocker_volatility
            + 0.1 * dependency_density
            + 0.05 * historical_stability
        )
        confidence_score = max(0.0, min(1.0, confidence_score))
        if confidence_score >= 0.75:
            confidence_level = "HIGH"
            reason = "Historical delivery signals are stable and planning accuracy is strong."
        elif confidence_score >= 0.45:
            confidence_level = "MEDIUM"
            reason = "Forecast confidence is moderate because some planning and execution signals are mixed."
        else:
            confidence_level = "LOW"
            reason = "The forecast is highly sensitive to blockers, carryover, and unstable velocity."

        return ForecastConfidence(
            confidence_score=float(round(confidence_score, 4)),
            confidence_level=confidence_level,
            confidence_reason=reason,
            confidence_inputs={
                "velocity_stability": round(velocity_stability, 4),
                "planning_accuracy": round(planning_accuracy, 4),
                "estimation_variance": round(estimation_variance, 4),
                "carryover_consistency": round(carryover_consistency, 4),
                "blocker_volatility": round(blocker_volatility, 4),
                "dependency_density": round(dependency_density, 4),
                "historical_stability": round(historical_stability, 4),
            },
        )

    def _build_forecast_drivers(
        self,
        scope_impact_days: float,
        remaining_days_blocker_loss: float,
        cp_remaining_days: float,
        spillover_delay_days: float,
        remaining_days_base_work: float,
    ) -> List[ForecastDriver]:
        """Build ranked drivers from deterministic forecast components."""
        drivers: List[ForecastDriver] = []
        if scope_impact_days > 0:
            drivers.append(ForecastDriver(
                name="Scope Growth",
                impact=float(round(scope_impact_days, 2)),
                reason="Current estimates exceed the baseline estimate for one or more work items.",
                supporting_metrics={"scope_growth_hours": float(round(self._scope_growth_hours(), 2))},
            ))
        if remaining_days_blocker_loss > 0:
            drivers.append(ForecastDriver(
                name="Blockers",
                impact=float(round(remaining_days_blocker_loss, 2)),
                reason="Blockers reduce effective throughput relative to the base scheduled velocity.",
                supporting_metrics={"estimated_blocker_velocity_impact": float(round(self.metrics.estimated_blocker_velocity_impact, 4))},
            ))
        if cp_remaining_days > 0:
            drivers.append(ForecastDriver(
                name="Critical Path",
                impact=float(round(cp_remaining_days, 2)),
                reason="Dependency sequencing requires serial work that extends the schedule beyond raw remaining effort.",
                supporting_metrics={"critical_path_remaining_hours": float(round(self.cp_result.critical_path_remaining_hours, 2))},
            ))
        if spillover_delay_days > 0:
            drivers.append(ForecastDriver(
                name="Carryover",
                impact=float(round(spillover_delay_days, 2)),
                reason="Predicted spillover erodes effective velocity and adds schedule delay.",
                supporting_metrics={"predicted_spillover_items": float(round(self._predicted_spillover_items(), 2))},
            ))
        if remaining_days_base_work > 0:
            drivers.append(ForecastDriver(
                name="Base Workload",
                impact=float(round(remaining_days_base_work, 2)),
                reason="Remaining effort still requires schedule time even before secondary effects are applied.",
                supporting_metrics={"remaining_effort_hours": float(round(self.metrics.remaining_effort_hours, 2))},
            ))
        return sorted(drivers, key=lambda d: d.impact, reverse=True)

    def _build_forecast_evidence(self) -> List[ForecastEvidence]:
        """Expose structured evidence values already available through ProjectMetrics."""
        return [
            ForecastEvidence(
                name="Effective project velocity",
                value=getattr(self.metrics, "effective_project_velocity", self.metrics.actual_avg_velocity),
                unit="hours/sprint",
                source="MetricsEngine",
            ),
            ForecastEvidence(name="Historical velocity", value=self.metrics.actual_avg_velocity, unit="hours/sprint", source="MetricsEngine"),
            ForecastEvidence(name="Remaining effort", value=self.metrics.remaining_effort_hours, unit="hours", source="MetricsEngine"),
            ForecastEvidence(name="Critical path remaining effort", value=self.cp_result.critical_path_remaining_hours, unit="hours", source="CriticalPathEngine"),
            ForecastEvidence(name="Carryover history", value=self.metrics.historical_total_carryover_items, unit="items", source="MetricsEngine"),
            ForecastEvidence(name="Planning accuracy", value=self.metrics.planning_metrics.planning_accuracy_score, unit="score", source="MetricsEngine"),
            ForecastEvidence(name="Dependency density", value=self.metrics.dependency_metrics.critical_dependency_density, unit="ratio", source="MetricsEngine"),
            ForecastEvidence(name="Blocker counts", value=self.metrics.active_blocker_count, unit="count", source="MetricsEngine"),
            ForecastEvidence(name="Resource utilization", value=self.metrics.avg_allocation_pct, unit="ratio", source="MetricsEngine"),
        ]

    def _build_assumptions(self) -> ForecastAssumptions:
        """Document forecast assumptions in machine-readable form."""
        return ForecastAssumptions(
            velocity_calculation_method="projected_velocity = effective_project_velocity * (1 - blocker_impact) * (1 - spillover_fraction * 0.5), floored at 25% of effective project velocity",
            blocker_adjustment_method="blocker_impact is applied as a multiplicative velocity reduction factor from ProjectMetrics",
            spillover_adjustment_method="predicted spillover is converted to equivalent hours and reduces effective throughput rather than adding a separate additive delay bucket",
            critical_path_handling="critical_path_remaining_hours is used as a lower bound for remaining work so serial dependency effort cannot be under-counted",
            timeline_anchoring="forecast uses sprint-based elapsed days and project start anchor date rather than current wall-clock time",
            capacity_assumptions={"velocity_floor_ratio": 0.25, "spillover_damping_ratio": 0.5},
        )

    def _build_explanation(self, expected_delay_days: float, confidence: ForecastConfidence, forecast_drivers: List[ForecastDriver]) -> ForecastExplanation:
        """Create structured explanation payload for downstream consumers."""
        if expected_delay_days <= 0:
            delay_signal = "on track"
            summary = "The deterministic schedule remains within the planned window."
        elif expected_delay_days <= 7:
            delay_signal = "slightly late"
            summary = "The deterministic schedule is projected to slip slightly beyond the target window."
        else:
            delay_signal = "late"
            summary = "The deterministic schedule is projected to miss the target window materially."

        primary_driver = forecast_drivers[0].name if forecast_drivers else "Base Workload"
        return ForecastExplanation(
            summary=summary,
            primary_driver=primary_driver,
            driver_names=[driver.name for driver in forecast_drivers],
            confidence_note=confidence.confidence_reason,
            delay_signal=delay_signal,
        )

    def _scope_growth_hours(self) -> float:
        return float(sum(max(0.0, wi.current_estimate_hrs - wi.estimated_effort_hrs) for wi in self.project_state.work_items))

    def _predicted_spillover_items(self) -> float:
        if self.spillover is None:
            return 0.0
        try:
            return float(sum(self.spillover.predicted_spillover_by_sprint.values()))
        except Exception:
            return 0.0

    def _calculate_schedule_elapsed_days(self, sprint_days: float) -> float:
        """Estimate elapsed project time using sprint schedule dates only."""
        completed_sprints = sum(
            1
            for sprint in self.project_state.sprints
            if (
                sprint.status == SprintStatus.COMPLETED
                or (isinstance(sprint.status, str) and sprint.status == SprintStatus.COMPLETED.value)
            )
        )

        days_from_completed = completed_sprints * sprint_days

        current_sprint = next(
            (
                sprint
                for sprint in self.project_state.sprints
                if (
                    sprint.status == SprintStatus.IN_PROGRESS
                    or (isinstance(sprint.status, str) and sprint.status == SprintStatus.IN_PROGRESS.value)
                )
            ),
            None,
        )
        if not current_sprint:
            return days_from_completed

        sprint_window_days = max(
            0.0,
            (current_sprint.end_date - current_sprint.start_date).total_seconds() / (24 * 3600),
        )
        return days_from_completed + min(sprint_window_days, sprint_days)