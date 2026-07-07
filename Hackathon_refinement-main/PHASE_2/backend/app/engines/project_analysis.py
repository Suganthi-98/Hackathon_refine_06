"""
ProjectAnalysis — the single computed truth for a session.

Problem
-------
Before this module existed, every API route independently re-ran the full
engine pipeline (Metrics → DependencyGraph → CriticalPath → Spillover →
Forecast → MonteCarlo → ImpactScoring → Risk) on every request.  This meant:

  • Each engine had its own private "truth" — numbers could disagree between
    the /forecast and /recommendations endpoints because they ran separate
    Monte Carlo simulations with different random seeds.
  • ~500 ms of CPU was wasted on every request re-deriving the same values
    from the same immutable ProjectState.
  • The architecture had drifted from:

        Workbook → ProjectState → single truth → all consumers

    to:

        Workbook → ProjectState
                      ↙      ↓      ↘
             Metrics  Forecast  Recommendations   (each with their own truth)

Solution
--------
``ProjectAnalysis`` wraps one ``ProjectState`` and runs every engine exactly
once.  The result is cached on the ``Session`` object so the whole lifetime
of a session always reads from the same numbers.

Usage (in routes)
-----------------
    from app.engines.project_analysis import ProjectAnalysis
    from app.storage import store

    analysis = store.get_analysis(session_id)   # cached after first call
    if not analysis:
        raise HTTPException(404, ...)

    # All fields are computed, typed, and consistent:
    analysis.metrics          → ProjectMetrics
    analysis.dag              → DependencyDAG
    analysis.cp_result        → CriticalPathResult
    analysis.spillover        → SpilloverAnalysis
    analysis.forecast         → ForecastResult
    analysis.monte_carlo      → MonteCarloResult
    analysis.impact_scores    → RiskScores
    analysis.risk_result      → RiskResult

Rebuild
-------
If ``ProjectState`` is mutated (scope change, descope), call
``store.invalidate_analysis(session_id)`` so the next request recomputes
from the new state rather than returning stale cached values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.domain.models import ProjectState
from app.engines.critical_path_engine import CriticalPathEngine, CriticalPathResult
from app.engines.dependency_engine import DependencyDAG, DependencyGraphEngine
from app.engines.forecast_engine import ForecastEngine, ForecastResult
from app.engines.impact_scoring_engine import ImpactScoringEngine, RiskScores
from app.engines.metrics_engine import MetricsEngine, ProjectMetrics
from app.engines.monte_carlo_engine import MonteCarloEngine, MonteCarloResult
from app.engines.risk_engine import RiskEngine, RiskResult
from app.engines.spillover_engine import SpilloverAnalysis, SpilloverAnalysisEngine

logger = logging.getLogger(__name__)

# Deterministic seed — must match the value used everywhere else so that
# on_time_probability from MonteCarlo agrees with expected_delay_days from Forecast.
_MONTE_CARLO_SEED: int = 42
_DEFAULT_SIMULATION_COUNT: int = 1000


@dataclass(frozen=True)
class ProjectAnalysis:
    """
    Immutable bag of all engine outputs for one ProjectState.

    Frozen so that no route can accidentally mutate shared analysis state.
    Routes that need a "what-if" clone should work on a deep-copied
    ProjectState and call ProjectAnalysis.build() on that copy.
    """

    project_state: ProjectState

    # ── Engine outputs (ordered by computation dependency) ──────────────────
    metrics: ProjectMetrics
    dag: DependencyDAG
    cp_result: CriticalPathResult
    spillover: SpilloverAnalysis
    forecast: ForecastResult
    monte_carlo: MonteCarloResult
    impact_scores: RiskScores
    risk_result: RiskResult

    # ── Factory ─────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        project_state: ProjectState,
        simulation_count: int = _DEFAULT_SIMULATION_COUNT,
        seed: int = _MONTE_CARLO_SEED,
    ) -> "ProjectAnalysis":
        """
        Run the full engine pipeline exactly once and return the result.

        Engine execution order is fixed by data dependency:

            MetricsEngine
                ↓
            DependencyGraphEngine  (parallel-safe with Metrics)
                ↓
            CriticalPathEngine     (needs DAG)
                ↓
            SpilloverAnalysisEngine (needs Metrics.average_item_effort)
                ↓
            ForecastEngine         (needs Metrics, CriticalPath, Spillover)
                ↓
            MonteCarloEngine       (needs Metrics, CriticalPath, Spillover)
                ↓
            ImpactScoringEngine    (needs DAG)
                ↓
            RiskEngine             (needs everything above)

        Parameters
        ----------
        project_state:
            Immutable project snapshot to analyse.  Never mutated here.
        simulation_count:
            Monte Carlo iteration count.  1000 gives stable probabilities;
            reduce to 100–200 only in tests.
        seed:
            Random seed for Monte Carlo.  Must stay at 42 (the project-wide
            default) in production so Forecast and MonteCarlo agree.
        """
        logger.info(
            "ProjectAnalysis.build() starting for project '%s' (session %s)",
            project_state.project_info.project_name,
            project_state.project_id,
        )

        # 1 — Metrics (pure aggregation over ProjectState)
        metrics: ProjectMetrics = MetricsEngine(project_state).calculate()

        # 2 — Dependency DAG
        dag: DependencyDAG = DependencyGraphEngine(project_state).build_dag()

        # 3 — Critical path (needs DAG)
        cp_result: CriticalPathResult = CriticalPathEngine(project_state, dag).analyze()

        # 4 — Spillover analysis (needs metrics.average_item_effort)
        spillover: SpilloverAnalysis = SpilloverAnalysisEngine(
            project_state, metrics.average_item_effort
        ).analyze()

        # 5 — Deterministic single-point forecast (needs metrics, cp, spillover)
        forecast: ForecastResult = ForecastEngine(
            project_state, metrics, cp_result, spillover
        ).calculate()

        # 6 — Monte Carlo (needs metrics, cp, spillover; seed kept constant)
        monte_carlo: MonteCarloResult = MonteCarloEngine(
            project_state=project_state,
            metrics=metrics,
            cp_result=cp_result,
            spillover=spillover,
            simulation_count=simulation_count,
            seed=seed,
        ).calculate()

        # 7 — Impact scoring (needs DAG)
        impact_scores: RiskScores = ImpactScoringEngine(project_state, dag).score()

        # 8 — Risk engine (needs all of the above)
        risk_result: RiskResult = RiskEngine(
            project_state=project_state,
            metrics=metrics,
            cp_result=cp_result,
            dag=dag,
            spillover=spillover,
            forecast=forecast,
            monte_carlo=monte_carlo,
            impact_scores=impact_scores,
        ).analyze()

        logger.info(
            "ProjectAnalysis.build() complete — delay=%.1fd, p_on_time=%.2f, risk=%.2f",
            forecast.expected_delay_days,
            monte_carlo.on_time_probability,
            risk_result.overall_risk_score,
        )

        return cls(
            project_state=project_state,
            metrics=metrics,
            dag=dag,
            cp_result=cp_result,
            spillover=spillover,
            forecast=forecast,
            monte_carlo=monte_carlo,
            impact_scores=impact_scores,
            risk_result=risk_result,
        )

    # ── Convenience accessors (thin aliases used by older route code) ────────

    @property
    def upstream(self):
        """
        Return an UpstreamEngineOutputs-compatible view so existing callers
        of ``recommendation_engine._compute_upstream()`` can be migrated
        incrementally without changing their attribute access patterns.

        Import lazily to avoid a circular dependency with simulation_engine.
        """
        from app.engines.recommendation_engine.models import UpstreamEngineOutputs
        return UpstreamEngineOutputs(
            metrics=self.metrics,
            dag=self.dag,
            cp_result=self.cp_result,
            spillover=self.spillover,
            forecast=self.forecast,
            monte_carlo=self.monte_carlo,
            impact_scores=self.impact_scores,
            risk_result=self.risk_result,
        )
