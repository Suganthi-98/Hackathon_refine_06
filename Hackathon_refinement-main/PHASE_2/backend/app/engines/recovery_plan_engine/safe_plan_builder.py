from typing import Dict, List, Set

from app.engines.recommendation_engine.models import Recommendation, ConfidenceLevel


class SafePlanBuilder:
    """Builds a Safe plan from simulated recommendation impacts.

    Strategy:
    - Sort recommendations by simulated on-time probability delta (desc),
      then by simulated delay reduction (desc), then priority_score.
    - Greedily select up to `max_actions` recommendations whose affected ids
      (items, resources, blockers) do not overlap with already-selected actions.
    - Skip LOW confidence recommendations.
    """

    @staticmethod
    def build(
        recommendations: List[Recommendation],
        simulation_map: Dict[str, object] | None = None,
        max_actions: int = 3,
    ) -> List[Recommendation]:
        if simulation_map is None:
            # No simulation info available — fall back to priority ordering
            sorted_recs = sorted(recommendations, key=lambda r: (-r.priority_score, r.recommendation_id))
        else:
            def key_for(rec: Recommendation):
                sim = simulation_map.get(rec.recommendation_id)
                # support both ScenarioResult-like and simplified dicts
                delta_prob = 0.0
                delta_delay = 0.0
                if sim is not None:
                    # ScenarioResult has monte_carlo_comparison
                    mc = getattr(sim, "monte_carlo_comparison", None)
                    fc = getattr(sim, "forecast_comparison", None)
                    if mc is not None:
                        delta_prob = getattr(mc, "simulated_on_time_probability", 0.0) - getattr(mc, "baseline_on_time_probability", 0.0)
                    if fc is not None:
                        delta_delay = getattr(fc, "baseline_delay_days", 0.0) - getattr(fc, "simulated_delay_days", 0.0)
                    # Support dict shape too
                    if isinstance(sim, dict):
                        delta_prob = sim.get("delta_on_time_probability", delta_prob)
                        delta_delay = sim.get("delta_expected_delay_days", delta_delay)

                return (-delta_prob, -delta_delay, -rec.priority_score, rec.recommendation_id)

            sorted_recs = sorted(recommendations, key=key_for)

        selected: List[Recommendation] = []
        used_items: Set[str] = set()
        used_resources: Set[str] = set()
        used_blockers: Set[str] = set()

        for rec in sorted_recs:
            if len(selected) >= max_actions:
                break

            if rec.confidence == ConfidenceLevel.LOW:
                continue

            if set(rec.affected_item_ids) & used_items:
                continue
            if set(rec.affected_resource_ids) & used_resources:
                continue
            if set(rec.affected_blocker_ids) & used_blockers:
                continue

            selected.append(rec)
            used_items.update(rec.affected_item_ids)
            used_resources.update(rec.affected_resource_ids)
            used_blockers.update(rec.affected_blocker_ids)

        return selected
