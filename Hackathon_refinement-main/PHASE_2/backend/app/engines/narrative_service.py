"""
NarrativeService — the only place an LLM is called in Sprint Whisperer.

This module is now a pure orchestrator.  All infrastructure concerns live
in app.ai:

    app.ai.config    → AISettings (API key, model, timeout, tokens, flags)
    app.ai.client    → ClaudeClient (auth, retry, timeout, tool-call)
    app.ai.prompts   → ADVISOR_SYSTEM_PROMPT, ADVISOR_OUTPUT_TOOL
    app.ai.cache     → NarrativeCache / InMemoryNarrativeCache / cache_key()
    app.ai.renderer  → render_recommendation_explanation, render_scenario_explanation,
                        render_executive_summary

Hard invariants (unchanged from original design — enforced by construction):

  1. The model receives ONLY an AdvisorInput snapshot — no ProjectState,
     no engines, no callables.  Nothing it could use to derive a new number.

  2. The model MUST respond via the submit_advisor_explanation tool.
     ClaudeClient raises AIResponseError if it returns plain text instead.

  3. Every numeric claim is a ClaimRef → resolved by the renderer from
     the real AdvisorInput value at render time.  The model never writes
     a number directly.

  4. Any failure (timeout, bad JSON, unresolvable claim, disabled flag)
     degrades to the existing deterministic template text.  This layer
     can never block or corrupt the deterministic pipeline.

  5. Results are cached per (model, AdvisorInput) so re-calling with
     the same facts never hits the API twice.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.ai.cache import InMemoryNarrativeCache, NarrativeCache, cache_key
from app.ai.client import ClaudeClient
from app.ai.config import AISettings, ai_settings
from app.ai.exceptions import AIError
from app.ai.renderer import (
    render_executive_summary,
    render_recommendation_explanation,
    render_scenario_explanation,
)
from app.engines.advisor_contract import (
    AdvisorInput,
    AdvisorOutput,
    AdvisorResponseStatus,
)

logger = logging.getLogger(__name__)


def _build_user_message(advisor_input: AdvisorInput) -> str:
    return (
        "Here is the deterministic snapshot to explain:\n\n"
        f"{advisor_input.model_dump_json(indent=2)}"
    )


class NarrativeService:
    """
    Orchestrates:
        AdvisorInput
              │
              ▼
        NarrativeCache (check)
              │ miss
              ▼
        ClaudeClient.generate()
              │
              ▼
        AdvisorOutput  (Pydantic validation)
              │
              ▼
        Renderer  (ClaimRef → resolved values)
              │
              ▼
        NarrativeCache (store)
              │
              ▼
        Dict[str, Any]  (API response shape)

    On any failure → _fallback_response() using deterministic template text.
    """

    def __init__(
        self,
        client: ClaudeClient,
        settings: Optional[AISettings] = None,
        cache: Optional[NarrativeCache] = None,
    ) -> None:
        self.client = client
        self.settings = settings or ai_settings
        self.cache = cache or InMemoryNarrativeCache()

    async def explain(
        self,
        advisor_input: AdvisorInput,
        fallback_text_by_recommendation: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Generate (or retrieve from cache) an AI narrative for the given
        AdvisorInput snapshot.

        Parameters
        ----------
        advisor_input
            Closed, read-only snapshot of already-computed engine facts.
        fallback_text_by_recommendation
            Maps recommendation_id → the deterministic description string
            produced by the recommendation engine.  Used when the model call
            fails entirely or the model skips a recommendation.

        Returns
        -------
        {
            "status": "ok" | "partial" | "fallback",
            "executive_summary": {...} | None,
            "recommendation_explanations": [{...}, ...],
            "scenario_explanation": {...} | None,
        }
        """
        if not self.settings.ai_advisor_enabled:
            return self._fallback_response(advisor_input, fallback_text_by_recommendation)

        key = cache_key(advisor_input, self.settings.ai_model)
        if self.settings.ai_cache_enabled:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached

        try:
            raw = await self.client.generate(_build_user_message(advisor_input))
            advisor_output = AdvisorOutput.model_validate(raw)
        except AIError as exc:
            logger.warning("Advisor call failed (%s); using template fallback", exc)
            return self._fallback_response(advisor_input, fallback_text_by_recommendation)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in NarrativeService.explain: %s", exc)
            return self._fallback_response(advisor_input, fallback_text_by_recommendation)

        result = self._render(advisor_output, advisor_input, fallback_text_by_recommendation)

        if self.settings.ai_cache_enabled:
            await self.cache.set(key, result)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _render(
        self,
        advisor_output: AdvisorOutput,
        advisor_input: AdvisorInput,
        fallback_text_by_recommendation: Dict[str, str],
    ) -> Dict[str, Any]:
        any_degraded = False

        # Executive summary
        rendered_exec_summary = None
        if advisor_output.executive_summary is not None:
            rendered_exec_summary = render_executive_summary(
                advisor_output.executive_summary, advisor_input
            )
            if not rendered_exec_summary["fully_resolved"]:
                any_degraded = True

        # Recommendation explanations
        rendered_recs = []
        seen_ids: set[str] = set()

        for explanation in advisor_output.recommendation_explanations:
            seen_ids.add(explanation.recommendation_id)
            rendered = render_recommendation_explanation(explanation, advisor_input)
            if not rendered["fully_resolved"]:
                any_degraded = True
            rendered_recs.append(rendered)

        # Any recommendation the model skipped falls back to its deterministic text.
        for rec_id, fallback_text in fallback_text_by_recommendation.items():
            if rec_id not in seen_ids:
                rendered_recs.append(
                    {
                        "recommendation_id": rec_id,
                        "sections": [
                            {"kind": "summary", "heading": "Summary", "body": fallback_text}
                        ],
                        "fully_resolved": True,
                        "source": "template_fallback",
                    }
                )

        # Scenario explanation
        rendered_scenario = None
        if advisor_output.scenario_explanation is not None:
            rendered_scenario = render_scenario_explanation(
                advisor_output.scenario_explanation, advisor_input
            )
            if not rendered_scenario["fully_resolved"]:
                any_degraded = True

        status = (
            AdvisorResponseStatus.PARTIAL if any_degraded else AdvisorResponseStatus.OK
        )
        return {
            "status": status.value,
            "executive_summary": rendered_exec_summary,
            "recommendation_explanations": rendered_recs,
            "scenario_explanation": rendered_scenario,
        }

    def _fallback_response(
        self,
        advisor_input: AdvisorInput,
        fallback_text_by_recommendation: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Deterministic-only fallback — no LLM involved, always succeeds.
        Numbers are read directly from the already-real AdvisorInput values.
        """
        fallback_exec_summary = None
        ctx = advisor_input.project_context
        if ctx is not None:
            delay_phrase = (
                f"{ctx.expected_delay_days:.0f} days late"
                if ctx.expected_delay_days > 0
                else f"{abs(ctx.expected_delay_days):.0f} days early"
            )
            fallback_exec_summary = {
                "headline": (
                    f"{ctx.current_sprint_name}: project is forecast to finish "
                    f"{delay_phrase}, with a {ctx.on_time_probability * 100:.0f}% "
                    f"chance of hitting the target date."
                ),
                "fully_resolved": True,
            }

        return {
            "status": AdvisorResponseStatus.FALLBACK.value,
            "executive_summary": fallback_exec_summary,
            "recommendation_explanations": [
                {
                    "recommendation_id": rec_id,
                    "sections": [
                        {"kind": "summary", "heading": "Summary", "body": text}
                    ],
                    "fully_resolved": True,
                    "source": "template_fallback",
                }
                for rec_id, text in fallback_text_by_recommendation.items()
            ],
            "scenario_explanation": None,
        }


    # ------------------------------------------------------------------
    # Recovery Advisor narrative
    # ------------------------------------------------------------------

    async def explain_recovery_plans(
        self,
        advisor_input: AdvisorInput,
    ) -> Dict[str, Any]:
        """
        Generate (or retrieve from cache) the Recovery Advisor narrative for
        the given AdvisorInput snapshot that contains recovery_plans.

        Returns
        -------
        {
            "status": "ok" | "partial" | "fallback",
            "narrative": {
                "situation_framing":       {"heading": str, "body": str},
                "strategy_rationale":      {"heading": str, "body": str},
                "alternatives_considered": {"heading": str, "body": str},
                "expected_outcomes":       {"heading": str, "body": str},
                "pm_guidance":             {"heading": str, "body": str},
            }
        }
        """
        if not self.settings.ai_advisor_enabled or not advisor_input.recovery_plans:
            return self._fallback_recovery_response(advisor_input)

        key = cache_key(advisor_input, self.settings.ai_model + ":recovery")
        if self.settings.ai_cache_enabled:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached

        try:
            from app.ai.prompts import RECOVERY_ADVISOR_SYSTEM_PROMPT

            # Temporarily swap the client's system prompt for the recovery mode prompt.
            # BoschClient always uses BOSCH_SYSTEM_PROMPT from the module; we send the
            # recovery prompt as the first user message prepended to the snapshot, which
            # is the safest approach across both providers without patching the client.
            user_message = (
                f"SYSTEM INSTRUCTION (recovery advisor mode):\n"
                f"{RECOVERY_ADVISOR_SYSTEM_PROMPT}\n\n"
                f"---\n\n"
                f"Here is the deterministic snapshot to explain:\n\n"
                f"{advisor_input.model_dump_json(indent=2)}"
            )
            raw = await self.client.generate(user_message)
        except AIError as exc:
            logger.warning("Recovery advisor call failed (%s); using fallback", exc)
            return self._fallback_recovery_response(advisor_input)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in explain_recovery_plans: %s", exc)
            return self._fallback_recovery_response(advisor_input)

        result = self._render_recovery_narrative(raw, advisor_input)

        if self.settings.ai_cache_enabled:
            await self.cache.set(key, result)

        return result

    def _render_recovery_narrative(
        self,
        raw: Dict[str, Any],
        advisor_input: AdvisorInput,
    ) -> Dict[str, Any]:
        """
        Resolve ClaimRefs in each of the five recovery narrative sections.

        The model returns a dict with five keys; each value is a section
        object with heading, body_template, and claims.  We resolve claims
        the same way as recommendation explanations — any unresolvable claim
        becomes "Not available" and status is marked PARTIAL.
        """
        from app.ai.renderer import render_section
        from app.engines.advisor_contract import NarrativeSection, AdvisorResponseStatus

        SECTION_KEYS = [
            "situation_framing",
            "strategy_rationale",
            "alternatives_considered",
            "expected_outcomes",
            "pm_guidance",
        ]

        narrative: Dict[str, Any] = {}
        any_degraded = False

        for key in SECTION_KEYS:
            section_data = raw.get(key)
            if not section_data or not isinstance(section_data, dict):
                # Model skipped this section — use deterministic fallback text
                narrative[key] = {
                    "heading": key.replace("_", " ").title(),
                    "body": self._fallback_recovery_section(key, advisor_input),
                }
                any_degraded = True
                continue

            try:
                section = NarrativeSection(
                    heading=section_data.get("heading", key.replace("_", " ").title()),
                    body_template=section_data.get("body_template", ""),
                    claims=[
                        __import__("app.engines.advisor_contract", fromlist=["ClaimRef"]).ClaimRef(**c)
                        for c in section_data.get("claims", [])
                    ],
                )
                text, ok = render_section(section, advisor_input)
                if not ok:
                    any_degraded = True
                narrative[key] = {"heading": section.heading, "body": text}
            except Exception as exc:
                logger.warning("Recovery section '%s' failed to render: %s", key, exc)
                narrative[key] = {
                    "heading": key.replace("_", " ").title(),
                    "body": self._fallback_recovery_section(key, advisor_input),
                }
                any_degraded = True

        status = AdvisorResponseStatus.PARTIAL if any_degraded else AdvisorResponseStatus.OK
        return {"status": status.value, "narrative": narrative}

    def _fallback_recovery_response(self, advisor_input: AdvisorInput) -> Dict[str, Any]:
        """
        Deterministic fallback — no LLM involved, built entirely from
        the already-computed RecoveryPlanArchetypeFacts in advisor_input.
        Always succeeds.
        """
        from app.engines.advisor_contract import AdvisorResponseStatus

        narrative: Dict[str, Any] = {}
        for key in [
            "situation_framing",
            "strategy_rationale",
            "alternatives_considered",
            "expected_outcomes",
            "pm_guidance",
        ]:
            narrative[key] = {
                "heading": key.replace("_", " ").title(),
                "body": self._fallback_recovery_section(key, advisor_input),
            }
        return {"status": AdvisorResponseStatus.FALLBACK.value, "narrative": narrative}

    @staticmethod
    def _fallback_recovery_section(key: str, advisor_input: AdvisorInput) -> str:
        """
        Build a minimal but data-grounded fallback sentence for each section
        using only fields that are guaranteed to exist in the recovery_plans list.
        """
        plans = advisor_input.recovery_plans
        if not plans:
            return "No recovery plan data available."

        recommended = next((p for p in plans if p.rank == 1), plans[0])
        alternatives = [p for p in plans if p.rank != 1]

        ctx = advisor_input.project_context
        metrics = advisor_input.metrics

        if key == "situation_framing":
            delay = f"{ctx.expected_delay_days:.0f} days late" if ctx else "behind schedule"
            blocker_str = f" with {metrics.active_blocker_count} active blockers" if metrics else ""
            return (
                f"The project is currently forecast to finish {delay}{blocker_str}. "
                f"Historical delivery patterns and current risk signals have been analysed "
                f"to generate the recovery options below."
            )

        if key == "strategy_rationale":
            prob_pct = round(recommended.deadline_probability * 100, 1)
            return (
                f"The {recommended.archetype} plan is recommended (rank 1 of {len(plans)}). "
                f"It offers a {prob_pct}% probability of hitting the deadline with "
                f"{recommended.execution_complexity.lower()} execution complexity. "
                f"{recommended.narrative_summary}"
            )

        if key == "alternatives_considered":
            if not alternatives:
                return "No alternative plans available for comparison."
            parts = []
            for alt in alternatives:
                prob_pct = round(alt.deadline_probability * 100, 1)
                parts.append(
                    f"{alt.archetype} ({prob_pct}% deadline probability, "
                    f"{alt.execution_complexity.lower()} complexity)"
                )
            return "Alternative plans considered: " + "; ".join(parts) + "."

        if key == "expected_outcomes":
            prob_pct = round(recommended.deadline_probability * 100, 1)
            delay = recommended.expected_delay_days
            delay_str = (
                "on time" if delay <= 0 else f"{delay:.1f} days late"
            )
            return (
                f"If the {recommended.archetype} plan is executed, the project has a "
                f"{prob_pct}% probability of meeting its deadline. "
                f"Expected delivery: {delay_str}. "
                f"Residual risk score: {recommended.overall_risk_score:.2f}."
            )

        if key == "pm_guidance":
            blocker_str = (
                f"Monitor the {metrics.active_blocker_count} active blocker(s) for resolution. "
                if metrics and metrics.active_blocker_count > 0
                else ""
            )
            return (
                f"{blocker_str}"
                f"Track velocity stability sprint-over-sprint. "
                f"Re-run the recovery plan analysis if new blockers are added or scope changes."
            )

        return "See deterministic plan details above."
