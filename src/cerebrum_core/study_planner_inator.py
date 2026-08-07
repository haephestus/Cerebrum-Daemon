"""
cerebrum_core.study_plan_center_inator
===============================================
service layer between the API router and StudyPlanRegisterInator —
mirrors learning_center_inator's active_analysis/passive_analysis
relationship to NoteChunkRegisterInator, but for study plan generation.

Router should never touch StudyPlanRegisterInator or the ollama_*_call
functions directly — everything goes through the functions here.
"""

import json
import logging
import uuid
from typing import Optional

from agents.rose import RosePrompts
from cerebrum_core.constants import STUDY_PLAN_SCHEMA
from database.planner import StudyPlanRegisterInator
from cerebrum_core.user_inator import ConfigManager, should_use_cloud
from common.ollama_compat.invoker_inator import (
    ollama_cloud_call,
    ollama_local_call2,
)

logger = logging.getLogger(__name__)


def _call_model_for_plan(prompt: str, schema: dict) -> dict:
    """
    Dispatches to cloud or local ollama depending on config, same
    cloud/local split used elsewhere in the codebase. Parses the JSON
    string response into a dict — format=schema returns structured JSON
    as text, it does not come back pre-parsed.
    """
    config = ConfigManager().load_config()
    use_cloud = should_use_cloud(config)

    if use_cloud:
        logger.info("[STUDY_PLAN] dispatching to ollama cloud")
        response_text = ollama_cloud_call(prompt=prompt, schema=schema)
    else:
        logger.info("[STUDY_PLAN] dispatching to ollama local")
        response_text = ollama_local_call2(prompt=prompt, schema=schema)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(
            f"[STUDY_PLAN] model returned non-JSON response: {response_text[:500]}"
        )
        raise ValueError(f"Model response was not valid JSON: {e}") from e


def generate_study_plan(
    user_id: str,
    user_profile: dict,
    target_role: str,
    context: Optional[str] = None,
    historical_plan_id: Optional[str] = None,
) -> dict:
    """
    Canonical study-plan generation entrypoint — the only path that should
    ever produce a plan. Runs as a background task from the router, same
    as active_analysis/passive_analysis in learning_center_inator.

    Returns {"plan_id": ..., **plan_data} for logging/testing; the
    registry write is the actual persistence step callers should read
    state from afterward (via GET /study_plan/{plan_id}).
    """
    registry = StudyPlanRegisterInator()

    historical_plan = None
    if historical_plan_id:
        historical_plan = registry.fetch_plan_inator(historical_plan_id)
        if historical_plan is None:
            logger.warning(
                "historical_plan_id=%s not found — generating fresh plan",
                historical_plan_id,
            )

    prompt_template = RosePrompts.get_prompt("holistic_study_plan_generator")
    if not prompt_template:
        raise RuntimeError("Prompt 'holistic_study_plan_generator' not found")

    filled_prompt = prompt_template.format(
        user_profile=json.dumps(user_profile),
        target_role=target_role,
        context=context or "",
        historical_plan=json.dumps(historical_plan) if historical_plan else "",
    )

    plan_data = _call_model_for_plan(prompt=filled_prompt, schema=STUDY_PLAN_SCHEMA)

    plan_id = str(uuid.uuid4())
    registry.register_inator(plan_id=plan_id, user_id=user_id, plan_data=plan_data)

    if historical_plan_id and historical_plan is not None:
        registry.bump_version_inator(
            old_plan_id=historical_plan_id, new_plan_id=plan_id
        )

    logger.info(
        "Study plan generated: plan_id=%s user_id=%s target_role=%s",
        plan_id,
        user_id,
        target_role,
    )

    return {"plan_id": plan_id, **plan_data}


def mark_phase_complete(plan_id: str, phase_id: int) -> None:
    registry = StudyPlanRegisterInator()
    registry.mark_phase_status_inator(plan_id, phase_id, "completed")


def mark_metric_achieved(metric_row_id: int) -> None:
    registry = StudyPlanRegisterInator()
    registry.mark_metric_achieved_inator(metric_row_id)


def fetch_review_queue(grace_days: int = 14) -> list[dict]:
    """Plans stuck past their duration — for a dashboard or notification job."""
    registry = StudyPlanRegisterInator()
    return registry.fetch_stale_plans_inator(grace_days=grace_days)
