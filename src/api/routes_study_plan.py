"""
cerebrum_core.routers.study_plan_router
===============================================
thin FastAPI layer — no direct registry or ollama_*_call access.
Every write path goes through study_plan_center_inator; reads that don't
need orchestration go straight to request.app.state.study_plan_registry,
same convention as router_learn.py's check_analysis_status.
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from cerebrum_core.study_planner_inator import (
    fetch_review_queue,
    generate_study_plan,
    mark_metric_achieved,
    mark_phase_complete,
)
from cerebrum_core.utils.user_context_inator import get_current_user_id

router_study_plan = APIRouter(prefix="/study_plan", tags=["Study Plan API"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST MODELS
# ============================================================================


class StudyPlanRequest(BaseModel):
    user_profile: dict
    target_role: str
    context: Optional[str] = None
    historical_plan_id: Optional[str] = None


class StudyPlanResponse(BaseModel):
    status: str
    user_id: str


# ============================================================================
# GENERATION (background task, same pattern as run_active_analysis)
# ============================================================================


@router_study_plan.post("/generate", response_model=StudyPlanResponse)
def request_study_plan(
    body: StudyPlanRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    background_tasks.add_task(
        generate_study_plan,
        user_id=user_id,
        user_profile=body.user_profile,
        target_role=body.target_role,
        context=body.context,
        historical_plan_id=body.historical_plan_id,
    )
    return StudyPlanResponse(status="pending", user_id=user_id)


# ============================================================================
# READS (direct registry access — no generation involved)
# ============================================================================


@router_study_plan.get("/{plan_id}")
def get_study_plan(plan_id: str, request: Request) -> Optional[dict]:
    registry = request.app.state.study_plan_registry
    plan: Optional[dict] = registry.fetch_plan_inator(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Plan not found: {plan_id}")
    return plan


@router_study_plan.get("/user/active")
def get_active_plans(
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    registry = request.app.state.study_plan_registry
    return {"user_id": user_id, "plans": registry.fetch_active_plans_inator(user_id)}


@router_study_plan.get("/{plan_id}/phases/incomplete")
def get_incomplete_phases(plan_id: str, request: Request):
    registry = request.app.state.study_plan_registry
    return {
        "plan_id": plan_id,
        "phases": registry.fetch_incomplete_phases_inator(plan_id),
    }


@router_study_plan.get("/{plan_id}/metrics/unachieved")
def get_unachieved_metrics(plan_id: str, request: Request):
    registry = request.app.state.study_plan_registry
    return {
        "plan_id": plan_id,
        "metrics": registry.fetch_unachieved_metrics_inator(plan_id),
    }


# ============================================================================
# STATE MUTATIONS (go through the service layer, not the registry directly)
# ============================================================================


@router_study_plan.post("/{plan_id}/phases/{phase_id}/complete")
def complete_phase(plan_id: str, phase_id: int):
    mark_phase_complete(plan_id, phase_id)
    return {"status": "completed", "plan_id": plan_id, "phase_id": phase_id}


@router_study_plan.post("/metrics/{metric_row_id}/achieve")
def achieve_metric(metric_row_id: int):
    mark_metric_achieved(metric_row_id)
    return {"status": "achieved", "metric_row_id": metric_row_id}


# ============================================================================
# ADMIN / REVIEW
# ============================================================================


@router_study_plan.get("/admin/review_queue")
def get_review_queue(grace_days: int = 14):
    return {"stale_plans": fetch_review_queue(grace_days=grace_days)}
