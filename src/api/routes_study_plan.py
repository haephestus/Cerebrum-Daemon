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

from cerebrum_core.database.planner.progress_service import (
    densify_phase,
    fetch_plan_progress,
    sweep_auto_complete,
)
from cerebrum_core.study_planner_inator import (
    fetch_review_queue,
    generate_study_plan,
    mark_metric_achieved,
    mark_phase_complete,
)
from cerebrum_core.utils.ollama_compat import invoker_inator
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


@router_study_plan.get("/user/all")
def get_user_plans(
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    Returns every plan belonging to the user regardless of status
    (draft/active/completed/archived). Filtering by status is a
    client-side concern now -- see DLearningCenterPage's plan toggle --
    so the registry method here does no WHERE-clause filtering itself.
    """
    registry = request.app.state.study_plan_registry
    return {"user_id": user_id, "plans": registry.fetch_all_plans_inator(user_id)}


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


@router_study_plan.get("/{plan_id}/progress")
def get_plan_progress(
    plan_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    Runs the auto-complete sweep first, then returns the combined
    progress payload — so opening this screen is what keeps task
    checkmarks fresh, rather than requiring a separate poll/cron for
    a single-user desktop-scale app. If this becomes a real background
    job later (multi-user, higher volume), move the sweep call out of
    the request path and into a scheduler, and this endpoint becomes a
    pure read.
    """
    plan_repo = request.app.state.study_plan_registry
    note_repo = request.app.state.note_registry

    if not plan_repo.check_inator(plan_id):
        raise HTTPException(status_code=404, detail="Plan not found")

    sweep_auto_complete(
        plan_id=plan_id, user_id=user_id, plan_repo=plan_repo, note_repo=note_repo
    )
    return fetch_plan_progress(
        plan_id=plan_id, user_id=user_id, plan_repo=plan_repo, note_repo=note_repo
    )


@router_study_plan.post("/{plan_id}/phases/{phase_id}/densify")
def densify_plan_phase(
    plan_id: str,
    phase_id: int,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    Generates the next chunk of week-by-day detail for one phase. Kept
    synchronous here for simplicity — if generation latency becomes a
    UX problem, wrap this in whatever background-job mechanism
    engram_generation_queue already uses for engram generation, and
    have the Flutter side poll /progress until new weeks show up
    instead of awaiting this call directly.
    """
    plan_repo = request.app.state.study_plan_registry
    note_repo = request.app.state.note_engram_repository

    if not plan_repo.check_inator(plan_id):
        raise HTTPException(status_code=404, detail="Plan not found")

    week_ids = densify_phase(
        plan_id=plan_id,
        phase_id=phase_id,
        user_id=user_id,
        plan_repo=plan_repo,
        note_repo=note_repo,
    )
    return {"plan_id": plan_id, "phase_id": phase_id, "week_ids": week_ids}


@router_study_plan.post("/tasks/{task_id}/complete")
def complete_task(
    task_id: int,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """Manual completion — auto_resolved stays False, distinguishing a
    user's own tap from the sweep's activity-derived completion."""
    plan_repo = request.app.state.study_plan_registry
    plan_repo.complete_task_inator(task_id, auto_resolved=False)
    return {"task_id": task_id, "status": "complete"}


@router_study_plan.post("/tasks/{task_id}/reopen")
def reopen_task(
    task_id: int,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    plan_repo = request.app.state.study_plan_registry
    plan_repo.reopen_task_inator(task_id)
    return {"task_id": task_id, "status": "pending"}


@router_study_plan.get("/{plan_id}/weeks/phase/{phase_id}")
def get_phase_weeks(
    plan_id: str,
    phase_id: int,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """All densified weeks for one phase — used when the user navigates
    away from the current week to browse earlier/later weeks in the
    same phase."""
    plan_repo = request.app.state.study_plan_registry
    return {"weeks": plan_repo.fetch_weeks_for_phase_inator(plan_id, phase_id)}
