"""
cerebrum_core.services.progress_service
======================================================
The layer that actually combines the two databases:
  - study_plan_registry.db  (plans/phases/weeks/days/tasks)
  - note_registry.db        (notes/topic_mastery/misconceptions/engram_attempts)

Nothing in study_plan_registry/*.py or note_engram_repository/*.py talks
to the other DB directly — that coupling lives only here, in Python,
since SQLite can't JOIN across separate files. Every function below
takes both repos in explicitly rather than importing/constructing them,
so this stays testable and so callers (routes, background jobs) control
lifetime/connection reuse themselves.

NOTE ON THE LLM CALL: I don't have the existing plan-generation module
(whatever defines holistic_study_plan_generator / _call_model_for_plan
from earlier discussion) in front of me, so densify_phase below takes a
`model_call_fn: Callable[[str], dict]` parameter instead of importing a
specific function — swap that for your actual generator call. If you
paste that module I'll wire this in exactly rather than leave it as an
injected callable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from os import replace
from typing import Callable, Optional

from agents.rose import RosePrompts
from cerebrum_core.constants import PHASE_WEEKS_SCHEMA
from common.ollama_compat import invoker_inator


def build_densify_prompt(
    *,
    phase: dict,
    weekly_rhythm: list,
    topic_masteries: list[dict],
    misconceptions: list[dict],
    week_start: int,
    week_end: int,
) -> str:

    phase_weeks_prompt_template = RosePrompts().get_prompt(
        "phase_weeks_prompt_template"
    )
    assert phase_weeks_prompt_template is not None
    phase_weeks_prompt_template = (
        phase_weeks_prompt_template.replace(
            "{phase_label}", str(phase.get("phase_label"))
        )
        .replace("{theme}", str(phase.get("theme")))
        .replace("{milestone}", str(phase.get("milestone")))
        .replace("{month_start}", str(phase.get("month_start")))
        .replace("{month_end}", str(phase.get("month_end")))
        .replace("{week_start}", str(week_start))
        .replace("{week_end}", str(week_end))
        .replace("{tracks_json}", str(phase.get("tracks", {})))
        .replace("{weekly_rhythm_json}", str(weekly_rhythm))
        .replace("{topic_mastery_json}", str(topic_masteries))
        .replace("{misconceptions_json}", str(misconceptions))
    )

    return phase_weeks_prompt_template


# ---------------------------------------------------------------------
# densify_phase — the generation entry point
# ---------------------------------------------------------------------


def densify_phase(
    *,
    plan_id: str,
    phase_id: int,
    user_id: str,
    plan_repo,  # StudyPlanRegisterInator
    note_repo,  # NoteEngramRepository
    weeks_per_call: int = 4,
) -> list[int]:
    """
    Generates day-by-day detail for the next `weeks_per_call` undensified
    weeks of one phase and writes them via insert_weeks_inator.

    Why weeks_per_call instead of the whole phase at once: a phase can
    span 5-6 months (~22-26 weeks) — generating all of it in one LLM
    call reintroduces the exact quality/reliability problem the
    active-phase-only strategy was meant to avoid, just one level down.
    Chunking to ~4 weeks keeps each generation call small and grounded
    in genuinely current mastery data (which will have changed by the
    time week 20 rolls around anyway).

    Returns the new week_ids written.
    """
    phase = plan_repo.fetch_phase_inator(plan_id, phase_id)
    if not phase:
        raise ValueError(f"No phase {phase_id} on plan {plan_id}")

    existing_weeks = plan_repo.fetch_weeks_for_phase_inator(plan_id, phase_id)
    already_densified = {w["week_number"] for w in existing_weeks}

    phase_week_start = ((phase["month_start"] or 0) * 4) + 1  # approx 4 weeks/month
    phase_week_end = ((phase["month_end"] or 0) + 1) * 4

    next_week = phase_week_start
    while next_week in already_densified:
        next_week += 1
    if next_week > phase_week_end:
        return []  # phase already fully densified

    week_start = next_week
    week_end = min(next_week + weeks_per_call - 1, phase_week_end)

    # Pull grounding context. Topic scope = whatever topics this phase's
    # tracks mention by name isn't reliable (tracks are free-text focus
    # areas, not topic strings) — so instead we pull the user's FULL
    # topic_mastery table and let the LLM select/reuse relevant topics
    # itself. For a user with a very large topic set this could get
    # long; truncate to the N most-recently-active topics if that
    # becomes a real problem.
    topic_masteries = (
        note_repo.get_all_topic_masteries_for_user(user_id)
        if hasattr(note_repo, "get_all_topic_masteries_for_user")
        else []
    )
    misconceptions = (
        note_repo.get_misconceptions_for_user(user_id)
        if hasattr(note_repo, "get_misconceptions_for_user")
        else []
    )

    plan_raw = plan_repo.fetch_plan_inator(plan_id) or {}
    weekly_rhythm = plan_raw.get("weekly_rhythm", [])

    prompt = build_densify_prompt(
        phase=phase,
        weekly_rhythm=weekly_rhythm,
        topic_masteries=topic_masteries,
        misconceptions=misconceptions,
        week_start=week_start,
        week_end=week_end,
    )

    response = invoker_inator.ollama_cloud_call(
        prompt=prompt, schema=PHASE_WEEKS_SCHEMA
    )  # expected to already conform to PHASE_WEEKS_SCHEMA
    weeks = response.get("weeks", [])

    week_ids = plan_repo.insert_weeks_inator(plan_id, phase_id, weeks)

    # The first newly-generated week becomes the active one if nothing
    # else is currently active for this plan (covers both "plan just
    # started" and "previous active week just completed" cases).
    current = plan_repo.fetch_current_week_inator(plan_id)
    if not current and week_ids:
        plan_repo.mark_week_status_inator(week_ids[0], "active")

    return week_ids


# ---------------------------------------------------------------------
# sweep_auto_complete — the engram-activity -> task-completion bridge
# ---------------------------------------------------------------------


def sweep_auto_complete(*, plan_id: str, user_id: str, plan_repo, note_repo) -> int:
    """
    For every pending practice/review task in this plan, checks whether
    the user has done any engram activity on that task's topic within
    the task's day window, and marks it complete if so.

    Day window = plan's created_at + (week_number - 1)*7 days +
    day_of_week, through the same instant + 1 day. This assumes the
    plan's calendar start is created_at — if plans instead have an
    explicit start_date separate from created_at, swap that in here.

    Called by whatever background job already exists for periodic
    plan maintenance (fetch_stale_plans_inator's caller is the obvious
    place to also trigger this, if such a job exists) or on-demand from
    a router endpoint hit when the user opens the Plan Progress screen.

    Returns count of tasks auto-completed.
    """
    plan = plan_repo.fetch_plan_inator(plan_id)
    if not plan:
        return 0

    all_plans = plan_repo.fetch_all_plans_inator(user_id=user_id)
    plan_row = next((p for p in all_plans if p["plan_id"] == plan_id), None)
    if not plan_row:
        return 0
    plan_start = datetime.fromisoformat(plan_row["created_at"]).replace(
        tzinfo=timezone.utc
    )

    pending = plan_repo.fetch_pending_topic_tasks_inator(plan_id)
    completed_count = 0

    for task in pending:
        day_start = plan_start + timedelta(
            days=(task["week_number"] - 1) * 7 + task["day_of_week"]
        )
        day_end = day_start + timedelta(days=1)

        activity = note_repo.get_topic_activity_since(
            user_id,
            task["topic"],
            day_start.isoformat(),
            day_end.isoformat(),
        )
        if activity > 0:
            plan_repo.complete_task_inator(task["task_id"], auto_resolved=True)
            completed_count += 1

    return completed_count


# ---------------------------------------------------------------------
# fetch_plan_progress — read-side aggregation for the widget
# ---------------------------------------------------------------------


def fetch_plan_progress(*, plan_id: str, user_id: str, plan_repo, note_repo) -> dict:
    """
    Combines task-completion ratio (study_plan_registry.db) with
    topic_mastery scores (note_registry.db) into one payload the
    Flutter Plan Progress widget can render directly: per-phase
    readiness, current week detail, and pace.
    """
    current_week = plan_repo.fetch_current_week_inator(plan_id)
    week_task_counts = (
        plan_repo.fetch_week_task_counts_inator(current_week["week_id"])
        if current_week
        else {"total": 0, "completed": 0, "ratio": 0.0}
    )

    # Topic mastery for whatever topics the current week actually covers
    # — not the whole user history, since this is meant to answer "how
    # ready is THIS week", not "how good is this user at everything".
    week_topic_scores = []
    if current_week:
        for topic in current_week["topics"]:
            tm = note_repo.get_topic_mastery(user_id, topic)
            if tm:
                week_topic_scores.append(
                    {"topic": topic, "overall_score": tm.overall_score}
                )

    avg_mastery = (
        sum(t["overall_score"] for t in week_topic_scores) / len(week_topic_scores)
        if week_topic_scores
        else None
    )

    incomplete_phases = plan_repo.fetch_incomplete_phases_inator(plan_id)
    unachieved_metrics = plan_repo.fetch_unachieved_metrics_inator(plan_id)

    return {
        "current_week": current_week,
        "current_week_task_progress": week_task_counts,
        "current_week_topic_mastery": week_topic_scores,
        "current_week_avg_mastery": avg_mastery,
        "incomplete_phases": incomplete_phases,
        "unachieved_metrics": unachieved_metrics,
    }
