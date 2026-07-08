"""
cerebrum_core.engrams.scheduling.scheduler
===========================================
Spaced repetition scheduler — SM-2 extended with cognitive level gating
and regression-aware interval adjustment.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ..core.scoring import (
    DEMOTION_THRESHOLD,
    LAPSE_THRESHOLD,
    PROMOTION_THRESHOLDS,
    compute_stability,
    detect_regression,
    update_dimension_scores,
)
from ..core.types import (
    DimensionScores,
    EngramMastery,
    MasteryState,
    QueuedEngram,
    SchedulingDecision,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MULTIPLIERS = {
    "easy": 2.8,
    "good": 1.8,
    "hard": 0.6,
    "lapse_minor": 0.4,
    "lapse_major": 0.2,
    "promotion": 1.0,  # interval reset on level change
}

MIN_INTERVAL_DAYS = 0.007  # ~10 minutes
MAX_INTERVAL_DAYS = 365.0
CONSECUTIVE_FOR_MASTERY = 4
CONSECUTIVE_FOR_PROMOTION = 3


# ---------------------------------------------------------------------------
# Core scheduling function
# ---------------------------------------------------------------------------


def compute_schedule(
    mastery: EngramMastery,
    new_score: float,
    recent_scores: list[float],
    incoming_dimensions: DimensionScores | None = None,
) -> SchedulingDecision:
    regression = detect_regression(mastery.current_score, new_score)
    stability = compute_stability(recent_scores + [new_score])

    new_state = mastery.state
    new_interval = mastery.interval_days
    new_level = mastery.current_level
    lapse = False
    promoted = False
    demoted = False
    new_consec = mastery.consecutive_correct + 1 if new_score >= 0.7 else 0

    # --- Lapse ---
    if new_score < LAPSE_THRESHOLD:
        lapse = True
        new_state = MasteryState.LAPSED
        new_consec = 0

        mult = (
            _MULTIPLIERS["lapse_major"]
            if regression.severity == "major"
            else _MULTIPLIERS["lapse_minor"]
        )
        new_interval = max(MIN_INTERVAL_DAYS, mastery.interval_days * mult)

        # Level demotion on severe lapse
        if (
            new_score < DEMOTION_THRESHOLD
            and mastery.current_level > 1
            and mastery.lapse_count >= 1
        ):
            new_level = mastery.current_level - 1
            demoted = True
            new_interval = 1.0

    # --- Normal progression ---
    else:
        mult = (
            _MULTIPLIERS["easy"]
            if new_score >= 0.90
            else _MULTIPLIERS["good"] if new_score >= 0.75 else _MULTIPLIERS["hard"]
        )
        new_interval = min(MAX_INTERVAL_DAYS, mastery.interval_days * mult)

        # State transitions
        if mastery.state == MasteryState.LAPSED:
            new_state = MasteryState.LEARNING
        elif mastery.state in (MasteryState.NEW, MasteryState.LEARNING):
            new_state = (
                MasteryState.REVIEW if new_consec >= 2 else MasteryState.LEARNING
            )
        elif mastery.state == MasteryState.REVIEW:
            if new_consec >= CONSECUTIVE_FOR_MASTERY and stability >= 0.7:
                new_state = MasteryState.MASTERED

        # Level promotion
        threshold = PROMOTION_THRESHOLDS.get(mastery.current_level, 0.92)
        if (
            mastery.current_level < 7
            and new_score >= threshold
            and new_consec >= CONSECUTIVE_FOR_PROMOTION
            and stability >= 0.65
        ):
            new_level = mastery.current_level + 1
            promoted = True
            new_state = MasteryState.LEARNING
            new_consec = 0
            new_interval = _MULTIPLIERS["promotion"]

    new_interval = max(MIN_INTERVAL_DAYS, min(MAX_INTERVAL_DAYS, new_interval))
    next_due = datetime.utcnow() + timedelta(days=new_interval)

    return SchedulingDecision(
        new_state=new_state,
        new_score=new_score,
        new_stability=stability,
        new_interval_days=new_interval,
        next_due_at=next_due,
        lapse_occurred=lapse,
        promotion_occurred=promoted,
        demotion_occurred=demoted,
        new_level=new_level,
        new_consecutive=new_consec,
    )


# ---------------------------------------------------------------------------
# Apply decision → updated mastery record
# ---------------------------------------------------------------------------


def apply_schedule(
    mastery: EngramMastery,
    decision: SchedulingDecision,
    new_score: float,
    incoming_dimensions: DimensionScores | None = None,
) -> EngramMastery:
    from copy import copy

    updated = copy(mastery)

    if incoming_dimensions:
        current_dims = DimensionScores(
            accuracy=mastery.score_accuracy,
            depth=mastery.score_depth,
            reasoning=mastery.score_reasoning,
            connections=mastery.score_connections,
            originality=mastery.score_originality,
            precision=mastery.score_precision,
            awareness_of_limits=mastery.score_awareness_of_limits,
        )
        new_dims = update_dimension_scores(current_dims, incoming_dimensions)
        updated.score_accuracy = new_dims.accuracy
        updated.score_depth = new_dims.depth
        updated.score_reasoning = new_dims.reasoning
        updated.score_connections = new_dims.connections
        updated.score_originality = new_dims.originality
        updated.score_precision = new_dims.precision
        updated.score_awareness_of_limits = new_dims.awareness_of_limits

    updated.state = decision.new_state
    updated.current_score = decision.new_score
    updated.stability = decision.new_stability
    updated.interval_days = decision.new_interval_days
    updated.next_due_at = decision.next_due_at.isoformat()
    updated.last_attempted_at = datetime.utcnow().isoformat()
    updated.attempt_count = mastery.attempt_count + 1
    updated.lapse_count = mastery.lapse_count + (1 if decision.lapse_occurred else 0)
    # Use the value computed in compute_schedule rather than recomputing it here,
    # since compute_schedule resets it to 0 on lapse and on level promotion —
    # recomputing from new_score alone would incorrectly undo that reset.
    updated.consecutive_correct = decision.new_consecutive
    updated.current_level = decision.new_level
    updated.updated_at = datetime.utcnow().isoformat()

    return updated


# ---------------------------------------------------------------------------
# Build daily study queue
# Priority: lapsed > overdue > due today > new
# ---------------------------------------------------------------------------


def build_study_queue(
    masteries: list[EngramMastery],
    max_new: int = 10,
    max_review: int = 100,
) -> list[QueuedEngram]:
    now = datetime.utcnow()
    queue: list[QueuedEngram] = []
    new_count = 0
    review_count = 0

    for m in masteries:
        if m.state == MasteryState.SUSPENDED:
            continue

        due_at = datetime.fromisoformat(m.next_due_at)
        is_overdue = due_at < now
        overdue_days = (now - due_at).total_seconds() / 86400

        if m.state == MasteryState.NEW:
            if new_count >= max_new:
                continue
            queue.append(
                QueuedEngram(
                    engram_id=m.engram_id, mastery=m, priority=50.0, reason="new"
                )
            )
            new_count += 1

        elif m.state == MasteryState.LAPSED:
            queue.append(
                QueuedEngram(
                    engram_id=m.engram_id,
                    mastery=m,
                    priority=100.0 + m.lapse_count,
                    reason="lapsed",
                )
            )
            review_count += 1

        elif is_overdue and review_count < max_review:
            priority = 80.0 + min(20.0, overdue_days * 2)
            queue.append(
                QueuedEngram(
                    engram_id=m.engram_id,
                    mastery=m,
                    priority=priority,
                    reason="overdue",
                )
            )
            review_count += 1

        elif not is_overdue and overdue_days > -1 and review_count < max_review:
            queue.append(
                QueuedEngram(
                    engram_id=m.engram_id, mastery=m, priority=60.0, reason="due"
                )
            )
            review_count += 1

    return sorted(queue, key=lambda q: q.priority, reverse=True)


# ---------------------------------------------------------------------------
# Topic-level triage: pull lapsed topic's engrams forward
# ---------------------------------------------------------------------------


def triage_topic_lapse(
    masteries: list[EngramMastery],
    topic_overall_score: float,
    lapse_threshold: float = LAPSE_THRESHOLD,
) -> list[EngramMastery]:
    if topic_overall_score >= lapse_threshold:
        return masteries

    now = datetime.utcnow().isoformat()
    result = []
    for m in masteries:
        if m.state in (MasteryState.MASTERED, MasteryState.SUSPENDED):
            result.append(m)
        else:
            from copy import copy

            updated = copy(m)
            updated.next_due_at = now
            result.append(updated)
    return result
