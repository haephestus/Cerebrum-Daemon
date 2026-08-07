import json
import logging
from typing import Optional

from fastapi import (APIRouter, BackgroundTasks, Depends, HTTPException, Query,
                     Request)
from pydantic import BaseModel

from agents.rose import RosePrompts
from cerebrum_core.engrams.core.types import (Engram, EngramType,
                                              FlashcardRating)
from cerebrum_core.learning_center_inator import (
    active_analysis, fetch_flashcards, fetch_long_questions, fetch_mcq,
    fetch_short_question, passive_analysis, submit_flashcard_rating,
    submit_long_question_answer, submit_mcq_answer,
    submit_short_question_answers)
from models.model_inator import NoteStorage
from common.archive_inator import (AnalysisArchiveInator,
                                                list_archived_note_ids)
from common.cache_inator import AnalysisCacheInator
from common.file_util_inator import CerebrumPaths
from cerebrum_core.user_context_inator import get_current_user_id

router_learn = APIRouter(prefix="/learn", tags=["Learning Center API"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class AnalysisResponse(BaseModel):
    status: str
    cached: bool
    version: float
    analysis: Optional[list[dict]] = None
    message: Optional[str] = None


class CacheStatusResponse(BaseModel):
    exists: bool
    is_current: bool
    cached_version: Optional[float] = None
    current_version: float
    needs_update: bool
    cached_at: Optional[str] = None


# ============================================================================
# ACTIVE ANALYSIS (TODO)
# ============================================================================


@router_learn.get("/analysis_status/{note_id}")
def check_analysis_status(note_id: str, request: Request):
    status = request.app.state.note_registry.get_analysis_status(note_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")
    return status


@router_learn.post("/active_analysis/{bubble_id}/{filename}")
def run_active_analysis(
    request: Request, bubble_id: str, filename: str, background_tasks: BackgroundTasks
):
    repo = request.app.state.note_registry
    repo.mark_analysis_status(filename, "pending")
    background_tasks.add_task(active_analysis, bubble_id, filename)
    return {"status": "pending", "bubble_id": bubble_id, "filename": filename}


# ============================================================================
# PASSIVE ANALYSIS - WITH CACHING
# ============================================================================


@router_learn.get(
    "/passive_analysis/{bubble_id}/{filename}", response_model=AnalysisResponse
)
def run_passive_analysis(
    request: Request,
    bubble_id: str,
    filename: str,
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force re-analysis, bypass cache"),
):
    note_path = CerebrumPaths().note_path(bubble_id, filename)

    if not note_path.exists():
        raise HTTPException(status_code=404, detail=f"Note not found: {filename}")

    try:
        note_data = json.loads(note_path.read_text(encoding="utf-8"))
        note = NoteStorage(**note_data)
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Failed to load note: {str(e)}")

    current_version = note.metadata.content_version
    cache_manager = AnalysisCacheInator(bubble_id=bubble_id, note_id=note.note_id)

    if not force:
        cached_analysis = cache_manager.get_cached_analysis(current_version)

        if cached_analysis:
            logger.info(f"Cache HIT for note {note.note_id} v{current_version}")
            return AnalysisResponse(
                status="completed",
                cached=True,
                version=current_version,
                analysis=cached_analysis,
                message="Retrieved from cache",
            )

        logger.info(f"Cache MISS for note {note.note_id} v{current_version}")
    else:
        logger.info(
            f"Force refresh requested for note {note.note_id} v{current_version}"
        )

    prompt = RosePrompts.get_prompt("rose_note_analyser")
    if not prompt:
        raise HTTPException(
            status_code=500, detail="Analysis prompt 'rose_note_analyser' not found"
        )

    logger.info(f"Scheduling analysis for note {note.note_id} v{current_version}")

    background_tasks.add_task(
        passive_analysis,
        bubble_id=bubble_id,  # ← added
        note=note,
        prompt=prompt,
    )
    request.app.state.note_registry.mark_analysed_inator(note_id=note.note_id)
    request.app.state.note_registry.mark_analysis_status(note.note_id, "pending")

    return AnalysisResponse(
        status="pending",
        cached=False,
        version=current_version,
        message="Analysis scheduled in background",
    )


# ============================================================================
# CACHE STATUS ENDPOINTS
# ============================================================================


@router_learn.get(
    "/analysis_status/{bubble_id}/{filename}", response_model=CacheStatusResponse
)
def get_analysis_status(bubble_id: str, filename: str):
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    note_path = notes_dir / filename

    if not note_path.exists():
        raise HTTPException(status_code=404, detail=f"Note not found: {filename}")

    try:
        note_data = json.loads(note_path.read_text(encoding="utf-8"))
        note = NoteStorage(**note_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load note: {str(e)}")

    current_version = note.metadata.content_version
    cache_manager = AnalysisCacheInator(bubble_id=bubble_id, note_id=note.note_id)
    cache_info = cache_manager.get_cache_info()

    if cache_info:
        cached_version = cache_info["content_version"]
        is_current = cached_version == current_version

        return CacheStatusResponse(
            exists=True,
            is_current=is_current,
            cached_version=cached_version,
            current_version=current_version,
            needs_update=not is_current,
            cached_at=cache_info["cached_at"],
        )

    return CacheStatusResponse(
        exists=False,
        is_current=False,
        current_version=current_version,
        needs_update=True,
    )


@router_learn.delete("/invalidate_analysis_cache/{bubble_id}/{filename}")
def invalidate_analysis_cache(bubble_id: str, filename: str):
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    note_path = notes_dir / filename

    if not note_path.exists():
        raise HTTPException(status_code=404, detail=f"Note not found: {filename}")

    try:
        note_data = json.loads(note_path.read_text(encoding="utf-8"))
        note = NoteStorage(**note_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load note: {str(e)}")

    cache_manager = AnalysisCacheInator(bubble_id=bubble_id, note_id=note.note_id)
    cache_manager.invalidate_cache()

    return {
        "detail": "Analysis cache invalidated",
        "note_id": note.note_id,
        "bubble_id": bubble_id,
    }


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================
@router_learn.get("/fetch/analysis")
def get_cached_note_analysis(
    bubble_id: str, note_id: str, version: float
) -> list[dict] | None:
    cache_manager = AnalysisCacheInator(bubble_id=bubble_id, note_id=note_id)
    return cache_manager.get_cached_analysis(content_version=version)


@router_learn.get("/fetch/analysis/full")
def get_full_cached_analysis(
    bubble_id: str,
    note_id: str,
    current_version: Optional[float] = None,
) -> Optional[dict]:
    """
    Return whatever analysis is cached (chunk_diagnostics + note_overview),
    regardless of the note's current content_version. Pass current_version
    to also get an is_current flag so the caller can show a staleness
    notice instead of just silently serving old content.
    """
    cache_manager = AnalysisCacheInator(bubble_id=bubble_id, note_id=note_id)
    info = cache_manager.get_cache_info()
    if info is None:
        return None

    cached_version = info["content_version"]
    chunks = cache_manager.get_cached_analysis(content_version=cached_version)
    overview = cache_manager.get_cached_overview(content_version=cached_version)

    return {
        "chunk_diagnostics": chunks,
        "note_overview": overview,
        "cached_version": cached_version,
        "cached_at": info.get("cached_at"),
        "is_current": (
            current_version is not None and cached_version == current_version
        ),
    }


@router_learn.get("/archive/{bubble_id}/{note_id}")
def get_archived_analysis(bubble_id: str, note_id: str):
    note_path = CerebrumPaths().note_path(
        bubble_id=bubble_id, filename=f"{note_id}.json"
    )
    if not note_path.exists():
        raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")

    try:
        note = NoteStorage(**json.loads(note_path.read_text(encoding="utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load note: {str(e)}")

    archive_path = CerebrumPaths().note_archive_path(bubble_id=bubble_id)
    archive_manager = AnalysisArchiveInator(
        note=note, archives_path=str(archive_path), chunks=[]
    )

    result = archive_manager.archive_browser_inator(bubble_id)
    if result is None or note_id not in result:
        raise HTTPException(
            status_code=404, detail=f"No archive found for note: {note_id}"
        )

    return result[note_id]


@router_learn.get("/archive/{bubble_id}")
def list_bubble_archives(bubble_id: str):
    archive_path = CerebrumPaths().note_archive_path(bubble_id=bubble_id)
    note_ids = list_archived_note_ids(str(archive_path))
    return {
        "bubble_id": bubble_id,
        "count": len(note_ids),
        "note_ids": note_ids,
    }


@router_learn.delete("/archive/clear/{bubble_id}")
def clear_bubble_cache(bubble_id: str, note_id: str):
    note_path = CerebrumPaths().note_path(
        bubble_id=bubble_id, filename=f"{note_id}.json"
    )
    if not note_path.exists():
        raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")

    try:
        note = NoteStorage(**json.loads(note_path.read_text(encoding="utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load note: {str(e)}")

    archive_path = CerebrumPaths().note_archive_path(bubble_id=bubble_id)
    AnalysisArchiveInator(
        note=note, archives_path=str(archive_path), chunks=[]
    ).archive_cleaner_inator()
    raise HTTPException(
        status_code=501, detail="Bulk cache clearing not yet implemented"
    )


# ============================================================================
# ENGRAM GENERATION (TODO)
# ============================================================================


def _sanitize_for_presentation(engram: Engram) -> dict:
    """Strip answer-bearing fields so this is safe to hand to a student
    before they've attempted the item."""
    from dataclasses import asdict

    content = asdict(engram.content)

    if engram.type == EngramType.MCQ:
        content.pop("correct_option", None)
        content.pop("explanation", None)
        content.pop("distractor_notes", None)
    elif engram.type == EngramType.SHORT_QUESTION:
        for q in content.get("questions", []):
            q.pop("expected_answer", None)
    elif engram.type == EngramType.LONG_QUESTION:
        content.pop("answer", None)
        for part in content.get("parts", []):
            part.pop("mark_scheme", None)
    # FLASHCARD: front/back are the point of a flashcard — nothing to strip

    return {
        "id": engram.id,
        "note_id": engram.note_id,
        "type": engram.type.value,
        "target_cognitive_level": engram.target_cognitive_level,
        "tags": engram.tags,
        "content": content,
    }


class EngramGenerationRequest(BaseModel):
    target_cognitive_level: int = 1


@router_learn.get("/engrams/list")
def list_engrams(
    request: Request,
    user_id: str = Depends(get_current_user_id),
    state: Optional[str] = Query(None, description="Filter by due date"),
    bubble_id: Optional[str] = Query(None, description="Filter to a specific bubble"),
    note_id: Optional[str] = Query(
        None, description="Filter to a specific note (requires bubble_id)"
    ),
    include_answers: bool = Query(
        False, description="Include answer-bearing fields (review/admin use only)"
    ),
):
    """
    Read-only: list engrams, scoped by what's provided.
      - no params            -> all engrams, every bubble/note
      - bubble_id only       -> all engrams in that bubble
      - bubble_id + note_id  -> engrams for that specific note

    By default, answer-bearing fields are stripped so this is safe to
    serve directly to a student. Pass include_answers=true for a
    review/admin view that shows correct answers/mark schemes.
    """
    repo = request.app.state.note_registry

    if note_id is not None:
        if bubble_id is None:
            raise HTTPException(
                status_code=400, detail="note_id requires bubble_id to also be provided"
            )
        note = repo.get_note(note_id)
        if not note:
            raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")
        if note.get("bubble_id") != bubble_id:
            raise HTTPException(
                status_code=400,
                detail=f"bubble_id {bubble_id} does not match note's bubble {note.get('bubble_id')}",
            )
        engrams = repo.get_note_engrams(note_id, user_id, state)
        scope = {"bubble_id": bubble_id, "note_id": note_id}

    elif bubble_id is not None:
        engrams = repo.get_bubble_engrams(bubble_id, user_id, state)
        scope = {"bubble_id": bubble_id}

    else:
        engrams = repo.get_all_engrams(user_id)
        scope = {}

    if include_answers:
        from dataclasses import asdict

        payload = [asdict(e) for e in engrams]
    else:
        payload = [_sanitize_for_presentation(e) for e in engrams]

    return {**scope, "count": len(engrams), "engrams": payload}


@router_learn.post("/engrams/{engram_type}/{bubble_id}/{note_id}")
def request_engram_generation(
    engram_type: str,
    bubble_id: str,
    note_id: str,
    body: EngramGenerationRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    Enqueue engram generation for a note. Does not generate inline —
    generation runs LLM calls per finding and belongs entirely in the
    background worker path (process_generation_queue), never in a
    request handler.

    Generation content (gaps, confused concepts, weak areas, etc.) is
    pulled automatically from the note's analysis JSON — nothing about
    what to target is passed in here.
    """
    valid_types = ["all", "mcq", "flashcard", "short_question", "long_question"]
    if engram_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid engram type. Must be one of: {valid_types}",
        )
    if not (1 <= body.target_cognitive_level <= 7):
        raise HTTPException(
            status_code=400, detail="target_cognitive_level must be between 1 and 7"
        )

    repo = request.app.state.note_registry
    note = repo.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")

    repo.queue_engram_generation(
        note_id=note_id,
        bubble_id=bubble_id,
        user_id=user_id,
        trigger="manual_request",
        target_cognitive_level=body.target_cognitive_level,
        target_type=engram_type,
        instructions=None,
    )
    return {"status": "queued", "note_id": note_id, "engram_type": engram_type}


@router_learn.get("/engrams/jobs/{bubble_id}/{note_id}")
def list_generation_jobs(bubble_id: str, note_id: str, request: Request):
    """Read-only: current pending/done/failed generation jobs for a note."""
    repo = request.app.state.note_registry
    # fetch_pending_generation_jobs only returns pending rows today —
    # if you want done/failed visibility here too, that needs a repo
    # method beyond what's on NoteEngramRepository currently (it only
    # exposes the pending-fetch, done-mark, failed-mark trio).
    jobs = repo.fetch_pending_generation_jobs(limit=50)
    return {"note_id": note_id, "jobs": [j for j in jobs if j["note_id"] == note_id]}


@router_learn.get("/engrams/{bubble_id}/{topic}")
def list_topic_engrams(
    bubble_id: str,
    topic: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """Read-only: engrams generated so far for a topic."""
    repo = request.app.state.note_registry
    engrams = repo.get_topic_engrams(user_id=user_id, topic=topic)
    return {"topic": topic, "count": len(engrams), "engrams": engrams}


class MCQSubmission(BaseModel):
    selected_option: str
    target_cognitive_level: int = 1


class FlashcardSubmission(BaseModel):
    self_rating: str
    target_cognitive_level: int = 1


class ShortQuestionResponseItem(BaseModel):
    question_index: int
    raw_answer: str


class ShortQuestionSubmission(BaseModel):
    responses: list[ShortQuestionResponseItem]
    target_cognitive_level: int = 1


class LongQuestionSubmission(BaseModel):
    raw_answer: str
    target_cognitive_level: int = 1


@router_learn.post("/engrams/mcq/{engram_id}/submit")
def submit_mcq(
    engram_id: str,
    body: MCQSubmission,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    repo = request.app.state.note_registry
    engram = repo.get_engram(engram_id)
    if not engram or engram.type.value != "mcq":
        raise HTTPException(404, f"MCQ engram not found: {engram_id}")
    attempt, mastery = submit_mcq_answer(
        repo,
        engram_id=engram_id,
        user_id=user_id,
        selected_option=body.selected_option,
        correct_option=engram.content.correct_option,  # server-side, not client-supplied
        target_cognitive_level=body.target_cognitive_level,
    )
    return {
        "attempt_id": attempt.id,
        "is_correct": attempt.score == 1.0,
        "mastery_state": mastery.state.value,
    }


@router_learn.post("/engrams/flashcard/{engram_id}/submit")
def submit_flashcard(
    engram_id: str,
    body: FlashcardSubmission,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    repo = request.app.state.note_registry
    attempt, mastery = submit_flashcard_rating(
        repo,
        engram_id=engram_id,
        user_id=user_id,
        rating=FlashcardRating(body.self_rating),
        target_cognitive_level=body.target_cognitive_level,
    )
    return {"attempt_id": attempt.id, "mastery_state": mastery.state.value}


@router_learn.post("/engrams/long_question/{engram_id}/submit")
def submit_long_question(
    engram_id: str,
    body: LongQuestionSubmission,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """Async grading — returns immediately with a job to poll, doesn't score inline."""
    repo = request.app.state.note_registry
    attempt_id, job_id = submit_long_question_answer(
        repo,
        engram_id=engram_id,
        user_id=user_id,
        raw_answer=body.raw_answer,
        target_cognitive_level=body.target_cognitive_level,
    )
    return {"attempt_id": attempt_id, "job_id": job_id, "status": "pending_grading"}


@router_learn.post("/engrams/short_question/{engram_id}/submit")
def submit_short_question(
    engram_id: str,
    body: ShortQuestionSubmission,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """Async grading — short answers are open-response, so this returns
    immediately with a job to poll and doesn't score inline (same contract as
    long_question). Each response is {question_index, raw_answer}."""
    repo = request.app.state.note_registry
    engram = repo.get_engram(engram_id)
    if not engram or engram.type.value != "short_question":
        raise HTTPException(404, f"Short-question engram not found: {engram_id}")
    attempt_id, job_id = submit_short_question_answers(
        repo,
        engram_id=engram_id,
        user_id=user_id,
        responses=[r.dict() for r in body.responses],
        target_cognitive_level=body.target_cognitive_level,
    )
    return {"attempt_id": attempt_id, "job_id": job_id, "status": "pending_grading"}


@router_learn.get("/engrams/grading/jobs/{job_id}")
def get_grading_job_status(
    job_id: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """Poll an async grading job (long_question / short_question submissions
    return a job_id). Reports the job status; once status is 'done', includes
    the attempt's overall score and grader. Returns 404 if the job doesn't
    exist or isn't owned by the caller (404 rather than 403 to avoid leaking
    which job ids exist)."""
    repo = request.app.state.note_registry
    job = repo.get_grading_job(job_id)
    if not job or job["attempt_user_id"] != user_id:
        raise HTTPException(404, f"Grading job not found: {job_id}")

    result = {
        "job_id": job["job_id"],
        "attempt_id": job["attempt_id"],
        "status": job["status"],
        "error": job["error"],
        "created_at": job["created_at"],
        "completed_at": job["completed_at"],
    }
    if job["status"] == "done":
        result["score"] = job["attempt_score"]
        result["grader"] = job["attempt_grader"]
    return result


@router_learn.get("/engrams/next/{engram_type}")
def get_next_engram(
    engram_type: str,
    request: Request,
    topic: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
):
    fetchers = {
        "mcq": fetch_mcq,
        "flashcard": fetch_flashcards,
        "short_question": fetch_short_question,
        "long_question": fetch_long_questions,
    }
    fn = fetchers.get(engram_type)
    if not fn:
        raise HTTPException(400, f"Invalid engram type: {engram_type}")
    result = fn(request.app.state.note_registry, user_id, topic)
    if result is None:
        return {"status": "nothing_due"}
    return result
