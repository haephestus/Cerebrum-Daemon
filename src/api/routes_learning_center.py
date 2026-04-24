import json
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel

from agents.rose import RosePrompts
from cerebrum_core.learning_center_inator import active_analysis, passive_analysis
from cerebrum_core.model_inator import NoteStorage
from cerebrum_core.utils.cache_inator import AnalysisCacheInator
from cerebrum_core.utils.file_util_inator import CerebrumPaths

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
    analysis: Optional[str] = None
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


@router_learn.post("/active_analysis/{bubble_id}/{filename}")
def run_active_analysis(request: Request, bubble_id: str, filename: str):
    """
    Run user-directed analysis with a specific question/prompt.

    TODO: Implement interactive analysis where user asks specific questions
    about their notes (e.g., "What are the key concepts?" or
    "Generate practice questions from this section")

    Args:
        bubble_id: ID of the study bubble
        filename: Note filename
        user_query: User's specific question/request

    Returns:
        Analysis result tailored to user query
    """

    result = active_analysis(bubble_id, filename)
    request.app.state.note_registry.mark_analysed_inator(
        note_id=filename.strip(".json")
    )
    return {"status": "Success", "result": result}


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
    """
    Run passive analysis on a note with intelligent caching.

    - Checks cache first (unless force=True)
    - Returns cached result if version matches
    - Schedules background analysis if needed

    Args:
        bubble_id: ID of the study bubble
        filename: Note filename (e.g., "my_note.json")
        force: If True, bypass cache and force fresh analysis

    Returns:
        AnalysisResponse with status and analysis (if cached)
    """
    # Load note
    note_path = CerebrumPaths().note_path(bubble_id, filename)

    if not note_path.exists():
        raise HTTPException(status_code=404, detail=f"Note not found: {filename}")

    try:
        note_data = json.loads(note_path.read_text(encoding="utf-8"))
        note = NoteStorage(**note_data)
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Failed to load note: {str(e)}")

    current_version = note.metadata.content_version

    # Initialize cache manager
    cache_manager = AnalysisCacheInator(bubble_id=bubble_id, note_id=note.note_id)

    # Check cache (unless force=True)
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

    # Get analysis prompt
    prompt = RosePrompts.get_prompt("rose_note_analyser")
    if not prompt:
        raise HTTPException(
            status_code=500, detail="Analysis prompt 'rose_note_analyser' not found"
        )

    # Schedule background analysis
    logger.info(f"Scheduling analysis for note {note.note_id} v{current_version}")

    background_tasks.add_task(
        passive_analysis,
        note=note,
        prompt=prompt,
        cache_manager=cache_manager,
    )
    request.app.state.note_registry.mark_analysed_inator(note_id=note.note_id)

    return AnalysisResponse(
        status="scheduled",
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
    """
    Check if analysis cache exists and is current.

    Useful for UI to show:
    - Whether analysis is available
    - If cached analysis is outdated
    - When to trigger fresh analysis

    Args:
        bubble_id: ID of the study bubble
        filename: Note filename

    Returns:
        CacheStatusResponse with cache metadata
    """
    # Load note
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

    # Check cache
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
    """
    Manually invalidate (delete) cached analysis for a note.

    Useful for:
    - Testing
    - Forcing fresh analysis
    - Clearing stale cache

    Args:
        bubble_id: ID of the study bubble
        filename: Note filename

    Returns:
        Success message
    """
    # Load note to get note_id
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    note_path = notes_dir / filename

    if not note_path.exists():
        raise HTTPException(status_code=404, detail=f"Note not found: {filename}")

    try:
        note_data = json.loads(note_path.read_text(encoding="utf-8"))
        note = NoteStorage(**note_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load note: {str(e)}")

    # Invalidate cache
    cache_manager = AnalysisCacheInator(bubble_id=bubble_id, note_id=note.note_id)

    cache_manager.invalidate_cache()

    return {
        "detail": "Analysis cache invalidated",
        "note_id": note.note_id,
        "bubble_id": bubble_id,
    }


# ============================================================================
# ENGRAM GENERATION (TODO)
# ============================================================================


@router_learn.get("/engrams/{engram_type}")
def generate_engram(
    engram_type: str, bubble_id: str, filename: str, background_tasks: BackgroundTasks
):
    """
    Generate learning materials (engrams) from note analysis.

    Engram types:
    - quiz: Multiple choice questions
    - flashcards: Spaced repetition cards
    - mock_exam: Comprehensive exam questions
    - summary: Key concepts summary

    TODO: Implement engram generation pipeline:
    1. Load cached analysis
    2. Generate specific engram type using LLM
    3. Store in bubble-specific folders
    4. Track in spaced repetition system

    Args:
        engram_type: Type of learning material to generate
        bubble_id: ID of the study bubble
        filename: Note filename
        background_tasks: FastAPI background tasks

    Returns:
        Generated engram or generation status
    """
    # TODO: Implement
    valid_types = ["quiz", "flashcards", "mock_exam", "summary"]

    if engram_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid engram type. Must be one of: {valid_types}",
        )

    raise HTTPException(
        status_code=501,
        detail=f"Engram generation for '{engram_type}' not yet implemented",
    )


# ============================================================================
# CACHE MANAGEMENT (TODO)
# ============================================================================
@router_learn.get("/fetch/analysis")
def get_cached_note_analysis(bubble_id: str, note_id: str, version: int) -> str | None:
    """
    Get cached analysis for a note.

    TODO: Show:
    - Total notes cached
    - Cache hit rate
    - Stale cache entries
    - Storage used

    Returns:
        Cache analysis
    """
    cache_manager = AnalysisCacheInator(bubble_id=bubble_id, note_id=note_id)
    cached_analysis = cache_manager.get_cached_analysis(content_version=version)
    return cached_analysis


@router_learn.get("/cache/stats/{bubble_id}")
def get_cache_stats(bubble_id: str):
    """
    Get cache statistics for a bubble.

    TODO: Show:
    - Total notes cached
    - Cache hit rate
    - Stale cache entries
    - Storage used

    Returns:
        Cache statistics
    """
    raise HTTPException(status_code=501, detail="Cache statistics not yet implemented")


@router_learn.delete("/cache/clear/{bubble_id}")
def clear_bubble_cache(bubble_id: str):
    """
    Clear all cached analysis for a bubble.

    TODO: Implement bubble-wide cache clearing

    Args:
        bubble_id: ID of the study bubble

    Returns:
        Success message with count of cleared entries
    """
    raise HTTPException(
        status_code=501, detail="Bulk cache clearing not yet implemented"
    )
