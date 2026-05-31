import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from api.routes_knowledgebase import embedding_task, markdown_converter_task
from cerebrum_core.model_inator import NoteStorage
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.note_util_inator import NoteChunkerInator, NoteToMarkdownInator

router_test = APIRouter(prefix="/test", tags=["Test routes"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router_test.get("/cache/note")
def get_note_cache(bubble_id: str, note_id: str):
    pass


# convert one file at a time
@router_test.post("/markdowninator")
async def convert_files(
    note_id: str, request: Request, background_task: BackgroundTasks
):
    """Queue unconverted files for markdown conversion."""
    file_registry = request.app.state.file_registry
    unconverted = file_registry.fetch_unconverted_file_inator()
    if note_id not in unconverted:
        return {"message": f"File: {note_id} already converted", "count": 0}
    background_task.add_task(markdown_converter_task, unconverted, file_registry)
    return {
        "message": f"Queued {len(unconverted)} files for conversion",
        "count": len(unconverted),
    }


# embed one file at a time
@router_test.post("/embeddinator")
async def embedd_files(request: Request, background_task: BackgroundTasks):
    """Queue converted files for embedding."""
    file_registry = request.app.state.file_registry
    unembedded = file_registry.fetch_unembedded_file_inator()
    if not unembedded:
        return {"message": "No files to embed", "count": 0}
    background_task.add_task(embedding_task, unembedded, file_registry)
    return {
        "message": f"Queued {len(unembedded)} files for embedding",
        "count": len(unembedded),
    }


def chunking_task(bubble_id: str, note_id: str) -> dict:
    note_path = (
        CerebrumPaths()
        .note_path(bubble_id=bubble_id, filename=note_id)
        .with_suffix(".json")
    )
    note = NoteStorage(**json.loads(note_path.read_text(encoding="utf-8")))
    flattened = NoteToMarkdownInator().flatten(note.content)

    _, documents = NoteChunkerInator(generate_artifacts=True).chunk(
        flattened_note=flattened,
        note_id=note_id,
        bubble_id=bubble_id,
    )

    logger.info(f"[CHUNK TASK] Note {note_id} → {len(documents)} chunks registered")
    return {"note_id": note_id, "chunks": len(documents)}


@router_test.post("/chunkinator")
async def chunk_note(
    bubble_id: str,
    note_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Chunk a note into segments and write the annotated markdown artifact.

    - Flattens the note to markdown
    - Splits into header/token-bounded chunks
    - Writes the chunked .md file to the derived path
    - Clears stale registry rows and re-registers all chunks

    Must be run before active_analysis can process a note.
    """
    note_path = (
        CerebrumPaths()
        .note_path(bubble_id=bubble_id, filename=note_id)
        .with_suffix(".json")
    )
    logger.info(f"[CHUNK ROUTE] Looking for note at: {note_path}")
    if not note_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Note {note_id} not found in bubble {bubble_id} — path: {note_path}",
        )

    background_tasks.add_task(chunking_task, bubble_id, note_id)

    return {
        "message": f"Chunking queued for note {note_id}",
        "bubble_id": bubble_id,
        "note_id": note_id,
    }
