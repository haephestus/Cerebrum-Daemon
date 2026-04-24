import logging

from fastapi import APIRouter, BackgroundTasks, Request

from api.routes_knowledgebase import embedding_task, markdown_converter_task

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
