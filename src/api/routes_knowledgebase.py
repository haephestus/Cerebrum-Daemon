"""
Complete knowledgebase routes with both file registry and vector store management.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cerebrum_core.knowledgebase_inator import FileMarkdownChunker, KnowledgebaseManager
from cerebrum_core.utils.embedd_inator import EmbeddInator
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.markdown_handler_inator import MarkdownConverter
from cerebrum_core.utils.registry.file_chunk_registry_inator import (
    FileChunkRegisterInator,
)
from cerebrum_core.utils.registry.file_registry_inator import FileRegisterInator

router = APIRouter(prefix="/knowledgebase")
archives_dir = CerebrumPaths().kb_archives_path()
markdown_files_dir = CerebrumPaths().kb_artifacts_path()
knowledgebase_dir = CerebrumPaths().kb_source_files_path()


# ========================================
# Background Tasks
# ========================================


def process_single_file_task(file_info: dict, file_registry: FileRegisterInator):
    """
    Process a single file: convert to markdown, chunk, and embed.
    """
    try:
        print(f"Processing: {file_info['original_name']}")
        filepath = Path(file_info["filepath"])

        if not filepath.exists():
            print(f"File not found: {filepath}")
            return

        # Step 1: Convert to Markdown with LLM sanitization
        converter = MarkdownConverter(filepath=filepath)
        markdown_path, metadata = converter.convert(metadata=None)

        # Step 2: Chunk Markdown
        chunker = FileMarkdownChunker()
        chunked_path = chunker.chunk(
            markdown_path=markdown_path, file_fingerprint=file_info["file_fingerprint"]
        )

        # Step 3: Update file registry (mark as converted)
        file_registry.mark_converted_inator(
            file_fingerprint=file_info["file_fingerprint"],
            domain=metadata.domain,
            subject=metadata.subject,
            sanitized_name=metadata.title,
        )

        # Step 4: Embed chunks
        embedding_manager = EmbeddInator(
            file_fingerprint=file_info["file_fingerprint"],
            original_name=file_info["original_name"],
        )
        embedding_manager.embed_from_chunked_markdown(
            chunked_markdown=chunked_path,
            collection_name=metadata.subject,
            domain=metadata.domain,
            subject=metadata.subject,
        )

        # Step 5: Mark as embedded
        file_registry.mark_embedded_inator(
            file_fingerprint=file_info["file_fingerprint"]
        )

        print(f"Completed: {file_info['original_name']}")

    except Exception as e:
        print(f"Failed processing {file_info['original_name']}: {e}")
        raise


def markdown_converter_task(
    unconverted_files: list[dict], file_registry: FileRegisterInator
):
    """
    Convert source files to Markdown with LLM-enriched metadata and chunk them.
    """
    for file_info in unconverted_files:
        try:
            print(f"Converting: {file_info['original_name']}")
            filepath = Path(file_info["filepath"])
            if not filepath.exists():
                print(f"File not found: {filepath}")
                continue

            # Convert to Markdown with LLM sanitization
            converter = MarkdownConverter(filepath=filepath)
            markdown_path, metadata = converter.convert(metadata=None)

            # Chunk Markdown
            chunker = FileMarkdownChunker()
            chunker.chunk(
                markdown_path=markdown_path,
                file_fingerprint=file_info["chunk_fingerprint"],
            )

            # Update file registry
            file_registry.mark_converted_inator(
                file_fingerprint=file_info["file_fingerprint"],
                domain=metadata.domain,
                subject=metadata.subject,
                sanitized_name=metadata.title,
            )

            print(f"Converted & chunked: {file_info['original_name']}")

        except Exception as e:
            print(f"Failed for {file_info['original_name']}: {e}")


def embedding_task(unembedded_files: list[dict], file_registry: FileRegisterInator):
    """
    Embed chunked Markdown files in vector database.
    """
    for file_info in unembedded_files:
        try:
            domain = file_info.get("domain", "default")
            subject = file_info.get("subject", "default")
            sanitized_name = file_info["sanitized_name"]

            # Locate chunked markdown file
            chunked_path = (
                markdown_files_dir / domain / subject / f"{sanitized_name}.chunked.md"
            )

            if not chunked_path.exists():
                print(f"Chunked markdown file not found: {chunked_path}")
                continue

            # Embed using byte-coordinate access
            embedding_manager = EmbeddInator(
                original_name=file_info["original_name"],
                file_fingerprint=file_info["file_fingerprint"],
            )
            embedding_manager.embed_from_chunked_markdown(
                chunked_markdown=chunked_path,
                collection_name=subject,
                domain=domain,
                subject=subject,
            )

            # Mark as embedded in registry
            file_registry.mark_embedded_inator(
                file_fingerprint=file_info["file_fingerprint"]
            )
            print(f"Embedded: {sanitized_name}")

        except Exception as e:
            print(f"Failed embedding {file_info['sanitized_name']}: {e}")
            print("Progress saved — will resume on next run.")


# ========================================
# File Registry Routes
# ========================================


@router.get("/show")
async def show_files(request: Request):
    """Show all source files in registry."""
    file_registry = request.app.state.file_registry
    return file_registry.show_all_inator() or []


@router.get("/show/chunks")
async def show_chunks(request: Request):
    """Show all source files in registry."""
    file_chunk_registry = request.app.state.file_chunk_registry
    return file_chunk_registry.show_all_inator() or []


@router.post("/upload")
async def upload_pdf(
    request: Request, file: UploadFile = File(...), priority: bool = False
):
    """
    Upload PDF to knowledgebase.

    Args:
        priority: If True, processes immediately. If False, queues for batch processing.
    """
    file_registry = request.app.state.file_registry

    if file.filename is None:
        raise ValueError("filename cannot be None")

    filepath = knowledgebase_dir / file.filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Register file
    file_registry.register_inator(file.filename, str(filepath))

    # Save file
    with filepath.open("wb") as f:
        content = await file.read()
        f.write(content)

    response = {
        "message": "PDF uploaded",
        "filename": file.filename,
        "path": str(filepath),
    }

    if priority:
        response["status"] = "queued for immediate processing"
    else:
        response["status"] = "queued for batch processing"

    return response


# batch embed and convert files
@router.post("/process/batch")
async def process_batch(
    request: Request,
    batch_size: int = 10,
):
    """
    Immediate batch processing:
    - N artifacts -> ONE chunked markdown
    - Embed after conversion
    - No background tasks
    """

    file_registry = request.app.state.file_registry
    unconverted = file_registry.fetch_unconverted_file_inator()

    if not unconverted:
        return {
            "message": "Nothing to process",
            "count": 0,
        }

    batches = []
    for i in range(0, len(unconverted), batch_size):
        batches.append(unconverted[i : i + batch_size])

    processed = 0

    for batch in batches:
        # 1️⃣ Convert → ONE markdown file
        markdown_path = markdown_converter_task(
            batch,
            file_registry,
        )

        # 2️⃣ Embed ONLY AFTER conversion
        if markdown_path:
            embedding_task(
                [markdown_path],
                file_registry,
            )
            processed += 1

    return {
        "message": "Batch processing complete",
        "batches_processed": processed,
        "batch_size": batch_size,
    }


@router.post("/process-file/{file_fingerprint}")
async def process_single_file(
    request: Request, file_fingerprint: str, background_tasks: BackgroundTasks
):
    """
    Process a single file immediately (convert + embed).
    Use for urgent documents or interactive workflows.

    Monitor progress via /stream-progress/{fingerprint}
    """
    file_registry = request.app.state.file_registry

    # Check if file exists
    if not file_registry.check_inator(file_fingerprint):
        raise HTTPException(status_code=404, detail="File not found")

    # Check if already processed
    if file_registry.check_inator(file_fingerprint, "embedded"):
        return {
            "message": "File already processed",
            "file_fingerprint": file_fingerprint,
            "status": "completed",
        }

    # Get file info
    all_files = file_registry.show_all_inator()
    file_info = next(
        (f for f in all_files if f["file_fingerprint"] == file_fingerprint), None
    )

    if not file_info:
        raise HTTPException(status_code=404, detail="File info not found")

    # Queue for immediate processing
    background_tasks.add_task(process_single_file_task, file_info, file_registry)

    return {
        "message": "File queued for processing",
        "file_fingerprint": file_fingerprint,
        "status": "processing",
        "progress_stream": f"/knowledgebase/stream-progress/{file_fingerprint}",
    }


@router.get("/stream-progress/{file_fingerprint}")
async def stream_progress(file_fingerprint: str):
    """
    Server-Sent Events (SSE) endpoint for real-time progress updates.

    Frontend usage:
        const eventSource = new EventSource('/knowledgebase/stream-progress/abc123');
        eventSource.onmessage = (event) => {
            const progress = JSON.parse(event.data);
            console.log(`Progress: ${progress.progress_pct}%`);
        };
    """

    async def event_generator():
        file_chunk_registry = FileChunkRegisterInator()
        last_progress = -1

        while True:
            try:
                # Get embedding progress
                progress = file_chunk_registry.get_embedding_progress(file_fingerprint)

                # Only send update if progress changed
                if progress["progress_pct"] != last_progress:
                    last_progress = progress["progress_pct"]

                    # SSE format: "data: {json}\n\n"
                    data = {
                        "file_fingerprint": file_fingerprint,
                        "total": progress["total"],
                        "completed": progress["completed"],
                        "remaining": progress["remaining"],
                        "progress_pct": progress["progress_pct"],
                        "status": (
                            "completed" if progress["remaining"] == 0 else "processing"
                        ),
                    }

                    yield f"data: {json.dumps(data)}\n\n"

                    # Exit if complete
                    if progress["remaining"] == 0 and progress["total"] > 0:
                        yield f"data: {json.dumps({'status': 'done'})}\n\n"
                        break

                # Poll every second
                await asyncio.sleep(1)

            except Exception as e:
                error_data = {"status": "error", "message": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/file-status/{file_fingerprint}")
async def get_file_status(request: Request, file_fingerprint: str):
    """
    Get current status of a file (polling alternative to SSE).

    Returns file conversion and embedding status.
    """
    file_registry = request.app.state.file_registry
    file_chunk_registry = FileChunkRegisterInator()

    if not file_registry.check_inator(file_fingerprint):
        raise HTTPException(status_code=404, detail="File not found")

    converted = file_registry.check_inator(file_fingerprint, "converted")
    embedded = file_registry.check_inator(file_fingerprint, "embedded")
    chunk_progress = file_chunk_registry.get_embedding_progress(file_fingerprint)

    return {
        "file_fingerprint": file_fingerprint,
        "converted": converted,
        "embedded": embedded,
        "chunk_progress": chunk_progress,
        "status": "completed" if embedded else "processing" if converted else "pending",
    }


class DeletePayload(BaseModel):
    filename: str
    filepath: str
    file_fingerprint: str
    collection_name: Optional[str] = None


@router.delete("/delete/")
async def remove_source_file(request: Request, payload: DeletePayload):
    """Remove file from knowledgebase and vector database."""
    file_registry = request.app.state.file_registry

    # Remove from registry and filesystem
    file_registry.remove_inator(
        payload.filename, payload.filepath, payload.file_fingerprint
    )

    # Remove from vector database across all collections
    try:
        manager = KnowledgebaseManager()
        count = manager.delete_by_fingerprint_all_collections(payload.file_fingerprint)
        print(f"Deleted {count} documents from vector stores")
    except Exception as e:
        print(f"Warning: Failed to delete from vector stores: {e}")

    return {"detail": "File removed from knowledgebase successfully"}


@router.post("/reset/{status}")
async def reset_registry(
    request: Request, status: str, file_fingerprint: Optional[str]
):
    """Reset conversion or embedding status in registry."""
    file_registry = request.app.state.file_registry

    if status not in ["converted", "embedded"]:
        raise HTTPException(
            status_code=400, detail="Status must be 'converted' or 'embedded'"
        )

    count = file_registry.reset_inator(status, file_fingerprint)

    # TODO: add method for clearing registry and cache
    return {"message": f"Reset {status} status", "affected_rows": count}


# ========================================
# Vector Store Management Routes
# ========================================


@router.get("/collections")
async def list_all_collections():
    """
    List all vector database collections with their info.

    Returns:
        List of collections with domain, subject, count, etc.
    """
    manager = KnowledgebaseManager()
    collections = manager.list_all_collections()

    return {
        "collections": collections,
        "count": len(collections),
    }


@router.get("/collections/{domain}/{subject}/{collection_name}")
async def get_collection_details(domain: str, subject: str, collection_name: str):
    """
    Get detailed information about a specific collection.

    Returns:
        Collection info with count, metadata, sample documents
    """
    manager = KnowledgebaseManager()

    try:
        info = manager.get_collection_info(collection_name, domain, subject)
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {e}")


@router.get("/collections/{domain}/{subject}/{collection_name}/count")
async def get_collection_count(domain: str, subject: str, collection_name: str):
    """Get document count for a specific collection."""
    manager = KnowledgebaseManager()

    try:
        store = manager.get_store(collection_name, domain, subject)
        count = store._collection.count()
        return {
            "domain": domain,
            "subject": subject,
            "collection": collection_name,
            "count": count,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {e}")


@router.delete("/collections/{domain}/{subject}/{collection_name}")
async def delete_collection(domain: str, subject: str, collection_name: str):
    """Delete an entire collection."""
    manager = KnowledgebaseManager()

    try:
        manager.delete_collection(collection_name, domain, subject)
        return {
            "message": "Collection deleted successfully",
            "domain": domain,
            "subject": subject,
            "collection": collection_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")


# ========================================
# Search Routes
# ========================================


class SearchRequest(BaseModel):
    query: str
    domains: Optional[list[str]]
    subjects: Optional[list[str]]
    k: int = 5


@router.post("/search")
async def search_collections(request: SearchRequest):
    """
    Search across multiple collections.

    Args:
        query: Search query text
        domains: Optional list of domains to search
        subjects: Optional list of subjects to search
        k: Number of results per collection

    Returns:
        List of matching documents with collection info
    """
    manager = KnowledgebaseManager()

    try:
        results = manager.search_across_collections(
            query=request.query,
            domains=request.domains,
            subjects=request.subjects,
            k=request.k,
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.get("/search/fingerprint/{fingerprint}")
async def find_by_fingerprint(file_fingerprint: str):
    """
    Find all documents with a specific fingerprint across all collections.

    Args:
        fingerprint: Document fingerprint to search for

    Returns:
        List of documents with collection info
    """
    manager = KnowledgebaseManager()

    try:
        documents = manager.get_documents_by_fingerprint(file_fingerprint)

        return {
            "file_fingerprint": file_fingerprint,
            "documents": documents,
            "count": len(documents),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# ========================================
# Delete Routes
# ========================================


class DeleteByMetadataRequest(BaseModel):
    collection_name: str
    domain: str = "default"
    subject: str = "default"
    metadata_filter: dict


@router.delete("/documents/by-metadata")
async def delete_by_metadata(request: DeleteByMetadataRequest):
    """
    Delete documents matching metadata criteria.

    Example request body:
    {
        "collection_name": "biology",
        "domain": "science",
        "subject": "biology",
        "metadata_filter": {"author": "Smith"}
    }
    """
    manager = KnowledgebaseManager()

    try:
        count = manager.delete_by_metadata(
            request.collection_name,
            request.metadata_filter,
            request.domain,
            request.subject,
        )

        return {
            "message": "Documents deleted successfully",
            "count": count,
            "filter": request.metadata_filter,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


@router.delete("/documents/delete/{file_fingerprint}")
async def delete_by_fingerprint(file_fingerprint: str):
    """
    Delete all documents with a specific fingerprint across ALL collections.

    This is useful when removing a source document from the knowledgebase.
    """
    manager = KnowledgebaseManager()

    try:
        count = manager.delete_by_fingerprint_all_collections(file_fingerprint)

        return {
            "message": "Documents deleted successfully",
            "file_fingerprint": file_fingerprint,
            "total_deleted": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


# ========================================
# Statistics Routes
# ========================================


@router.get("/stats")
async def get_statistics():
    """
    Get overall knowledgebase statistics.

    Returns:
        Total collections, documents, domains, subjects
    """
    manager = KnowledgebaseManager()

    collections = manager.list_all_collections()

    total_docs = sum(c["count"] for c in collections)
    unique_domains = len(set(c["domain"] for c in collections))
    unique_subjects = len(set(c["subject"] for c in collections))

    return {
        "total_collections": len(collections),
        "total_documents": total_docs,
        "unique_domains": unique_domains,
        "unique_subjects": unique_subjects,
        "collections": collections,
    }
