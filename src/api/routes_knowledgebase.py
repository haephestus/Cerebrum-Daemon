"""
Complete knowledgebase routes with both file registry and vector store management.
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from cerebrum_core.knowledgebase_inator import FileMarkdownChunker, KnowledgebaseManager
from cerebrum_core.utils.user_context_inator import get_current_user_id
from cerebrum_core.utils.chunking_queue_inator import QueuedJob, file_processing_queue
from cerebrum_core.utils.embedd_inator import EmbeddInator
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.markdown_handler_inator import MarkdownConverter
from cerebrum_core.database.file_chunk_registry_inator import (
    FileChunkRegisterInator,
)
from cerebrum_core.database.file_registry_inator import FileRegisterInator
from cerebrum_core.database.figure_registry_inator import FigureRegisterInator
from cerebrum_core.database.concept_index_inator import ConceptIndexInator

router = APIRouter(prefix="/knowledgebase")
knowledgebase_dir = CerebrumPaths().kb_source_files_path()


# ========================================
# Background Tasks
# ========================================


def _convert_and_chunk_step(file_info: dict, file_registry: FileRegisterInator):
    """Stage 1: PDF -> markdown (LLM sanitize) -> chunked markdown -> registry."""
    filepath = Path(file_info["filepath"])
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    converter = MarkdownConverter(filepath=filepath)
    markdown_path, metadata = converter.convert(metadata=None)

    chunker = FileMarkdownChunker()
    chunked_path = chunker.chunk(
        markdown_path=markdown_path,
        file_fingerprint=file_info["file_fingerprint"],
        doc_type=metadata.doc_type,
    )

    file_registry.mark_converted_inator(
        file_fingerprint=file_info["file_fingerprint"],
        domain=metadata.domain,
        subject=metadata.subject,
        sanitized_name=metadata.title,
        doc_type=metadata.doc_type,
    )

    return chunked_path, metadata


def _embed_step(
    file_info: dict,
    domain: str,
    subject: str,
    chunked_path: Path,
    file_registry: FileRegisterInator,
):
    """Stage 2: embed chunked markdown into the vector store."""
    embedding_manager = EmbeddInator(
        file_fingerprint=file_info["file_fingerprint"],
        original_name=file_info["original_name"],
    )
    embedding_manager.embed_from_chunked_markdown(
        chunked_markdown=chunked_path,
        collection_name=subject,
        domain=domain,
        subject=subject,
    )
    file_registry.mark_embedded_inator(file_fingerprint=file_info["file_fingerprint"])


def process_single_file_task(file_info: dict, file_registry: FileRegisterInator):
    """
    Process a single file: convert to markdown, chunk, and embed.
    Resumable at the FILE level, not just the chunk level — skips
    conversion entirely (including the LLM rename call) if this file
    was already converted in a previous run, so retrying a file that
    only failed at embedding doesn't re-pay for PDF conversion + LLM
    sanitization every time.
    """
    fingerprint = file_info["file_fingerprint"]
    name = file_info["original_name"]

    try:
        if file_registry.check_inator(fingerprint, "embedded"):
            print(f"Already fully processed: {name}")
            return

        if file_registry.check_inator(fingerprint, "converted"):
            print(f"Skipping conversion (already done): {name}")
            row = file_registry.get_by_fingerprint(fingerprint)
            if row is None:
                raise RuntimeError(
                    f"Registry says {fingerprint} is converted but no row found"
                )
            chunked_path = (
                CerebrumPaths()
                .kb_artifacts_path(row["domain"], row["subject"], row["sanitized_name"])
                .with_name(f"{row['sanitized_name']}.chunked.md")
            )
            _embed_step(
                file_info, row["domain"], row["subject"], chunked_path, file_registry
            )
        else:
            print(f"Processing: {name}")
            chunked_path, metadata = _convert_and_chunk_step(file_info, file_registry)
            _embed_step(
                file_info,
                metadata.domain,
                metadata.subject,
                chunked_path,
                file_registry,
            )

        print(f"Completed: {name}")

    except Exception as e:
        print(f"Failed processing {name}: {e}")
        raise


@router.post("/process-file/{file_fingerprint}")
async def process_single_file(request: Request, file_fingerprint: str):
    """
    Queue a file for processing (convert + embed). Runs one file at a
    time process-wide — see /queue/status for position.
    """
    file_registry = request.app.state.file_registry

    if not file_registry.check_inator(file_fingerprint):
        raise HTTPException(status_code=404, detail="File not found")

    if file_registry.check_inator(file_fingerprint, "embedded"):
        return {
            "message": "File already processed",
            "file_fingerprint": file_fingerprint,
            "status": "completed",
        }

    row = file_registry.get_by_fingerprint(file_fingerprint)
    if not row:
        raise HTTPException(status_code=404, detail="File info not found")

    await file_processing_queue.enqueue(
        QueuedJob(
            file_fingerprint=file_fingerprint,
            original_name=row["original_name"],
            fn=lambda: process_single_file_task(row, file_registry),
        )
    )

    return {
        "message": "File queued for processing",
        "file_fingerprint": file_fingerprint,
        "status": "queued",
        "queue": file_processing_queue.status(),
        "progress_stream": f"/knowledgebase/stream-progress/{file_fingerprint}",
    }


@router.get("/queue/status")
async def get_queue_status():
    return file_processing_queue.status()


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
                doc_type=metadata.doc_type,
            )

            # Update file registry
            file_registry.mark_converted_inator(
                file_fingerprint=file_info["file_fingerprint"],
                domain=metadata.domain,
                subject=metadata.subject,
                sanitized_name=metadata.title,
                doc_type=metadata.doc_type,
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
            chunked_path = CerebrumPaths().kb_artifacts_path(
                domain,
                subject,
                sanitized_name,
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


def _user_org_ids(request: Request, user_id: str) -> list:
    """The caller's org ids, resolved from note_registry — the bridge that
    lets the file registry (a separate DB) scope files to a user's orgs."""
    return request.app.state.note_registry.get_user_org_ids(user_id)


@router.get("/show")
async def show_files(
    request: Request, user_id: str = Depends(get_current_user_id)
):
    """Show source files visible to the caller: all public files plus any
    privately granted to them or one of their orgs."""
    file_registry = request.app.state.file_registry
    org_ids = _user_org_ids(request, user_id)
    return file_registry.show_all_inator(user_id=user_id, org_ids=org_ids) or []


# ========================================
# File access (discretionary sharing)
# ========================================


class FileAccessGrant(BaseModel):
    principal_type: str  # 'user' or 'org'
    principal_id: str


@router.get("/access/{file_fingerprint}")
async def list_file_access(
    file_fingerprint: str,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """List a file's grants. Empty list = public. Caller must currently be
    able to see the file (public, or granted to them/their org)."""
    file_registry = request.app.state.file_registry
    org_ids = _user_org_ids(request, user_id)
    if not file_registry.filter_visible([file_fingerprint], user_id, org_ids):
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "file_fingerprint": file_fingerprint,
        "access": file_registry.list_access(file_fingerprint),
    }


@router.post("/access/{file_fingerprint}")
async def grant_file_access(
    file_fingerprint: str,
    body: FileAccessGrant,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """Grant a user/org access to a file. The first grant flips the file from
    public to private. Only someone who can currently see the file may share
    it (prevents scoping a file you can't see); org grants require the caller
    to belong to that org."""
    file_registry = request.app.state.file_registry
    org_ids = _user_org_ids(request, user_id)
    if not file_registry.filter_visible([file_fingerprint], user_id, org_ids):
        raise HTTPException(status_code=404, detail="File not found")
    if body.principal_type == "org" and body.principal_id not in org_ids:
        raise HTTPException(
            status_code=403, detail="You can only grant access to orgs you belong to"
        )
    try:
        file_registry.grant_access(
            file_fingerprint, body.principal_type, body.principal_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"file_fingerprint": file_fingerprint, "granted": body.model_dump()}


@router.delete("/access/{file_fingerprint}")
async def revoke_file_access(
    file_fingerprint: str,
    body: FileAccessGrant,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """Revoke a grant. Removing the last grant makes the file public again."""
    file_registry = request.app.state.file_registry
    org_ids = _user_org_ids(request, user_id)
    if not file_registry.filter_visible([file_fingerprint], user_id, org_ids):
        raise HTTPException(status_code=404, detail="File not found")
    removed = file_registry.revoke_access(
        file_fingerprint, body.principal_type, body.principal_id
    )
    return {"file_fingerprint": file_fingerprint, "revoked": removed}


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


# ========================================
# File Serving Routes
# ========================================


@router.get("/file/{file_fingerprint}")
async def serve_source_file(request: Request, file_fingerprint: str):
    """
    Stream the original source PDF for a fingerprint.

    Always served through the API (never the raw folder) so access stays
    behind daemon auth and behaves identically over localhost or a zrok
    tunnel. FileResponse honors HTTP Range requests, so a client PDF
    viewer can pull a single page without downloading the whole document.
    """
    file_registry = request.app.state.file_registry

    row = file_registry.get_by_fingerprint(file_fingerprint)
    if row is None:
        raise HTTPException(status_code=404, detail="File not found in registry")

    filepath = Path(row["filepath"])
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Source file missing on disk")

    # media_type omitted -> inferred from the file extension (application/pdf
    # for PDFs); inline disposition lets a browser/PDF viewer render in place.
    return FileResponse(
        path=filepath,
        filename=row["original_name"],
        content_disposition_type="inline",
    )


@router.get("/locate/{file_fingerprint}/{chunk_index}")
async def locate_chunk(request: Request, file_fingerprint: str, chunk_index: int):
    """
    Resolve a (file_fingerprint, chunk_index) — the identifiers a search
    result / LLM citation carries — to its page in the source PDF.

    Navigation only: the client jumps its PDF viewer to pdf_page_start.
    Byte offsets are deliberately NOT returned — they index the chunked
    markdown artifact, not the PDF, so they're meaningless to a viewer.
    Use /explain-text to pull the actual text for an explanation flow.
    """
    file_chunk_registry = request.app.state.file_chunk_registry

    chunk = file_chunk_registry.get_chunk(file_fingerprint, chunk_index)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return {
        "file_fingerprint": file_fingerprint,
        "chunk_index": chunk["chunk_index"],
        "chunk_type": chunk["chunk_type"],
        "pdf_page_start": chunk["pdf_page_start"],
        "pdf_page_end": chunk["pdf_page_end"],
        "chapter_title": chunk["chapter_title"],
        "section_title": chunk["section_title"],
        "section_path": json.loads(chunk["section_path"])
        if chunk["section_path"]
        else [],
        "question_number": chunk["question_number"],
        "marks": chunk["marks"],
        "zone": chunk["zone"],
    }


@router.get("/explain-text/{file_fingerprint}/{chunk_index}")
async def explain_text(request: Request, file_fingerprint: str, chunk_index: int):
    """
    Return the actual text of a chunk, resolved by slicing the chunked
    markdown artifact at the chunk's byte span (the same access pattern
    EmbeddInator uses). This is the "feed the LLM / explain this passage"
    endpoint — byte offsets stay a server-side detail; the caller gets text.
    """
    file_registry = request.app.state.file_registry
    file_chunk_registry = request.app.state.file_chunk_registry

    chunk = file_chunk_registry.get_chunk(file_fingerprint, chunk_index)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    row = file_registry.get_by_fingerprint(file_fingerprint)
    if row is None:
        raise HTTPException(status_code=404, detail="File not found in registry")
    if not (row["domain"] and row["subject"] and row["sanitized_name"]):
        raise HTTPException(
            status_code=409, detail="File not yet converted — no chunked artifact"
        )

    chunked_path = (
        CerebrumPaths()
        .kb_artifacts_path(row["domain"], row["subject"], row["sanitized_name"])
        .with_name(f"{row['sanitized_name']}.chunked.md")
    )
    if not chunked_path.exists():
        raise HTTPException(status_code=404, detail="Chunked artifact missing on disk")

    file_bytes = chunked_path.read_bytes()
    content = file_bytes[chunk["byte_start"] : chunk["byte_end"]].decode(
        "utf-8", errors="replace"
    )

    return {
        "file_fingerprint": file_fingerprint,
        "chunk_index": chunk["chunk_index"],
        "chunk_type": chunk["chunk_type"],
        "pdf_page_start": chunk["pdf_page_start"],
        "pdf_page_end": chunk["pdf_page_end"],
        "chapter_title": chunk["chapter_title"],
        "section_title": chunk["section_title"],
        "section_path": json.loads(chunk["section_path"])
        if chunk["section_path"]
        else [],
        "question_number": chunk["question_number"],
        "marks": chunk["marks"],
        "zone": chunk["zone"],
        "content": content,
    }


@router.get("/figures/{file_fingerprint}")
async def list_figures(file_fingerprint: str):
    """List every extracted figure for a document (page, bbox, caption)."""
    return FigureRegisterInator().list_figures(file_fingerprint)


@router.get("/figure/{file_fingerprint}/{figure_index}")
async def serve_figure(request: Request, file_fingerprint: str, figure_index: int):
    """
    Render a single figure as a PNG by cropping its bbox out of the source
    PDF page (with a small margin). The client shows this crop when a chunk
    that references the figure is surfaced; the caption is the LLM-facing
    anchor (see /figures).
    """
    import pymupdf

    file_registry = request.app.state.file_registry
    row = file_registry.get_by_fingerprint(file_fingerprint)
    if row is None:
        raise HTTPException(status_code=404, detail="File not found in registry")

    fig = FigureRegisterInator().get_figure(file_fingerprint, figure_index)
    if fig is None:
        raise HTTPException(status_code=404, detail="Figure not found")

    filepath = Path(row["filepath"])
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Source file missing on disk")

    try:
        x0, y0, x1, y1 = json.loads(fig["bbox"])
        doc = pymupdf.open(filepath)
        page = doc[fig["pdf_page"] - 1]
        clip = pymupdf.Rect(x0 - 4, y0 - 4, x1 + 4, y1 + 4) & page.rect
        pix = page.get_pixmap(clip=clip, dpi=150)
        png = pix.tobytes("png")
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render figure: {e}")

    return Response(content=png, media_type="image/png")


@router.get("/concepts/{file_fingerprint}")
async def list_concepts(file_fingerprint: str):
    """List concept/definition seeds harvested for a document (glossary today)."""
    return ConceptIndexInator().get_by_fingerprint(file_fingerprint)


@router.post("/reset/{status}")
async def reset_registry(
    request: Request, status: str, file_fingerprint: Optional[str] = None
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
        count = len(store.index_to_docstore_id)
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
async def search_collections(
    body: SearchRequest,
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    Search across multiple collections.

    Args:
        query: Search query text
        domains: Optional list of domains to search
        subjects: Optional list of subjects to search
        k: Number of results per collection

    Returns:
        List of matching documents with collection info

    Results are post-filtered to what the caller may see: hits from private
    files not granted to the user (or one of their orgs) are dropped. This is
    enforced at the file-registry level (filter_visible) so no chunk/vector
    metadata has to carry ownership — see routes #3.
    """
    manager = KnowledgebaseManager()
    file_registry = request.app.state.file_registry
    org_ids = _user_org_ids(request, user_id)

    try:
        results = manager.search_across_collections(
            query=body.query,
            domains=body.domains,
            subjects=body.subjects,
            k=body.k,
        )

        fingerprints = [
            (r.get("metadata") or {}).get("file_fingerprint") for r in results
        ]
        visible = set(file_registry.filter_visible(fingerprints, user_id, org_ids))
        # A hit with no fingerprint in metadata can't be access-checked; drop it
        # rather than risk leaking a private file that lost its provenance.
        results = [
            r
            for r in results
            if (r.get("metadata") or {}).get("file_fingerprint") in visible
        ]

        return {
            "query": body.query,
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


# routes.py — add to imports


@router.delete("/delete/{file_fingerprint}")
async def delete_file(request: Request, file_fingerprint: str):
    """
    Fully remove a file: registry entry, source file on disk, markdown
    artifacts (.md/.chunked.md/.pageoffsets.json), and every matching
    document across all vector store collections.
    """
    file_registry = request.app.state.file_registry

    if not file_registry.check_inator(file_fingerprint):
        raise HTTPException(status_code=404, detail="File not found in registry")

    try:
        removed_row = file_registry.remove_inator(file_fingerprint)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Markdown artifacts only exist once a file has been converted —
    # domain/subject/sanitized_name are NULL in the registry until then,
    # so this is skipped (not an error) for files deleted pre-conversion.
    artifacts_warning = None
    if (
        removed_row["domain"]
        and removed_row["subject"]
        and removed_row["sanitized_name"]
    ):
        try:
            artifacts_dir = CerebrumPaths().kb_artifacts_path(
                removed_row["domain"],
                removed_row["subject"],
                removed_row["sanitized_name"],
            )
            if artifacts_dir.exists():
                shutil.rmtree(artifacts_dir)
        except Exception as e:
            artifacts_warning = str(e)

    # Structural-layer cleanup: figures + concept seeds keyed by fingerprint.
    FigureRegisterInator().delete_by_fingerprint(file_fingerprint)
    ConceptIndexInator().delete_by_fingerprint(file_fingerprint)

    vector_count = 0
    try:
        manager = KnowledgebaseManager()
        vector_count = manager.delete_by_fingerprint_all_collections(file_fingerprint)
    except Exception as e:
        return {
            "detail": "File removed from registry, but vector store cleanup failed",
            "file_fingerprint": file_fingerprint,
            "original_name": removed_row["original_name"],
            "vector_documents_deleted": 0,
            "warning": str(e),
        }

    result = {
        "detail": "File removed from knowledgebase successfully",
        "file_fingerprint": file_fingerprint,
        "original_name": removed_row["original_name"],
        "vector_documents_deleted": vector_count,
    }
    if artifacts_warning:
        result["warning"] = f"Markdown artifacts cleanup failed: {artifacts_warning}"
    return result


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
