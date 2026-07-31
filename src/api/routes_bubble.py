import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import jsonpatch
import ulid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from agents.rose import RosePrompts
from cerebrum_core.constants import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from cerebrum_core.learning_center_inator import passive_analysis
from cerebrum_core.model_inator import (
    ContentDiff,
    CreateStudyBubble,
    InkDiff,
    NoteBase,
    NoteContent,
    NoteOut,
    NoteStorage,
    StudyBubble,
    UserConfig,
)
from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.archive_inator import AnalysisArchiveInator
from cerebrum_core.utils.cache_inator import AnalysisCacheInator
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.note_util_inator import (
    _delete_note_files,
    _load_note,
    _load_note_skip_ink,
    _note_dir,
    _note_exists,
    _note_needs_ink_migration,
    _save_note,
    diff_collapser_inator,
)
from cerebrum_core.utils.retrieve_inator import RetrieverInator

bubble_router = APIRouter(prefix="/bubbles", tags=["Study Bubble API"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------ UTILITIES ------------------------------ #
# TODO: move to note_util_inator?
def hash_obj(obj: Any) -> str:
    """Return MD5 hash of object JSON strin."""
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def get_user_config():
    return ConfigManager().load_config()


# TODO: move to note_util_inator?
def ensure_valid_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the document has valid AppFlowy structure with delta fields.
    """
    if not document:
        return {
            "type": "page",
            "children": [{"type": "paragraph", "data": {"delta": [{"insert": ""}]}}],
        }

    # Ensure children exist
    if "children" not in document or not isinstance(document["children"], list):
        document["children"] = [
            {"type": "paragraph", "data": {"delta": [{"insert": ""}]}}
        ]
        return document

    # Validate each child has delta
    for child in document["children"]:
        if isinstance(child, dict):
            if child.get("type") == "paragraph":
                if "data" not in child:
                    child["data"] = {}
                if isinstance(child["data"], dict) and "delta" not in child["data"]:
                    child["data"]["delta"] = [{"insert": ""}]

    return document


# TODO: move to note_util_inator?
def extract_total_text(doc):
    text_chunks = []
    for child in doc.get("children", []):
        for op in child["data"].get("delta", []):
            text_chunks.append(op.get("insert", ""))
    return "".join(text_chunks)


# TODO: move to note_util_inator?
def calculate_version_increment(old_doc: dict, new_doc: dict) -> float:
    """
    Rules:
      - If text added > 100 chars OR new children > 10 → major bump (+1)
      - Else → minor bump (+0.01)
    """
    # Entire text
    old_text = extract_total_text(old_doc)
    new_text = extract_total_text(new_doc)

    added_chars = len(new_text) - len(old_text)

    # Child changes (block additions)
    old_children = old_doc.get("children", [])
    new_children = new_doc.get("children", [])
    added_children = max(0, len(new_children) - len(old_children))

    # ----- Decision -----

    if added_chars > 125 or added_children > 10:
        return 1.0  # major bump
    elif added_chars > 75 or added_children > 5:
        return 0.1  # medium bump
    elif added_chars > 0:
        return 0.01  # minor bump
    else:
        return 0.0  # unchanged


# ------------------------------ NOTE PACKAGING ------------------------------ #
#
# Note storage (folder-vs-legacy-flat-file handling, atomic writes, etc.)
# now lives in cerebrum_core.utils.note_util_inator — see that module's
# docstring for the full layout. This file just imports the helpers it
# needs (_note_dir, _note_exists, _load_note, _load_note_skip_ink,
# _note_needs_ink_migration, _save_note, _delete_note_files) rather than
# defining them, so learning_center_inator.py (and anything else that
# touches note storage) can share the exact same logic instead of
# re-implementing — or subtly diverging from — it.


# --------------------------- STUDY BUBBLE CRUD -------------------------- #


@bubble_router.get("/", response_model=List[StudyBubble])
def list_study_bubbles():
    STUDY_BUBBLES_ROOT = CerebrumPaths().bubbles_root_dir()
    """
    List all study bubbles.
    """
    bubbles = []
    for folder in STUDY_BUBBLES_ROOT.iterdir():
        if not folder.is_dir():
            continue

        info_file = folder / "info.json"
        if not info_file.exists():
            continue

        data = json.loads(info_file.read_text())

        bubbles.append(StudyBubble(**data))

    return bubbles


@bubble_router.post("/create")
def create_study_bubble(data: CreateStudyBubble) -> StudyBubble:
    """
    Create a study bubble folder and info file.
    """
    bubble_id = hashlib.md5(data.name.encode()).hexdigest()
    bubble = CerebrumPaths().bubble_path(bubble_id)

    if bubble.exists():
        raise HTTPException(status_code=400, detail="Bubble already exists")

    CerebrumPaths().init_bubble_dirs(bubble_id=bubble_id)

    bubble_data = StudyBubble(
        id=bubble_id,
        name=data.name,
        description=data.description,
        domains=data.domains,
        user_goals=data.user_goals,
        created_at=datetime.now(),
    )

    info_file = bubble / "info.json"
    info_file.write_text(bubble_data.model_dump_json(indent=4), encoding="utf-8")

    return bubble_data


@bubble_router.get("/{bubble_id}")
def get_study_bubble(bubble_id: str) -> StudyBubble:
    """
    Fetch a single study bubble's info.
    """

    bubble_path = CerebrumPaths().bubble_path(bubble_id)
    info_file = bubble_path / "info.json"

    if not info_file.exists():
        raise HTTPException(status_code=404, detail="Study bubble not found")

    data = json.loads(info_file.read_text())
    return StudyBubble(**data)


@bubble_router.delete("/{bubble_id}")
def delete_study_bubble(bubble_id: str):
    """
    Delete a bubble and its notes.
    """
    bubble_path = CerebrumPaths().bubble_path(bubble_id)

    if not bubble_path.exists():
        raise HTTPException(status_code=404, detail="Study bubble not found")

    # Recursively delete the folder
    shutil.rmtree(bubble_path)

    return {"detail": "Study bubble deleted successfully"}


# ------------------------------- NOTES CRUD ------------------------------ #


# List notes
@bubble_router.get("/{bubble_id}/notes", response_model=List[NoteOut])
def list_notes_in_bubble(bubble_id: str):
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)

    filenames: set[str] = set()

    # Folder-form notes
    for entry in notes_dir.iterdir():
        if entry.is_dir() and (entry / "content.json").exists():
            filenames.add(f"{entry.name}.json")

    # Legacy flat-file notes not yet migrated
    for file in notes_dir.glob("*.json"):
        filenames.add(file.name)

    notes = []
    for filename in filenames:
        storage_data = _load_note_skip_ink(notes_dir, filename)
        content_obj = NoteContent(**storage_data["content"])
        notes.append(
            NoteOut(
                title=storage_data["title"],
                content=content_obj,
                # Intentionally not loading ink for a list view — see
                # the note-packaging comment in note_util_inator.py.
                ink=[],
                filename=filename,
                version=storage_data["metadata"]["content_version"],
            )
        )
    return notes


# Create a new note
@bubble_router.post("/{bubble_id}/create/notes", response_model=NoteOut)
def create_note(request: Request, bubble_id: str, note: NoteBase):
    note_registry = request.app.state.note_registry
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    notes_dir.mkdir(parents=True, exist_ok=True)
    note.content.document = ensure_valid_document(note.content.document)

    # safe_title = note.title.replace(" ", "_")
    note_id = ulid.ulid()
    filename = f"{note_id}.json"

    # NOTE: registry gets the note's identifying path as
    # `notes_dir / filename` — same as before this change. That path no
    # longer literally points at a file (the real files are
    # `notes_dir/<note_id>/content.json` and `.../ink.json` now), but I
    # don't have visibility into what note_registry actually does with
    # this path. If it does more than use it as an opaque identifier
    # (e.g. checks existence, watches it, reads it directly), this will
    # need to point at `_note_dir(notes_dir, filename)` instead — flagging
    # rather than guessing.
    note_registry.register_inator(
        note_id=note_id, bubble_id=bubble_id, filepath=str(notes_dir / filename)
    )
    # Avoid collisions
    # Obsolete? because of uuids?
    """
    counter = 1
    while filepath.exists():
        filename = f"{safe_title}_{counter}.json"
        filepath = notes_dir / filename
        counter += 1
    """

    storage = NoteStorage(
        note_id=note_id,
        title=note.title,
        bubble_id=bubble_id,
        content=note.content,
        ink=note.ink or [],
    )
    storage.metadata.content_hash = hash_obj(storage.content.model_dump())
    storage.metadata.ink_hash = hash_obj(storage.ink)

    _save_note(notes_dir, filename, storage, write_ink=True)

    return NoteOut(
        title=storage.title,
        content=storage.content,
        ink=storage.ink or [],
        filename=filename,
        version=int(storage.metadata.content_version),
    )


# Get a single note
@bubble_router.get("/{bubble_id}/notes/get/{filename}", response_model=NoteOut)
def get_note(bubble_id: str, filename: str):
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)

    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")

    stored_note = _load_note(notes_dir, filename)

    # Ensure the document is valid before returning
    stored_note.content.document = ensure_valid_document(stored_note.content.document)

    return NoteOut(
        title=stored_note.title,
        content=stored_note.content,
        ink=stored_note.ink,
        filename=filename,
        version=stored_note.metadata.content_version,
        analyse_note=stored_note.analyse_note,
    )


'''
@bubble_router.post("/{bubble_id}/debug/notes")
async def debug_create_note(bubble_id: str, request: Request):
    """Temporary debug endpoint"""
    body = await request.json()
    logger.info(f"Received body: {json.dumps(body, indent=2)}")
    return {"received": body}
'''


@bubble_router.post(
    "/{bubble_id}/notes/toggle_analysis/{filename}", response_model=NoteOut
)
def toggle_note_analysis(bubble_id: str, filename: str):
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")

    # 1. Load (transparently handles folder vs legacy flat file)
    stored_note = _load_note(notes_dir, filename)

    # 2. Flat toggle (No if-else blocks)
    stored_note.analyse_note = not stored_note.analyse_note

    # 3. Save directly (Bypassing registry, diff history, and background
    #    tasks). write_ink=True unconditionally — this isn't autosave
    #    frequency, so migrating any legacy note here is a fine trade-off
    #    over the rare redundant ink write.
    _save_note(notes_dir, filename, stored_note, write_ink=True)

    # 4. Standard Response matching your schema
    return NoteOut(
        filename=filename,
        title=stored_note.title,
        content=stored_note.content,
        ink=stored_note.ink,
        bubble_id=stored_note.bubble_id,
        version=int(stored_note.metadata.content_version),
        analyse_note=stored_note.analyse_note,
    )


# Update a note
@bubble_router.put("/{bubble_id}/notes/update/{filename}", response_model=NoteOut)
def update_note(
    request: Request,
    bubble_id: str,
    filename: str,
    note: NoteBase,
    background_tasks: BackgroundTasks,
):
    note_registry = request.app.state.note_registry
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)

    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")

    if not note_registry.check_inator(note_id=note.note_id):
        note_registry.register_inator(
            note_id=note.note_id,
            bubble_id=note.bubble_id,
            filepath=str(notes_dir / filename),
        )
    # ------------------------------------------------------------------
    # Load existing note (transparently handles folder vs legacy flat)
    # ------------------------------------------------------------------
    stored_note = _load_note(notes_dir, filename)

    # Ensure document validity
    note.content.document = ensure_valid_document(note.content.document)

    old_doc = stored_note.content.document
    new_doc = note.content.document

    # ------------------------------------------------------------------
    # Versioning decision
    # ------------------------------------------------------------------
    increment = calculate_version_increment(old_doc, new_doc)
    is_created = stored_note.metadata.content_version == 0
    is_major = increment >= 1.0

    old_content = stored_note.content.model_dump()
    new_content = note.content.model_dump()
    print(f"Contents equal: {old_content == new_content}")
    print(f"Version increment: {increment}")
    # ------------------------------------------------------------------
    # CONTENT DIFF + VERSION BUMP
    # ------------------------------------------------------------------
    if old_content != new_content:
        patch_ops = jsonpatch.make_patch(old_content, new_content).patch

        stored_note.history.content.append(
            ContentDiff(
                version=stored_note.metadata.content_version,
                ts=datetime.now(),
                ops=patch_ops,
            )
        )
        print()

        # Apply version bump
        if is_major:
            stored_note.metadata.content_version = (
                int(stored_note.metadata.content_version) + 1
            )
        else:
            stored_note.metadata.content_version += increment

        stored_note.content = note.content
        stored_note.metadata.content_hash = hash_obj(new_content)

    # ------------------------------------------------------------------
    # INK SNAPSHOT (deliberately NOT diffed) + PACKAGING
    # ------------------------------------------------------------------
    # Ink is snapshotted, not diffed — see the packaging comment in
    # note_util_inator.py. `needs_ink_write` covers both "ink actually
    # changed" and "this note is mid-migration and ink.json doesn't
    # exist yet" — content-only autosaves on an already-migrated note
    # hit neither condition, so ink.json isn't touched at all.
    new_ink_hash = hash_obj(note.ink)
    ink_changed = new_ink_hash != stored_note.metadata.ink_hash
    needs_ink_write = ink_changed or _note_needs_ink_migration(notes_dir, filename)

    if ink_changed:
        stored_note.metadata.ink_version += 1
        stored_note.metadata.ink_hash = new_ink_hash

    stored_note.ink = note.ink

    # ------------------------------------------------------------------
    # DIFF COMPRESSION
    # ------------------------------------------------------------------
    stored_note = diff_collapser_inator(stored_note)

    # ------------------------------------------------------------------
    # SAVE NOTE
    # ------------------------------------------------------------------
    _save_note(notes_dir, filename, stored_note, write_ink=needs_ink_write)

    # ------------------------------------------------------------------
    # 🚀 AUTOMATIC ANALYSIS TRIGGER (MAJOR BUMPS ONLY)
    # ------------------------------------------------------------------
    # Left keyed on content version only, on purpose — analysing a
    # Sketch blob isn't something rose_note_analyser does today. If
    # handwriting/ink analysis becomes a feature, this trigger is where
    # it'd need a second condition alongside `is_major or is_created`.
    should_analyze = is_major or is_created
    if should_analyze:
        logger.info(
            f"Major version bump detected for note {stored_note.note_id} "
            f"(v{stored_note.metadata.content_version}) — scheduling analysis"
        )

        prompt = RosePrompts.get_prompt("rose_note_analyser")

        if prompt:
            cache_manager = AnalysisCacheInator(
                bubble_id=bubble_id,
                note_id=stored_note.note_id,
            )

            # Avoid duplicate scheduling
            cache_info = cache_manager.get_cache_info()
            if (
                not cache_info
                or cache_info["content_version"] != stored_note.metadata.content_version
            ):
                background_tasks.add_task(
                    passive_analysis,
                    note=stored_note,
                    prompt=prompt,
                    bubble_id=bubble_id,
                )
        else:
            logger.warning("Analysis prompt not found — skipping analysis")

    # ------------------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------------------
    print(f"DEBUG: Received ink length: {len(note.ink)}")
    return NoteOut(
        filename=filename,
        title=stored_note.title,
        content=stored_note.content,
        ink=stored_note.ink,
        bubble_id=stored_note.bubble_id,
        version=int(stored_note.metadata.content_version),
    )


# ------------------------------------------------------------------
# Rename a note
# ------------------------------------------------------------------
class RenamePayload(BaseModel):
    title: str


@bubble_router.put("/{bubble_id}/notes/rename/{filename}", response_model=NoteOut)
def rename_note(bubble_id: str, filename: str, payload: RenamePayload):
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)

    if not _note_exists(notes_dir, filename):
        raise HTTPException(404, "Note not found")

    # Load (transparently handles folder vs legacy flat file)
    note = _load_note(notes_dir, filename)
    note.title = payload.title

    # write_ink=True unconditionally, same reasoning as
    # toggle_note_analysis: not autosave-frequency, so migrating any
    # legacy note here is a fine trade-off over the rare redundant write.
    _save_note(notes_dir, filename, note, write_ink=True)

    return NoteOut(**note.model_dump(), filename=filename)


# Delete a note
@bubble_router.delete("/{bubble_id}/notes/delete/{filename}")
def delete_note(request: Request, bubble_id: str, filename: str):
    note_registry = request.app.state.note_registry
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    note_id = filename.strip(".json")

    # 1. Check if it exists. If not, maybe it's already gone—return 204 or 404
    if not _note_exists(notes_dir, filename):
        # Optional: You might want to call the registry clean up anyway here
        raise HTTPException(status_code=404, detail="Note not found")

    # 2. Load BEFORE deleting — need this for the archive cleaner. Was
    #    `json.loads(filepath.read_text())` directly against a single
    #    flat file; `_load_note` handles folder vs legacy transparently.
    stored_note = _load_note(notes_dir, filename)

    # 3. Clean up the Archive (External/Related records)
    try:
        AnalysisArchiveInator(
            note=stored_note,
            archives_path=str(_note_dir(notes_dir, filename)),
        ).archive_cleaner_inator()
    except Exception as e:
        logger.error(f"Archive cleaning failed for {note_id}: {e}")

    # 4. Remove from Registry (Database)
    # Ensure this happens BEFORE the files are removed
    note_registry.remove_inator(note_id=note_id, filepath=str(notes_dir / filename))

    # 5. Optionally: Call your reset_inator if you need to set flags to 0
    # note_registry.reset_inator(status="cached", note_id=note_id)

    # 6. Physical deletion — the note's folder (content.json + ink.json)
    #    and/or a legacy flat file, whichever exists.
    _delete_note_files(notes_dir, filename)

    return {"detail": "Note deleted successfully"}


# ---------------------------- CHAT ENDPOINT ------------------------------ #


class Query(BaseModel):
    text: str


# TODO: index notes directly linked to the current bubbleid
@bubble_router.post("/{bubble_id}/chat")
async def chat_in_bubble(
    # bubble_id: str,
    query: Query,
    config: UserConfig = Depends(get_user_config),
):
    """
    Chat inside a specific study bubble.
    """
    archives_root = CerebrumPaths().kb_archives_path()
    chat_model = config.models.chat_model or DEFAULT_CHAT_MODEL
    embedding_model = config.models.embedding_model or DEFAULT_EMBED_MODEL
    translation_prompt = RosePrompts.get_prompt("rose_query_translator")
    processor = RetrieverInator(
        archives_root=str(archives_root),
        embedding_model=embedding_model,
    )

    # TODO: find a better alternative than assert
    assert translation_prompt is not None
    # TRANSLATE USER QUERY
    # CONSTRUCT CONTEXT
    # TODO: cache responses for bubbles
    # RETRIEVE from knowledgebase and from note/.archives
    # GENERATE RESPONSE
    return
