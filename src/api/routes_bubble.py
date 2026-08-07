import hashlib
import json
import logging
import shutil
from datetime import datetime
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
from cerebrum_core.utils.user_context_inator import get_current_user_id

bubble_router = APIRouter(prefix="/bubbles", tags=["Study Bubble API"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------ UTILITIES ------------------------------ #
def hash_obj(obj: Any) -> str:
    """Return MD5 hash of object JSON strin."""
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def get_user_config():
    return ConfigManager().load_config()


def ensure_valid_document(document: Dict[str, Any]) -> Dict[str, Any]:
    if not document:
        return {
            "type": "page",
            "children": [{"type": "paragraph", "data": {"delta": [{"insert": ""}]}}],
        }
    if "children" not in document or not isinstance(document["children"], list):
        document["children"] = [
            {"type": "paragraph", "data": {"delta": [{"insert": ""}]}}
        ]
        return document
    for child in document["children"]:
        if isinstance(child, dict):
            if child.get("type") == "paragraph":
                if "data" not in child:
                    child["data"] = {}
                if isinstance(child["data"], dict) and "delta" not in child["data"]:
                    child["data"]["delta"] = [{"insert": ""}]
    return document


def extract_total_text(doc):
    text_chunks = []
    for child in doc.get("children", []):
        for op in child["data"].get("delta", []):
            text_chunks.append(op.get("insert", ""))
    return "".join(text_chunks)


def calculate_version_increment(old_doc: dict, new_doc: dict) -> float:
    old_text = extract_total_text(old_doc)
    new_text = extract_total_text(new_doc)
    added_chars = len(new_text) - len(old_text)
    old_children = old_doc.get("children", [])
    new_children = new_doc.get("children", [])
    added_children = max(0, len(new_children) - len(old_children))
    if added_chars > 125 or added_children > 10:
        return 1.0
    elif added_chars > 75 or added_children > 5:
        return 0.1
    elif added_chars > 0:
        return 0.01
    else:
        return 0.0


# ------------------------------ OWNERSHIP ------------------------------ #
def _load_bubble_or_404(bubble_id: str) -> dict:
    """Loads a bubble's info.json as a raw dict, or raises 404."""
    bubble_path = CerebrumPaths().bubble_path(bubble_id)
    info_file = bubble_path / "info.json"
    if not info_file.exists():
        raise HTTPException(status_code=404, detail="Study bubble not found")
    return json.loads(info_file.read_text())


def _assert_bubble_owner(bubble_id: str, user_id: str) -> dict:
    """
    Loads a bubble and verifies the requesting user owns it.

    ASSUMPTION: this assumes info.json has a "user_id" field once you
    add it to the StudyBubble model (see note below create_study_bubble).
    Bubbles created before that migration won't have this field --
    data.get("user_id") returns None for them, which will always fail
    the ownership check. You'll want a one-off script to backfill
    user_id onto existing bubble info.json files, or just accept that
    pre-migration bubbles become inaccessible (fine if this is still
    early dev with disposable data).
    """
    data = _load_bubble_or_404(bubble_id)
    if data.get("user_id") != user_id:
        # 404, not 403 -- don't reveal that a bubble exists but isn't yours
        raise HTTPException(status_code=404, detail="Study bubble not found")
    return data


# --------------------------- STUDY BUBBLE CRUD -------------------------- #


@bubble_router.get("/", response_model=List[StudyBubble])
def list_study_bubbles(user_id: str = Depends(get_current_user_id)):
    """List study bubbles belonging to the requesting user."""
    STUDY_BUBBLES_ROOT = CerebrumPaths().bubbles_root_dir()
    bubbles = []
    for folder in STUDY_BUBBLES_ROOT.iterdir():
        if not folder.is_dir():
            continue
        info_file = folder / "info.json"
        if not info_file.exists():
            continue
        data = json.loads(info_file.read_text())
        if data.get("user_id") != user_id:
            continue
        bubbles.append(StudyBubble(**data))
    return bubbles


@bubble_router.post("/create")
def create_study_bubble(
    data: CreateStudyBubble, user_id: str = Depends(get_current_user_id)
) -> StudyBubble:
    """
    Create a study bubble folder and info file, owned by the requesting
    user.

    REQUIRES a model_inator.py change: add `user_id: str` to the
    StudyBubble pydantic model. I don't have that file's contents, so
    I'm not editing it blindly -- add the field yourself, or paste me
    the file and I'll do it precisely. Without it, `StudyBubble(**data)`
    calls throughout this file (including here) will reject the extra
    "user_id" key unless the model already allows extra fields.
    """
    bubble_id = hashlib.md5(data.name.encode()).hexdigest()
    bubble = CerebrumPaths().bubble_path(bubble_id)

    if bubble.exists():
        raise HTTPException(status_code=400, detail="Bubble already exists")

    CerebrumPaths().init_bubble_dirs(bubble_id=bubble_id)

    bubble_data = StudyBubble(
        id=bubble_id,
        user_id=user_id,
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
def get_study_bubble(
    bubble_id: str, user_id: str = Depends(get_current_user_id)
) -> StudyBubble:
    """Fetch a single study bubble's info -- only if you own it."""
    data = _assert_bubble_owner(bubble_id, user_id)
    return StudyBubble(**data)


@bubble_router.delete("/{bubble_id}")
def delete_study_bubble(bubble_id: str, user_id: str = Depends(get_current_user_id)):
    """Delete a bubble and its notes -- only if you own it."""
    _assert_bubble_owner(bubble_id, user_id)
    bubble_path = CerebrumPaths().bubble_path(bubble_id)
    shutil.rmtree(bubble_path)
    return {"detail": "Study bubble deleted successfully"}


# ------------------------------- NOTES CRUD ------------------------------ #
# Notes don't carry their own owner check -- ownership is inherited from
# the bubble they live in (a bubble belongs to exactly one user, per the
# schema comment on notes.user_id: "owner; every note belongs to exactly
# one user"). So every note route below calls _assert_bubble_owner first;
# if that passes, the user is allowed to touch notes in that bubble.


@bubble_router.get("/{bubble_id}/notes", response_model=List[NoteOut])
def list_notes_in_bubble(bubble_id: str, user_id: str = Depends(get_current_user_id)):
    _assert_bubble_owner(bubble_id, user_id)
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)

    filenames: set[str] = set()
    for entry in notes_dir.iterdir():
        if entry.is_dir() and (entry / "content.json").exists():
            filenames.add(f"{entry.name}.json")
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
                ink=[],
                filename=filename,
                version=storage_data["metadata"]["content_version"],
            )
        )
    return notes


@bubble_router.post("/{bubble_id}/create/notes", response_model=NoteOut)
def create_note(
    request: Request,
    bubble_id: str,
    note: NoteBase,
    user_id: str = Depends(get_current_user_id),
):
    """
    REQUIRES a note_registry.register_inator signature change: it needs
    to accept and store user_id, since notes.user_id is NOT NULL in your
    schema. I don't have that file, so I'm passing user_id through here
    on the assumption the signature will accept it -- if register_inator
    doesn't take a user_id kwarg yet, this will TypeError until you (or
    I, if you share that file) add it.
    """
    _assert_bubble_owner(bubble_id, user_id)

    note_registry = request.app.state.note_registry
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    notes_dir.mkdir(parents=True, exist_ok=True)
    note.content.document = ensure_valid_document(note.content.document)

    note_id = ulid.ulid()
    filename = f"{note_id}.json"

    note_registry.register_inator(
        note_id=note_id,
        bubble_id=bubble_id,
        user_id=user_id,
        filepath=str(notes_dir / filename),
    )

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


@bubble_router.get("/{bubble_id}/notes/get/{filename}", response_model=NoteOut)
def get_note(
    bubble_id: str, filename: str, user_id: str = Depends(get_current_user_id)
):
    _assert_bubble_owner(bubble_id, user_id)
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)

    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")

    stored_note = _load_note(notes_dir, filename)
    stored_note.content.document = ensure_valid_document(stored_note.content.document)

    return NoteOut(
        title=stored_note.title,
        content=stored_note.content,
        ink=stored_note.ink,
        filename=filename,
        version=stored_note.metadata.content_version,
        analyse_note=stored_note.analyse_note,
    )


@bubble_router.post(
    "/{bubble_id}/notes/toggle_analysis/{filename}", response_model=NoteOut
)
def toggle_note_analysis(
    bubble_id: str, filename: str, user_id: str = Depends(get_current_user_id)
):
    _assert_bubble_owner(bubble_id, user_id)
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")

    stored_note = _load_note(notes_dir, filename)
    stored_note.analyse_note = not stored_note.analyse_note
    _save_note(notes_dir, filename, stored_note, write_ink=True)

    return NoteOut(
        filename=filename,
        title=stored_note.title,
        content=stored_note.content,
        ink=stored_note.ink,
        bubble_id=stored_note.bubble_id,
        version=int(stored_note.metadata.content_version),
        analyse_note=stored_note.analyse_note,
    )


@bubble_router.put("/{bubble_id}/notes/update/{filename}", response_model=NoteOut)
def update_note(
    request: Request,
    bubble_id: str,
    filename: str,
    note: NoteBase,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    _assert_bubble_owner(bubble_id, user_id)
    note_registry = request.app.state.note_registry
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)

    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")

    if not note_registry.check_inator(note_id=note.note_id):
        note_registry.register_inator(
            note_id=note.note_id,
            bubble_id=note.bubble_id,
            user_id=user_id,
            filepath=str(notes_dir / filename),
        )

    stored_note = _load_note(notes_dir, filename)
    note.content.document = ensure_valid_document(note.content.document)

    old_doc = stored_note.content.document
    new_doc = note.content.document

    increment = calculate_version_increment(old_doc, new_doc)
    is_created = stored_note.metadata.content_version == 0
    is_major = increment >= 1.0

    old_content = stored_note.content.model_dump()
    new_content = note.content.model_dump()

    if old_content != new_content:
        patch_ops = jsonpatch.make_patch(old_content, new_content).patch
        stored_note.history.content.append(
            ContentDiff(
                version=stored_note.metadata.content_version,
                ts=datetime.now(),
                ops=patch_ops,
            )
        )
        if is_major:
            stored_note.metadata.content_version = (
                int(stored_note.metadata.content_version) + 1
            )
        else:
            stored_note.metadata.content_version += increment

        stored_note.content = note.content
        stored_note.metadata.content_hash = hash_obj(new_content)

    new_ink_hash = hash_obj(note.ink)
    ink_changed = new_ink_hash != stored_note.metadata.ink_hash
    needs_ink_write = ink_changed or _note_needs_ink_migration(notes_dir, filename)

    if ink_changed:
        stored_note.metadata.ink_version += 1
        stored_note.metadata.ink_hash = new_ink_hash

    stored_note.ink = note.ink
    stored_note = diff_collapser_inator(stored_note)

    _save_note(notes_dir, filename, stored_note, write_ink=needs_ink_write)

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

    return NoteOut(
        filename=filename,
        title=stored_note.title,
        content=stored_note.content,
        ink=stored_note.ink,
        bubble_id=stored_note.bubble_id,
        version=int(stored_note.metadata.content_version),
    )


class RenamePayload(BaseModel):
    title: str


@bubble_router.put("/{bubble_id}/notes/rename/{filename}", response_model=NoteOut)
def rename_note(
    bubble_id: str,
    filename: str,
    payload: RenamePayload,
    user_id: str = Depends(get_current_user_id),
):
    _assert_bubble_owner(bubble_id, user_id)
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)

    if not _note_exists(notes_dir, filename):
        raise HTTPException(404, "Note not found")

    note = _load_note(notes_dir, filename)
    note.title = payload.title
    _save_note(notes_dir, filename, note, write_ink=True)

    return NoteOut(**note.model_dump(), filename=filename)


@bubble_router.delete("/{bubble_id}/notes/delete/{filename}")
def delete_note(
    request: Request,
    bubble_id: str,
    filename: str,
    user_id: str = Depends(get_current_user_id),
):
    _assert_bubble_owner(bubble_id, user_id)
    note_registry = request.app.state.note_registry
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    note_id = filename.strip(".json")

    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")

    stored_note = _load_note(notes_dir, filename)

    try:
        AnalysisArchiveInator(
            note=stored_note,
            archives_path=str(_note_dir(notes_dir, filename)),
        ).archive_cleaner_inator()
    except Exception as e:
        logger.error(f"Archive cleaning failed for {note_id}: {e}")

    note_registry.remove_inator(note_id=note_id, filepath=str(notes_dir / filename))
    _delete_note_files(notes_dir, filename)

    return {"detail": "Note deleted successfully"}


# ---------------------------- CHAT ENDPOINT ------------------------------ #


class Query(BaseModel):
    text: str


@bubble_router.post("/{bubble_id}/chat")
async def chat_in_bubble(
    bubble_id: str,
    query: Query,
    config: UserConfig = Depends(get_user_config),
    user_id: str = Depends(get_current_user_id),
):
    """Chat inside a specific study bubble -- only if you own it."""
    _assert_bubble_owner(bubble_id, user_id)

    archives_root = CerebrumPaths().kb_archives_path()
    chat_model = config.models.chat_model or DEFAULT_CHAT_MODEL
    embedding_model = config.models.embedding_model or DEFAULT_EMBED_MODEL
    translation_prompt = RosePrompts.get_prompt("rose_query_translator")
    processor = RetrieverInator(
        archives_root=str(archives_root),
        embedding_model=embedding_model,
    )

    assert translation_prompt is not None
    return
