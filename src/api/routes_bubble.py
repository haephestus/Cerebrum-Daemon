import hashlib
import json
import logging
import shutil
from datetime import datetime
from typing import Any, Dict, List

import jsonpatch
import ulid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agents.rose import RosePrompts
from cerebrum_core.constants import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from cerebrum_core.learning_center_inator import passive_analysis
from models.model_inator import (
    ContentDiff,
    CreateStudyBubble,
    Note,
    NoteInput,
    NoteManifest,
    NoteOut,
    Page,
    PageManifest,
    StudyBubble,
    UserConfig,
)
from cerebrum_core.user_inator import ConfigManager
from common.archive_inator import AnalysisArchiveInator
from common.cache_inator import AnalysisCacheInator
from common.file_util_inator import CerebrumPaths
from notes.note_util_inator import (
    _delete_note_files,
    _load_note,
    _load_note_skip_ink,
    _note_dir,
    _note_exists,
    _note_needs_ink_migration,
    _save_note,
    diff_collapser_inator,
)
from vectorstore.retrieve_inator import RetrieverInator
from cerebrum_core.user_context_inator import get_current_user_id

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


def _note_out(note: Note, filename: str) -> NoteOut:
    """A Note wrapped as the HTTP response model (adds `filename`)."""
    return NoteOut(manifest=note.manifest, pages=note.pages, filename=filename)


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
        # New per-page folder layout uses manifest.json; the previous single
        # layout used content.json — accept either so both are listed.
        if entry.is_dir() and (
            (entry / "manifest.json").exists() or (entry / "content.json").exists()
        ):
            filenames.add(f"{entry.name}.json")
    for file in notes_dir.glob("*.json"):  # legacy flat notes
        filenames.add(file.name)

    notes = []
    for filename in filenames:
        note = _load_note_skip_ink(notes_dir, filename)
        notes.append(_note_out(note, filename))
    return notes


@bubble_router.post("/{bubble_id}/create/notes", response_model=NoteOut)
def create_note(
    request: Request,
    bubble_id: str,
    note: NoteInput,
    user_id: str = Depends(get_current_user_id),
):
    _assert_bubble_owner(bubble_id, user_id)

    note_registry = request.app.state.note_registry
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    notes_dir.mkdir(parents=True, exist_ok=True)

    note_id = ulid.ulid()
    filename = f"{note_id}.json"

    note_registry.register_inator(
        note_id=note_id,
        bubble_id=bubble_id,
        user_id=user_id,
        filepath=str(notes_dir / filename),
    )

    # A note always has at least one page; synthesise an empty first page if the
    # client sent none.
    incoming = note.pages or [Page(page_id="p1", page_index=0)]
    pages: list[Page] = []
    for idx, p in enumerate(incoming):
        doc = ensure_valid_document(p.document)
        meta = PageManifest(
            page_id=p.page_id,
            page_index=p.page_index if p.page_index else idx,
            content_hash=hash_obj({"document": doc}),
            ink_hash=hash_obj(p.ink),
        )
        pages.append(
            Page(
                page_id=p.page_id,
                page_index=meta.page_index,
                document=doc,
                ink=p.ink or [],
                metadata=meta,
            )
        )

    manifest = NoteManifest(
        title=note.title,
        note_id=note_id,
        bubble_id=bubble_id,
        content_hash=hash_obj({p.page_id: p.document for p in pages}),
    )
    stored = Note(manifest=manifest, pages=pages)

    _save_note(notes_dir, filename, stored, write_ink=True)
    # Reload so the response carries exactly the persisted pages.
    saved = _load_note(notes_dir, filename)
    return _note_out(saved, filename)


@bubble_router.get("/{bubble_id}/notes/get/{filename}", response_model=NoteOut)
def get_note(
    bubble_id: str, filename: str, user_id: str = Depends(get_current_user_id)
):
    _assert_bubble_owner(bubble_id, user_id)
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)

    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")

    stored_note = _load_note(notes_dir, filename)
    for p in stored_note.pages:
        p.document = ensure_valid_document(p.document)

    return _note_out(stored_note, filename)


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
    stored_note.manifest.analyse_note = not stored_note.manifest.analyse_note
    _save_note(notes_dir, filename, stored_note, write_ink=True)

    return _note_out(stored_note, filename)


@bubble_router.put("/{bubble_id}/notes/update/{filename}", response_model=NoteOut)
def update_note(
    request: Request,
    bubble_id: str,
    filename: str,
    note: NoteInput,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    """Reconcile the whole note from the client's full `pages[]`:
      * a page_id present in both → diff its document, bump that page's version
        (and note-level version) if it changed; union its ink hash;
      * a page_id only in the payload → ADD it;
      * a stored page_id absent from the payload → DELETE it (dropped here, its
        folder + analysis.json GC'd by _save_note);
      * page_index carries reorder (folders never rename — order lives in the
        note manifest's page_order map).
    """
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

    # Safety: an empty `pages[]` would reconcile to zero pages and wipe the note.
    # A real note always has ≥1 page, so treat this as a no-op rather than a wipe
    # (guards a client that posts a malformed/empty body).
    if not note.pages:
        logger.warning(
            "update_note: empty pages[] for %s — skipping to avoid wiping the note",
            filename,
        )
        return _note_out(stored_note, filename)

    stored_by_id = {p.page_id: p for p in stored_note.pages}

    is_created = stored_note.manifest.content_version == 0
    any_major = False
    content_changed = False
    ink_changed_any = False

    reconciled: list[Page] = []
    for idx, incoming in enumerate(note.pages):
        new_doc = ensure_valid_document(incoming.document)
        prev = stored_by_id.get(incoming.page_id)

        if prev is None:
            # ADD — a page_id the server hasn't seen.
            meta = PageManifest(
                page_id=incoming.page_id,
                page_index=incoming.page_index if incoming.page_index else idx,
                content_hash=hash_obj({"document": new_doc}),
                ink_hash=hash_obj(incoming.ink),
            )
            reconciled.append(
                Page(
                    page_id=incoming.page_id,
                    page_index=meta.page_index,
                    document=new_doc,
                    ink=incoming.ink or [],
                    metadata=meta,
                )
            )
            content_changed = True
            any_major = True  # a new page is a structural (major) change
            ink_changed_any = True  # ensure the new page's ink.json is written
            continue

        # EXISTING — diff this page in place.
        page = prev
        page.page_index = incoming.page_index if incoming.page_index else idx

        old_content = {"document": page.document}
        new_content = {"document": new_doc}
        if old_content != new_content:
            content_changed = True
            increment = calculate_version_increment(page.document, new_doc)
            is_major = increment >= 1.0
            page.history.content.append(
                ContentDiff(
                    version=page.metadata.content_version,
                    ts=datetime.now(),
                    ops=jsonpatch.make_patch(old_content, new_content).patch,
                )
            )
            if is_major:
                page.metadata.content_version = int(page.metadata.content_version) + 1
                any_major = True
            else:
                page.metadata.content_version += increment
            page.document = new_doc
            page.metadata.content_hash = hash_obj(new_content)

        new_ink_hash = hash_obj(incoming.ink)
        if new_ink_hash != page.metadata.ink_hash:
            page.metadata.ink_version += 1
            page.metadata.ink_hash = new_ink_hash
            page.ink = incoming.ink
            ink_changed_any = True

        reconciled.append(page)

    # Pages not carried into `reconciled` are deletions — _save_note GCs their
    # folders (and analysis.json) since they're no longer present.
    stored_note.pages = reconciled
    stored_note.manifest.page_order = {p.page_id: p.page_index for p in reconciled}

    # Persist the note's identity from the authoritative URL path. The manifest
    # historically never stored bubble_id (it defaults to ""), so NoteOut echoed
    # an EMPTY bubble_id — and a client that merged the response back lost its
    # bubble_id and sent later saves to `/bubbles//notes/update/...` (404). Stamp
    # it (and note_id) here so it's saved and returned correctly from now on.
    stored_note.manifest.bubble_id = bubble_id
    if not stored_note.manifest.note_id:
        stored_note.manifest.note_id = note.note_id

    if content_changed:
        if any_major:
            stored_note.manifest.content_version = (
                int(stored_note.manifest.content_version) + 1
            )
        else:
            stored_note.manifest.content_version += 0.01
        stored_note.manifest.content_hash = hash_obj(
            {p.page_id: p.document for p in reconciled}
        )
    stored_note.manifest.last_modified = datetime.now()

    stored_note = diff_collapser_inator(stored_note)

    needs_ink_write = ink_changed_any or _note_needs_ink_migration(notes_dir, filename)
    _save_note(notes_dir, filename, stored_note, write_ink=needs_ink_write)

    should_analyze = any_major or is_created
    if should_analyze:
        logger.info(
            f"Major version bump detected for note {stored_note.note_id} "
            f"(v{stored_note.manifest.content_version}) — scheduling analysis"
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
                or cache_info["content_version"]
                != stored_note.manifest.content_version
            ):
                background_tasks.add_task(
                    passive_analysis,
                    note=stored_note,
                    prompt=prompt,
                    bubble_id=bubble_id,
                )
        else:
            logger.warning("Analysis prompt not found — skipping analysis")

    return _note_out(stored_note, filename)


# Images embedded in a note live alongside it: notes/<note_id>/images/<id>.<ext>.
# A note references an image by URL (the GET route below); the bytes are stored
# in the note's own folder so they travel/GC with the note (delete the note →
# its images go too, via shutil.rmtree of the note dir).
_ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


@bubble_router.post("/{bubble_id}/notes/{filename}/images")
async def upload_note_image(
    bubble_id: str,
    filename: str,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """Store an uploaded image in the note's own `images/` folder and return the
    URL the client embeds in the AppFlowy image block. The image id is an
    unguessable ULID — that id is the capability that the (auth-exempt) GET route
    relies on, since <img>/Image.network can't attach the daemon key."""
    _assert_bubble_owner(bubble_id, user_id)
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    if not _note_exists(notes_dir, filename):
        raise HTTPException(status_code=404, detail="Note not found")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_IMAGE_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{ext}'. Allowed: "
            f"{', '.join(sorted(_ALLOWED_IMAGE_EXTS))}",
        )

    images_dir = _note_dir(notes_dir, filename) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_name = f"{ulid.ulid()}{ext}"
    (images_dir / image_name).write_bytes(await file.read())

    note_id = filename[:-5] if filename.endswith(".json") else filename
    # Relative URL — the client prepends its configured base URL, so the same
    # note works regardless of which host/port the daemon is reached on.
    return {
        "image_name": image_name,
        "url": f"/bubbles/{bubble_id}/notes/{note_id}/images/{image_name}",
    }


@bubble_router.get("/{bubble_id}/notes/{note_id}/images/{image_name}")
def get_note_image(bubble_id: str, note_id: str, image_name: str):
    """Serve a note image. Auth-exempt (see DaemonAuthMiddleware) because it's
    loaded via Image.network, which can't send the daemon key; the unguessable
    ULID in `image_name` is the access capability. Path-traversal guarded."""
    if "/" in image_name or "\\" in image_name or ".." in image_name:
        raise HTTPException(status_code=400, detail="Invalid image name")
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    image_path = _note_dir(notes_dir, note_id) / "images" / image_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


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
    note.manifest.title = payload.title
    _save_note(notes_dir, filename, note, write_ink=True)

    return _note_out(note, filename)


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
