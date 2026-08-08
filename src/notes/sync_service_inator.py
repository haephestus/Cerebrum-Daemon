"""
notes.sync_service_inator — the hub half of note sync (gap 1 / stream C).

Receives a replica's version of a note, merges it into the stored one with the
version-vector engine, and persists the result. This is what a device's push
lands on (the LAN daemon or cloud hub). Pull-since-cursor is deferred — it needs
per-note change tracking and the client's other half to exercise.
"""

from __future__ import annotations

from common.file_util_inator import CerebrumPaths
from models.model_inator import NoteStorage
from notes.note_util_inator import _load_note, _note_exists, _save_note
from notes.sync_merge_inator import NoteMergeResult, merge_note


def sync_push_note(
    bubble_id: str, note_id: str, incoming: NoteStorage
) -> NoteMergeResult:
    """Merge an incoming note version into the stored one and persist.

    First push of a note (nothing stored yet) is accepted as-is. Otherwise the
    stored and incoming replicas are merged page-by-page (LWW on concurrent,
    ink union), and the merged note is saved. Returns the merge result so the
    caller can hand back the merged note + which pages had concurrent conflicts.
    """
    notes_dir = CerebrumPaths().note_root_dir(bubble_id)
    filename = f"{note_id}.json"

    if _note_exists(notes_dir, filename):
        stored = _load_note(notes_dir, filename)
        result = merge_note(stored, incoming)
    else:
        result = NoteMergeResult(note=incoming, conflicted_pages=[])

    _save_note(notes_dir, filename, result.note)
    return result
