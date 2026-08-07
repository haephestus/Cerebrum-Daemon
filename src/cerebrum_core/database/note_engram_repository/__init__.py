"""
cerebrum_core.engrams.storage.note_engram_repository
======================================================
Merged replacement for the old pair of:
  - cerebrum_core.database.note_registry.NoteRegisterInator
  - cerebrum_core.engrams.storage.sqlite_repository.SQLiteRepository

Why merged: both classes owned a table describing "a note" (note_registry.note_id
vs notes.id), living in two separate .db files with no way to JOIN them. This
repository owns a single `notes` table used both for ingestion-pipeline tracking
(cached/analysed/filepath) and as the note content engrams hang off of.

This package is one class, NoteEngramRepository, assembled from mixins —
each file below owns one table or one closely related group of tables,
and contributes methods without any per-mixin state of its own:

    _base.py             connection/lock/schema plumbing every mixin relies on
    schema.py             the CREATE TABLE statements (see that file for the
                           per-table rationale/changelog)
    content_codecs.py     read/write (de)serialization for the four engram
                           content shapes (mcq/flashcard/short_question/long_question)
    notes.py               NotesMixin   — notes table (registry + content)
    users.py                UsersMixin   — users table
    engrams.py               EngramsMixin — engrams table + typed constructors
    attempts.py               AttemptsMixin — engram_attempts + *_responses
    mastery.py                  MasteryMixin  — engram_mastery + topic_mastery
    misconceptions.py            MisconceptionsMixin — misconceptions table
    grading_jobs.py                GradingJobsMixin — grading_jobs table
    generation_queue.py             GenerationQueueMixin — engram_generation_queue

Drop-in compatibility: every method that used to live on NoteRegisterInator
(register_inator, mark_cached_inator, mark_analysed_inator,
fetch_uncached_notes_inator, fetch_unanalysed_notes_inator, check_inator,
show_all_inator, remove_inator, reset_inator) keeps its exact name and
signature here, so existing call sites only need to change the import:

    - from cerebrum_core.database.note_registry_inator import NoteRegisterInator
    + from cerebrum_core.engrams.storage.note_engram_repository import NoteEngramRepository as NoteRegisterInator

The mastery/engram side implements the current MasteryRepository ABC
(mastery_service.py): get_topic_masteries, upsert_topic_mastery,
get_topic_mastery, get_topic_engrams(user_id, topic).

Usage (existing connection, e.g. in-memory DB for tests):
    import sqlite3
    from cerebrum_core.engrams.storage.note_engram_repository import NoteEngramRepository

    db   = sqlite3.connect(":memory:")
    repo = NoteEngramRepository(db)  # schema created automatically

Usage (production, resolves the on-disk registry path for you):
    from cerebrum_core.engrams.storage.note_engram_repository import NoteEngramRepository

    repo = NoteEngramRepository.open()  # or NoteEngramRepository.open("registry/notes.db")
"""

from __future__ import annotations

from ...engrams.core.mastery_service import MasteryRepository
from ._base import _RepositoryBase
from .attempts import AttemptsMixin
from .engrams import EngramsMixin
from .generation_queue import GenerationQueueMixin
from .grading_jobs import GradingJobsMixin
from .mastery import MasteryMixin
from .misconceptions import MisconceptionsMixin
from .notes import NotesMixin
from .orgs import OrgsMixin
from .topics import TopicsMixin
from .users import UsersMixin

__all__ = ["NoteEngramRepository"]


class NoteEngramRepository(
    _RepositoryBase,
    NotesMixin,
    TopicsMixin,
    UsersMixin,
    OrgsMixin,
    EngramsMixin,
    AttemptsMixin,
    MasteryMixin,
    MisconceptionsMixin,
    GradingJobsMixin,
    GenerationQueueMixin,
    MasteryRepository,
):
    """
    Owns notes (ingestion tracking + content), engrams, attempts, mastery,
    misconceptions, and job queues in a single SQLite file.

    This class itself defines no methods — everything comes from the
    mixins above, each scoped to one table or table-family (see the
    module docstring for the full map). _RepositoryBase supplies the
    connection/lock plumbing every mixin assumes is available via
    self._get_conn() / self._lock.

    Thread-safety: unlike the old SQLiteRepository (one held connection),
    this opens a short-lived connection per call, guarded by a lock for
    writes — matching NoteRegisterInator's pattern, since this is what
    gets used from a threaded FastAPI app via app.state.
    """
