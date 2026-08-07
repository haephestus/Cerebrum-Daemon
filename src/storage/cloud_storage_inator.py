"""
storage.cloud_storage_inator  —  MOCK / SCAFFOLD (not wired yet)
=============================================================================
Design sketch for the cloud storage abstraction (deployment plan steps 4–5).
NOTHING here is imported by the running daemon yet — it's a typed skeleton with
TODOs so the shape is agreed before we spend real integration effort against
Cloudflare. Local mode keeps using the existing SQLite/FS code paths unchanged.

Three seams, each selected by common.deploy_config_inator.is_cloud():

  1. Database      — local: sqlite3 file  | cloud: Cloudflare D1 (SQLite dialect
                      over an HTTP REST API, NOT a socket/file).
  2. ObjectStore   — local: local FS       | cloud: Cloudflare R2 (S3-compatible).
  3. NoteBlobStore — the note's authoritative AppFlowy document delta + ink +
                      history (JSON, backend-owned). local: today's
                      content.json/ink.json folder store | cloud: durable
                      D1 (document/history) + R2 (ink). NOT a string, NOT
                      client-offloaded — the server processes/versions it.

Key correction baked into this design: **D1 is SQLite, not MySQL.** So the SQL
(schema.py, migrations.py, every mixin) ports almost verbatim. The ONLY thing
that changes is how a "connection" runs a statement — file cursor vs HTTP call.
That's why the Database seam below mimics just the slice of the sqlite3
Connection API the repositories actually use (execute / executemany /
executescript / commit / row access), so the mixins don't have to change.

References to the real code this must slot into:
  - cerebrum_core/database/note_engram_repository/_base.py :: _get_conn()
  - cerebrum_core/database/file_registry_inator.py  (raw sqlite3.connect calls)
  - cerebrum_core/utils/file_util_inator.py :: CerebrumPaths (FS roots)
"""

from __future__ import annotations

from typing import Any, BinaryIO, Iterator, Optional, Protocol, Sequence, runtime_checkable

# ---------------------------------------------------------------------------
# 1. DATABASE SEAM
# ---------------------------------------------------------------------------


@runtime_checkable
class Cursorish(Protocol):
    """The slice of sqlite3.Cursor the repositories rely on."""

    def fetchone(self) -> Optional[Sequence[Any]]: ...
    def fetchall(self) -> list[Sequence[Any]]: ...
    @property
    def rowcount(self) -> int: ...
    @property
    def description(self) -> Any: ...


@runtime_checkable
class Connectionish(Protocol):
    """The slice of sqlite3.Connection the repositories rely on. If the D1
    adapter satisfies this, `_get_conn()` can return it and NO mixin changes."""

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Cursorish: ...
    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> Cursorish: ...
    def executescript(self, script: str) -> Cursorish: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


class D1Connection:
    """
    MOCK. A sqlite3.Connection-compatible facade over Cloudflare D1's HTTP API.

    TODO(step 4):
      - Auth/config from env: CLOUDFLARE_ACCOUNT_ID, D1_DATABASE_ID,
        CLOUDFLARE_API_TOKEN. Fail loudly if missing in cloud mode.
      - Endpoint: POST https://api.cloudflare.com/client/v4/accounts/{acct}
        /d1/database/{db}/query  (or /raw). Body: {"sql": ..., "params": [...]}.
      - Map execute() -> single query; executemany() -> D1 /query batch (D1
        supports an array of statements in one call, run as an implicit txn);
        executescript() -> split on ';' and batch (schema bootstrap path).
      - Return a Cursorish wrapping the JSON `results` array. Provide row access
        that behaves like sqlite3.Row (index AND key) since mixins use both
        `row[0]` and `dict(row)` / `row["col"]`. `description` from result meta.
      - rowcount from D1 `meta.changes`; lastrowid from `meta.last_row_id`.
      - commit()/rollback(): D1 auto-commits per request. For multi-statement
        atomicity use the batch endpoint (one HTTP call = one txn). The
        repository's `_transaction()` contextmanager (see _base.py) will need a
        cloud variant that buffers statements and flushes them as one batch.
      - Connection pooling: there are no long-lived connections; each call is
        HTTP. Consider an httpx.Client kept on the adapter for keep-alive.
      - Gotchas: D1 has no PRAGMA foreign_keys/journal_mode (ignore or no-op);
        WAL/busy_timeout are meaningless. 10GB DB cap, per-query row/time limits.
    """

    def __init__(self, account_id: str, database_id: str, api_token: str):
        raise NotImplementedError("D1Connection is a scaffold — TODO step 4")


# ---------------------------------------------------------------------------
# 2. OBJECT STORE SEAM  (uploaded PDFs / artifacts)
# ---------------------------------------------------------------------------


@runtime_checkable
class ObjectStore(Protocol):
    def put(self, key: str, data: BinaryIO, content_type: str = "application/octet-stream") -> None: ...
    def open_stream(self, key: str) -> Iterator[bytes]: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def presigned_url(self, key: str, expires_s: int = 3600) -> str: ...


class LocalFsObjectStore:
    """MOCK. Wraps the current local-FS behaviour so callers can be written
    against ObjectStore now and keep working in local mode.

    TODO(step 4): back onto CerebrumPaths().kb_root_dir(); `key` is the relative
    path under it; open_stream() yields file chunks; presigned_url() returns a
    daemon route (there's no signing locally) that streams the file.
    """

    def __init__(self, root=None):
        raise NotImplementedError("LocalFsObjectStore is a scaffold — TODO step 4")


class R2ObjectStore:
    """MOCK. Cloudflare R2 (S3-compatible) object store for cloud mode.

    TODO(step 4):
      - Config: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET,
        endpoint https://{account}.r2.cloudflarestorage.com.
      - Prefer boto3/aioboto3 with signature v4 (R2 speaks the S3 API) — verify
        it's an available dependency before committing to it.
      - put(): stream upload (multipart for large PDFs) so we don't buffer whole
        files in a serverless instance's small memory.
      - open_stream(): range/streamed GET, wired into FastAPI StreamingResponse.
      - presigned_url(): generate a real S3 presigned GET so the client pulls
        large files straight from R2 instead of proxying bytes through the daemon
        (this is the preferred download path in cloud mode).
      - delete()/exists() map to S3 DeleteObject/HeadObject.
    """

    def __init__(self, bucket: str, endpoint: str, access_key: str, secret_key: str):
        raise NotImplementedError("R2ObjectStore is a scaffold — TODO step 4")


# ---------------------------------------------------------------------------
# 3. NOTE STORAGE SEAM  (AppFlowy document + ink + history)
# ---------------------------------------------------------------------------
#
# CORRECTION to an earlier assumption: a note is NOT a plain string. It's an
# **AppFlowy Editor document delta** (a block tree; text lives in
# block.data.delta[].insert) plus an **ink layer** (scribble/Sketch strokes),
# plus backend-maintained **metadata + version history** (NoteMetadata +
# NoteHistory's ContentDiff/InkDiff op lists). See model_inator.py
# (NoteContent/NoteStorage) and note_util_inator.py (_save_note/_load_note ->
# notes/<id>/content.json + ink.json).
#
# The backend OWNS this entirely: it persists the JSON, applies diff-ops for
# versioning, and flattens the delta -> markdown (NoteToMarkdownInator) to feed
# the chunk/embed/analyse pipeline. The client (note_editor_controller ->
# bubbles_api.createNote/updateNote) is a thin editor that POSTs
# {title, content:{document}, ink:[...]} and stores nothing authoritative.
#
# Current reality: the whole note is one monolithic JSON file on disk
# (notes/<id>/content.json = {title, content:{document}, ink, metadata,
# history}). The DB `notes.content` column is effectively DEAD — its only
# writer, create_note(), has no callers, so it stays ''.
#
# Target model (NORMALISE, per product direction): keep ONLY the delta as JSON;
# give every other field its own column/row. Applies to both modes — local
# SQLite and cloud D1 share one normalised shape, retiring the content.json blob:
#
#   notes (extend existing table):
#     title            TEXT
#     document         TEXT   -- the AppFlowy delta JSON (NoteContent.document);
#                                repurpose the dead `content` column or add this
#     content_hash     TEXT
#     content_version  REAL
#     ink_hash         TEXT
#     ink_version      REAL
#     last_modified    TEXT   -- (or reuse updated_at)
#     ink              TEXT   -- local; in cloud -> R2 object (strokes get large)
#   note_history (NEW table — history is a list, so it's rows not a blob):
#     note_id, kind ('content'|'ink'), version, ts, ops (JSON)
#
# The delta and each diff's `ops` stay JSON (nested block trees — not worth
# shredding); everything scalar becomes a column, history becomes rows.
#
# Consequences for cloud mode:
#   - "notes as strings" and "client-offload" are OUT — the server holds the
#     document (it processes + versions it); the JSON is intrinsic to the editor.
#   - D1 stores the normalised notes row + note_history; R2 holds ink blobs
#     (hash mirrored in notes.ink_hash). Migrate content.json/ink.json into this
#     shape on cutover.


class NoteBlobStore(Protocol):
    """Reads/writes a note's authoritative JSON blobs, keyed by (note_id,
    user_id). `document` is the AppFlowy NoteContent.document dict; `ink` is the
    list of stroke dicts. Mirrors note_util_inator's content.json / ink.json."""

    def put(self, note_id: str, user_id: str, document: dict, ink: list) -> None: ...
    def get(self, note_id: str, user_id: str) -> Optional[dict]: ...  # {document, ink, metadata, history}


class LocalFolderNoteStore:
    """MOCK. local mode — the existing on-disk folder store.

    TODO(step 5):
      - Wrap note_util_inator's _save_note/_load_note (notes/<id>/content.json +
        ink.json) behind this interface so callers are storage-agnostic. No
        behaviour change locally; this just gives the folder store the same
        shape as the cloud one.
    """

    def __init__(self, notes_dir=None):
        raise NotImplementedError("LocalFolderNoteStore is a scaffold — TODO step 5")


class CloudNoteStore:
    """MOCK. cloud mode — durable, NORMALISED note storage; server stays source
    of truth.

    TODO(step 5):
      - notes row: title, document (delta JSON), content_hash/version,
        ink_hash/version, last_modified — all first-class columns in D1, NOT a
        blob. get() rehydrates a NoteStorage from the row + note_history.
      - note_history table: one row per ContentDiff/InkDiff (kind, version, ts,
        ops JSON). Reuse the existing diff-op versioning logic as-is.
      - ink -> R2 object (notes/<user_id>/<note_id>/ink.json), hash mirrored in
        notes.ink_hash. Stream large ink rather than buffering.
      - Keep flatten->markdown (NoteToMarkdownInator) server-side; only the
        source of the document JSON moves (D1 row instead of content.json).
      - Migrate content.json/ink.json folders into this shape on cutover.
    """

    def __init__(self, d1: "D1Connection", ink_store: "R2ObjectStore"):
        raise NotImplementedError("CloudNoteStore is a scaffold — TODO step 5")


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------


def get_database_connection() -> Connectionish:
    """TODO(step 4): local -> existing sqlite3 conn (from _base._get_conn);
    cloud -> D1Connection(from env). Route the repositories' _get_conn through
    here so a single switch flips the whole DB layer."""
    raise NotImplementedError("storage factory is a scaffold — TODO steps 4–5")


def get_object_store() -> ObjectStore:
    """TODO(step 4): local -> LocalFsObjectStore ; cloud -> R2ObjectStore."""
    raise NotImplementedError("storage factory is a scaffold — TODO steps 4–5")


def get_note_store() -> NoteBlobStore:
    """TODO(step 5): local -> LocalFolderNoteStore() (wraps the current
    content.json/ink.json store) ; cloud -> CloudNoteStore(D1, R2)."""
    raise NotImplementedError("storage factory is a scaffold — TODO steps 4–5")
