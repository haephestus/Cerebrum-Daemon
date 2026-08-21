# Phase 0 spec — foundation

> Status: **complete** (all items shipped; see [[roadmap]]). Kept as reference.

## Checklist
- [x] FastAPI service skeleton + lifespan wiring (`cerebrum_inator.py`): 5 registries on `app.state`, deferred manifest bake, worker gating via `is_local()`
- [x] Middleware: DaemonAuth (X-Daemon-Key local / bearer cloud), CORS ordered outermost so preflight passes
- [x] SQLite layer: registries (`file_registry`, `file_chunk_registry`, `note_chunk_registry`) + `NoteEngramRepository` with migrations (`database/note_engram_repository/migrations.py`, standalone `migrate_add_note_owner.py`)
- [x] Auth surface: accounts, login, password reset request/verify/update, orgs + members
- [x] Ollama compat: invoker + parser, master-manifest scrape → `<config>/models_manifest.json`
- [x] Model management endpoints incl. downloads and cloud/local selection

## Notes
- Daemon key minted once, printed at startup, persisted at `<config>/daemon_api_key.txt`.
- Serverless cold-boot budget (~10s Leapcell) already constrained startup design here — see ADR-0007.
