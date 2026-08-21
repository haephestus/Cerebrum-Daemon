# Architecture — Cerebrum-Daemon

> Backend engine of Cerebrum (learning assistant). Python 3.12 target / 3.13 dev env. AGPL-3.0.
> Sibling: Flutter client `haephestus/Cerebrum` (offline-first). Contracts: [[cross-repo/contracts]].

## Shape
```
Flutter client ──HTTP──▶ cerebrum_inator.py (FastAPI, uvicorn :8000)
                            │ DaemonAuthMiddleware (X-Daemon-Key local | bearer cloud)
                            ├─ api/        9 routers: user·org·knowledgebase·bubble·
                            │              learning_center·study_plan·suggested_reading·sync·test
                            ├─ cerebrum_core/   domain logic: KB, engrams (gen/grading/
                            │              scheduler/mastery), analysis, planner,
                            │              suggested reading, profiles, users/topics
                            ├─ notes/      block_chunker·chunk_analyser·markdown_handler·
                            │              chunking_queue·sync_merge·sync_service·note_util
                            ├─ database/   SQLite registries + NoteEngramRepository
                            │             (+migrations) + planner package + sync_store
                            ├─ common/    ollama_compat·license_policy·cache·archive·
                            │             content_fetch·deploy_config·resync recovery
                            └─ agents/    rose.py — grounded RAG answer prompts
                                   ▼
                     Ollama (local LLMs) · FAISS stores · SQLite · note file tree
```

## Runtime model (lifespan in `cerebrum_inator.py`)
- Five registries bound to `app.state`: FileRegisterInator, NoteEngramRepository, FileChunkRegisterInator, NoteChunkRegisterInator, StudyPlanRegisterInator.
- Ollama catalog scraped once → `<config>/models_manifest.json`, deferred to a thread so the listener binds immediately.
- **Background workers gated by `is_local()`** (ADR-0007): file-processing queue (`notes/chunking_queue_inator`), engram generation-queue worker, grading worker draining `grading_jobs`. Serverless freezes stall loops → cloud needs an always-on worker service ([[plan/ship]]).
- Grading worker launched with `vector_store=None, embedder=None`: answer-embedding/regression path deliberately off.

## Storage
- SQLite everywhere (ADR-0002) + JSON files on disk for note content: `<base>/<bubble_id>/notes/<note_id>/content.json`, per-page folders since 1e58dc4.
- FAISS vector stores organized `domain/subject/collection` (ADR-0003; README's ChromaDB claim is stale).
- Recovery: `common/resync_inator.py` re-inserts notes.id rows from disk folders, reconnecting engrams/mastery/attempts by id.

## Auth
Local: shared daemon key minted at boot, printed to stdout, persisted at `<config>/daemon_api_key.txt`; validated by middleware before any route. Cloud: user bearer tokens; password reset flow via email/reset-token modules. CORS wraps auth so preflight passes; current wildcard+credentials combo is a known P3 fix ([[plan/hardening]]).

## Deployment modes
Local daemon (primary) or serverless cloud (Leapcell ~10s cold-boot budget cited in-code). `deploy_config_inator.is_local()` is the switch for workers and auth mode.
