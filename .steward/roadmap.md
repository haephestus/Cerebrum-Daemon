# Cerebrum-Daemon — Roadmap

> Phase source of truth. Structure syncs to the PM tracker on save.
> One numbering scheme everywhere — this doc, [[todo]], and every in-code
> `TODO: PHASE N` anchor use the *same* phase numbers.
>
> **Phase 0 is the first phase.**
>
> Population note (2026-08-21): skeleton phases kept as scaffolded; items below
> are mapped from actual code/git state. `[x]` only where a commit or shipped
> module demonstrates it. See [[decisions]] ADR-0008 for the mapping rationale.

## Phase 0 — foundation

> core tech exists: data model, API service, storage, auth

See [[plan/foundation]] for detailed spec.

- [x] FastAPI service: lifespan wiring of 5 registries, CORS + DaemonAuthMiddleware, OpenAPI security schemes (`cerebrum_inator.py`)
- [x] SQLite persistence layer: file/note-chunk registries + `NoteEngramRepository` package with schema migrations (`database/`)
- [x] Users, orgs, accounts, password-reset flow (`routes_user`, `routes_org`, `api/auth/`)
- [x] Dual auth modes: X-Daemon-Key (local) / bearer token (cloud) (`daemon_auth_inator`, `daemon_auth_middleware`)
- [x] Ollama compat layer + boot-time model-manifest bake (`common/ollama_compat/`)
- [x] Model management: installed/online/cloud model listing, download, chat/embedding/cloud selection (`routes_user`)

## Phase 1 — core feature

> the main capability works end to end

See [[plan/core-feature]] for detailed spec.

- [x] KB ingestion pipeline end-to-end: upload → Markdown conversion → chunking → embedding → domain vector stores; fingerprints, figures, concepts, collections, access control (`kb_ingest_inator`, `knowledgebase_inator`, `routes_knowledgebase`)
- [x] Retrieval + grounded RAG chat via Rose agent, context-only answering policy (`agents/rose.py`, `POST /{bubble_id}/chat`)
- [x] Note analysis (active/passive) with caching + versioning; page-aware since 817b974 (`learning_center_inator`, `routes_learning_center`)
- [x] Notes/bubbles CRUD: page model, block-aligned chunking, durable block ids, image upload/serving (commits 2ba4fc5, 68fc5e2)

## Phase 2 — features

> secondary capabilities added

See [[plan/features]] for detailed spec.

- [x] Engram generation queue: mcq / flashcard / short_question / long_question (`engram_generator_inator`, generation-queue worker)
- [x] Engram grading: sync mcq/flashcard submit; async short/long via `grading_jobs` drained by worker (`grading/worker.py`, `ai_grading.py`)
- [x] SM-2 scheduler extended with cognitive-level gating + regression-aware intervals; daemon-owned mastery state (`engrams/scheduler/`, `mastery_service.py`)
- [x] Study planner: plan generation, phases, weeks, metrics, densify, progress, admin review queue (`study_planner_inator`, `database/planner/`, `routes_study_plan`)
- [x] Learning profiles + inference (`learning_profile_inator`, `learning_profile_inference_inator`)
- [x] Offline sync gap-1 streams A/B/C: block-aligned chunking + registry repair (2ba4fc5), page model + version vectors (1e58dc4), version-vector merge engine + sync store + push service (3fc91a5); `/sync push|pull|replica-id` live ([[features/offline-sync]])
- [x] Suggested reading gap-3 Phase 0: KB-first offline provider, orchestrator, license policy, accept/dismiss lifecycle (c30d1f3, 8d92210) ([[features/suggested-reading]])
- [ ] Suggested reading later phases: online tiers + accept→ingest→KB loop
- [ ] Gap-1 closeout: reconcile remaining stream scope against client offline-first layer
- [ ] Grading answer-embedding/regression path — currently gated off (`vector_store=None, embedder=None` at worker launch, `cerebrum_inator.py`)

## Phase 3 — hardening

> tests, failure surfaces, security pass

See [[plan/hardening]] for detailed spec.

- [ ] Inventory `src/tests/` coverage vs subsystems; prioritize sync/merge engine + contract surfaces
- [ ] Fix CORS `allow_origins=["*"]` combined with `allow_credentials=True` (invalid per browser spec; `cerebrum_inator.py`)
- [ ] Repo hygiene: `.gitignore` is empty (0 bytes) while `src/logs/cerebrum_debug.log`, `src/.env/`, `src/.direnv/` live in-tree
- [ ] README corrections: ChromaDB → FAISS (§3), stale root-level module names (§3–4), "engram generation planned" claim (§8, landed in ba1e2f8/35e048d)
- [ ] Failure-mode pass: concurrent-write paths beyond 5701a7c's manifest fix; worker crash/restart semantics
- [ ] Security pass: upload surface, image serving path traversal, password-reset tokens, cloud bearer scope

## Phase 4 — ship

> deployable and backed up

See [[plan/ship]] for detailed spec.

- [ ] Dual-mode deployment story documented: local daemon vs serverless (Leapcell ~10s cold-boot budget already shapes startup; `deploy_config_inator.is_local()` gates workers)
- [ ] Cloud-mode replacement for in-process workers (generation + grading queues stall when process freezes)
- [ ] Backup/restore procedure for SQLite DBs + vector stores + note file tree
- [ ] Release tagging across main/develop/staging/feature branch layout


## What to resist

- **Don't put environment work ahead of product.**
- **Don't let the LLM invent project structure.** Deterministic, always.
- **Don't let the LLM compute money.** Ever.
- **Don't reintroduce a second phase numbering.**
- **Don't touch wire shapes without grepping `CROSS-REPO CONTRACT`.**
