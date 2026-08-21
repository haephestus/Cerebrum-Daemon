# Decisions — Cerebrum-Daemon

> Managed by Steward. Record the reasons behind the shape of this project:
> one entry per decision, newest last. The `Status` line is the contract —
> `accepted`, `proposed`, or `superseded` (link to the entry that replaced it).

## ADR-0001 — Local-first daemon, frontend-agnostic
- **Status**: accepted
- **Date**: 2026-04-28 (repo split; b87ded5)
- **Context**: The AI layer (ingestion, RAG, analysis) must run on consumer hardware without data leaving the machine, while the Flutter client stays presentation-only.
- **Decision**: Backend is a persistent local daemon exposing HTTP, modeled on Ollama's operating model ("start it once, let it run"). Client talks to `localhost`; backend and frontend live in separate repos (`Cerebrum` ⇄ `Cerebrum-Daemon`).
- **Consequences**: Any client can target the API; cloud deployment exists but local remains the primary mode. Wire shapes become load-bearing → ADR-0004.

## ADR-0002 — SQLite-only persistence, registries + repository split
- **Status**: accepted
- **Date**: pre-history (present from initial commits)
- **Context**: Single-user-ish local daemon; no DB server should be a setup requirement on consumer machines.
- **Decision**: All persistence is SQLite: lightweight registries (`file_registry`, `file_chunk_registry`, `note_chunk_registry`) plus the domain-heavy `NoteEngramRepository` package with its own schema/migrations, plus `database/planner/`. Note *content* additionally lives as JSON files on disk (`<bubble>/notes/<note_id>/`).
- **Consequences**: Zero-config startup; concurrent-write discipline required by hand (see 5701a7c manifest fix). Recovery tooling like `common/resync_inator.py` exists to rebuild `notes.id` rows from disk.

## ADR-0003 — FAISS for vector stores (ChromaDB dropped)
- **Status**: accepted
- **Date**: 2026-08 (visible in current tree)
- **Context**: README §3 still names ChromaDB, but `chromadb` is absent from `src/requirements.txt` while `faiss-cpu==1.14.3` is pinned; FAISS usage spans KB search, cache, archive, and engram storage.
- **Decision**: FAISS-backed domain-specific vector stores (`domain/subject/collection` hierarchy in `knowledgebase_inator`); engrams keep their own store under `engrams/storage/vector_store.py`.
- **Consequences**: Docs drift (README stale). Persistence/backups must cover FAISS index files alongside SQLite ([[plan/ship]]).

## ADR-0004 — Cross-repo wire contracts are load-bearing and grep-marked
- **Status**: accepted
- **Date**: 2026-08 (docs/cross-repo-contracts.md)
- **Context**: The Flutter client is offline-first: notes, images, attempts, mastery, engram content are cached on-device and reconciled on reconnect. Silent breakage modes include dropped answers, duplicate records, notes losing pages.
- **Decision**: Every coupling point carries an in-code `CROSS-REPO CONTRACT` marker; `docs/cross-repo-contracts.md` indexes them. Key shapes frozen: whole-page-set saves with per-page LWW, verbatim client-supplied `note_id`s, optional `bubble_id=md5(name)`, empty-bubble→404, idempotent attempt replay via `INSERT OR IGNORE`, daemon-owned `MasteryState`, `question_index` matched against `question_number`.
- **Consequences**: Changing any shape requires updating both repos together. Registry mirrored at [[cross-repo/contracts]].

## ADR-0005 — Version-vector sync engine (gap 1), not ad-hoc syncing
- **Status**: accepted
- **Date**: 2026-08 (commits 2ba4fc5, 1e58dc4, 3fc91a5)
- **Context**: Notes must reconcile across offline devices; naive last-write-wins at file level loses edits.
- **Decision**: Per-node sync bookkeeping in three tables (`replica_identity`, `sync_outbox`, `sync_cursor`); version vectors per note; per-page LWW merge (`notes/sync_merge_inator.py`); push/pull endpoints per bubble/note; a device can talk to >1 hub (LAN daemon + cloud), each with its own cursor.
- **Consequences**: Chunking had to become block-aligned (stream A) so merges resolve below page granularity; durable block ids and stable `page_id`s are now contract-level requirements (ADR-0004).

## ADR-0006 — License-gated ingestion for suggested reading
- **Status**: accepted
- **Date**: 2026-08 (8d92210)
- **Context**: "Free to read" ≠ "free to use commercially." Fetching/embedding third-party content into the KB has different legal posture than linking it.
- **Decision**: `license_policy_inator` decides ingest-vs-pointer per license. Commercial build (default, env `CEREBRUM_COMMERCIAL_USE=true`): CC-BY/CC-BY-SA/CC0/PD ingest; CC-BY-NC/ND/unknown/free-read pointer-only. Non-commercial builds additionally allow NC.
- **Consequences**: Online tiers of suggested reading can ship without legal exposure; ingest pipeline reuse is gated behind policy checks.

## ADR-0007 — Background workers run local-only; cold-boot budget shapes startup
- **Status**: accepted
- **Date**: 2026-08 (cerebrum_inator.py lifespan comments)
- **Context**: On serverless (Leapcell ~10s budget) the process freezes between requests; in-process drain loops silently stall.
- **Decision**: File-processing, generation-queue, and grading workers start only when `deploy_config_inator.is_local()`; model-manifest bake runs as a deferred background task so the listener binds immediately.
- **Consequences**: Cloud mode needs a separate always-on worker service before grading/generation work there ([[plan/ship]]); until then cloud is request-scoped only.

## ADR-0008 — Steward phase skeleton retained, real work mapped onto it
- **Status**: accepted
- **Date**: 2026-08-21
- **Context**: Vault was scaffolded with generic phases (foundation/core/features/hardening/ship) while the codebase was months ahead; renumbering would orphan the "one numbering scheme" rule.
- **Decision**: Keep scaffolded phase numbers/titles as-is; map shipped work onto them with evidence-tagged `[x]`. Project-native "gap 1"/"gap 3" language maps into Phase 2 items, not new phases.
- **Consequences**: Roadmap now reflects reality; future phases must extend Phase 2+ checklists rather than invent parallel numbering.

See [[todo]] for what's open.
