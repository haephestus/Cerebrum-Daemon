# Phase 1 spec — core feature

> Status: **complete** (all items shipped; see [[roadmap]]). Kept as reference.

## Checklist
- [x] KB ingestion end-to-end (`kb_ingest_inator` → `knowledgebase_inator`): Markdown conversion (AppFlowy JSON, PDF), chunking, embedding into FAISS `domain/subject/collection` stores; fingerprints; figures + concept index; per-file access control
- [x] Retrieval + RAG chat: Rose agent (`agents/rose.py`) with context-only policy and explicit "not enough information" fallback; exposed at bubble chat route
- [x] Note analysis active/passive with cache/versioning; page-aware analysis since 817b974
- [x] Notes/bubbles CRUD incl. images and durable block ids

## Verification anchors (commits)
- 67c4898 / 817b974 — page-aware chunking + per-page analysis
- 68fc5e2 — note image upload/serving, persist bubble_id, durable block ids
- ba1e2f8 — improved engram generation (early P2 spillover)

## Notes
- README §3–§4 describe this era's architecture under old root-level module names; current layout is `src/{api,cerebrum_core,database,notes,common,agents}`. Fix tracked in [[plan/hardening]].
