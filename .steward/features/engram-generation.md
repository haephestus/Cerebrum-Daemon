# Feature: Engram generation

**Status**: DONE (P2)

## Goals
Produce practice material (engrams) from a learner's own notes, asynchronously.

## Scope
- Four engram types (`EngramType`): `mcq`, `flashcard`, `short_question`, `long_question` (`cerebrum_core/engrams/core/types.py`)
- Request: `POST /engrams/{engram_type}/{bubble_id}/{note_id}` → enqueued, not inline; status via `/engrams/jobs/{bubble_id}/{note_id}` plus per-type listing/fetch routes
- Generation queue drained by a dedicated worker (`run_generation_queue_worker`, local mode only — ADR-0007)
- Content persisted through `note_engram_repository/engrams.py`; engrams keep their own vector store (`engrams/storage/vector_store.py`)
- Landing commits 35e048d ("added engram generation") → ba1e2f8 ("improved")

## Dependencies
Note analysis output + KB retrieval as source material ([[features/note-analysis]], [[features/kb-ingestion]]); model management for generator model selection.

## Notes
- Downstream consumer is [[features/engram-grading-mastery]]: attempts, mastery, and the scheduler all key off generated content.
- README §8 still calls this "planned but not yet implemented" — stale; fix tracked in [[plan/hardening]].
