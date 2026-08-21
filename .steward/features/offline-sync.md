# Feature: Offline sync (gap 1)

**Status**: IN PROGRESS (P2) — streams A/B/C committed; closeout vs client pending

## Goals
Notes edited on offline devices reconcile without loss or duplication, across multiple hubs.

## Scope
- **Stream A** (2ba4fc5 + 21a1440): block-aligned note chunking + `chunk_blocks` join; `note_chunks` registry repair; owner-resolution fix
- **Stream B** (1e58dc4): page model — `Page`/`PageMetadata`, `version_vector` per note, `note_pages` accessor
- **Stream C** (3fc91a5): version-vector merge engine (`notes/sync_merge_inator.py`), push service (`notes/sync_service_inator.py`), sync store (`database/sync_store_inator.py`: replica_identity / sync_outbox / sync_cursor)
- Endpoints: `/sync/push/{bubble}/{note}`, `/sync/pull/{bubble}/{note}`, `/sync/replica-id`
- Client-side contract: whole-page-set saves, absent `page_id` = DELETE, LWW reconcile (ADR-0004/0005)

## Dependencies
[[features/bubbles-notes]] block ids; [[features/users-orgs-auth]] identity.

## Notes
- Open: reconcile remaining stream scope against the client's offline-first layer ([[roadmap]] P2).
- Multi-hub is designed-in: a device holds one cursor *per peer* (LAN daemon + cloud).
