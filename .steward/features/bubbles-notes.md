# Feature: Bubbles & notes CRUD

**Status**: DONE (P1 core; offline-sync aspects continue under [[features/offline-sync]])

## Goals
Study bubbles group notes; notes are structured, page-based documents with images, fully CRUD-able and offline-tolerant.

## Scope
- Bubble CRUD; client-supplied `bubble_id=md5(name)` optional param; identity from Depends user_id not body (CONTRACT)
- Notes: list (empty→404 CONTRACT), get, create honouring client `note_id` verbatim → `<note_id>.json` filename (CONTRACT), update (whole-page-set saves, absent page_id = DELETE, per-page LWW — CONTRACT), rename, delete
- Note model: `Page`/`PageMetadata` + `version_vector` + note_pages accessor (1e58dc4); block-aligned chunking + `chunk_blocks` join (2ba4fc5); durable block ids (68fc5e2)
- Images: upload + absolute-URL response that client does NOT embed (stores `cerebrum-image://<note_id>/<name>` ref) (CONTRACT); concurrent-write manifest corruption fixed 5701a7c

## Dependencies
Database layer registries; disk layout `<base>/<bubble_id>/notes/<note_id>/`.

## Notes
- Recovery path for orphaned DB rows exists (`common/resync_inator.py` rebuilds notes.id from disk folders, reconnecting engrams/mastery/attempts).
