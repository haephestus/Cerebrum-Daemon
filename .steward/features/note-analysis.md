# Feature: Note analysis

**Status**: DONE (P1, page-aware since 817b974)

## Goals
Structured feedback on user notes grounded strictly in retrieved source material.

## Scope
- Active analysis (per bubble/note/filename) and passive analysis paths through `learning_center_inator` (mirrors planner's service-layer discipline: router never touches registries directly)
- Page-aware since 817b974: `page_id` recorded in `note_chunks`, per-page `analysis.json`
- Caching + versioning: analysis cached per note version; invalidate endpoint; `analysis_status` polling; full-analysis fetch variant
- Archive endpoints per bubble/note (+ clear)

## Dependencies
[[features/kb-ingestion]] retrieval; [[features/bubbles-notes]] page model.

## Notes
- Analysis keys on `page_id`s — churning them breaks client sync (CROSS-REPO CONTRACT).
