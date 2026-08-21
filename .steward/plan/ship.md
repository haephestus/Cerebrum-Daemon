# Phase 4 spec — ship

> Status: **not started**.

## Subtasks
- [ ] Document dual-mode deployment: local daemon (primary, Ollama-style) vs serverless cloud (Leapcell ~10s cold-boot budget)
- [ ] Cloud worker service: generation-queue + grading workers are gated to `is_local()` (ADR-0007) — serverless freezes stall them; needs always-on companion or managed queue
- [ ] Backup/restore runbook covering: all SQLite DBs, FAISS index files, note file tree (`<base>/<bubble_id>/notes/<note_id>/`), config dir incl. daemon key + models_manifest.json
- [ ] Branch strategy resolution: main / develop / staging / feature all exist on origin; define release flow and tag scheme
- [ ] Version the API explicitly (OpenAPI still says "1.0.0")

## Notes
- Ship is blocked by [[plan/hardening]]'s security pass; do not deploy cloud mode before CORS/token scope fixes.
