# Phase 3 spec — hardening

> Status: **not started**. Items below are observed facts from the 2026-08-21 vault population pass, not speculation.

## Subtasks
### Tests
- [ ] Inventory `src/tests/` coverage against subsystems (api routes ×9, cerebrum_core domains, notes/sync_merge engine, database repos)
- [ ] Priority targets: sync merge engine (pure logic, high blast radius), attempts idempotency (`INSERT OR IGNORE` CONTRACT), grading job state machine (`pending|processing|done|failed`)
### Failure surfaces
- [ ] Concurrent-write audit beyond 5701a7c (manifest fix): remaining SQLite multi-writer paths, outbox drain races
- [ ] Worker crash/restart semantics: generation + grading queues must resume, not duplicate
### Security pass
- [ ] CORS `allow_origins=["*"]` + `allow_credentials=True` — invalid combo per browser spec; pick real origins or drop credentials
- [ ] Upload surface: file-type enforcement, archive handling (`archive_inator`), image serving path traversal on `{image_name}`
- [ ] Password-reset token lifecycle (request/verify/update) — rate limits, token entropy, expiry
- [ ] Cloud bearer scope vs local daemon key equivalence review
### Hygiene
- [ ] `.gitignore` is empty (0 bytes): ignore `src/.env/`, `src/.direnv/`, `src/logs/`, venvs, `__pycache__`; untrack `src/logs/cerebrum_debug.log`
- [ ] Root `TODO.md` duplicates `.steward/todo.md` template — remove one
- [ ] README fixes: ChromaDB → FAISS (§3); root-level module names → current `src/` layout (§3–4); "engram generation planned" is false (§8); requirements say Python 3.12 but local dev runs 3.13 (.direnv/.env)

## Notes
- Do not start P4 ship before the security pass lands ([[plan/ship]] depends on it).
