# Cerebrum-Daemon — Immediate todos

> Working todos for today or this session. Phase structure lives in
> [[roadmap]] and syncs to the PM tracker. Keep this file for ad-hoc
> items that don't belong in a phase.

## Tasks
- [ ] Finish suggested-reading client implementation (HEAD 41897c1 was saved mid-work: "save before suggestend reading client impelementation")
- [ ] Suggested reading gap-3 next slice: online tiers + accept→ingest→KB loop ([[plan/features]])
- [ ] Gap-1 sync closeout vs client offline-first layer — verify remaining stream scope ([[features/offline-sync]])
- [ ] Decide fate of grading answer-embedding/regression path (gated off at `vector_store=None`) or descope it explicitly ([[decisions]])

## Notes
- Root `TODO.md` is a stale copy of this file's template — delete or symlink; don't maintain both.
- README §3/§4/§8 are stale (FAISS not ChromaDB; old module names; engrams shipped). Tracked under [[plan/hardening]], not blocking.
- Daemon key auth reminder: key printed at startup, stored in `<config>/daemon_api_key.txt`.
- Link to [[decisions]] for architecture choices · Link to [[roadmap]] for phase structure
