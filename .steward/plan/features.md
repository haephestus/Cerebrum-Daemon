# Phase 2 spec — features

> Status: **in progress** — 8 of 11 items shipped. This is the live phase.

## Subtasks
- [x] Engram generation queue (mcq/flashcard/short_question/long_question)
- [x] Grading: sync mcq/flashcard; async short/long via grading_jobs worker
- [x] Mastery + SM-2 scheduler w/ cognitive gating + regression detection
- [x] Study planner: generate/phases/weeks/metrics/densify/progress/review-queue/tasks complete-reopen
- [x] Learning profiles + inference
- [x] Offline sync gap-1 streams A/B/C ([[features/offline-sync]])
- [x] Suggested reading gap-3 Phase 0 ([[features/suggested-reading]])
- [ ] Suggested reading: online tiers + accept→ingest→KB loop
  - Provider-agnostic orchestrator already slots tiers in without caller changes
  - Ingest side must pass through license policy gate first (ADR-0006)
- [ ] Gap-1 closeout vs client offline-first layer (client-side work was in flight at HEAD 41897c1)
- [ ] Decide/descope grading answer-embedding regression path (worker launched with `vector_store=None, embedder=None`)
- [ ] Async file-processing scheduling TODO (`cerebrum_inator.py` line ~69): schedule md_conversion + md_chunking so files are ready for analysis

## Definition of done for the phase
All wire shapes above stable and contract-grepped; no gated-off code paths left undecided.
