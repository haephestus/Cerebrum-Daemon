# Feature index — Cerebrum-Daemon

> Every feature asked for or planned, with status and wiki-links.
> Status legend: `DONE` · `IN PROGRESS (Pn)` · `PLANNED (Pn)` · `CANDIDATE`

## Phase map
| Phase | Features | Status |
|-------|----------|--------|
| P0 foundation | [[features/users-orgs-auth]], [[features/model-management]] | DONE |
| P1 core | [[features/kb-ingestion]], [[features/rag-chat]], [[features/note-analysis]], [[features/bubbles-notes]] | DONE |
| P2 features | [[features/engram-generation]], [[features/engram-grading-mastery]], [[features/study-planner]], [[features/learning-profiles]], [[features/offline-sync]], [[features/suggested-reading]] | 4 DONE · 2 IN PROGRESS |
| P3 hardening | tests, CORS/security pass, hygiene, README fixes ([[plan/hardening]]) | PLANNED (P3) |
| P4 ship | dual-mode deploy, worker service, backups ([[plan/ship]]) | PLANNED (P4) |

## Features
| Feature | Status | Spec |
|---------|--------|------|
| Users / orgs / auth | DONE | [[features/users-orgs-auth]] |
| Model management | DONE | [[features/model-management]] |
| KB ingestion pipeline | DONE | [[features/kb-ingestion]] |
| RAG chat (Rose) | DONE | [[features/rag-chat]] |
| Note analysis | DONE | [[features/note-analysis]] |
| Bubbles & notes CRUD | DONE | [[features/bubbles-notes]] |
| Engram generation | DONE | [[features/engram-generation]] |
| Engram grading & mastery | DONE (regression path gated off) | [[features/engram-grading-mastery]] |
| Study planner | DONE | [[features/study-planner]] |
| Learning profiles | DONE | [[features/learning-profiles]] |
| Offline sync (gap 1) | IN PROGRESS (P2) | [[features/offline-sync]] |
| Suggested reading (gap 3) | IN PROGRESS (P2) | [[features/suggested-reading]] |

## Candidates
| Feature | Source | Why it fits | Phase |
|---------|--------|-------------|-------|
| Answer-embedding/regression path for grading | gated-off worker args (`vector_store=None`) | closes the loop on short/long answer scoring quality | P2/P3 |
| Cloud worker service | ADR-0007 consequence | generation/grading stall serverless otherwise | P4 |
| Projects router | commented-out include in `cerebrum_inator.py` | only if product wants project-level grouping | CANDIDATE |

See [[roadmap]] for phase details, [[cross-repo/contracts]] for wire-shape constraints.
