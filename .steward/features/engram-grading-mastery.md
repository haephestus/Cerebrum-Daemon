# Feature: Engram grading & mastery

**Status**: DONE (P2); answer-embedding/regression path explicitly gated OFF

## Goals
Score attempts consistently and maintain a daemon-owned mastery state that drives spaced repetition.

## Scope
- Sync submit paths: mcq + flashcard (`question_index` matched against `question_number` — CONTRACT; mismatch silently scores zero)
- Async paths: short/long question submit → `grading_jobs` queue → AI grading worker (`ai_grading.py`, `context_retrieval.py` pulls grounded context from cached→KB retriever); client polls `/engrams/grading/jobs/{id}` branching on `pending|processing|done|failed`; `done` carries score+grader (CONTRACT)
- Idempotent replay: attempts use `INSERT OR IGNORE`; offline queue resubmits never duplicate; re-submits skip re-enqueueing grading jobs (CONTRACT, load-bearing in `attempts.py`)
- Mastery: `MasteryState` values are contract-level (client renders foreign states); SM-2 scheduler extended with cognitive-level gating, promotion/demotion/lapse thresholds, stability computation, regression detection (`scheduler/`)
- Misconceptions tracking table present

## Dependencies
[[features/engram-generation]]; [[features/rag-chat]] retrieval for grounded grading.

## Notes
- Worker launched with `vector_store=None, embedder=None`: answer-embedding/regression path gated off until a concrete answer vector store exists ([[roadmap]] P2 open item).
