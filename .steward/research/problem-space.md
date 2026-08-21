# Problem space — Cerebrum

## Problem
Students need an AI study assistant that works without their data leaving the machine, without subscriptions to cloud LLM providers, and without assuming reliable connectivity.

## Users
Single learners studying from documents (PDFs, rich-text notes); multi-device offline-capable usage is a first-class flow, not an edge case.

## Product thesis
Learning = ingest sources → build a personal knowledge base → practice against it (engrams) → get grounded feedback → repeat on a spaced schedule. Every feature in the daemon maps to one link of that chain:
- ingest → [[features/kb-ingestion]]
- knowledge base → FAISS domain stores
- practice → [[features/engram-generation]] / [[features/engram-grading-mastery]]
- feedback → [[features/note-analysis]] + AI grading
- schedule → SM-2 scheduler + [[features/study-planner]]
- offline → [[features/offline-sync]] + [[features/suggested-reading]]

## Constraints
- Consumer hardware: local models via Ollama, CPU FAISS
- Privacy: local-first daemon (ADR-0001); cloud mode exists but local is primary
- Offline-first client dictates wire contracts (ADR-0004)
- Legal: third-party content ingestion gated by license policy (ADR-0006)

## Non-goals
Being a general-purpose chat product; multi-tenant SaaS in the near term (orgs exist but deployment posture is single-user local first).
