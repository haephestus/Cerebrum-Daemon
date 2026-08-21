# Feature: Suggested reading (gap 3)

**Status**: IN PROGRESS (P2) — Phase 0 shipped; online tiers + ingest loop open

## Goals
Recommend reading material from the learner's weak areas and pull accepted sources into the KB under license policy.

## Scope
- Phase 0 (c30d1f3, 8d92210): KB-first **offline** provider — Seed built from a note's analysis overview (topic + weak areas + confused links + knowledge gaps); Tier-1 knowledge-base source; candidates persisted
- Accept/dismiss lifecycle with persistence (`note_engram_repository/suggested_reading.py`)
- License policy gate (ADR-0006): commercial build ingests CC-BY/-SA/CC0/PD only; NC/ND/unknown/free-read are pointer-only; env `CEREBRUM_COMMERCIAL_USE`
- Ranking bias toward learner profile: book sources (openstax/gutenberg/doab/ncbi_bookshelf) weighted for book-preferring profiles
- Endpoints: `/note/{bubble}/{note_id}`, `/list`, `/accept/{id}`, `/dismiss/{id}`
- Deliberately provider-agnostic so online tiers slot in without caller changes

## Dependencies
[[features/note-analysis]] overview as seed source; ingestion pipeline for the future accept→ingest loop; [[features/learning-profiles]] ranking bias.

## Notes
- HEAD commit 41897c1 was saved mid-client-implementation — finish that first ([[todo]]).
