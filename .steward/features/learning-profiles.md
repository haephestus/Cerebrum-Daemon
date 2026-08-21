# Feature: Learning profiles

**Status**: DONE (P2)

## Goals
Model each learner's effective profile so downstream features (planner difficulty, suggested-reading ranking) personalize.

## Scope
- Explicit profile CRUD: GET/PUT learning-profile under user routes
- Inferred profile: `learning_profile_inference_inator` derives traits from behavior (attempts/mastery signals)
- Persistence via `note_engram_repository/learning_profile.py`

## Dependencies
[[features/engram-grading-mastery]] signals; feeds [[features/study-planner]] and [[features/suggested-reading]] (book-vs-article ranking bias).

## Notes
- Landing commit 24fd07f also rehomed folders by-utility — profiles shipped alongside a layout refactor.
