# Cross-repo contracts — Cerebrum-Daemon

> Wire shapes other code depends on. In-code `CROSS-REPO CONTRACT` markers
> are the source of truth; `docs/cross-repo-contracts.md` is the repo-side
> index this file mirrors for Steward.

## Load-bearing contracts (current)
| Endpoint / thing | Who depends on it | Breaks if… |
|------------------|-------------------|------------|
| `routes_bubble.update_note` | client sends WHOLE page set every save; absent `page_id` = DELETE; per-page LWW | partial payloads required, or `page_id`s churned (per-page analysis keys on them) |
| `routes_bubble.create_note` | client-supplied `note.note_id` honoured verbatim → filename `<note_id>.json` | server mints ids → local folder key ≠ server filename → sync duplicates |
| `routes_bubble.create_study_bubble` | optional `bubble_id=md5(name)` query param; identity from Depends user_id, not body | bubble_id becomes required, or `CreateStudyBubble` requires user_id → 422s |
| `routes_bubble.list_notes_in_bubble` | empty bubble → 404 (client maps to `[]`); entries carry `ink: []` | ink returned here, or 200-empty vs 404 semantics flip |
| `routes_bubble` image routes | client stores bytes locally + `cerebrum-image://<note_id>/<name>` ref; upload returns absolute URL not embedded in content | route or filename keying changes → offline image render breaks |
| `routes_learning_center` submit models | field names are the wire; `attempt_id`/`attempted_at` client-owned; mcq/flashcard sync, short/long async via job_id | field renamed or sync/async flipped without client update |
| `routes_learning_center.get_grading_job_status` | client polls `/engrams/grading/jobs/{id}`, branches on `pending\|processing\|done\|failed`; done carries score+grader | route renamed or status value changed → grades never resolve |
| `routes_learning_center._sanitize_for_presentation` | client fetches answers via `include_answers=true` for offline compare; reveal gated client-side | role-gating include_answers or changing stripped fields |
| `ai_grading.questions_by_index` | submitted `question_index` matched against `question_number` | treated as 0-based position → answers silently scored 0 |
| `attempts.create_attempt/_insert_attempt` | `INSERT OR IGNORE` makes offline submit replays idempotent; short/long skip re-enqueueing grading jobs | plain INSERT → retries raise/duplicate |
| `mastery_service.MasteryState/_update_mastery` | daemon owns mastery; client keeps provisional SM-2 estimate and adopts returned state | state values change without client coordination |

## Planned contracts (gated by phase)
| Endpoint / thing | Phase | Purpose |
|------------------|-------|---------|
| `/sync push/pull/replica-id` payload shapes | P2 closeout ([[features/offline-sync]]) | multi-device reconciliation; cursors per peer hub |
| Suggested-reading accept→ingest result shapes | P2 ([[features/suggested-reading]]) | online tiers + KB ingestion loop |

See [[cross-repo/_index]] for linked repos.
