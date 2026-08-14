# Cross-repo contracts (daemon ⇄ client)

The Python daemon (`Cerebrum-Daemon/src`) and the Flutter client
(`Cerebrum/src`) share **wire contracts**: request/response shapes and
identity/ordering rules the client's offline-first layer is built around. Change
a shape here without updating the client and things break *silently* on the
client (dropped answers, duplicate records, un-renderable images, notes losing
pages).

Every coupling point is annotated in-code with a greppable marker:

```
grep -rn "CROSS-REPO CONTRACT" src                 # here (daemon)
grep -rn "CROSS-REPO CONTRACT" ../Cerebrum/src/lib # client
```

**Read the annotation next to the code before changing an endpoint/model shape.**
This file is an index; the annotations are the source of truth.

## The load-bearing contracts (daemon side)

| Daemon | What the client depends on | Breaks if… |
|---|---|---|
| `api/routes_bubble.py` `update_note` | Client sends the WHOLE page set every save; absent `page_id` = DELETE, per-`page_id` LWW reconcile | you require partial payloads, or churn `page_id`s (per-page analysis keys on them) |
| `api/routes_bubble.py` `create_note` | Honours client-supplied `note.note_id` verbatim → filename `<note_id>.json` | stops honouring it → client's local folder key ≠ server filename → duplicates on sync |
| `api/routes_bubble.py` `create_study_bubble` | `bubble_id` optional query param (client sends `md5(name)`); identity from Depends `user_id`, not body | `bubble_id` required-without-default, or `CreateStudyBubble` *requires* `user_id` → client 422s |
| `api/routes_bubble.py` `list_notes_in_bubble` | Empty bubble → 404 (client maps to `[]`); list entries carry `ink: []` | returning ink here (client re-fetches full note anyway) or 200-with-empty vs 404 semantics |
| `api/routes_bubble.py` image routes | Client stores bytes locally + a `cerebrum-image://<note_id>/<name>` ref; upload returns a URL it does NOT embed | image route / filename-keying changes → offline image render breaks |
| `api/routes_learning_center.py` submit models | Field names are the wire; `attempt_id`/`attempted_at` are client-owned; mcq/flashcard sync, short/long async (`job_id`) | rename a field, or flip a type sync↔async, without updating the client |
| `api/routes_learning_center.py` `get_grading_job_status` | Client polls `/engrams/grading/jobs/{id}` and branches on `status` (`pending\|processing\|done\|failed`); `done` carries `score`+`grader` | rename route or a status value → client grades never resolve |
| `api/routes_learning_center.py` `_sanitize_for_presentation` | Client fetches answers via `include_answers=true` for offline compare + offline MCQ grading (reveal gated client-side) | role-gating `include_answers` or changing stripped fields breaks offline compare |
| `cerebrum_core/engrams/grading/ai_grading.py` (`questions_by_index`) | Submitted `question_index` is matched against `question_number` | treat it as a 0-based position → client answers silently scored 0 |
| `database/note_engram_repository/attempts.py` `create_attempt` / `_insert_attempt` | `INSERT OR IGNORE` = the client's offline queue can replay a submit without duplicating the attempt (short/long also skip re-enqueuing a grading job) | revert to plain INSERT → offline retries raise / duplicate |
| `cerebrum_core/engrams/core/mastery_service.py` (`MasteryState`, `_update_mastery`) | Daemon owns mastery; the client keeps only a provisional SM-2 estimate and adopts the returned `mastery_state` | change `MasteryState` values without telling the client (it shows foreign states) |

## Note

The client is offline-first: notes, images, engram attempts, mastery, and engram
*content* are cached on-device and reconciled here on reconnect. Most daemon
changes are safe as long as the **shapes above** stay stable; when they must
change, grep the client annotations and update both sides together.
