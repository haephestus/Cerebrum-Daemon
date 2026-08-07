"""
cerebrum_core.engrams.storage.note_engram_repository.mastery
================================================================
Per-engram mastery (engram_mastery) and topic-level aggregate mastery
(topic_mastery). These implement the MasteryRepository ABC's
mastery-facing methods: get_topic_masteries, upsert_topic_mastery,
get_topic_mastery (get_topic_engrams lives in engrams.py, since it reads
the engrams table, not engram_mastery/topic_mastery).
"""

from __future__ import annotations

import sqlite3

from cerebrum_core.engrams.core.types import EngramMastery, MasteryState, TopicMastery


class MasteryMixin:
    # -----------------------------------------------------------------------
    # Per-engram mastery
    # -----------------------------------------------------------------------

    def get_mastery(self, engram_id: str, user_id: str) -> "EngramMastery | None":
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM engram_mastery WHERE engram_id = ? AND user_id = ?",
                (engram_id, user_id),
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_mastery(row) if row else None

    def upsert_mastery(self, mastery: EngramMastery) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO engram_mastery
                  (id, engram_id, user_id, state, current_score, stability,
                   interval_days, next_due_at, last_attempted_at, attempt_count,
                   lapse_count, consecutive_correct, current_level,
                   score_accuracy, score_depth, score_reasoning, score_connections,
                   score_originality, score_precision, score_awareness_of_limits,
                   updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(engram_id, user_id) DO UPDATE SET
                  state=excluded.state, current_score=excluded.current_score,
                  stability=excluded.stability, interval_days=excluded.interval_days,
                  next_due_at=excluded.next_due_at,
                  last_attempted_at=excluded.last_attempted_at,
                  attempt_count=excluded.attempt_count, lapse_count=excluded.lapse_count,
                  consecutive_correct=excluded.consecutive_correct,
                  current_level=excluded.current_level,
                  score_accuracy=excluded.score_accuracy, score_depth=excluded.score_depth,
                  score_reasoning=excluded.score_reasoning,
                  score_connections=excluded.score_connections,
                  score_originality=excluded.score_originality,
                  score_precision=excluded.score_precision,
                  score_awareness_of_limits=excluded.score_awareness_of_limits,
                  updated_at=excluded.updated_at
                """,
                (
                    mastery.id,
                    mastery.engram_id,
                    mastery.user_id,
                    mastery.state.value,
                    mastery.current_score,
                    mastery.stability,
                    mastery.interval_days,
                    mastery.next_due_at,
                    mastery.last_attempted_at,
                    mastery.attempt_count,
                    mastery.lapse_count,
                    mastery.consecutive_correct,
                    mastery.current_level,
                    mastery.score_accuracy,
                    mastery.score_depth,
                    mastery.score_reasoning,
                    mastery.score_connections,
                    mastery.score_originality,
                    mastery.score_precision,
                    mastery.score_awareness_of_limits,
                    mastery.updated_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_topic_masteries(self, user_id: str, topic: str) -> list[EngramMastery]:
        """Matches MasteryRepository.get_topic_masteries; groups on the topic
        ENTITY (notes.topic_id), resolved from the given name via its canonical
        slug — so a differently-cased/spaced name still finds the right topic.
        Returns [] if this user has no such topic."""
        conn = self._get_conn()
        try:
            topic_id = self._lookup_topic_id(conn, user_id, topic)
            if topic_id is None:
                return []
            rows = conn.execute(
                """
                SELECT em.* FROM engram_mastery em
                JOIN engrams e ON e.id = em.engram_id
                JOIN notes n   ON n.id = e.note_id
                WHERE em.user_id = ? AND n.topic_id = ?
                """,
                (user_id, topic_id),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_mastery(r) for r in rows]

    def get_all_due_masteries(self, user_id: str) -> list[EngramMastery]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM engram_mastery
                WHERE user_id = ? AND state != 'suspended'
                  AND next_due_at <= datetime('now', '+1 day')
                ORDER BY next_due_at ASC
                """,
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_mastery(r) for r in rows]

    # -----------------------------------------------------------------------
    # Topic mastery (aggregate table; matches MasteryRepository's
    # upsert_topic_mastery / get_topic_mastery, and TopicMastery.topic)
    # -----------------------------------------------------------------------

    def upsert_topic_mastery(self, tm: TopicMastery) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO topic_mastery
                  (id, user_id, topic, topic_id, factual_score, applied_score,
                   conceptual_score, doctoral_score, overall_score, engram_count,
                   lapsed_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, topic) DO UPDATE SET
                  topic_id=COALESCE(excluded.topic_id, topic_mastery.topic_id),
                  factual_score=excluded.factual_score,
                  applied_score=excluded.applied_score,
                  conceptual_score=excluded.conceptual_score,
                  doctoral_score=excluded.doctoral_score,
                  overall_score=excluded.overall_score,
                  engram_count=excluded.engram_count,
                  lapsed_count=excluded.lapsed_count,
                  updated_at=excluded.updated_at
                """,
                (
                    tm.id,
                    tm.user_id,
                    tm.topic,
                    tm.topic_id,
                    tm.factual_score,
                    tm.applied_score,
                    tm.conceptual_score,
                    tm.doctoral_score,
                    tm.overall_score,
                    tm.engram_count,
                    tm.lapsed_count,
                    tm.updated_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_topic_mastery(self, user_id: str, topic: str) -> "TopicMastery | None":
        conn = self._get_conn()
        try:
            # Prefer the topic entity (id) so a differently-cased name still
            # resolves; fall back to a name match for any legacy row whose
            # topic_id hasn't been backfilled yet.
            topic_id = self._lookup_topic_id(conn, user_id, topic)
            if topic_id is not None:
                row = conn.execute(
                    "SELECT * FROM topic_mastery WHERE user_id = ? AND topic_id = ?",
                    (user_id, topic_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM topic_mastery WHERE user_id = ? AND topic = ?",
                    (user_id, topic),
                ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return TopicMastery(
            id=row["id"],
            user_id=row["user_id"],
            topic=row["topic"],
            topic_id=row["topic_id"],
            factual_score=row["factual_score"],
            applied_score=row["applied_score"],
            conceptual_score=row["conceptual_score"],
            doctoral_score=row["doctoral_score"],
            overall_score=row["overall_score"],
            engram_count=row["engram_count"],
            lapsed_count=row["lapsed_count"],
            updated_at=row["updated_at"],
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _row_to_mastery(row: sqlite3.Row) -> EngramMastery:
        return EngramMastery(
            id=row["id"],
            engram_id=row["engram_id"],
            user_id=row["user_id"],
            state=MasteryState(row["state"]),
            current_score=float(row["current_score"]),
            stability=float(row["stability"]),
            interval_days=float(row["interval_days"]),
            next_due_at=row["next_due_at"],
            last_attempted_at=row["last_attempted_at"],
            attempt_count=int(row["attempt_count"]),
            lapse_count=int(row["lapse_count"]),
            consecutive_correct=int(row["consecutive_correct"]),
            current_level=int(row["current_level"]),
            score_accuracy=float(row["score_accuracy"] or 0),
            score_depth=float(row["score_depth"] or 0),
            score_reasoning=float(row["score_reasoning"] or 0),
            score_connections=float(row["score_connections"] or 0),
            score_originality=float(row["score_originality"] or 0),
            score_precision=float(row["score_precision"] or 0),
            score_awareness_of_limits=float(row["score_awareness_of_limits"] or 0),
            updated_at=row["updated_at"],
        )

    """
    ADDITIONS needed for study_plan_progress_service.densify_phase's context
    gathering. Two separate small methods for two separate existing files —
    paste each into its respective mixin.
    """

    # =============================================================================
    # ADD to mastery.py, MasteryMixin (near get_topic_mastery)
    # =============================================================================

    def get_all_topic_masteries_for_user(self, user_id: str) -> list[dict]:
        """
        Bulk version of get_topic_mastery — every topic this user has any
        mastery data for, not just one. This is what densify_phase feeds
        the LLM as "here's what topics already exist and how the user is
        doing on them" so it reuses topic strings instead of minting
        near-duplicates.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
               SELECT topic, factual_score, applied_score, conceptual_score,
                      doctoral_score, overall_score, engram_count, lapsed_count
               FROM topic_mastery
               WHERE user_id = ?
               ORDER BY overall_score ASC
               """,
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # ORDER BY overall_score ASC is deliberate: weakest topics first, so if
    # the prompt ever needs to truncate this list for token-budget reasons,
    # truncating from the bottom keeps the topics that most need review
    # tasks, and drops the ones the user has already mastered.
    # =============================================================================
    # ADD to misconceptions.py, MisconceptionsMixin (near upsert_misconception)
    # =============================================================================

    def get_misconceptions_for_user(self, user_id: str, limit: int = 50) -> list[dict]:
        """
        Read-side companion to upsert_misconception (which only wrote,
        never read). Ordered by occurrences DESC so the most persistent,
        recurring misunderstandings surface first — those are the ones
        densify_phase should be weighting review tasks toward, not a
        one-off slip from months ago.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
               SELECT concept, description, occurrences, last_seen
               FROM misconceptions
               WHERE user_id = ?
               ORDER BY occurrences DESC, last_seen DESC
               LIMIT ?
               """,
                (user_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
