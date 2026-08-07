"""
cerebrum_core.engrams.core.types
=================================
All dataclasses, enums, and constants for the Engram Mastery System.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EngramType(str, Enum):
    MCQ = "mcq"
    FLASHCARD = "flashcard"
    SHORT_QUESTION = "short_question"
    LONG_QUESTION = "long_question"


class MasteryState(str, Enum):
    NEW = "new"
    LEARNING = "learning"
    REVIEW = "review"
    MASTERED = "mastered"
    LAPSED = "lapsed"
    SUSPENDED = "suspended"


class GraderType(str, Enum):
    PENDING = "pending"
    AUTO = "auto"
    AI = "ai"
    HUMAN = "human"


class FlashcardRating(str, Enum):
    AGAIN = "again"
    HARD = "hard"
    GOOD = "good"
    EASY = "easy"


class GenerationTrigger(str, Enum):
    MISCONCEPTION = "misconception"
    LEVEL_PROMOTION = "level_promotion"
    TOPIC_GAP = "topic_gap"
    MANUAL = "manual"


# Cognitive level 1–7 (Bloom's extended to PhD level)
CognitiveLevel = Literal[1, 2, 3, 4, 5, 6, 7]

COGNITIVE_LEVELS: dict[int, str] = {
    1: "Recall",
    2: "Comprehension",
    3: "Application",
    4: "Analysis",
    5: "Synthesis",
    6: "Evaluation",
    7: "Doctoral",
}

# Engram types allowed at each cognitive level
LEVEL_ENGRAM_RESTRICTIONS: dict[int, list[EngramType]] = {
    1: [
        EngramType.MCQ,
        EngramType.FLASHCARD,
        EngramType.SHORT_QUESTION,
        EngramType.LONG_QUESTION,
    ],
    2: [
        EngramType.MCQ,
        EngramType.FLASHCARD,
        EngramType.SHORT_QUESTION,
        EngramType.LONG_QUESTION,
    ],
    3: [EngramType.MCQ, EngramType.SHORT_QUESTION, EngramType.LONG_QUESTION],
    4: [EngramType.MCQ, EngramType.SHORT_QUESTION, EngramType.LONG_QUESTION],
    5: [EngramType.MCQ, EngramType.LONG_QUESTION],
    6: [EngramType.LONG_QUESTION],
    7: [EngramType.LONG_QUESTION],
}


# ---------------------------------------------------------------------------
# Source material
# ---------------------------------------------------------------------------


@dataclass
class Note:
    id: str
    title: str
    content: str
    topic: str
    tags: list[str] = field(default_factory=list)
    subtopic: Optional[str] = None
    source: Optional[str] = None
    version: int = 1
    embedding_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Engram content types
# ---------------------------------------------------------------------------


@dataclass
class MCQContent:
    finding_index: str
    question_number: int
    stem: str
    options: dict[str, str]  # {"A": "...", "B": "...", ...}
    correct_option: str  # "A" | "B" | "C" | "D"
    explanation: str
    severity: str
    distractor_notes: dict[str, str] = field(default_factory=dict)


@dataclass
class FlashcardContent:
    finding_index: str
    card_number: int
    front: str
    back: str
    bridge_concept: str
    severity: str
    diagnostic_note: str


@dataclass
class QuizQuestion:
    finding_index: int
    question_number: int
    level: str
    stem: str
    expected_answer: str
    hint: str
    context_anchored: int
    severity: str


@dataclass
class QuizContent:
    questions: list[QuizQuestion]


@dataclass
class RubricCriteria:
    accuracy: str
    depth: str
    reasoning: str
    connections: str
    precision: str
    originality: Optional[str] = None  # required at level 5+
    awareness_of_limits: Optional[str] = None  # required at level 6+


@dataclass
class LongQuestionContent:
    finding_index: int
    question_stem: str
    answer: str
    severity: str
    total_marks: int
    parts: list[LongQuestionPart]


@dataclass
class LongQuestionPart:
    part: str
    level: str
    question: str
    marks: int
    mark_scheme: str
    note: str


EngramContent = MCQContent | FlashcardContent | QuizContent | LongQuestionContent


@dataclass
class Engram:
    id: str
    bubble_id: str
    note_id: str
    type: EngramType
    target_cognitive_level: int
    content: EngramContent
    tags: list[str] = field(default_factory=list)
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Attempts
# ---------------------------------------------------------------------------


@dataclass
class EngramAttempt:
    id: str
    engram_id: str
    user_id: str
    target_cognitive_level: int
    attempted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    score: Optional[float] = None
    grader: GraderType = GraderType.PENDING
    time_spent_ms: Optional[int] = None
    note_version: Optional[int] = None
    context_snapshot: Optional[list[str]] = None


@dataclass
class MCQResponse:
    attempt_id: str
    selected_option: str
    correct_option: str
    is_correct: bool
    distractor_key: Optional[str] = None


@dataclass
class FlashcardResponse:
    attempt_id: str
    self_rating: FlashcardRating
    time_to_flip_ms: Optional[int] = None


@dataclass
class QuizResponse:
    """A student's answer to one sub-question of a short_question engram.

    Open-response, async-graded (mirrors LongQuestionResponse, not MCQ):
    the student submits free-text `raw_answer`, and score/is_correct/
    feedback/misconceptions/graded_at stay None until the grading worker
    lands the AI grade. `is_correct` is a convenience flag derived from
    `score` crossing a pass threshold, not an exact string match.
    """

    id: str
    attempt_id: str
    question_index: int
    raw_answer: str
    score: Optional[float] = None
    is_correct: Optional[bool] = None
    feedback: Optional[str] = None
    misconceptions: list[dict[str, str]] = field(default_factory=list)
    graded_at: Optional[str] = None


@dataclass
class DimensionScores:
    accuracy: float = 0.0
    depth: float = 0.0
    reasoning: float = 0.0
    connections: float = 0.0
    originality: float = 0.0
    precision: float = 0.0
    awareness_of_limits: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "depth": self.depth,
            "reasoning": self.reasoning,
            "connections": self.connections,
            "originality": self.originality,
            "precision": self.precision,
            "awareness_of_limits": self.awareness_of_limits,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "DimensionScores":
        return cls(
            accuracy=d.get("accuracy", 0.0),
            depth=d.get("depth", 0.0),
            reasoning=d.get("reasoning", 0.0),
            connections=d.get("connections", 0.0),
            originality=d.get("originality", 0.0),
            precision=d.get("precision", 0.0),
            awareness_of_limits=d.get(
                "awareness_of_limits", d.get("awarenessOfLimits", 0.0)
            ),
        )


@dataclass
class LongQuestionResponse:
    attempt_id: str
    raw_answer: str
    word_count: int = 0
    ai_feedback: Optional[str] = None
    concepts_demonstrated: list[str] = field(default_factory=list)
    concepts_missed: list[str] = field(default_factory=list)
    misconceptions: list[dict[str, str]] = field(default_factory=list)
    dimension_scores: Optional[DimensionScores] = None
    level_demonstrated: Optional[int] = None
    regression_detected: bool = False
    vector_id: Optional[str] = None
    graded_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Mastery
# ---------------------------------------------------------------------------


@dataclass
class EngramMastery:
    id: str
    engram_id: str
    user_id: str
    state: MasteryState = MasteryState.NEW
    current_score: float = 0.0
    stability: float = 0.0
    interval_days: float = 1.0
    next_due_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_attempted_at: Optional[str] = None
    attempt_count: int = 0
    lapse_count: int = 0
    consecutive_correct: int = 0
    current_level: int = 1
    score_accuracy: float = 0.0
    score_depth: float = 0.0
    score_reasoning: float = 0.0
    score_connections: float = 0.0
    score_originality: float = 0.0
    score_precision: float = 0.0
    score_awareness_of_limits: float = 0.0
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TopicMastery:
    id: str
    user_id: str
    topic: str
    topic_id: Optional[str] = None  # FK to the topics entity (topic identity)
    factual_score: float = 0.0
    applied_score: float = 0.0
    conceptual_score: float = 0.0
    doctoral_score: float = 0.0
    overall_score: float = 0.0
    engram_count: int = 0
    lapsed_count: int = 0
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MasteryVector:
    factual: float
    applied: float
    conceptual: float
    doctoral: float


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class GradingResult:
    score: float
    dimension_scores: DimensionScores
    level_demonstrated: int
    feedback: str
    concepts_demonstrated: list[str]
    concepts_missed: list[str]
    misconceptions: list[dict[str, str]]
    regression_from_last: bool
    suggested_next_level: int


@dataclass
class ShortAnswerGrade:
    """AI grade for a single short-answer sub-question."""

    question_index: int
    score: float
    is_correct: bool
    feedback: str
    misconceptions: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ShortQuestionGradingResult:
    """Whole-attempt result for a short_question engram: one ShortAnswerGrade
    per sub-question plus the aggregate score and roll-ups the mastery/
    generation path needs. `overall_score` is the mean of per-question
    scores; `misconceptions` and `concepts_missed` are flattened across
    sub-questions so apply_short_question_grading_result can feed the same
    misconception/generation path long questions use.
    """

    overall_score: float
    grades: list[ShortAnswerGrade]
    misconceptions: list[dict[str, str]] = field(default_factory=list)
    concepts_missed: list[str] = field(default_factory=list)
    suggested_next_level: int = 1
    regression_from_last: bool = False


@dataclass
class GradingJob:
    id: str
    attempt_id: str
    status: str
    priority: int
    attempts: int
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


@dataclass
class SchedulingDecision:
    new_state: MasteryState
    new_score: float
    new_stability: float
    new_interval_days: float
    next_due_at: datetime
    lapse_occurred: bool
    promotion_occurred: bool
    demotion_occurred: bool
    new_level: int
    new_consecutive: int


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class QueuedEngram:
    engram_id: str
    mastery: EngramMastery
    priority: float
    reason: str  # 'lapsed' | 'overdue' | 'due' | 'new'
