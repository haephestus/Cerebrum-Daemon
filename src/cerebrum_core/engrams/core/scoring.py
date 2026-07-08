"""
cerebrum_core.engrams.core.scoring
===================================
Score computation for each engram type plus shared utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import DimensionScores, FlashcardRating

# ---------------------------------------------------------------------------
# Flashcard rating → score
# ---------------------------------------------------------------------------

_FLASHCARD_SCORE_MAP: dict[FlashcardRating, float] = {
    FlashcardRating.AGAIN: 0.00,
    FlashcardRating.HARD: 0.30,
    FlashcardRating.GOOD: 0.75,
    FlashcardRating.EASY: 1.00,
}


def flashcard_rating_to_score(rating: FlashcardRating) -> float:
    return _FLASHCARD_SCORE_MAP[rating]


# ---------------------------------------------------------------------------
# MCQ
# ---------------------------------------------------------------------------


def score_mcq(is_correct: bool) -> float:
    return 1.0 if is_correct else 0.0


# ---------------------------------------------------------------------------
# Quiz (partial credit)
# ---------------------------------------------------------------------------


def score_short_answer(responses: list[dict]) -> float:
    if not responses:
        return 0.0
    correct = sum(1 for r in responses if r.get("is_correct"))
    return correct / len(responses)


# ---------------------------------------------------------------------------
# Long question — dimension-weighted score
# Weights shift with cognitive level: higher levels demand synthesis/originality
# ---------------------------------------------------------------------------

_DIMENSION_WEIGHTS: dict[int, dict[str, float]] = {
    1: {"accuracy": 0.70, "precision": 0.20, "depth": 0.10},
    2: {"accuracy": 0.50, "precision": 0.20, "depth": 0.20, "reasoning": 0.10},
    3: {"accuracy": 0.30, "depth": 0.25, "reasoning": 0.25, "precision": 0.20},
    4: {
        "accuracy": 0.20,
        "depth": 0.25,
        "reasoning": 0.30,
        "connections": 0.15,
        "precision": 0.10,
    },
    5: {
        "reasoning": 0.25,
        "connections": 0.25,
        "depth": 0.20,
        "originality": 0.15,
        "accuracy": 0.15,
    },
    6: {
        "reasoning": 0.20,
        "connections": 0.20,
        "originality": 0.20,
        "awareness_of_limits": 0.20,
        "depth": 0.20,
    },
    7: {
        "originality": 0.25,
        "connections": 0.25,
        "awareness_of_limits": 0.20,
        "reasoning": 0.20,
        "depth": 0.10,
    },
}


def score_long_question(dimensions: DimensionScores, level: int) -> float:
    weights = _DIMENSION_WEIGHTS.get(level, _DIMENSION_WEIGHTS[7])
    dim_dict = dimensions.to_dict()
    total = sum(dim_dict.get(k, 0.0) * w for k, w in weights.items())
    weight_sum = sum(weights.values())
    return min(1.0, total / weight_sum) if weight_sum > 0 else 0.0


# ---------------------------------------------------------------------------
# Rolling average with exponential decay (recent attempts count more)
# ---------------------------------------------------------------------------


def rolling_average(scores: list[float], decay: float = 0.8) -> float:
    if not scores:
        return 0.0
    weighted_sum = 0.0
    weight_sum = 0.0
    weight = 1.0
    for score in reversed(scores):  # most recent first
        weighted_sum += score * weight
        weight_sum += weight
        weight *= decay
    return weighted_sum / weight_sum


# ---------------------------------------------------------------------------
# Stability: how consistent recent attempts have been (0 = volatile, 1 = rock-solid)
# ---------------------------------------------------------------------------


def compute_stability(recent_scores: list[float]) -> float:
    if len(recent_scores) < 2:
        return 0.0
    mean = sum(recent_scores) / len(recent_scores)
    variance = sum((s - mean) ** 2 for s in recent_scores) / len(recent_scores)
    return max(0.0, 1.0 - math.sqrt(variance) * 2)


# ---------------------------------------------------------------------------
# Rolling update of dimension scores (exponential moving average)
# ---------------------------------------------------------------------------


def update_dimension_scores(
    current: DimensionScores,
    incoming: DimensionScores,
    alpha: float = 0.3,
) -> DimensionScores:
    def ema(cur: float, inc: float) -> float:
        return cur * (1 - alpha) + inc * alpha

    return DimensionScores(
        accuracy=ema(current.accuracy, incoming.accuracy),
        depth=ema(current.depth, incoming.depth),
        reasoning=ema(current.reasoning, incoming.reasoning),
        connections=ema(current.connections, incoming.connections),
        originality=ema(current.originality, incoming.originality),
        precision=ema(current.precision, incoming.precision),
        awareness_of_limits=ema(
            current.awareness_of_limits, incoming.awareness_of_limits
        ),
    )


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


@dataclass
class RegressionResult:
    occurred: bool
    delta: float
    severity: str  # 'none' | 'minor' | 'major'


def detect_regression(current_score: float, new_score: float) -> RegressionResult:
    delta = new_score - current_score
    if delta >= 0:
        return RegressionResult(occurred=False, delta=delta, severity="none")
    if delta < -0.4:
        return RegressionResult(occurred=True, delta=delta, severity="major")
    if delta < -0.2:
        return RegressionResult(occurred=True, delta=delta, severity="minor")
    return RegressionResult(occurred=False, delta=delta, severity="none")


# ---------------------------------------------------------------------------
# Promotion thresholds (score required to advance to next level)
# ---------------------------------------------------------------------------

PROMOTION_THRESHOLDS: dict[int, float] = {
    1: 0.80,
    2: 0.82,
    3: 0.85,
    4: 0.85,
    5: 0.88,
    6: 0.90,
    7: 0.92,
}

LAPSE_THRESHOLD = 0.50
DEMOTION_THRESHOLD = 0.45
