"""
engram_mastery.grading.ai_grader
=================================
AI grading pipeline for long-form questions.
Builds context-rich prompts, calls the Ollama grading model (cloud or local),
parses long_question results.

TODO: this module previously called the Anthropic API directly via httpx.
It now goes through the same ollama_cloud_call / ollama_local_call invokers
that engram_generator_inator.py uses, so both halves of the system (engram
generation and engram grading) go through one model-calling path. Confirm
this is the intended long-term setup vs. e.g. keeping Anthropic for grading
specifically because grading is more judgment-heavy than generation.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

from cerebrum_core.user_inator import should_use_cloud
from common.ollama_compat.invoker_inator import (
    ollama_cloud_call,
    ollama_local_call,
)

from ..core.types import (
    COGNITIVE_LEVELS,
    DimensionScores,
    EngramMastery,
    GradingResult,
    LongQuestionContent,
    QuizContent,
    ShortAnswerGrade,
    ShortQuestionGradingResult,
)

# A per-question score at or above this is flagged is_correct=True. Purely a
# convenience/reporting flag (mastery uses the raw float score, not this) —
# 0.6 matches "clearly demonstrated the idea", below the LAPSE_THRESHOLD gap.
SHORT_ANSWER_PASS_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Output schema for the grading model
# Mirrors the schema_id / input / output convention used for the FLASHCARD_SCHEMA,
# MCQ_SCHEMA, etc. in engram_generator_inator.py, so grading fits the same
# long_question-output pattern as generation.
# ---------------------------------------------------------------------------

GRADING_SCHEMA: dict = {
    "schema_id": "engram_grading_v1",
    "max_tokens": 1500,
    "system": "You are an academic examiner. Always respond with valid JSON only.",
    "output": {
        "overallScore": "float (0.0-1.0)",
        "dimensionScores": {
            "accuracy": "float (0.0-1.0)",
            "depth": "float (0.0-1.0)",
            "reasoning": "float (0.0-1.0)",
            "connections": "float (0.0-1.0)",
            "originality": "float (0.0-1.0)",
            "precision": "float (0.0-1.0)",
            "awarenessOfLimits": "float (0.0-1.0)",
        },
        "levelDemonstrated": "integer (1-7)",
        "conceptsDemonstrated": ["string"],
        "conceptsMissed": ["string"],
        "misconceptions": [{"concept": "string", "description": "string"}],
        "regressionFromLast": "boolean",
        "suggestedNextLevel": "integer (1-7)",
        "feedback": "string",
    },
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# TODO: update grader according to new type structures


@dataclass
class GradingContext:
    question: LongQuestionContent
    user_answer: str
    target_cognitive_level: int
    note_chunks: list[str]
    past_answers: list[str] = None  # type: ignore[assignment]
    mastery: Optional[EngramMastery] = None

    def __post_init__(self) -> None:
        if self.past_answers is None:
            self.past_answers = []


def build_grading_prompt(ctx: GradingContext) -> str:
    level_name = COGNITIVE_LEVELS[ctx.target_cognitive_level]
    is_high = ctx.target_cognitive_level >= 5

    # NOTE: question.answer is a plain reference-answer string — see
    # types.LongQuestionContent.answer: str and the long_question_content.answer
    # TEXT column in sqlite_repository.py. It is NOT a RubricCriteria object;
    # nothing in the generation path (LONG_QUESTION_SCHEMA -> add_long_question) ever
    # produces per-dimension rubric text. The actual mark scheme lives on
    # each LongQuestionPart (part/level/question/marks/mark_scheme), so the
    # rubric shown to the grading model is built from those instead.
    rubric_lines = [
        f"  - Part {p.part} ({p.level}, {p.marks} marks): {p.mark_scheme}"
        for p in ctx.question.parts
    ]
    rubric_text = (
        "\n".join(rubric_lines)
        if rubric_lines
        else "  - No part-level mark scheme provided; grade holistically "
        "against the reference answer below."
    )
    reference_answer_text = (
        f"\n## REFERENCE ANSWER (for grading guidance — do not penalise "
        f"the student for not matching it verbatim)\n{ctx.question.answer}\n"
        if ctx.question.answer
        else ""
    )

    note_context = ""
    if ctx.note_chunks:
        chunks = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(ctx.note_chunks))
        note_context = f"\n## RELEVANT SOURCE MATERIAL\n{chunks}"

    past_context = ""
    if ctx.past_answers:
        answers = "\n\n".join(
            f"[Previous {i+1}]\n{a}" for i, a in enumerate(ctx.past_answers[:3])
        )
        past_context = f"\n## USER'S PREVIOUS ANSWERS (most recent first)\n{answers}"

    mastery_context = ""
    if ctx.mastery:
        m = ctx.mastery
        mastery_context = (
            f"\n## STUDENT HISTORY\n"
            f"- Attempts: {m.attempt_count}\n"
            f"- Current score: {m.current_score * 100:.0f}%\n"
            f"- Lapses: {m.lapse_count}\n"
            f"- Stability: {m.stability * 100:.0f}%"
        )

    high_level_note = ""
    if is_high:
        high_level_note = (
            f"\nAt this level ({level_name}), you are assessing doctoral-level thinking. You expect:\n"
            "- Original synthesis beyond what was taught\n"
            "- Critical awareness of the limits and assumptions of the field\n"
            "- Connection of ideas across domains\n"
            "- Independent reasoning that goes beyond memorised frameworks\n"
            "A technically correct but shallow answer scores LOW at this level.\n"
        )

    return (
        f"You are a rigorous academic examiner assessing a student at cognitive level "
        f"{ctx.target_cognitive_level}/7 ({level_name}).\n"
        f"{high_level_note}\n"
        f"## QUESTION\n{ctx.question.question_stem}\n\n"
        f"## ASSESSMENT RUBRIC (Level {ctx.target_cognitive_level} — {level_name})\n{rubric_text}\n"
        f"{reference_answer_text}"
        f"{note_context}{past_context}{mastery_context}\n\n"
        f"## STUDENT'S ANSWER\n{ctx.user_answer}\n\n"
        "---\n\n"
        "Assess this answer strictly. Respond ONLY with a JSON object:\n\n"
        "{\n"
        '  "overallScore": <0.0–1.0>,\n'
        '  "dimensionScores": {\n'
        '    "accuracy": <0.0–1.0>,\n'
        '    "depth": <0.0–1.0>,\n'
        '    "reasoning": <0.0–1.0>,\n'
        '    "connections": <0.0–1.0>,\n'
        '    "originality": <0.0–1.0>,\n'
        '    "precision": <0.0–1.0>,\n'
        '    "awarenessOfLimits": <0.0–1.0>\n'
        "  },\n"
        '  "levelDemonstrated": <1–7>,\n'
        '  "conceptsDemonstrated": ["concept1", "concept2"],\n'
        '  "conceptsMissed": ["concept3"],\n'
        '  "misconceptions": [\n'
        '    {"concept": "...", "description": "what they got wrong and why"}\n'
        "  ],\n"
        '  "regressionFromLast": <true|false>,\n'
        '  "suggestedNextLevel": <1–7>,\n'
        '  "feedback": "Detailed prose feedback referencing the student\'s actual words."\n'
        "}\n\n"
        "Do not include any text outside the JSON object."
    )


# ---------------------------------------------------------------------------
# Parse AI response
# ---------------------------------------------------------------------------


def parse_grading_response(raw: str) -> GradingResult:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v or 0)))

    def clamp_level(v: int) -> int:
        return max(1, min(7, int(v or 1)))

    ds = parsed.get("dimensionScores", {})

    return GradingResult(
        score=clamp(parsed.get("overallScore", 0)),
        dimension_scores=DimensionScores(
            accuracy=clamp(ds.get("accuracy", 0)),
            depth=clamp(ds.get("depth", 0)),
            reasoning=clamp(ds.get("reasoning", 0)),
            connections=clamp(ds.get("connections", 0)),
            originality=clamp(ds.get("originality", 0)),
            precision=clamp(ds.get("precision", 0)),
            awareness_of_limits=clamp(ds.get("awarenessOfLimits", 0)),
        ),
        level_demonstrated=clamp_level(parsed.get("levelDemonstrated", 1)),
        concepts_demonstrated=parsed.get("conceptsDemonstrated", []),
        concepts_missed=parsed.get("conceptsMissed", []),
        misconceptions=parsed.get("misconceptions", []),
        regression_from_last=bool(parsed.get("regressionFromLast", False)),
        suggested_next_level=clamp_level(parsed.get("suggestedNextLevel", 1)),
        feedback=parsed.get("feedback", ""),
    )


# ---------------------------------------------------------------------------
# Ollama call (cloud or local)
# ---------------------------------------------------------------------------


async def call_grading_model(
    prompt: str,
    use_cloud: Optional[bool] = None,
    api_key: Optional[str] = None,
    schema: Optional[dict] = None,
) -> str:
    """
    ollama_cloud_call / ollama_local_call are plain synchronous functions
    (engram_generator_inator.py calls them without await), but this pipeline
    is async top to bottom (run_grading_pipeline, process_grading_job in
    worker.py). Both calls are pushed onto a worker thread via
    asyncio.to_thread so a slow model call doesn't block the event loop —
    and therefore doesn't stall every other in-flight grading job or the
    generation-queue poll loop running in the same process.

    use_cloud defaults to should_use_cloud() (the single cloud/local resolver,
    which reads ollama.toggle_cloud and requires an API key) when not given
    explicitly — callers that want a specific path (e.g. worker.py's use_cloud
    plumbing) can still override per-call.
    """
    resolved_use_cloud = (
        use_cloud if use_cloud is not None else should_use_cloud()
    )
    resolved_schema = schema if schema is not None else GRADING_SCHEMA

    if resolved_use_cloud:
        response = await asyncio.to_thread(
            ollama_cloud_call,
            prompt=prompt,
            schema=resolved_schema,
        )
    else:
        response = await asyncio.to_thread(
            ollama_local_call,
            prompt=prompt,
            analyses_schema=resolved_schema,
        )

    # Mirrors engram_generator_inator.py's _parse_engram(), which handles the
    # invoker returning an already-parsed dict vs. a raw JSON string.
    # parse_grading_response() below assumes `raw` is always a str, so
    # normalize here rather than there.
    content = (
        json.dumps(response) if isinstance(response, dict) else str(response or "")
    )

    if not content:
        raise ValueError("Empty response from grading model")
    return content


# ---------------------------------------------------------------------------
# Full async pipeline
# ---------------------------------------------------------------------------


@dataclass
class GradingPipelineInput:
    attempt_id: str
    engram_id: str
    user_id: str
    question: LongQuestionContent
    user_answer: str
    target_cognitive_level: int
    mastery: Optional[EngramMastery] = None
    note_chunks: list[str] = None  # type: ignore[assignment]
    past_answers: list[str] = None  # type: ignore[assignment]
    api_key: Optional[str] = None
    use_cloud: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.note_chunks is None:
            self.note_chunks = []
        if self.past_answers is None:
            self.past_answers = []


@dataclass
class GradingPipelineOutput:
    attempt_id: str
    result: GradingResult
    raw_response: str
    graded_at: str


async def run_grading_pipeline(inp: GradingPipelineInput) -> GradingPipelineOutput:
    from datetime import datetime

    prompt = build_grading_prompt(
        GradingContext(
            question=inp.question,
            user_answer=inp.user_answer,
            target_cognitive_level=inp.target_cognitive_level,
            note_chunks=inp.note_chunks,
            past_answers=inp.past_answers,
            mastery=inp.mastery,
        )
    )
    raw_response = await call_grading_model(
        prompt, use_cloud=inp.use_cloud, api_key=inp.api_key
    )
    result = parse_grading_response(raw_response)

    return GradingPipelineOutput(
        attempt_id=inp.attempt_id,
        result=result,
        raw_response=raw_response,
        graded_at=datetime.utcnow().isoformat(),
    )


# ===========================================================================
# Short-question grading (open-response, async — mirrors the long-question
# pipeline above but grades MANY sub-questions per attempt in one model call).
# ===========================================================================

SHORT_QUESTION_GRADING_SCHEMA: dict = {
    "schema_id": "engram_short_question_grading_v1",
    "max_tokens": 2000,
    "system": "You are an academic examiner. Always respond with valid JSON only.",
    "output": {
        "questions": [
            {
                "questionIndex": "integer",
                "score": "float (0.0-1.0)",
                "isCorrect": "boolean",
                "conceptsMissed": ["string"],
                "misconceptions": [{"concept": "string", "description": "string"}],
                "feedback": "string",
            }
        ],
        "suggestedNextLevel": "integer (1-7)",
        "regressionFromLast": "boolean",
    },
}


@dataclass
class ShortAnswerItem:
    """One sub-question paired with the student's answer, for grading."""

    question_index: int
    level: str
    stem: str
    expected_answer: str
    raw_answer: str


@dataclass
class ShortQuestionGradingContext:
    items: list[ShortAnswerItem]
    target_cognitive_level: int
    note_chunks: list[str] = field(default_factory=list)
    mastery: Optional[EngramMastery] = None


def build_short_question_grading_prompt(ctx: ShortQuestionGradingContext) -> str:
    level_name = COGNITIVE_LEVELS[ctx.target_cognitive_level]

    question_blocks = []
    for item in ctx.items:
        question_blocks.append(
            f"### Question index {item.question_index} "
            f"(target level: {item.level})\n"
            f"PROMPT: {item.stem}\n"
            f"EXPECTED ANSWER (grading guidance — do not require verbatim match): "
            f"{item.expected_answer}\n"
            f"STUDENT ANSWER: {item.raw_answer}\n"
        )
    questions_text = "\n".join(question_blocks)

    note_context = ""
    if ctx.note_chunks:
        chunks = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(ctx.note_chunks))
        note_context = f"\n## RELEVANT SOURCE MATERIAL\n{chunks}\n"

    mastery_context = ""
    if ctx.mastery:
        m = ctx.mastery
        mastery_context = (
            f"\n## STUDENT HISTORY\n"
            f"- Attempts: {m.attempt_count}\n"
            f"- Current score: {m.current_score * 100:.0f}%\n"
            f"- Lapses: {m.lapse_count}\n"
        )

    return (
        f"You are a rigorous examiner grading short open-response answers from a "
        f"student working at cognitive level {ctx.target_cognitive_level}/7 "
        f"({level_name}).\n"
        "Grade EACH question independently on how well the student's answer "
        "demonstrates the expected understanding — reward correct meaning in the "
        "student's own words, do not penalise wording that differs from the "
        "expected answer, and do not reward empty or off-topic answers.\n"
        f"{note_context}{mastery_context}\n"
        f"## QUESTIONS\n{questions_text}\n"
        "---\n\n"
        "Respond ONLY with a JSON object of this exact shape:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "questionIndex": <int, echo the index above>,\n'
        '      "score": <0.0-1.0>,\n'
        '      "isCorrect": <true|false>,\n'
        '      "conceptsMissed": ["..."],\n'
        '      "misconceptions": [{"concept": "...", "description": "..."}],\n'
        '      "feedback": "Specific feedback referencing the student\'s words."\n'
        "    }\n"
        "  ],\n"
        '  "suggestedNextLevel": <1-7>,\n'
        '  "regressionFromLast": <true|false>\n'
        "}\n\n"
        "Return one entry per question above, echoing its questionIndex. "
        "Do not include any text outside the JSON object."
    )


def parse_short_question_grading_response(
    raw: str, submitted_indices: list[int]
) -> ShortQuestionGradingResult:
    """Parse the grader's JSON into a ShortQuestionGradingResult.

    `submitted_indices` is the authoritative list of question_indexes the
    student actually answered — the parser keys off it (not off whatever the
    model echoes back) so a model that drops or invents a question can't
    desync the graded rows from the submitted rows. A missing entry defaults
    to score 0.
    """
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v or 0)))

    def clamp_level(v: int) -> int:
        return max(1, min(7, int(v or 1)))

    by_index: dict[int, dict] = {}
    for q in parsed.get("questions", []):
        try:
            by_index[int(q.get("questionIndex"))] = q
        except (TypeError, ValueError):
            continue

    grades: list[ShortAnswerGrade] = []
    all_misconceptions: list[dict[str, str]] = []
    all_concepts_missed: list[str] = []
    for idx in submitted_indices:
        q = by_index.get(idx, {})
        score = clamp(q.get("score", 0))
        misconceptions = q.get("misconceptions", []) or []
        concepts_missed = q.get("conceptsMissed", []) or []
        grades.append(
            ShortAnswerGrade(
                question_index=idx,
                score=score,
                is_correct=bool(
                    q.get("isCorrect", score >= SHORT_ANSWER_PASS_THRESHOLD)
                ),
                feedback=q.get("feedback", ""),
                misconceptions=misconceptions,
            )
        )
        all_misconceptions.extend(misconceptions)
        all_concepts_missed.extend(concepts_missed)

    overall = sum(g.score for g in grades) / len(grades) if grades else 0.0

    return ShortQuestionGradingResult(
        overall_score=overall,
        grades=grades,
        misconceptions=all_misconceptions,
        concepts_missed=all_concepts_missed,
        suggested_next_level=clamp_level(parsed.get("suggestedNextLevel", 1)),
        regression_from_last=bool(parsed.get("regressionFromLast", False)),
    )


@dataclass
class ShortQuestionGradingPipelineInput:
    attempt_id: str
    engram_id: str
    user_id: str
    content: QuizContent
    answers: dict[int, str]  # question_index -> raw_answer
    target_cognitive_level: int
    mastery: Optional[EngramMastery] = None
    note_chunks: list[str] = field(default_factory=list)
    api_key: Optional[str] = None
    use_cloud: Optional[bool] = None


@dataclass
class ShortQuestionGradingPipelineOutput:
    attempt_id: str
    result: ShortQuestionGradingResult
    raw_response: str
    graded_at: str


async def run_short_question_grading_pipeline(
    inp: ShortQuestionGradingPipelineInput,
) -> ShortQuestionGradingPipelineOutput:
    from datetime import datetime

    # Pair each submitted answer with its question, keying off the engram's
    # QuizContent so expected_answer/level/stem come from the source of truth,
    # not the client. Answers with no matching question are dropped.
    #
    # CROSS-REPO CONTRACT ⇄ client: the submitted `question_index` is matched
    # against `question_number` here — so the client MUST send
    # question_index == question_number (NOT a 0-based position). A mismatch
    # silently drops the answer (scored 0). Client side: short_question.dart
    # builds {question_index: q.questionNumber}.
    questions_by_index = {q.question_number: q for q in inp.content.questions}
    items: list[ShortAnswerItem] = []
    for q_index, raw_answer in inp.answers.items():
        q = questions_by_index.get(q_index)
        if q is None:
            continue
        items.append(
            ShortAnswerItem(
                question_index=q_index,
                level=q.level,
                stem=q.stem,
                expected_answer=q.expected_answer,
                raw_answer=raw_answer,
            )
        )

    submitted_indices = [item.question_index for item in items]

    prompt = build_short_question_grading_prompt(
        ShortQuestionGradingContext(
            items=items,
            target_cognitive_level=inp.target_cognitive_level,
            note_chunks=inp.note_chunks,
            mastery=inp.mastery,
        )
    )
    raw_response = await call_grading_model(
        prompt,
        use_cloud=inp.use_cloud,
        api_key=inp.api_key,
        schema=SHORT_QUESTION_GRADING_SCHEMA,
    )
    result = parse_short_question_grading_response(raw_response, submitted_indices)

    return ShortQuestionGradingPipelineOutput(
        attempt_id=inp.attempt_id,
        result=result,
        raw_response=raw_response,
        graded_at=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# Paraphrase trap generator
# ---------------------------------------------------------------------------


async def generate_paraphrase_trap(
    original_question: str,
    note_chunks: list[str],
    target_cognitive_level: int,
    use_cloud: bool = True,
    api_key: Optional[str] = None,
) -> str:
    prompt = (
        f"You are creating a paraphrase trap for a cognitive level {target_cognitive_level}/7 question.\n\n"
        f'Original question: "{original_question}"\n\n'
        f"Context from notes:\n{chr(10).join(note_chunks[:2])}\n\n"
        f"Rewrite the question so it:\n"
        "1. Tests the same underlying concept\n"
        "2. Uses different framing, vocabulary, and scenario\n"
        "3. Cannot be answered by recognising the original wording\n"
        f"4. Is appropriate for level {target_cognitive_level}/7\n\n"
        "Return ONLY the new question text, no explanation."
    )
    return await call_grading_model(prompt, use_cloud=use_cloud, api_key=api_key)
