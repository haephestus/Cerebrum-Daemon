DEFAULT_CLOUD_MODEL = "gemma4:31b-cloud"
DEFAULT_CHAT_MODEL = "llama3.2:3b"
DEFAULT_EMBED_MODEL = "mxbai-embed-large:335m"
# ---------------------------------------------------------------------
# PHASE_WEEKS_SCHEMA — response_format schema for the week-generation call
# ---------------------------------------------------------------------

PHASE_WEEKS_SCHEMA = {
    "type": "object",
    "properties": {
        "weeks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "week_number": {
                        "type": "integer",
                        "description": "Absolute week number from plan start, 1-indexed.",
                    },
                    "focus_summary": {
                        "type": "string",
                        "description": "One-line theme for the week, e.g. 'CRISPR delivery mechanisms + first pipeline test'.",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Topic strings this week covers. MUST reuse an "
                            "existing topic string from the provided "
                            "topic_mastery context where the subject matter "
                            "genuinely matches — only mint a new topic string "
                            "for material that has no existing coverage."
                        ),
                    },
                    "days": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day_of_week": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 6,
                                    "description": "0=Monday .. 6=Sunday.",
                                },
                                "tasks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string"},
                                            "task_type": {
                                                "type": "string",
                                                "enum": [
                                                    "study",
                                                    "practice",
                                                    "build",
                                                    "review",
                                                    "milestone_check",
                                                ],
                                            },
                                            "topic": {
                                                "type": ["string", "null"],
                                                "description": (
                                                    "Required (non-null) for "
                                                    "practice/review tasks — "
                                                    "these auto-complete off "
                                                    "engram activity under this "
                                                    "topic. Null for build/"
                                                    "milestone_check, which are "
                                                    "manually marked done."
                                                ),
                                            },
                                            "target_minutes": {"type": "integer"},
                                            "source_hint": {
                                                "type": ["string", "null"],
                                                "description": (
                                                    "Why this task exists — cite "
                                                    "the mastery gap or "
                                                    "misconception it addresses "
                                                    "if one was provided in "
                                                    "context, else null."
                                                ),
                                            },
                                        },
                                        "required": [
                                            "label",
                                            "task_type",
                                            "target_minutes",
                                        ],
                                    },
                                },
                            },
                            "required": ["day_of_week", "tasks"],
                        },
                    },
                },
                "required": ["week_number", "focus_summary", "topics", "days"],
            },
        }
    },
    "required": ["weeks"],
}

STUDY_PLAN_SCHEMA = {
    "type": "object",
    "required": [
        "plan_overview",
        "phases",
        "weekly_rhythm",
        "regional_opportunity_map",
        "success_metrics",
        "immediate_next_actions",
    ],
    "properties": {
        "plan_overview": {
            "type": "object",
            "required": [
                "target_role",
                "total_duration_months",
                "starting_position",
                "guiding_principle",
            ],
            "properties": {
                "target_role": {"type": "string"},
                "total_duration_months": {"type": "integer"},
                "starting_position": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["domain", "readiness_pct", "notes"],
                        "properties": {
                            "domain": {"type": "string"},
                            "readiness_pct": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "notes": {"type": "string"},
                        },
                    },
                },
                "guiding_principle": {"type": "string"},
            },
        },
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "phase_id",
                    "phase_label",
                    "month_range",
                    "theme",
                    "tracks",
                    "milestone",
                ],
                "properties": {
                    "phase_id": {"type": "integer"},
                    "phase_label": {"type": "string"},
                    "month_range": {"type": "string"},
                    "theme": {"type": "string"},
                    "tracks": {
                        "type": "object",
                        "required": [
                            "income",
                            "technical_skill",
                            "domain_knowledge",
                            "project",
                        ],
                        "properties": {
                            "income": {
                                "type": "object",
                                "required": ["roles", "target_range"],
                                "properties": {
                                    "roles": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "target_range": {"type": "string"},
                                },
                            },
                            "technical_skill": {
                                "type": "object",
                                "required": ["focus_areas"],
                                "properties": {
                                    "focus_areas": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                            "domain_knowledge": {
                                "type": "object",
                                "required": ["focus_areas", "self_test"],
                                "properties": {
                                    "focus_areas": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "self_test": {"type": "string"},
                                },
                            },
                            "project": {
                                "type": "object",
                                "required": ["name", "description", "requirements"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "requirements": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                    "milestone": {"type": "string"},
                },
            },
        },
        "weekly_rhythm": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["day_or_block", "focus", "description"],
                "properties": {
                    "day_or_block": {"type": "string"},
                    "focus": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "success_metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["month_marker", "checkpoint", "is_binary_check"],
                "properties": {
                    "month_marker": {"type": "string"},
                    "checkpoint": {"type": "string"},
                    "is_binary_check": {"type": "boolean"},
                },
            },
        },
        "immediate_next_actions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

ANALYSIS_SCHEMA = {
    "type": "object",
    "required": ["chunk_diagnostics", "note_overview"],
    "properties": {
        "chunk_diagnostics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["chunk_id", "chunk_excerpt", "findings"],
                "properties": {
                    "chunk_id": {"type": "string"},
                    "chunk_excerpt": {"type": "string"},
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "finding_index",
                                "type",
                                "severity",
                                "confidence",
                                "context_coverage",
                                "student_claim",
                                "correct_understanding",
                                "gap_explanation",
                            ],
                            "properties": {
                                "finding_index": {"type": "integer"},
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "misconception",
                                        "weak_point",
                                        "incorrect",
                                        "missing_concept",
                                    ],
                                },
                                "severity": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "context_coverage": {"type": "boolean"},
                                "student_claim": {"type": "string"},
                                "correct_understanding": {"type": "string"},
                                "gap_explanation": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "note_overview": {
            "type": "object",
            "required": [
                "topic",
                "mastery_signal",
                "progress_delta",
                "concept_map",
                "progress",
                "regressions",
                "knowledge_gaps_summary",
                "priority_study_areas",
                "remediation_order",
                "suggested_sources",
            ],
            "properties": {
                "topic": {"type": "string"},
                "mastery_signal": {
                    "type": "string",
                    "enum": ["novice", "developing", "proficient", "advanced"],
                },
                "progress_delta": {
                    "type": "string",
                    "enum": [
                        "baseline",
                        "regressed",
                        "stagnant",
                        "improved",
                        "significantly_improved",
                    ],
                },
                "concept_map": {
                    "type": "object",
                    "required": ["strong_areas", "weak_areas", "confused_links"],
                    "properties": {
                        "strong_areas": {"type": "array", "items": {"type": "string"}},
                        "weak_areas": {"type": "array", "items": {"type": "string"}},
                        "confused_links": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": [
                                    "concept_a",
                                    "concept_b",
                                    "confusion_description",
                                ],
                                "properties": {
                                    "concept_a": {"type": "string"},
                                    "concept_b": {"type": "string"},
                                    "confusion_description": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "progress": {"type": "array", "items": {"type": "string"}},
                "regressions": {"type": "array", "items": {"type": "string"}},
                "knowledge_gaps_summary": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "priority_study_areas": {"type": "array", "items": {"type": "string"}},
                "remediation_order": {"type": "array", "items": {"type": "string"}},
                "suggested_sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "title",
                            "type",
                            "link_or_citation",
                            "addresses_findings",
                            "reason",
                        ],
                        "properties": {
                            "title": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "book",
                                    "article",
                                    "paper",
                                    "video",
                                    "course",
                                    "online",
                                ],
                            },
                            "link_or_citation": {"type": "string"},
                            "addresses_findings": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Shared JSON schema enforced on every chunk's LLM response
# ---------------------------------------------------------------------------
CHUNK_ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "required": ["chunk_diagnostics", "note_overview"],
    "properties": {
        "chunk_diagnostics": {
            # chunk_id and chunk_excerpt are injected post-hoc — not required
            # from the LLM so Ollama's enforcer doesn't fight us on them.
            "type": "array",
            "items": {
                "type": "object",
                "required": ["findings"],
                "properties": {
                    "chunk_id": {"type": "string"},
                    "chunk_excerpt": {"type": "string"},
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "finding_index",
                                "type",
                                "severity",
                                "confidence",
                                "context_coverage",
                                "student_claim",
                                "correct_understanding",
                                "gap_explanation",
                            ],
                            "properties": {
                                "finding_index": {"type": "integer"},
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "misconception",
                                        "weak_point",
                                        "incorrect",
                                        "missing_concept",
                                    ],
                                },
                                "severity": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "context_coverage": {"type": "boolean"},
                                "student_claim": {"type": "string"},
                                "correct_understanding": {"type": "string"},
                                "gap_explanation": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "note_overview": {
            "type": "object",
            "required": [
                "topic",
                "mastery_signal",
                "progress_delta",
                "concept_map",
                "progress",
                "regressions",
                "knowledge_gaps_summary",
                "priority_study_areas",
                "remediation_order",
                "suggested_sources",
            ],
            "properties": {
                "topic": {"type": "string"},
                "mastery_signal": {
                    "type": "string",
                    "enum": ["novice", "developing", "proficient", "advanced"],
                },
                "progress_delta": {
                    "type": "string",
                    "enum": [
                        "baseline",
                        "regressed",
                        "stagnant",
                        "improved",
                        "significantly_improved",
                    ],
                },
                "concept_map": {
                    "type": "object",
                    "required": ["strong_areas", "weak_areas", "confused_links"],
                    "properties": {
                        "strong_areas": {"type": "array", "items": {"type": "string"}},
                        "weak_areas": {"type": "array", "items": {"type": "string"}},
                        "confused_links": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": [
                                    "concept_a",
                                    "concept_b",
                                    "confusion_description",
                                ],
                                "properties": {
                                    "concept_a": {"type": "string"},
                                    "concept_b": {"type": "string"},
                                    "confusion_description": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "progress": {"type": "array", "items": {"type": "string"}},
                "regressions": {"type": "array", "items": {"type": "string"}},
                "knowledge_gaps_summary": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "priority_study_areas": {"type": "array", "items": {"type": "string"}},
                "remediation_order": {"type": "array", "items": {"type": "string"}},
                "suggested_sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "title",
                            "type",
                            "link_or_citation",
                            "addresses_findings",
                            "reason",
                        ],
                        "properties": {
                            "title": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "book",
                                    "article",
                                    "paper",
                                    "video",
                                    "course",
                                    "online",
                                ],
                            },
                            "link_or_citation": {"type": "string"},
                            "addresses_findings": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}


FLASHCARD_SCHEMA: dict = {
    "schema_id": "engram_flashcard_v1",
    "max_tokens": 512,
    "system": "You are an expert flashcard designer for adaptive learning systems.",
    "input": {
        "topic": "string",
        "mastery_signal": "string",
        "strong_areas": ["string"],
        "chunk_excerpt": "string",
        "finding_index": "integer",
        "finding_type": "string",
        "finding_severity": "high | medium | low",
        "finding_confidence": "float (0.0 - 1.0)",
        "gap_explanation": "string",
        "student_claim": "string",
        "correct_understanding": "string",
        "retrieved_docs": ["string"],
        "severity_card_count": "integer (derived: high→3, medium→2, low→1)",
    },
    "output": {
        "type": "array",
        "items": {
            "finding_index": "integer",
            "card_number": "integer",
            "front": "string",
            "back": "string",
            "bridge_concept": "string | null",
            "severity": "string",
            "diagnostic_note": "string | null",
        },
    },
}
MCQ_SCHEMA: dict = {
    "schema_id": "engram_mcq_v1",
    "max_tokens": 512,
    "system": "You are an expert MCQ designer for adaptive student assessment.",
    "input": {
        "topic": "string",
        "mastery_signal": "string",
        "concept_a": "string",
        "concept_b": "string",
        "confusion_description": "string",
        "chunk_excerpt": "string",
        "finding_index": "integer",
        "finding_type": "string",
        "finding_severity": "high | medium | low",
        "finding_confidence": "float (0.0 - 1.0)",
        "gap_explanation": "string",
        "student_claim": "string",
        "correct_understanding": "string",
        "retrieved_docs": ["string"],
        "severity_mcq_count": "integer (derived: high→3, medium→2, low→1)",
    },
    "output": {
        "type": "array",
        "items": {
            "finding_index": "integer",
            "question_number": "integer",
            "stem": "string",
            "options": {"A": "string", "B": "string", "C": "string", "D": "string"},
            "correct_option": "string",
            "correct_explanation": "string",
            "distractor_notes": {
                "misconception_option": "string",
                "confused_link_option": "string",
            },
            "severity": "string",
        },
    },
}
SHORT_QUESTION_SCHEMA: dict = {
    "schema_id": "engram_short_question_v1",
    "max_tokens": 512,
    "system": "You are an expert short_question designer for adaptive learning systems.",
    "input": {
        "topic": "string",
        "mastery_signal": "string",
        "progress_delta": "string",
        "strong_areas": ["string"],
        "weak_areas": ["string"],
        "knowledge_gaps_summary": ["string"],
        "chunk_excerpt": "string",
        "finding_index": "integer",
        "finding_type": "string",
        "finding_severity": "high | medium | low",
        "finding_confidence": "float (0.0 - 1.0)",
        "gap_explanation": "string",
        "student_claim": "string",
        "correct_understanding": "string",
        "context_coverage": "boolean",
        "retrieved_docs": ["string"],
        "severity_short_question_count": "integer (derived: high→3, medium→2, low→1)",
        "target_cognitive_level": "integer (1-7) — sets the scaffold's level ceiling",
    },
    "output": {
        "type": "array",
        "items": {
            "type": "object",
            "required": [
                "finding_index",
                "question_number",
                "level",
                "stem",
                "expected_answer",
                "context_anchored",
                "severity",
            ],
            "properties": {
                "finding_index": {"type": "integer"},
                "question_number": {"type": "integer"},
                "level": {
                    "type": "string",
                    "enum": [
                        "recall",
                        "understand",
                        "apply",
                        "analyse",
                        "synthesise",
                        "evaluate",
                        "doctoral",
                    ],
                },
                "stem": {"type": "string"},
                "expected_answer": {"type": "string"},
                "hint": {"type": ["string", "null"]},
                "context_anchored": {"type": "boolean"},
                "severity": {"type": "string", "enum": ["high", "medium", "low"]},
            },
        },
    },
}

LONG_QUESTION_SCHEMA: dict = {
    "schema_id": "engram_long_question_v1",
    "max_tokens": 512,
    "system": "You are an expert long_question question designer for academic assessment.",
    "input": {
        "topic": "string",
        "mastery_signal": "string",
        "remediation_order": ["string"],
        "regression_prompt": "string",
        "chunk_excerpt": "string",
        "finding_index": "integer",
        "finding_type": "string",
        "finding_severity": "high | medium | low",
        "finding_confidence": "float (0.0 - 1.0)",
        "gap_explanation": "string",
        "student_claim": "string",
        "correct_understanding": "string",
        "concept_a": "string",
        "concept_b": "string",
        "confusion_description": "string",
        "retrieved_docs": ["string"],
        "target_cognitive_level": "integer (1-7) — sets the scaffold's level ceiling",
    },
    "output": {
        "type": "object",
        "required": [
            "finding_index",
            "question_stem",
            "parts",
            "severity",
            "total_marks",
        ],
        "properties": {
            "finding_index": {"type": "integer"},
            "question_stem": {"type": "string"},
            "answer": {"type": ["string", "null"]},
            "parts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["part", "level", "question", "marks", "mark_scheme"],
                    "properties": {
                        "part": {"type": "string"},
                        "level": {
                            "type": "string",
                            "enum": [
                                "recall",
                                "understand",
                                "apply",
                                "analyse",
                                "synthesise",
                                "evaluate",
                                "doctoral",
                            ],
                        },
                        "question": {"type": "string"},
                        "marks": {"type": "integer"},
                        "mark_scheme": {"type": "string"},
                        "note": {"type": ["string", "null"]},
                    },
                },
            },
            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
            "total_marks": {"type": "integer"},
        },
    },
}
