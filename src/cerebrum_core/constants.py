DEFAULT_CLOUD_MODEL = "gemma4:31b-cloud"
DEFAULT_CHAT_MODEL = "llama3.2:3b"
DEFAULT_EMBED_MODEL = "mxbai-embed-large:335m"


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
QUIZ_SCHEMA: dict = {
    "schema_id": "engram_short_answer_v1",
    "max_tokens": 512,
    "system": "You are an expert short_answer designer for adaptive learning systems.",
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
        "severity_short_answer_count": "integer (derived: high→3, medium→2, low→1)",
    },
    "output": {
        "type": "array",
        "items": {
            "finding_index": "integer",
            "question_number": "integer",
            "level": "recall | explain | apply",
            "stem": "string",
            "expected_answer": "string",
            "hint": "string",
            "context_anchored": "boolean",
            "severity": "string",
        },
    },
}
LFQ_SCHEMA: dict = {
    "schema_id": "engram_long_answer_v1",
    "max_tokens": 512,
    "system": "You are an expert long_answer question designer for academic assessment.",
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
    },
    "output": {
        "type": "object",
        "properties": {
            "finding_index": "integer",
            "question_stem": "string",
            "answer": "...",
            "parts": [
                {
                    "part": "string",
                    "level": "recall | explain | apply | analyse",
                    "question": "string",
                    "marks": "integer",
                    "mark_scheme": "string",
                    "note": "string | null",
                }
            ],
            "severity": "string",
            "total_marks": "integer",
        },
    },
}
SEARCH_SCHEMA: dict = {
    "schema_id": "semantic_search_query_v1",
    "max_tokens": 512,
    "system": "You are a semantic search query architect for an educational knowledge base.",
    "input": {
        "topic": "string",
        "knowledge_gaps_summary": ["string"],
        "gap_explanation": "string",
        "student_claim": "string",
        "concept_a": "string",
        "concept_b": "string",
        "confusion_description": "string",
        "correct_understanding": "string",
        "weak_areas": ["string"],
        "priority_study_areas": ["string"],
        "chunk_excerpt": "string",
    },
    "output": {
        "type": "array",
        "items": {
            "query_id": "integer",
            "signal_source": "string",
            "query_string": "string",
            "priority": "CRITICAL | HIGH | MEDIUM | CONTEXTUAL",
            "serves_engrams": ["flashcard | mcq | short_answer | long_answer"],
            "retrieval_intent": "string",
        },
    },
}
