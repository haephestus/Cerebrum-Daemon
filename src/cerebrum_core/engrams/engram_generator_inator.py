import json
import logging
import re
from pathlib import Path

# import sqlite3
# from pathlib import Path
from typing import Dict

from agents.rose import RosePrompts
from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.llm_invoker_inator import ollama_structured
from cerebrum_core.utils.retrieve_inator import RetrieverInator

# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings


FLASHCARD_SCHEMA = dict = {
    "schema_id": "engram_flashcard_v1",
    "model": "claude-sonnet-4-20250514",
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
MCQ_SCHEMA = dict = {
    "schema_id": "engram_mcq_v1",
    "model": "claude-sonnet-4-20250514",
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
QUIZ_SCHEMA = dict = {
    "schema_id": "engram_quiz_v1",
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 512,
    "system": "You are an expert quiz designer for adaptive learning systems.",
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
        "severity_quiz_count": "integer (derived: high→3, medium→2, low→1)",
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
LFQ_SCHEMA = dict = {
    "schema_id": "engram_structured_v1",
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 512,
    "system": "You are an expert structured question designer for academic assessment.",
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
SEARCH_SCHEMA = dict = {
    "schema_id": "semantic_search_query_v1",
    "model": "claude-sonnet-4-20250514",
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
            "serves_engrams": ["flashcard | mcq | quiz | structured"],
            "retrieval_intent": "string",
        },
    },
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EngramGenerator:

    def __init__(self, bubble_id: str, note_id: str):
        self.bubble_id = bubble_id
        self.note_id = note_id
        self.bubble_cache_path = (
            CerebrumPaths().cache_root_dir() / "bubble_cache" / self.bubble_id
        )
        self.archives_path = CerebrumPaths().kb_archives_path()
        self.embedding_model = ConfigManager().load_config().models.embedding_model
        self.chat_model = ConfigManager().load_config().models.chat_model
        self.analyses = self._analysis_retriever()

    # ── STEP 0: load analysis JSONs ────────────────────────────────────
    def _analysis_retriever(self) -> list[Dict]:
        analysis_dir = CerebrumPaths().note_analysis_dir(self.bubble_id, self.note_id)
        analysis_files = sorted(analysis_dir.glob("*.json"))
        if not analysis_files:
            raise FileNotFoundError(f"No analysis JSON files found in {analysis_dir}")
        results = []
        for file in analysis_files:
            analysis_file = json.loads(file.read_text(encoding="utf-8"))
            overview = analysis_file["analysis"]["note_overview"]
            findings = [
                {**f, "chunk_excerpt": chunk["chunk_excerpt"]}
                for chunk in analysis_file["analysis"]["chunk_diagnostics"]
                for f in chunk["findings"]
            ]
            results.append(
                {
                    "topic": overview["topic"],
                    "bubble_id": analysis_file["bubble_id"],
                    "mastery_signal": overview["mastery_signal"],
                    "strong_areas": overview["concept_map"]["strong_areas"],
                    "knowledge_gaps": overview["knowledge_gaps_summary"],
                    "priority_areas": overview["priority_study_areas"],
                    "weak_areas": overview["concept_map"]["weak_areas"],
                    "gap_explanations": [f["gap_explanation"] for f in findings],
                    "correct_understandings": [
                        f["correct_understanding"] for f in findings
                    ],
                    "findings": findings,
                }
            )
        logger.info(f"Loaded {len(results)} analysis files from {analysis_dir}")
        return results

    # ── STEP 1: retrieve + fill + cache ───────────────────────────────
    def retrieval_pass(self, engram_prompt: str, schema_id: str) -> list[Path]:
        """
        Retrieves context for each analysis, fills prompts per finding,
        writes one debug cache file per finding.
        Returns list of cache file paths written.
        """
        assert self.embedding_model is not None
        translation_prompt_template = RosePrompts.get_prompt("rose_analysis_to_query")
        assert translation_prompt_template is not None

        cache_files = []

        for i, analysis in enumerate(self.analyses):
            # retrieve once per analysis (embedding model active here)
            retriever = RetrieverInator(
                archives_root=str(self.archives_path),
                embedding_model=self.embedding_model,
            )
            filled_translation = (
                translation_prompt_template.replace("{topic}", str(analysis["topic"]))
                .replace("{bubble_id}", str(analysis["bubble_id"]))
                .replace("{knowledge_gaps}", str(analysis["knowledge_gaps"]))
                .replace("{priority_areas}", str(analysis["priority_areas"]))
                .replace("{weak_areas}", str(analysis["weak_areas"]))
                .replace("{gap_explanations}", str(analysis["gap_explanations"]))
            )
            translated = retriever.translator_inator(filled_translation)
            retriever.constructor_inator(translated_query=translated)
            retriever.retrieve_inator()
            chunks = retriever.context_inator()
            context_text = self._clean_chunks(chunks)

            logger.info(
                "Retrieved context for topic: %s (%d chars)",
                analysis["topic"],
                len(context_text),
            )

            # fill one prompt per finding, write to cache
            for j, finding in enumerate(analysis["findings"]):
                severity = finding.get("severity", "medium")
                severity_count = {"high": "3", "medium": "2", "low": "1"}.get(
                    severity, "2"
                )
                cache_file = (
                    self._prompt_cache_dir(schema_id)
                    / f"{self.note_id.lower()}_{i}_{j}.json"
                )

                if cache_file.exists():
                    logger.info("Cache hit, skipping → %s", cache_file)
                    cache_files.append(cache_file)
                    continue

                filled_prompt = (
                    engram_prompt.replace("{topic}", str(analysis["topic"]))
                    .replace("{retrieved_docs}", context_text)
                    .replace(
                        "{mastery_signal}",
                        str(analysis.get("mastery_signal", "unknown")),
                    )
                    .replace("{strong_areas}", str(analysis.get("strong_areas", [])))
                    .replace("{chunk_excerpt}", str(finding.get("chunk_excerpt", "")))
                    .replace("{finding_index}", str(finding.get("finding_index", j)))
                    .replace("{finding_type}", str(finding.get("type", "")))
                    .replace("{finding_severity}", severity)
                    .replace(
                        "{finding_confidence}", str(finding.get("confidence", 0.5))
                    )
                    .replace(
                        "{gap_explanation}", str(finding.get("gap_explanation", ""))
                    )
                    .replace("{student_claim}", str(finding.get("student_claim", "")))
                    .replace(
                        "{correct_understanding}",
                        str(finding.get("correct_understanding", "")),
                    )
                    .replace("{severity_card_count}", severity_count)
                    .replace("{severity_mcq_count}", severity_count)
                    .replace("{severity_quiz_count}", severity_count)
                )

                cache_file = self._write_prompt_cache(
                    filled_prompt=filled_prompt,
                    analysis=analysis,
                    finding=finding,
                    schema_id=schema_id,
                    i=i,
                    j=j,
                )
                cache_files.append(cache_file)
                logger.info(
                    "Prompt cached (%d chars) → %s", len(filled_prompt), cache_file
                )

        return cache_files

    # ── STEP 2: generate from cache ────────────────────────────────────
    def generation_pass(self, schema_id: str, analysis_schema: Dict) -> list:
        """
        Reads all cached prompt files for this note+schema,
        sends each to Ollama, saves engram output.
        Embedding model is NOT needed here — only chat model active.
        """
        cache_dir = self._prompt_cache_dir(schema_id)
        cache_files = sorted(cache_dir.glob("*.json"))

        if not cache_files:
            raise FileNotFoundError(
                f"No cached prompts found in {cache_dir}. " f"Run retrieval_pass first."
            )

        engram_dir = CerebrumPaths().engram_path(self.bubble_id)
        engram_dir.mkdir(parents=True, exist_ok=True)
        responses = []

        for cache_file in cache_files:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            filled_prompt = "\n".join(payload["filled_prompt_lines"])
            meta = payload["meta"]

            logger.info(
                "Generating engram %d-%d | topic: %s | %d chars",
                meta["analysis_index"],
                meta["finding_index"],
                meta["topic"],
                meta["prompt_chars"],
            )

            try:
                response = ollama_structured(
                    prompt=filled_prompt,
                    analyses_schema=analysis_schema,
                )
            except Exception as e:
                logger.error(
                    "Failed engram %d-%d: %s",
                    meta["analysis_index"],
                    meta["finding_index"],
                    e,
                )
                continue

            output_file = (
                engram_dir / f"{self.note_id}_{schema_id}_"
                f"{meta['analysis_index']}_{meta['finding_index']}.json"
            )
            output_file.write_text(
                json.dumps(
                    {
                        "note_id": meta["note_id"],
                        "bubble_id": meta["bubble_id"],
                        "topic": meta["topic"],
                        "schema_id": schema_id,
                        "analysis_index": meta["analysis_index"],
                        "finding_index": meta["finding_index"],
                        "finding_type": meta["finding_type"],
                        "finding_severity": meta["finding_severity"],
                        "engram": (
                            json.loads(response)
                            if isinstance(response, str)
                            else response
                        ),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            logger.info(
                "Saved engram %d-%d → %s",
                meta["analysis_index"],
                meta["finding_index"],
                output_file,
            )
            responses.append(response)

        return responses

    # ── HELPERS ────────────────────────────────────────────────────────
    def _prompt_cache_dir(self, schema_id: str) -> Path:
        d = self.bubble_cache_path / "debug" / "engrams" / schema_id / self.note_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_prompt_cache(
        self,
        filled_prompt: str,
        analysis: Dict,
        finding: Dict,
        schema_id: str,
        i: int,
        j: int,
    ) -> Path:
        import textwrap

        def wrap(text: str, width: int = 80) -> list[str]:
            lines = str(text).splitlines()
            wrapped = []
            for line in lines:
                if len(line) <= width:
                    wrapped.append(line)
                else:
                    wrapped.extend(textwrap.wrap(line, width=width) or [""])
            return wrapped

        payload = {
            "meta": {
                "note_id": self.note_id,
                "bubble_id": self.bubble_id,
                "analysis_index": i,
                "finding_index": j,
                "topic": analysis["topic"],
                "finding_type": finding.get("type", ""),
                "finding_severity": finding.get("severity", ""),
                "prompt_chars": len(filled_prompt),
            },
            "finding": {
                "chunk_excerpt": wrap(finding.get("chunk_excerpt", "")),
                "student_claim": wrap(finding.get("student_claim", "")),
                "correct_understanding": wrap(finding.get("correct_understanding", "")),
                "gap_explanation": wrap(finding.get("gap_explanation", "")),
            },
            "filled_prompt_lines": wrap(filled_prompt),
        }

        cache_file = (
            self._prompt_cache_dir(schema_id) / f"{self.note_id.lower()}_{i}_{j}.json"
        )
        cache_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return cache_file

    def _clean_chunks(self, chunks: list[str]) -> str:
        cleaned = []
        for chunk in chunks:
            lines = chunk.split("\n")
            content = "\n".join(
                l
                for l in lines
                if not re.match(
                    r"^(chunk_index|source_block_ids|parent_chunk_index|"
                    r"token_count|header_1|byte_start|byte_end|"
                    r"block_ids|hunk_index|-->)",
                    l.strip(),
                )
            ).strip()
            if content:
                cleaned.append(content)
        return "\n\n".join(cleaned)


# ── entry point ────────────────────────────────────────────────────────
engram = EngramGenerator(
    bubble_id="genetics",
    note_id="01KDTN8K9G360YEQXRKVZA4ZQT",
)

schema_id = FLASHCARD_SCHEMA["schema_id"]
flashcard_prompt = RosePrompts.get_prompt("rose_flashcard_generator")
assert flashcard_prompt is not None

# Pass 1 — embedding model only
engram.retrieval_pass(engram_prompt=flashcard_prompt, schema_id=schema_id)

# Pass 2 — chat model only, reads from cache
responses = engram.generation_pass(
 schema_id=schema_id,
 analysis_schema=FLASHCARD_SCHEMA,
)
print(responses)
