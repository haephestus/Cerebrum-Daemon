# TODO: this generator is disconnected from the rest of the engram-mastery
# system. It does not import types.py, does not go through
# MasteryRepository/SQLiteRepository, and does not use vector_store.py's
# EmbeddingProvider/VectorStore interfaces — it has its own retrieval path
# via RetrieverInator and writes finished engrams to loose JSON files under
# CerebrumPaths().engram_path(), not into the `engrams` table.
#
# It IS consistent with ai_grading.py in one respect: both now call
# ollama_cloud_call / ollama_local_call rather than the Anthropic API, so at
# least the model-calling layer matches across generation and grading.
#
# To actually wire this in as "notes -> engrams" per the intended flow:
#   1. Instead of _analysis_retriever() reading note-analysis JSON off disk,
#      this should likely be triggered by rows in engram_generation_queue
#      (written by mastery_service.queue_engram_generation — see the TODO
#      there) so misconception-triggered regeneration actually runs this
#      class instead of just accumulating unread queue rows.
#   2. generation_pass() should call repo.create_engram(...) (this now
#      exists on SQLiteRepository, writing into the typed mcq_content /
#      flashcard_content / short_answer_questions / long_question_content tables)
#      instead of / in addition to writing the JSON file to engram_dir, so
#      generated engrams become queryable through get_engram /
#      get_topic_engrams and actually show up to students via build_study_queue.
#   3. The four content schemas below (FLASHCARD_SCHEMA, MCQ_SCHEMA,
#      QUIZ_SCHEMA, LFQ_SCHEMA) don't map one-to-one onto types.py's
#      MCQContent / FlashcardContent / QuizContent / LongQuestionContent
#      field names (e.g. "stem" here vs. "question" in types.py; this
#      schema's "correct_option"/"correct_explanation" vs. MCQContent's
#      "correct"/"explanation"). Needs a mapping layer before _parse_engram()'s
#      output can become an Engram.
import json
import logging
import re
from pathlib import Path
from typing import Dict

from agents.rose import RosePrompts
from cerebrum_core.constants import (
    FLASHCARD_SCHEMA,
    LFQ_SCHEMA,
    MCQ_SCHEMA,
    QUIZ_SCHEMA,
)
from cerebrum_core.engrams.storage.sqlite_repository import NoteEngramRepository
from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.ollama_compat.invoker_inator import (
    ollama_cloud_call,
    ollama_local_call,
    ollama_local_call2,
)
from cerebrum_core.utils.retrieve_inator import RetrieverInator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EngramGenerator:

    def __init__(self, bubble_id: str, note_id: str):
        self.bubble_id = bubble_id
        self.note_id = note_id
        self.target_cognitive_level: int | None = None
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
        print(analysis_dir)
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
                if "severity" not in finding:
                    logger.warning(
                        "Finding %d in analysis %d missing 'severity' (schema "
                        "requires it) — defaulting to 'medium'. Analysis file "
                        "may be malformed.",
                        finding.get("finding_index", j),
                        i,
                    )
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
                    .replace("{severity_short_answer_count}", severity_count)
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
    # ── STEP 2: generate from cache ────────────────────────────────────
    # Severity ("high"/"medium"/"low") measures how damaging a gap is —
    # it drives item COUNT (severity_card_count etc. in retrieval_pass)
    # and nothing else. It is not a difficulty proxy: a high-severity
    # misconception can need a simple corrective fact (low Bloom's level),
    # and a low-severity weak_point can only surface at a higher level.
    # cognitive_level is set from, in priority order:
    #   1. self.target_cognitive_level, when generate_engram_for_level
    #      explicitly requested a level — always wins.
    #   2. TYPE_TO_COGNITIVE_LEVEL, as a fallback for un-targeted base
    #      generation (_mcq_generator etc.), keyed on WHAT KIND of gap
    #      this is rather than how severe it is.
    TYPE_TO_COGNITIVE_LEVEL = {
        "missing_concept": 1,  # not there yet — needs introducing/recalling first
        "incorrect": 2,  # one wrong detail, otherwise intact — needs correcting w/ understanding
        "weak_point": 3,  # fragile/inconsistent — needs exercising via application
        "misconception": 4,  # actively wrong model — needs breaking down to dislodge
    }
    # Deliberately capped at 4 (Analyse). Synthesise/Evaluate/Doctoral (5-7)
    # should only ever be reached via an explicit target_cognitive_level —
    # never as a silent fallback from an automatically-detected finding.
    _TYPE_TO_COGNITIVE_LEVEL = {
        "missing_concept": 1,
        "incorrect": 2,
        "weak_point": 3,
        "misconception": 4,
    }

    # ── STEP 2: generate from cache ────────────────────────────────────
    def generation_pass(self, schema_id: str, engram_schema: Dict) -> list[Dict]:
        """
        Reads all cached prompt files for this note+schema,
        sends each to Ollama, and writes one flat JSON file per generated
        engram item (no "engram" nesting, no "items" list wrapper).
        Embedding model is NOT needed here — only chat model active.

        Returns a list of per-item outcome dicts:
            {
                "analysis_index": int,
                "finding_index": int,
                "engram_type": str,
                "engram_id": str | None,   # None means this item never made it into the repo
                "error": str | None,       # populated for any failure stage
            }
        One dict per generated item (short_answer responses may contribute several
        dicts sharing one engram_id). A cache file that fails at the LLM
        call or JSON-parse stage contributes exactly one dict with
        engram_id=None and error set — there's no per-item detail to report
        yet at that point.
        """
        repo = NoteEngramRepository()
        cache_dir = self._prompt_cache_dir(schema_id)
        cache_files = sorted(cache_dir.glob("*.json"))
        if not cache_files:
            raise FileNotFoundError(
                f"No cached prompts found in {cache_dir}. " f"Run retrieval_pass first."
            )
        engram_dir = CerebrumPaths().engram_path(self.bubble_id)
        engram_dir.mkdir(parents=True, exist_ok=True)
        outcomes: list[Dict] = []
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
                response = ollama_cloud_call(
                    prompt=filled_prompt,
                    schema=engram_schema,
                    OLLAMA_API_KEY=ConfigManager().load_config().ollama.api_key,
                )
            except Exception as e:
                logger.error(
                    "Failed engram %d-%d: %s",
                    meta["analysis_index"],
                    meta["finding_index"],
                    e,
                )
                outcomes.append(
                    {
                        "analysis_index": meta["analysis_index"],
                        "finding_index": meta["finding_index"],
                        "engram_type": self._engram_type(schema_id),
                        "engram_id": None,
                        "error": f"ollama_call_failed: {e}",
                    }
                )
                continue
            parsed_engram = self._parse_engram(response, schema_id)
            base_record = {
                "note_id": meta["note_id"],
                "bubble_id": meta["bubble_id"],
                "topic": meta["topic"],
                "schema_id": schema_id,
                "analysis_index": meta["analysis_index"],
                "finding_index": meta["finding_index"],
                "finding_type": meta["finding_type"],
                "finding_severity": meta["finding_severity"],
                "engram_type": parsed_engram["engram_type"],
            }
            # Parsing failed entirely — still write one flat file with the
            # error info merged in, so there's always something on disk to debug from.
            if "parse_error" in parsed_engram:
                output_file = (
                    engram_dir / f"{self.note_id.upper()}_{schema_id}_"
                    f"{meta['analysis_index']}_{meta['finding_index']}.json"
                )
                error_fields = {
                    k: v
                    for k, v in parsed_engram.items()
                    if k not in ("engram_type", "items")
                }
                output_file.write_text(
                    json.dumps(
                        {**base_record, **error_fields}, indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
                logger.error(
                    "Saved engram %d-%d with parse error → %s",
                    meta["analysis_index"],
                    meta["finding_index"],
                    output_file,
                )
                outcomes.append(
                    {
                        "analysis_index": meta["analysis_index"],
                        "finding_index": meta["finding_index"],
                        "engram_type": parsed_engram["engram_type"],
                        "engram_id": None,
                        "error": f"parse_error: {parsed_engram['parse_error']}",
                    }
                )
                continue

            cognitive_level = getattr(self, "target_cognitive_level", None)
            if cognitive_level is None:
                finding_type = meta.get("finding_type", "")
                if finding_type not in self._TYPE_TO_COGNITIVE_LEVEL:
                    logger.warning(
                        "Finding %d-%d has unrecognised/missing type %r — "
                        "defaulting cognitive_level to 1 (recall).",
                        meta["analysis_index"],
                        meta["finding_index"],
                        finding_type,
                    )
                cognitive_level = self.TYPE_TO_COGNITIVE_LEVEL.get(finding_type, 1)

            tags = [meta["finding_type"], meta["topic"]]

            # short_answer is the one type where multiple items in this response
            # belong to ONE engram (many short_answer_questions rows) rather than
            # one engram each — reset per cache_file, reused across the
            # items loop below.
            short_answer_engram_id = None

            # One flat file per item — no list, no "items"/"engram" nesting.
            items = parsed_engram["items"] or [{}]
            for idx, item in enumerate(items):
                suffix = f"_{idx}" if len(items) > 1 else ""
                output_file = (
                    engram_dir / f"{self.note_id.upper()}_{schema_id}_"
                    f"{meta['analysis_index']}_{meta['finding_index']}{suffix}.json"
                )
                output_file.write_text(
                    json.dumps({**base_record, **item}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info(
                    "Saved engram %d-%d%s → %s",
                    meta["analysis_index"],
                    meta["finding_index"],
                    suffix,
                    output_file,
                )

                item_engram_id = None
                item_error = None
                try:
                    if "mcq" in schema_id:
                        item_engram_id = repo.add_mcq(
                            meta["note_id"], item, cognitive_level, tags
                        )
                    elif "flashcard" in schema_id:
                        item_engram_id = repo.add_flashcard(
                            meta["note_id"], item, cognitive_level, tags
                        )
                    elif "short_answer" in schema_id:
                        # reuse short_answer_engram_id across items so every
                        # question in this response lands on the same
                        # short_answer engram; None on the first iteration lets
                        # add_short_answer generate a fresh id, which we then
                        # capture and pass on subsequent iterations.
                        short_answer_engram_id = repo.add_short_answer(
                            meta["note_id"],
                            [item],
                            cognitive_level,
                            tags,
                            engram_id=short_answer_engram_id,
                        )
                        item_engram_id = short_answer_engram_id
                    elif "long_answer" in schema_id:
                        item_engram_id = repo.add_long_question(
                            meta["note_id"], item, cognitive_level, tags
                        )
                except Exception as e:
                    logger.error(
                        "Failed to write engram %d-%d%s to repo: %s",
                        meta["analysis_index"],
                        meta["finding_index"],
                        suffix,
                        e,
                    )
                    item_error = f"repo_write_failed: {e}"

                outcomes.append(
                    {
                        "analysis_index": meta["analysis_index"],
                        "finding_index": meta["finding_index"],
                        "engram_type": parsed_engram["engram_type"],
                        "engram_id": item_engram_id,
                        "error": item_error,
                    }
                )

        return outcomes

    # ── HELPERS ────────────────────────────────────────────────────────
    def _persist_engram(
        self,
    ):
        pass

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
                line
                for line in lines
                if not re.match(
                    r"^(chunk_index|source_block_ids|parent_chunk_index|"
                    r"token_count|header_1|byte_start|byte_end|"
                    r"block_ids|hunk_index|-->)",
                    line.strip(),
                )
            ).strip()
            if content:
                cleaned.append(content)
        return "\n\n".join(cleaned)

    def _extract_json_block(self, text: str) -> str:
        """Strip markdown code fences (```json ... ``` or ``` ... ```) from a raw LLM string."""
        text = text.strip()
        fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()
        # fallback: strip stray leading/trailing fences even if malformed
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _engram_type(self, schema_id: str) -> str:
        """Purely a label derived from schema_id — does not drive any parsing branch."""
        schema_id = schema_id.lower()
        if "flashcard" in schema_id:
            return "flashcard"
        if "mcq" in schema_id:
            return "mcq"
        if "short_answer" in schema_id:
            return "short_answer"
        if "long_answer" in schema_id:
            return "long_answer"
        return "unknown"

    def _parse_engram(self, response, schema_id: str = "") -> Dict:
        """
        Engram-agnostic parser. Handles dicts, JSON strings, markdown-fenced JSON,
        and single-object vs list payloads. Does NOT branch on engram type —
        whatever keys the model returned are passed through as-is.

        Returns:
            {"engram_type": str, "items": [ {...}, ... ]}
            or, on failure:
            {"engram_type": str, "items": [], "parse_error": str, "raw": <original response>}
        """
        engram_type = self._engram_type(schema_id)

        # 1. Get to a Python object, however the response arrives.
        if isinstance(response, (dict, list)):
            parsed = response
        elif isinstance(response, str) and response.strip():
            cleaned = self._extract_json_block(response)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as error:
                logger.error(
                    "Failed to parse engram response: %s | raw=%r",
                    error,
                    response[:200],
                )
                return {
                    "engram_type": engram_type,
                    "items": [],
                    "parse_error": str(error),
                    "raw": response,
                }
        else:
            return {"engram_type": engram_type, "items": []}

        # 2. Normalize shape: always end up with a list of item dicts, keys untouched.
        if isinstance(parsed, dict):
            items = [parsed]
        elif isinstance(parsed, list):
            items = [item for item in parsed if isinstance(item, dict)]
        else:
            logger.error(
                "Unexpected engram payload type %s for schema %s",
                type(parsed),
                schema_id,
            )
            return {
                "engram_type": engram_type,
                "items": [],
                "parse_error": f"unexpected top-level type: {type(parsed).__name__}",
                "raw": response if isinstance(response, str) else str(response),
            }

        return {"engram_type": engram_type, "items": items}


# ── entry point ────────────────────────────────────────────────────────
def _short_answer_generator(bubble_id, note_id):
    engram = EngramGenerator(
        bubble_id=bubble_id,
        note_id=note_id,
    )

    schema_id = QUIZ_SCHEMA["schema_id"]
    flashcard_prompt = RosePrompts.get_prompt("rose_short_answer_generator")
    assert flashcard_prompt is not None

    # Pass 1 — embedding model only
    engram.retrieval_pass(engram_prompt=flashcard_prompt, schema_id=schema_id)

    # Pass 2 — chat model only, reads from cache
    engram.generation_pass(
        schema_id=schema_id,
        engram_schema=QUIZ_SCHEMA,
    )


def _mcq_generator(bubble_id, note_id):
    engram = EngramGenerator(
        bubble_id=bubble_id,
        note_id=note_id,
    )

    schema_id = MCQ_SCHEMA["schema_id"]
    flashcard_prompt = RosePrompts.get_prompt("rose_mcq_generator")
    assert flashcard_prompt is not None

    # Pass 1 — embedding model only
    engram.retrieval_pass(engram_prompt=flashcard_prompt, schema_id=schema_id)

    # Pass 2 — chat model only, reads from cache
    engram.generation_pass(
        schema_id=schema_id,
        engram_schema=MCQ_SCHEMA,
    )


def _lfq_generator(bubble_id, note_id):
    engram = EngramGenerator(
        bubble_id=bubble_id,
        note_id=note_id,
    )

    schema_id = LFQ_SCHEMA["schema_id"]
    flashcard_prompt = RosePrompts.get_prompt("rose_lfq_generator")
    assert flashcard_prompt is not None

    # Pass 1 — embedding model only
    engram.retrieval_pass(engram_prompt=flashcard_prompt, schema_id=schema_id)

    # Pass 2 — chat model only, reads from cache
    engram.generation_pass(
        schema_id=schema_id,
        engram_schema=LFQ_SCHEMA,
    )


def _flashacard_generator(bubble_id, note_id):
    engram = EngramGenerator(
        bubble_id=bubble_id,
        note_id=note_id,
    )

    schema_id = FLASHCARD_SCHEMA["schema_id"]
    flashcard_prompt = RosePrompts.get_prompt("rose_flashcard_generator")
    assert flashcard_prompt is not None

    # Pass 1 — embedding model only
    engram.retrieval_pass(engram_prompt=flashcard_prompt, schema_id=schema_id)

    # Pass 2 — chat model only, reads from cache
    engram.generation_pass(
        schema_id=schema_id,
        engram_schema=FLASHCARD_SCHEMA,
    )


def _main() -> None:

    bubble_id = "1edae102638a8cd7882e6de1c1e9639e"
    note_id = "01KTC4MWWA4YNSYNTYDYBEKB52"

    _flashacard_generator(bubble_id, note_id)
    _mcq_generator(bubble_id, note_id)
    _short_answer_generator(bubble_id, note_id)
    _lfq_generator(bubble_id, note_id)


if __name__ == "__main__":
    _main()
