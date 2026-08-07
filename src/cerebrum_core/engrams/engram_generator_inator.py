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
#      [DONE — see generate_engram_for_level() below, which is what
#      mastery_service.process_generation_queue() calls per job.]
#   2. generation_pass() should call repo.create_engram(...) (this now
#      exists on SQLiteRepository, writing into the typed mcq_content /
#      flashcard_content / short_question / long_question_content tables)
#      instead of / in addition to writing the JSON file to engram_dir, so
#      generated engrams become queryable through get_engram /
#      get_topic_engrams and actually show up to students via build_study_queue.
#   3. The four content schemas below (FLASHCARD_SCHEMA, MCQ_SCHEMA,
#      SHORT_QUESTION_SCHEMA, LONG_QUESTION_SCHEMA) don't map one-to-one onto types.py's
#      MCQContent / FlashcardContent / QuizContent / LongQuestionContent
#      field names (e.g. "stem" here vs. "question" in types.py; this
#      schema's "correct_option"/"correct_explanation" vs. MCQContent's
#      "correct"/"explanation"). Needs a mapping layer before _parse_engram()'s
#      output can become an Engram.
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

from agents.rose import RosePrompts
from cerebrum_core.constants import (
    FLASHCARD_SCHEMA,
    LONG_QUESTION_SCHEMA,
    MCQ_SCHEMA,
    SHORT_QUESTION_SCHEMA,
)
from cerebrum_core.database.note_engram_repository import NoteEngramRepository
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
                    "progress_delta": overview.get("progress_delta", ""),
                    "strong_areas": overview["concept_map"]["strong_areas"],
                    "weak_areas": overview["concept_map"]["weak_areas"],
                    "confused_links": overview["concept_map"].get("confused_links", []),
                    "knowledge_gaps": overview["knowledge_gaps_summary"],
                    "priority_areas": overview["priority_study_areas"],
                    "remediation_order": overview.get("remediation_order", []),
                    "regressions": overview.get("regressions", []),
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
        assert self.embedding_model is not None
        translation_prompt_template = RosePrompts.get_prompt("rose_analysis_to_query")
        assert translation_prompt_template is not None

        cache_files = []

        for i, analysis in enumerate(self.analyses):
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

            # Note-level fields shared across all findings in this analysis.
            # confused_links is note-level, not per-finding — pick the entry
            # whose concept_a/concept_b appears in this finding's text if
            # possible, else fall back to the first (usually only) entry.
            confused_links = analysis.get("confused_links", [])
            regression_prompt = "; ".join(analysis.get("regressions", []))
            remediation_order = analysis.get("remediation_order", [])
            progress_delta = analysis.get("progress_delta", "")

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

                # target_cognitive_level is decided HERE — once, per finding —
                # rather than being recomputed later in generation_pass. This
                # is the only place that needs to know both self.target_cognitive_level
                # (explicit request, e.g. from generate_engram_for_level) and
                # finding_type (for the untargeted fallback), since it has to
                # be baked into the cached prompt text before the LLM ever
                # sees it. generation_pass just reads meta["target_cognitive_level"].
                finding_type = finding.get("type", "")
                if self.target_cognitive_level is not None:
                    target_cognitive_level = self.target_cognitive_level
                else:
                    if finding_type not in self._TYPE_TO_COGNITIVE_LEVEL:
                        logger.warning(
                            "Finding %d-%d has unrecognised/missing type %r — "
                            "defaulting target_cognitive_level to 1 (recall).",
                            i,
                            j,
                            finding_type,
                        )
                    target_cognitive_level = self.TYPE_TO_COGNITIVE_LEVEL.get(
                        finding_type, 1
                    )

                cache_file = (
                    self._prompt_cache_dir(schema_id)
                    / f"{self.note_id.lower()}_{i}_{j}.json"
                )

                # A path match alone isn't a real cache hit: the file may
                # predate target_cognitive_level being tracked at all (no
                # "target_cognitive_level" key in meta), or may have been
                # written for a *different* target_cognitive_level than this
                # call is asking for (cache is keyed by note/analysis/finding
                # per schema, not per level). Either case must regenerate,
                # or generation_pass silently reuses the wrong level's prompt.
                if cache_file.exists():
                    try:
                        cached_meta = json.loads(
                            cache_file.read_text(encoding="utf-8")
                        )["meta"]
                    except (json.JSONDecodeError, KeyError):
                        cached_meta = {}
                    if (
                        cached_meta.get("target_cognitive_level")
                        == target_cognitive_level
                    ):
                        logger.info("Cache hit, skipping → %s", cache_file)
                        cache_files.append(cache_file)
                        continue
                    logger.info(
                        "Cache stale for %s (cached level=%r, requested=%r) — "
                        "regenerating",
                        cache_file,
                        cached_meta.get("target_cognitive_level"),
                        target_cognitive_level,
                    )

                # Pick the most relevant confused_link for this finding: prefer
                # one whose concept_a/concept_b text overlaps the finding's
                # student_claim or gap_explanation; else just use the first
                # available entry; else empty strings if there are none at all.
                link = self._pick_confused_link(confused_links, finding)
                concept_a = link.get("concept_a", "") if link else ""
                concept_b = link.get("concept_b", "") if link else ""
                confusion_description = (
                    link.get("confusion_description", "") if link else ""
                )

                filled_prompt = (
                    engram_prompt.replace("{topic}", str(analysis["topic"]))
                    .replace("{retrieved_docs}", context_text)
                    .replace(
                        "{mastery_signal}",
                        str(analysis.get("mastery_signal", "unknown")),
                    )
                    .replace("{progress_delta}", str(progress_delta))
                    .replace("{strong_areas}", str(analysis.get("strong_areas", [])))
                    .replace("{weak_areas}", str(analysis.get("weak_areas", [])))
                    .replace(
                        "{knowledge_gaps_summary}",
                        str(analysis.get("knowledge_gaps", [])),
                    )
                    .replace(
                        "{priority_study_areas}",
                        str(analysis.get("priority_areas", [])),
                    )
                    .replace("{remediation_order}", str(remediation_order))
                    .replace("{regression_prompt}", regression_prompt)
                    .replace("{concept_a}", str(concept_a))
                    .replace("{concept_b}", str(concept_b))
                    .replace("{confusion_description}", str(confusion_description))
                    .replace("{chunk_excerpt}", str(finding.get("chunk_excerpt", "")))
                    .replace("{finding_index}", str(finding.get("finding_index", j)))
                    .replace("{finding_type}", str(finding.get("type", "")))
                    .replace("{finding_severity}", severity)
                    .replace(
                        "{finding_confidence}", str(finding.get("confidence", 0.5))
                    )
                    .replace(
                        "{context_coverage}",
                        str(finding.get("context_coverage", False)),
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
                    .replace("{severity_short_question_count}", severity_count)
                    .replace("{target_cognitive_level}", str(target_cognitive_level))
                )

                cache_file = self._write_prompt_cache(
                    filled_prompt=filled_prompt,
                    analysis=analysis,
                    finding=finding,
                    schema_id=schema_id,
                    i=i,
                    j=j,
                    target_cognitive_level=target_cognitive_level,
                )
                cache_files.append(cache_file)
                logger.info(
                    "Prompt cached (%d chars) → %s", len(filled_prompt), cache_file
                )

        return cache_files

    @staticmethod
    def _pick_confused_link(
        confused_links: list[Dict], finding: Dict
    ) -> Optional[Dict]:
        """Best-effort match of a note-level confused_link entry to a specific
        finding. confused_links has no finding_index of its own, so this is a
        heuristic: prefer a link whose concept_a/concept_b text appears in the
        finding's student_claim or gap_explanation; else fall back to the
        first available link; else None."""
        if not confused_links:
            return None
        haystack = (
            str(finding.get("student_claim", ""))
            + " "
            + str(finding.get("gap_explanation", ""))
        ).lower()
        for link in confused_links:
            a = str(link.get("concept_a", "")).lower()
            b = str(link.get("concept_b", "")).lower()
            if (a and a in haystack) or (b and b in haystack):
                return link
        return confused_links[0]

    # ── STEP 2: generate from cache ────────────────────────────────────
    # Severity ("high"/"medium"/"low") measures how damaging a gap is —
    # it drives item COUNT (severity_card_count etc. in retrieval_pass)
    # and nothing else. It is not a difficulty proxy: a high-severity
    # misconception can need a simple corrective fact (low Bloom's level),
    # and a low-severity weak_point can only surface at a higher level.
    # target_cognitive_level is set from, in priority order:
    #   1. self.target_cognitive_level, when generate_engram_for_level
    #      explicitly requested a level — always wins.
    #   2. TYPE_TO_COGNITIVE_LEVEL, as a fallback for un-targeted base
    #      generation (_mcq_generator etc.), keyed on WHAT KIND of gap
    #      this is rather than how severe it is.
    # NOTE: this decision is made once, in retrieval_pass, and cached into
    # meta["target_cognitive_level"] — generation_pass below just reads it
    # back rather than recomputing it, so there's a single source of truth.
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
        One dict per generated item (short_question responses may contribute several
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

            # target_cognitive_level was already decided once in retrieval_pass
            # (the only place that needs to reconcile self.target_cognitive_level
            # vs. the finding-type fallback, since it has to be baked into the
            # prompt text). Read it back rather than recomputing it here, so
            # there's a single source of truth instead of two logic paths
            # that could disagree.
            target_cognitive_level = meta["target_cognitive_level"]

            tags = [meta["finding_type"], meta["topic"]]

            # short_question is the one type where multiple items in this response
            # belong to ONE engram (many short_question rows) rather than
            # one engram each — reset per cache_file, reused across the
            # items loop below.
            short_question_engram_id = None

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
                            meta["note_id"],
                            meta["bubble_id"],
                            item,
                            target_cognitive_level,
                            tags,
                        )
                    elif "flashcard" in schema_id:
                        item_engram_id = repo.add_flashcard(
                            meta["note_id"],
                            meta["bubble_id"],
                            item,
                            target_cognitive_level,
                            tags,
                        )
                    elif "short_question" in schema_id:
                        # reuse short_question_engram_id across items so every
                        # question in this response lands on the same
                        # short_question engram; None on the first iteration lets
                        # add_short_question generate a fresh id, which we then
                        # capture and pass on subsequent iterations.
                        short_question_engram_id = repo.add_short_question(
                            meta["note_id"],
                            meta["bubble_id"],
                            [item],
                            target_cognitive_level,
                            tags,
                            engram_id=short_question_engram_id,
                        )
                        item_engram_id = short_question_engram_id
                    elif "long_question" in schema_id:
                        item_engram_id = repo.add_long_question(
                            meta["note_id"],
                            meta["bubble_id"],
                            item,
                            target_cognitive_level,
                            tags,
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
        target_cognitive_level: int,
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
                "target_cognitive_level": target_cognitive_level,
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
        if "short_question" in schema_id:
            return "short_question"
        if "long_question" in schema_id:
            return "long_question"
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


# NOTE: generate_engram_for_level lives in learning_center_inator.py, not
# here — it builds a level_suffix-augmented prompt and calls
# EngramGenerator.retrieval_pass()/.generation_pass() directly (setting
# self.target_cognitive_level before retrieval_pass, same as required
# below). Nothing in this file needs its own copy of that dispatcher.


# ── entry point ────────────────────────────────────────────────────────
def _short_question_generator(bubble_id, note_id):
    engram = EngramGenerator(
        bubble_id=bubble_id,
        note_id=note_id,
    )

    schema_id = SHORT_QUESTION_SCHEMA["schema_id"]
    flashcard_prompt = RosePrompts.get_prompt("rose_short_question_generator")
    assert flashcard_prompt is not None

    # Pass 1 — embedding model only
    engram.retrieval_pass(engram_prompt=flashcard_prompt, schema_id=schema_id)

    # Pass 2 — chat model only, reads from cache
    engram.generation_pass(
        schema_id=schema_id,
        engram_schema=SHORT_QUESTION_SCHEMA,
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


def _long_question_generator(bubble_id, note_id):
    engram = EngramGenerator(
        bubble_id=bubble_id,
        note_id=note_id,
    )

    schema_id = LONG_QUESTION_SCHEMA["schema_id"]
    flashcard_prompt = RosePrompts.get_prompt("rose_long_question_generator")
    assert flashcard_prompt is not None

    # Pass 1 — embedding model only
    engram.retrieval_pass(engram_prompt=flashcard_prompt, schema_id=schema_id)

    # Pass 2 — chat model only, reads from cache
    engram.generation_pass(
        schema_id=schema_id,
        engram_schema=LONG_QUESTION_SCHEMA,
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
    _short_question_generator(bubble_id, note_id)
    _long_question_generator(bubble_id, note_id)


if __name__ == "__main__":
    _main()
