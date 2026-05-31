import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

from agents.rose import RosePrompts
from cerebrum_core.constants import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from cerebrum_core.model_inator import ArchivedNote, NoteStorage, TranslatedQuery
from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.archive_inator import AnalysisArchiveInator
from cerebrum_core.utils.cache_inator import RetrievalCacheInator
from cerebrum_core.utils.file_util_inator import (
    CerebrumPaths,
    knowledgebase_index_inator,
)
from cerebrum_core.utils.note_util_inator import NoteChunkerInator, NoteToMarkdownInator

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


# Claude helped big time T_T (review it though)
class NoteAnalyserInator:
    """
    Ingests notes and converts them into queries for knowledgebase retrieval.
    Handles chunking, archiving, and semantic analysis of notes.
    """

    def __init__(self, note: NoteStorage, generate_artifact: bool = True) -> None:
        """
        Initialize the note analyzer.

        Args:
            note: Pre-loaded NoteStorage object
            notes_path: Path to notes directory
            generate_artifact: Whether to generate markdown artifacts
        """
        self.note = note
        self.notes_path = CerebrumPaths().note_path(
            note.bubble_id,
            note.note_id,
        )
        self.generate_artifact = generate_artifact

        # Initialize state
        self.markdown_artifact: str = ""
        self.chunks: list[Document] = []
        self.translation_results: list[TranslatedQuery] = []
        self.constructed_query: dict = {"routes": []}
        self.retrieved_docs: list[Document] = []

        # Paths
        self.kb_archives = CerebrumPaths().kb_archives_path()
        self.bubble_cache_path = (
            CerebrumPaths().cache_root_dir() / "bubble_cache" / note.bubble_id
        )
        self.archive_path = CerebrumPaths().note_archive_path(
            bubble_id=self.note.bubble_id
        )

        # LLM configs
        config = ConfigManager().load_config()
        self.embedding_model = config.models.embedding_model or DEFAULT_EMBED_MODEL
        self.chat_model = config.models.chat_model or DEFAULT_CHAT_MODEL

        # Initialize on creation
        self._initialize()

    def _initialize(self) -> None:
        pass

    def analyser_inator(self, prompt: str, top_k_chunks: int = 5) -> dict:
        filename = self.note.note_id
        archived_data = self._load_archived_data() or {}

        if filename not in archived_data.keys():
            logging.info(f"Note {self.note.note_id} not in archive, will archive")
            self._archive_note()
        else:
            logging.info(f"Note {self.note.note_id} found in archive")

        # 1. Translate chunks → queries
        self.translation_results = self._note_to_query()
        logging.info(f"Translated {len(self.translation_results)} queries")

        # 2. Build routes from translated queries
        self._constructor_inator()
        logging.info(f"Constructed {len(self.constructed_query['routes'])} routes")

        if not self.constructed_query["routes"]:
            logging.warning("No valid routes constructed - cannot retrieve documents")
            return {"error": "No valid knowledge base paths found for this note"}

        # Now retrieve documents
        cache_manager = RetrievalCacheInator(
            note_id=self.note.note_id,
            bubble_id=self.note.bubble_id,
        )
        cached_docs = cache_manager.deterministic_fetcher()

        if cached_docs is not None:
            logging.info(
                f"Using cache retrieval results for analysis of note: {self.note.note_id}"
            )
            self.retrieved_docs = cached_docs
        else:
            logging.info("No cache found, performing fresh retrieval")
            self._retrieve_inator(k=top_k_chunks)
            RetrievalCacheInator(
                note_id=self.note.note_id,
                bubble_id=self.note.bubble_id,
            ).cache_populator_inator(self.retrieved_docs)

        logging.info(f"Retrieved {len(self.retrieved_docs)} total documents")

        if not self.retrieved_docs:
            logging.warning("No documents retrieved from knowledge base")
            return {"error": "No relevant context found in knowledge base"}

        # Build context from retrieved documents
        context_text = self._build_context(top_k_chunks)
        logging.info(f"Built context with {len(context_text)} characters")

        # Flatten and chunk note
        flattened_note = NoteToMarkdownInator().flatten(self.note.content)
        NoteChunkerInator().chunk_markdown(
            markdown_text=flattened_note, note_id=self.note.note_id
        )
        logging.info("Notes chunked successfully")

        # Run schema-constrained analysis
        result = self._run_analysis(
            prompt=prompt,
            archived_data=archived_data,
            context_text=context_text,
        )
        logging.info(
            f"Analysis complete: {len(result.get('chunk_diagnostics', []))} chunks diagnosed"
        )
        return result

    def _ollama_structured(self, prompt: str) -> str:
        """
        Call Ollama's HTTP API directly with a JSON schema format constraint.
        LangChain's OllamaLLM only supports format='json', not schema dicts,
        so we bypass it here for structured output enforcement.
        """
        config = ConfigManager().load_config()
        base_url = getattr(config.models, "ollama_base_url", "http://127.0.0.1:11434")

        payload = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
            "format": ANALYSIS_SCHEMA,
        }

        try:
            resp = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama HTTP request failed: {e}")
            raise

    def _run_analysis(
        self, prompt: str, archived_data: dict, context_text: str
    ) -> dict:
        """
        Build the final prompt, invoke the LLM with schema-constrained JSON output,
        and attach authoritative metadata before returning.
        """
        # Pass chunks as labelled blocks so the model can reference chunk_ids correctly
        chunk_map = "\n\n".join(
            f"[{chunk.metadata.get('chunk_id', f'chunk_{i:02d}')}]\n{chunk.page_content}"
            for i, chunk in enumerate(self.chunks)
        )

        def _serialise_archived(data: dict) -> str:
            """Serialize archived_data regardless of whether values are ArchivedNote, dataclass, or plain dict."""

            def _to_dict(obj):
                if hasattr(obj, "model_dump"):  # Pydantic v2
                    return obj.model_dump()
                if hasattr(obj, "dict"):  # Pydantic v1
                    return obj.dict()
                if hasattr(obj, "__dataclass_fields__"):  # dataclass
                    import dataclasses

                    return dataclasses.asdict(obj)
                return obj  # already a plain dict / primitive

            return json.dumps(
                {k: _to_dict(v) for k, v in data.items()},
                ensure_ascii=False,
                default=str,  # last-resort fallback for any remaining types
            )

        final_prompt = (
            prompt.replace("{archived_data}", _serialise_archived(archived_data))
            .replace("{current_note}", chunk_map)
            .replace("{context}", context_text)
        )

        logging.info("Invoking LLM with schema-constrained JSON output")
        raw = self._ollama_structured(final_prompt)
        logging.info(f"Raw analysis length: {len(raw)} chars")

        try:
            parsed: dict = json.loads(raw)
        except json.JSONDecodeError as e:
            logging.error(
                f"Schema-constrained output was still invalid JSON: {e}\n{raw[:500]}"
            )
            raise ValueError(
                "rose_note_analyser returned invalid JSON despite schema enforcement"
            ) from e

        # Inject metadata — never trust the model to fill these correctly
        parsed["metadata"] = {
            "note_id": self.note.note_id,
            "bubble_id": self.note.bubble_id,
            "content_version": self.note.metadata.content_version,
            "note_title": getattr(self.note.metadata, "title", ""),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "chunks_count": len(self.chunks),
            "queries_count": len(self.translation_results),
            "retrieved_docs": len(self.retrieved_docs),
        }

        return parsed

    def _load_archived_data(self) -> dict[str, ArchivedNote] | None:
        """Load archived note data for this bubble."""
        archive_manager = AnalysisArchiveInator(
            note=self.note,
            archives_path=str(self.archive_path),
            chunks=self.chunks,
        )
        if not archive_manager:
            return None
        return archive_manager.archive_browser_inator(self.note.bubble_id)

    def _archive_note(self) -> None:
        if not self.chunks:
            flattened = NoteToMarkdownInator().flatten(self.note.content)
            _, raw_chunks = NoteChunkerInator().chunk(
                flattened_note=flattened,
                note_id=self.note.note_id,
                bubble_id=self.note.bubble_id,
            )
            logging.info(f"Raw chunk metadata keys: {raw_chunks[0].metadata.keys()}")
            # Normalize metadata keys for archiving
            self.chunks = [
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "note_id": self.note.note_id,
                        "chunk_id": f"chunk_{i:02d}",
                        "source_block_ids": chunk.metadata.get("source_block_ids", []),
                        "fingerprint": chunk.metadata.get("chunk_fingerprint")
                        or chunk.metadata.get("fingerprint"),
                        "header": chunk.metadata.get("header", ""),
                        "generated_at": None,
                        "header_level": next(
                            (
                                v
                                for k, v in chunk.metadata.items()
                                if k.startswith("header_")
                            ),
                            None,
                        ),
                        "content_version": self.note.metadata.content_version,
                    },
                )
                for i, chunk in enumerate(raw_chunks)
            ]

        if not self.chunks:
            logging.warning(f"No chunks for note {self.note.note_id}, skipping archive")
            return

        AnalysisArchiveInator(
            note=self.note,
            archives_path=str(self.archive_path),
            chunks=self.chunks,
        ).archive_populator_inator()

        logging.info(f"Archived note {self.note.note_id}")

    def _build_context(self, top_k: int) -> str:
        seen = set()
        dedup_docs = []
        for doc in self.retrieved_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                dedup_docs.append(doc)

        context_docs = dedup_docs[:top_k]

        parts = []
        for i, doc in enumerate(context_docs):
            source = (
                doc.metadata.get("source")
                or doc.metadata.get("title")
                or doc.metadata.get("chunk_id")
                or f"ref_{i + 1}"
            )
            parts.append(f"[REF {i + 1} | {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def _note_to_query(self) -> list[TranslatedQuery]:
        """
        Translate note chunks into knowledge base queries.

        Returns:
            List of translated queries with routing information
        """
        translation_prompt_template = RosePrompts().get_prompt("rose_note_to_query")
        if not translation_prompt_template:
            raise ValueError("Prompt 'rose_note_to_query' not found in RosePrompts")

        available_stores, _ = knowledgebase_index_inator(Path(self.kb_archives))
        translated_queries: list[TranslatedQuery] = []

        for chunk in self.chunks:
            parsed_query = None
            raw_output = None
            try:
                filled_prompt = translation_prompt_template.format(
                    user_note=chunk.page_content,
                    available_stores=available_stores,
                )

                raw_output = OllamaLLM(model=self.chat_model).invoke(filled_prompt)

                logging.info("=" * 80)
                logging.info(f"CHUNK {chunk.metadata['chunk_id']} RAW LLM OUTPUT:")
                logging.info(f"{raw_output}")
                logging.info("=" * 80)

                parsed_query = self._parse_llm_json_output(raw_output)
                logging.info(f"Parsed query keys: {parsed_query.keys()}")

                parsed_query.update(
                    {
                        "chunk_id": chunk.metadata.get("chunk_id"),
                        "chunk_fingerprint": chunk.metadata.get("fingerprint"),
                        "header": chunk.metadata.get("header", ""),
                        "header_level": chunk.metadata.get("header_level"),
                    }
                )

                tq = TranslatedQuery(**parsed_query)
                translated_queries.append(tq)

            except KeyError as e:
                logging.warning(
                    f"Failed to translate chunk {chunk.metadata.get('chunk_id', 'unknown')}: "
                    f"Missing key {e}. Parsed data: {parsed_query}"
                )
                continue
            except Exception as e:
                logging.error(
                    f"Failed to translate chunk {chunk.metadata.get('chunk_id', 'unknown')}: {e}. "
                    f"Raw output: {raw_output[:500] if raw_output else 'None'}. "
                    f"Parsed data: {parsed_query}",
                    exc_info=True,
                )
                continue

        return translated_queries

    def _constructor_inator(self) -> dict[str, Any]:
        available_stores, _ = knowledgebase_index_inator(Path(self.kb_archives))

        logging.info(f"available_stores raw: {available_stores}")

        # Cartesian product (old/wrong)
        cartesian_paths = set()
        for domain in available_stores["domains"]:
            for subject in available_stores["subjects"]:
                cartesian_paths.add((domain, subject))
        logging.info(f"Cartesian paths (OLD): {cartesian_paths}")

        # Zip pairs (new/correct)
        zip_paths = set(
            zip(
                available_stores["domains"],
                available_stores["subjects"],
            )
        )
        logging.info(f"Zip paths (NEW): {zip_paths}")

        valid_paths = zip_paths
        seen_collections: set[tuple] = set()

        for query in self.translation_results:
            for route in query.subqueries:
                if not route.domain or not route.subject:
                    continue
                if (route.domain, route.subject) not in valid_paths:
                    continue

                collection_key = (route.domain, route.subject)
                if collection_key in seen_collections:
                    continue
                seen_collections.add(collection_key)

                path = self.kb_archives / route.domain / route.subject
                self.constructed_query["routes"].append(
                    {
                        "subquery": route,
                        "path": str(path),
                        "domain": route.domain,
                        "subject": route.subject,
                    }
                )
                logging.info(
                    f"Constructed {len(self.constructed_query['routes'])} unique routes"
                )
        return self.constructed_query

    def _retrieve_inator(self, k: int = 3) -> list[Document]:
        seen_content: set[str] = set()

        for route in self.constructed_query["routes"]:
            try:
                store = Chroma(
                    collection_name=route["subject"],
                    persist_directory=route["path"],
                    embedding_function=OllamaEmbeddings(model=self.embedding_model),
                )
                retriever = store.as_retriever(
                    search_type="mmr", search_kwargs={"k": k, "fetch_k": 15}
                )
                results = retriever.invoke(route["subquery"].text)

                new_docs = [d for d in results if d.page_content not in seen_content]
                for doc in new_docs:
                    seen_content.add(doc.page_content)
                    self.retrieved_docs.append(doc)

                logging.info(
                    f"Retrieved {len(new_docs)} new docs (of {len(results)}) "
                    f"for {route['domain']}/{route['subject']}"
                )
            except Exception as e:
                logging.error(f"Failed to retrieve from {route['path']}: {e}")
                continue

        return self.retrieved_docs

    def _parse_llm_json_output(self, output: str) -> dict:
        """
        Safely parse JSON from LLM output.

        Args:
            output: Raw LLM output string

        Returns:
            Parsed dictionary

        Raises:
            ValueError: If JSON cannot be parsed
        """
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            import re

            json_block_match = re.search(
                r"```json\s*(\{.*?\})\s*```", output, re.DOTALL
            )
            if json_block_match:
                try:
                    return json.loads(json_block_match.group(1))
                except json.JSONDecodeError:
                    pass

            match = re.search(r"\{.*\}", output, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            logging.error(f"Could not parse JSON. Raw output: {output[:500]}")
            raise ValueError(f"Could not parse JSON from: {output[:200]}...")

    def refresh_note(self, updated_note: NoteStorage) -> None:
        """
        Refresh analyzer with updated note content.

        Args:
            updated_note: New NoteStorage object
        """
        self.note = updated_note
        self.markdown_artifact = ""
        self.chunks = []
        self.translation_results = []
        self.constructed_query = {"routes": []}
        self.retrieved_docs = []
        self._initialize()
        logging.info(f"Refreshed analyzer with note {updated_note.note_id}")

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Document]:
        """Get specific chunk by ID."""
        for chunk in self.chunks:
            if chunk.metadata.get("chunk_id") == chunk_id:
                return chunk
        return None

    def get_chunks_by_header(self, header: str) -> list[Document]:
        """Get all chunks matching a header."""
        return [
            chunk for chunk in self.chunks if chunk.metadata.get("header") == header
        ]

    def export_artifact(self, output_path: Path) -> None:
        """
        Export markdown artifact to file.

        Args:
            output_path: Where to save the artifact
        """
        if not self.markdown_artifact:
            raise ValueError("No artifact generated yet")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.markdown_artifact, encoding="utf-8")
        logging.info(f"Exported artifact to {output_path}")

    def __repr__(self) -> str:
        return (
            f"NoteAnalyserInator(note_id={self.note.note_id}, "
            f"chunks={len(self.chunks)}, "
            f"queries={len(self.translation_results)})"
        )
