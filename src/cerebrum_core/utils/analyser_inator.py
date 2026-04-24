import json
import logging
from pathlib import Path
from typing import Any, Optional

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
            CerebrumPaths().cache_root_dir() / "bubble_cache" / "notes"
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

    def analyser_inator(self, prompt: str, top_k_chunks: int = 5) -> str:
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
            return "No valid knowledge base paths found for this note"

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

        # Log retrieval results
        logging.info(f"Retrieved {len(self.retrieved_docs)} total documents")

        # Check if we have any retrieved documents
        if not self.retrieved_docs:
            logging.warning("No documents retrieved from knowledge base")
            return "No relevant context found in knowledge base"

        # Build context from retrieved documents
        context_text = self._build_context(top_k_chunks)
        logging.info(f"Built context with {len(context_text)} characters")

        # Prepare note content
        # read note content from caches instead
        # chunk by chunk analysis can happend at this point
        flattened_note = NoteToMarkdownInator().flatten(self.note.content)
        NoteChunkerInator().chunk_markdown(
            markdown_text=flattened_note, note_id=self.note.note_id
        )
        logging.info("Notes chunked successfully")

        # Generate analysis
        final_prompt = prompt.format(
            archived_data=archived_data,
            current_note=flattened_note,
            context=context_text,
        )

        logging.info("Invoking LLM for final analysis")
        response = OllamaLLM(model=self.chat_model).invoke(final_prompt)
        logging.info(f"Analysis complete, response length: {len(response)} characters")

        return response

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

            # Normalize metadata keys for archiving
            self.chunks = [
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "note_id": self.note.note_id,
                        "chunk_id": chunk.metadata.get("chunk_index"),
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
                for chunk in raw_chunks
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
        """
        Build context string from retrieved documents.

        Args:
            top_k: Number of top documents to include

        Returns:
            Formatted context string with summaries
        """
        # deduplicate results
        seen = set()
        dedup_docs = []

        for doc in self.retrieved_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                dedup_docs.append(doc)

        context_docs = dedup_docs[:top_k]

        # Summarize chunks
        context_summaries = []
        for doc in context_docs:
            summary_prompt = f"""
            Summarize the following text in 1-2 sentences, keeping only the key factual information:
            {doc.page_content}
            """
            summary = OllamaLLM(model=self.chat_model).invoke(summary_prompt)
            context_summaries.append(summary.strip())

        return "\n\n".join(context_summaries)

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
                    user_note=chunk.page_content,  # ✅ Fixed: changed from user_query
                    available_stores=available_stores,
                )

                raw_output = OllamaLLM(model=self.chat_model).invoke(filled_prompt)

                # LOG THE FULL RAW OUTPUT
                logging.info("=" * 80)
                logging.info(f"CHUNK {chunk.metadata['chunk_id']} RAW LLM OUTPUT:")
                logging.info(f"{raw_output}")
                logging.info("=" * 80)

                parsed_query = self._parse_llm_json_output(raw_output)

                # Log what we got from parsing
                logging.info(f"Parsed query keys: {parsed_query.keys()}")

                # The parsed_query should now have: rewritten, subqueries, domain, subject
                # Add chunk metadata
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

        # DEBUG
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

        # Use the correct one
        valid_paths = zip_paths

        seen_routes = set()

        # After building valid_paths, deduplicate routes by (domain, subject) only
        # One route per unique collection — let MMR handle diversity within it
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
            # Try direct parse first
            return json.loads(output)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re

            # Look for ```json ... ``` blocks
            json_block_match = re.search(
                r"```json\s*(\{.*?\})\s*```", output, re.DOTALL
            )
            if json_block_match:
                try:
                    return json.loads(json_block_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Look for any JSON object
            match = re.search(r"\{.*\}", output, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            # Log the actual output to debug
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
