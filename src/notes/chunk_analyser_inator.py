import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal, Optional

import jsonschema
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from agents.rose import RosePrompts
from cerebrum_core.constants import (
    CHUNK_ANALYSIS_SCHEMA,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
)
from database.note_chunk_registry_inator import NoteChunkRegisterInator
from models.model_inator import ArchivedNote  # [ADDED] ArchivedNote
from models.model_inator import TranslatedQuery
from cerebrum_core.user_inator import ConfigManager
from common.archive_inator import AnalysisArchiveInator  # [ADDED]
from common.cache_inator import AnalysisCacheInator, RetrievalCacheInator
from vectorstore.embeddings_inator import get_embeddings
from vectorstore.faiss_store_inator import get_or_create_store
from common.file_util_inator import (
    CerebrumPaths,
    knowledgebase_index_inator,
)
from common.ollama_compat.invoker_inator import ollama_local_call


# ---------------------------------------------------------------------------
# Result envelope yielded by the generator for each chunk
# ---------------------------------------------------------------------------
@dataclass
class ChunkResult:
    """
    Yielded by ``chunk_stream_inator`` after each chunk completes.

    status values:
      "cached"   — fingerprint unchanged, result loaded from bubble cache.
      "analysed" — chunk was freshly analysed and cached.
      "error"    — analysis failed; ``error`` contains the exception message.
                   Cache is NOT written, so the next run will retry this chunk.
    """

    chunk_index: int
    note_id: str
    status: Literal["cached", "analysed", "error"]
    analysis: dict = field(default_factory=dict)  # validated against ANALYSIS_SCHEMA
    chunk_metadata: dict = field(default_factory=dict)
    error: Optional[str] = None


class ChunkAnalyserInator:
    """
    Analyses a note chunk-by-chunk using byte coordinates from the registry.

    Flow per chunk:
      1. Fetch chunk content from the chunked markdown artifact via byte_start/byte_end.
      2. Compare the on-disk chunk_fingerprint against the registry record.
      3. If unchanged  → return the cached analysis from the bubble cache.
      4. If changed    → translate chunk → retrieve KB docs → run LLM analysis
                         → inject authoritative chunk_id / chunk_excerpt
                         → serialise corrected analysis → store in bubble cache.

    Historical context:
      On first use the note is archived via AnalysisArchiveInator so that
      subsequent runs can pass prior analysis results to the LLM through the
      {archived_data} prompt placeholder, giving the model a longitudinal view
      of the student's progress.
    """

    def __init__(
        self, bubble_id: str, note_id: str, note_chunks, note=None
    ) -> None:  # [ADDED] note param
        self.note_id = note_id
        self.note_chunks = note_chunks
        self.bubble_id = bubble_id
        self.note = note  # [ADDED] NoteStorage reference needed for archiving

        self.note_chunk_registry = NoteChunkRegisterInator()

        # LLM / embedding config
        config = ConfigManager().load_config()
        self.embedding_model = config.models.embedding_model or DEFAULT_EMBED_MODEL
        self.chat_model = config.models.chat_model or DEFAULT_CHAT_MODEL

        # Paths
        paths = CerebrumPaths()
        self.kb_archives = paths.kb_archives_path()
        self.bubble_cache_path = (
            paths.cache_root_dir() / "bubble_cache" / self.bubble_id
        )
        self.archive_path = paths.note_archive_path(bubble_id=self.bubble_id)  # [ADDED]
        self.analysis_path = paths.note_analysis_dir(
            bubble_id=self.bubble_id, note_id=self.note_id
        )

        self.retrieval_cache = RetrievalCacheInator(
            note_id=self.note_id,
            bubble_id=self.bubble_id,
        )
        # Registry fingerprint lookup: chunk_index → chunk_fingerprint
        self._registry_fingerprints: dict[int, str] = {
            row.chunk_index: row.chunk_fingerprint
            for row in self.note_chunk_registry.fetch_chunks_inator(note_id)
        }

        logging.info(
            f"[INIT] ChunkAnalyserInator — note_id={note_id} bubble_id={bubble_id} "
            f"registry_chunks={len(self._registry_fingerprints)}"
        )

        # Runtime state
        self.chunk_analyses: dict[int, str] = {}  # chunk_index → analysis JSON
        self.retrieved_docs: list[Document] = []
        self.constructed_query: dict = {"routes": []}
        self.translation_results: list[TranslatedQuery] = []

        # [ADDED] Historical context — loaded once per session, shared across all chunks
        self._archived_data: dict | None = None
        self._archived_data_loaded: bool = False

        # Cancellation token
        self.stop_event: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_stream_inator(
        self,
        prompt: str,
        top_k_chunks: int = 3,
    ) -> Generator[ChunkResult, None, None]:
        self.stop_event.clear()

        raw_response = ""
        cached_md_path = self._resolve_markdown_artifact_path()
        logging.info(f"[STREAM] Resolved markdown artifact path: {cached_md_path}")

        # [ADDED] Load/create archive once before the chunk loop so every chunk
        # in this session shares the same historical snapshot.
        archived_data = self._ensure_archived_data()
        serialised_archive = self._serialise_archived(archived_data)
        logging.info(
            f"[ARCHIVE] Historical context loaded — {len(archived_data)} entries, "
            f"{len(serialised_archive)} chars"
        )

        for chunk_row in self.note_chunks:
            if self.stop_event.is_set():
                logging.info(
                    f"[STREAM] Stop requested — halting before chunk "
                    f"{self._attr(chunk_row, 'chunk_index')}"
                )
                return

            chunk_index = self._attr(chunk_row, "chunk_index")
            current_fingerprint = self._attr(chunk_row, "chunk_fingerprint")
            byte_start = self._attr(chunk_row, "byte_start")
            byte_end = self._attr(chunk_row, "byte_end")

            logging.info(
                f"[STREAM] ── Chunk {chunk_index} ──────────────────────────────────"
            )
            logging.info(
                f"[STREAM] bytes={byte_start}:{byte_end} fingerprint={current_fingerprint[:12]}…"
            )

            # 1. Fingerprint check → serve from cache if unchanged
            registry_fingerprint = self._registry_fingerprints.get(chunk_index)
            if registry_fingerprint == current_fingerprint:
                cached_raw = self._load_chunk_cache(chunk_index)
                if cached_raw is not None:
                    logging.info(
                        f"[STREAM] Chunk {chunk_index}: fingerprint unchanged — serving from cache"
                    )
                    cached_analysis = self._load_chunk_cache(chunk_index)
                    if cached_analysis is not None:
                        yield ChunkResult(
                            chunk_index=chunk_index,
                            note_id=self.note_id,
                            status="cached",
                            analysis=cached_analysis,
                            chunk_metadata={
                                "note_id": self.note_id,
                                "chunk_index": chunk_index,
                                "byte_start": byte_start,
                                "byte_end": byte_end,
                            },
                        )
                    continue
                logging.info(
                    f"[STREAM] Chunk {chunk_index}: fingerprint unchanged but no cache — re-analysing"
                )

            # 2. Fetch raw content from markdown artifact
            try:
                chunk_content, chunk_metadata = self.chunk_fetcher_inator(
                    cached_note_path=str(cached_md_path),
                    chunk_index=chunk_index,
                    chunk_start=byte_start,
                    chunk_end=byte_end,
                )
                if "-->" in chunk_content:
                    chunk_content = chunk_content[
                        chunk_content.index("-->") + 3 :
                    ].strip()
                logging.info(
                    f"[FETCH] Chunk {chunk_index}: {len(chunk_content)} chars fetched"
                )
                logging.info(
                    f"[FETCH] Chunk {chunk_index} content preview:\n{chunk_content[:300]}"
                )
            except Exception as exc:
                logging.error(f"[FETCH] Chunk {chunk_index}: fetch failed — {exc}")
                yield ChunkResult(
                    chunk_index=chunk_index,
                    note_id=self.note_id,
                    status="error",
                    error=str(exc),
                )
                self._reset_retrieval_state()
                continue

            # 3. Translate → build routes → retrieve KB docs
            try:
                logging.info(
                    f"[TRANSLATE] Chunk {chunk_index}: translating to KB query…"
                )
                translated = self._translate_chunk(chunk_content, chunk_metadata)
                if translated:
                    self.translation_results.append(translated)
                    logging.info(
                        f"[TRANSLATE] Chunk {chunk_index}: rewritten={str(getattr(translated, 'rewritten', ''))[:120]}"
                    )
                    logging.info(
                        f"[TRANSLATE] Chunk {chunk_index}: subqueries={[str(sq) for sq in translated.subqueries]}"
                    )
                else:
                    logging.warning(
                        f"[TRANSLATE] Chunk {chunk_index}: translation returned None"
                    )

                logging.info(f"[ROUTES] Chunk {chunk_index}: building routes…")
                self._constructor_inator()
                route_labels = [
                    f"{r['domain']}/{r['subject']}"
                    for r in self.constructed_query["routes"]
                ]
                logging.info(
                    f"[ROUTES] Chunk {chunk_index}: {len(route_labels)} routes — {route_labels}"
                )

                # ── Retrieval: cache-first, then live ──────────────────────────
                cache_manager = RetrievalCacheInator(
                    note_id=self.note_id,
                    bubble_id=self.bubble_id,
                )
                cached_docs = cache_manager.deterministic_fetcher()

                if cached_docs is not None:
                    logging.info(
                        f"[ARCHIVE] Chunk {chunk_index}: using {len(cached_docs)} cached docs"
                    )
                    self.retrieved_docs = cached_docs
                # AFTER
                if self.constructed_query["routes"]:
                    cached_docs = self.retrieval_cache.deterministic_fetcher()
                    if cached_docs:
                        logging.info(
                            f"[RETRIEVE] Chunk {chunk_index}: {len(cached_docs)} docs from retrieval cache"
                        )
                        self.retrieved_docs = cached_docs
                    else:
                        logging.info(
                            f"[RETRIEVE] Chunk {chunk_index}: retrieval cache miss — hitting KB live (top_k={top_k_chunks})…"
                        )
                        self._retrieve_inator(k=top_k_chunks)
                        logging.info(
                            f"[RETRIEVE] Chunk {chunk_index}: {len(self.retrieved_docs)} docs retrieved — caching"
                        )
                        self.retrieval_cache.cache_populator_inator(self.retrieved_docs)
                else:
                    logging.warning(
                        f"[ROUTES] Chunk {chunk_index}: no valid KB routes — checking retrieval cache as fallback…"
                    )
                    cached_docs = self.retrieval_cache.deterministic_fetcher()
                    if cached_docs:
                        logging.info(
                            f"[RETRIEVE] Chunk {chunk_index}: {len(cached_docs)} docs from retrieval cache (no-route fallback)"
                        )
                        self.retrieved_docs = cached_docs
                    else:
                        logging.warning(
                            f"[RETRIEVE] Chunk {chunk_index}: no routes and no retrieval cache — context will be empty"
                        )
                    logging.warning(
                        f"[ROUTES] Chunk {chunk_index}: no valid KB routes — skipping retrieval"
                    )
                # ── End retrieval block ────────────────────────────────────────

                context_text = self._build_context(top_k_chunks)
                logging.info(
                    f"[CONTEXT] Chunk {chunk_index}: {len(context_text)} chars built"
                )
                logging.info(
                    f"[CONTEXT] Chunk {chunk_index} context preview:\n{context_text[:500]}"
                )

            except Exception as exc:
                logging.error(
                    f"[TRANSLATE/RETRIEVE] Chunk {chunk_index}: failed — {exc}",
                    exc_info=True,
                )
                yield ChunkResult(
                    chunk_index=chunk_index,
                    note_id=self.note_id,
                    status="error",
                    chunk_metadata=chunk_metadata,
                    error=str(exc),
                )
                self._reset_retrieval_state()
                continue

            # 4. LLM analysis — schema-constrained JSON output
            # [CHANGED] {archived_data} is now populated with real historical context
            filled_prompt = (
                prompt.replace("{current_note}", chunk_content)
                .replace("{context}", context_text)
                .replace("{archived_data}", serialised_archive)  # [CHANGED] was ""
            )
            logging.info(
                f"[PROMPT] Chunk {chunk_index}: filled prompt ({len(filled_prompt)} chars) preview:\n"
                f"{filled_prompt[:400]}"
            )

            try:
                debug_cache_dir = (
                    self.bubble_cache_path / "debug" / "prompts" / self.note_id
                )
                debug_cache_dir.mkdir(parents=True, exist_ok=True)
                debug_cache_file = (
                    debug_cache_dir / f"{self.note_id}_chunk_{chunk_index}.json"
                )
                debug_payload = {
                    "chunk_index": chunk_index,
                    "chunk_content": chunk_content,
                    "archived_data_entries": len(
                        archived_data
                    ),  # [ADDED] debug visibility
                    "retrieved_docs": [
                        {
                            "source": doc.metadata.get("source")
                            or doc.metadata.get("title")
                            or doc.metadata.get("chunk_id")
                            or f"ref_{i}",
                            "content": doc.page_content,
                        }
                        for i, doc in enumerate(self.retrieved_docs)
                    ],
                    "filled_prompt": filled_prompt,
                }
                debug_cache_file.write_text(
                    json.dumps(debug_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logging.info(f"[DEBUG] prompt cache written → {debug_cache_file}")

                logging.info(f"[LLM] Chunk {chunk_index}: invoking _ollama_local_call…")
                raw_response = ollama_local_call(
                    prompt=filled_prompt, analyses_schema=CHUNK_ANALYSIS_SCHEMA
                )
                logging.info(
                    f"[LLM] Chunk {chunk_index}: raw response ({len(raw_response)} chars) preview:\n"
                    f"{raw_response[:400]}"
                )
 
                analysis = json.loads(raw_response)

            except json.JSONDecodeError as exc:
                logging.error(
                    f"[LLM] Chunk {chunk_index}: non-JSON response — {exc}\n"
                    f"raw: {raw_response[:300]}"
                )
                yield ChunkResult(
                    chunk_index=chunk_index,
                    note_id=self.note_id,
                    status="error",
                    chunk_metadata=chunk_metadata,
                    error=f"Non-JSON response: {exc}",
                )
                self._reset_retrieval_state()
                continue
            except Exception as exc:
                logging.error(
                    f"[LLM] Chunk {chunk_index}: invocation failed — {exc}",
                    exc_info=True,
                )
                yield ChunkResult(
                    chunk_index=chunk_index,
                    note_id=self.note_id,
                    status="error",
                    chunk_metadata=chunk_metadata,
                    error=str(exc),
                )
                self._reset_retrieval_state()
                continue

            # 5. Inject authoritative metadata
            authoritative_chunk_id = f"chunk_{chunk_index}"
            authoritative_excerpt = chunk_content

            chunk_diagnostics = analysis.get("chunk_diagnostics", [])
            if isinstance(chunk_diagnostics, dict):
                logging.warning(
                    f"[INJECT] Chunk {chunk_index}: chunk_diagnostics was a bare dict — wrapping in list"
                )
                chunk_diagnostics = [chunk_diagnostics]
                analysis["chunk_diagnostics"] = chunk_diagnostics

            if not isinstance(chunk_diagnostics, list) or not all(
                isinstance(diag, dict) for diag in chunk_diagnostics
            ):
                logging.error(
                    f"[INJECT] Chunk {chunk_index}: chunk_diagnostics is malformed — "
                    f"expected list[dict], got {type(chunk_diagnostics)}: {str(chunk_diagnostics)[:200]}"
                )
                yield ChunkResult(
                    chunk_index=chunk_index,
                    note_id=self.note_id,
                    status="error",
                    chunk_metadata=chunk_metadata,
                    error="chunk_diagnostics malformed: expected list[dict]",
                )
                self._reset_retrieval_state()
                continue

            for diag in chunk_diagnostics:
                diag["chunk_id"] = authoritative_chunk_id
                diag["chunk_excerpt"] = authoritative_excerpt

            corrected_json = json.dumps(analysis, indent=2, ensure_ascii=False)

            # 6. Only write cache after a fully successful analysis + injection
            self._store_chunk_cache(chunk_index, corrected_json)
            self._registry_fingerprints[chunk_index] = current_fingerprint
            self.chunk_analyses[chunk_index] = corrected_json

            logging.info(
                f"[STREAM] Chunk {chunk_index}: analysis complete — "
                f"diagnostics={len(analysis.get('chunk_diagnostics', []))} "
                f"mastery={analysis.get('note_overview', {}).get('mastery_signal', '?')}"
            )

            self._reset_retrieval_state()

            yield ChunkResult(
                chunk_index=chunk_index,
                note_id=self.note_id,
                status="analysed",
                analysis=analysis,
                chunk_metadata=chunk_metadata,
            )

    def chunk_fetcher_inator(
        self,
        cached_note_path: str,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
    ) -> tuple[str, dict]:
        path = Path(cached_note_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Cached markdown artifact not found: {cached_note_path}"
            )

        with path.open("rb") as fh:
            fh.seek(chunk_start)
            raw_bytes = fh.read(chunk_end - chunk_start)

        content = raw_bytes.decode("utf-8", errors="replace")

        metadata = {
            "note_id": self.note_id,
            "chunk_index": chunk_index,
            "byte_start": chunk_start,
            "byte_end": chunk_end,
        }

        logging.info(
            f"[FETCH] chunk_index={chunk_index} bytes={chunk_start}:{chunk_end} "
            f"chars={len(content)}"
        )
        return content, metadata

    # ------------------------------------------------------------------
    # [ADDED] Historical context / archiving
    # ------------------------------------------------------------------

    def _ensure_archived_data(self) -> dict:
        """
        Load the bubble's archive on first call, archive the current note if
        it is missing, and cache the result for the lifetime of this session.

        Returns a dict of {note_id: ArchivedNote} (may be empty if no note
        reference was provided via ``self.note``).
        """
        if self._archived_data_loaded:
            return self._archived_data or {}

        self._archived_data_loaded = True

        if self.note is None:
            logging.warning(
                "[ARCHIVE] No NoteStorage reference supplied — historical context disabled. "
                "Pass note=<NoteStorage> to ChunkAnalyserInator.__init__ to enable it."
            )
            self._archived_data = {}
            return {}

        archive_manager = AnalysisArchiveInator(
            note=self.note,
            archives_path=str(self.archive_path),
            chunks=[],  # chunks not needed for browsing
        )
        existing = archive_manager.archive_browser_inator(self.bubble_id) or {}
        logging.info(
            f"[ARCHIVE] archive_browser_inator returned {len(existing)} entries "
            f"for bubble {self.bubble_id}"
        )

        # Archive the note if it is not yet present
        if self.note_id not in existing:
            logging.info(
                f"[ARCHIVE] note_id={self.note_id} not in archive — creating entry"
            )
            self._archive_note()
            # Re-read so the fresh entry is included in the context
            existing = archive_manager.archive_browser_inator(self.bubble_id) or {}
            logging.info(f"[ARCHIVE] re-read after archive — {len(existing)} entries")
        else:
            logging.info(f"[ARCHIVE] note_id={self.note_id} found in existing archive")

        self._archived_data = existing
        return existing

    def _archive_note(self) -> None:
        """
        Archive the note using the SAME chunks the analyser is already
        iterating (self.note_chunks, from the registry) — not a fresh,
        independent re-chunking. This keeps archive chunk_id aligned with
        the registry chunk_index used everywhere else (ChunkResult,
        _registry_fingerprints, bubble cache), so historical entries for
        "chunk 3" actually correspond to the same span of the note as
        chunk_index=3 does in chunk_stream_inator.
        """
        if self.note is None:
            logging.warning("[ARCHIVE] Cannot archive — no NoteStorage reference")
            return

        if not self.note_chunks:
            logging.warning(
                f"[ARCHIVE] No registry chunks for note {self.note_id} — skipping archive"
            )
            return

        cached_md_path = self._resolve_markdown_artifact_path()

        archive_chunks = []
        for chunk_row in self.note_chunks:
            chunk_index = self._attr(chunk_row, "chunk_index")
            chunk_fingerprint = self._attr(chunk_row, "chunk_fingerprint")
            byte_start = self._attr(chunk_row, "byte_start")
            byte_end = self._attr(chunk_row, "byte_end")

            try:
                content, _ = self.chunk_fetcher_inator(
                    cached_note_path=str(cached_md_path),
                    chunk_index=chunk_index,
                    chunk_start=byte_start,
                    chunk_end=byte_end,
                )
            except Exception as exc:
                logging.error(
                    f"[ARCHIVE] Failed to fetch chunk {chunk_index} for archiving: {exc}"
                )
                continue

            if "-->" in content:
                content = content[content.index("-->") + 3 :].strip()

            archive_chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "note_id": self.note_id,
                        # authoritative chunk_id, matching registry chunk_index —
                        # the same id used in ChunkResult.chunk_index and the
                        # bubble cache elsewhere in this class
                        "chunk_id": f"chunk_{chunk_index:02d}",
                        "fingerprint": chunk_fingerprint,
                        "header": "",
                        "header_level": None,
                        "content_version": self.note.metadata.content_version,
                    },
                )
            )

        if not archive_chunks:
            logging.warning(
                f"[ARCHIVE] No chunks fetched for note {self.note_id} — skipping archive"
            )
            return

        AnalysisArchiveInator(
            note=self.note,
            archives_path=str(self.archive_path),
            chunks=archive_chunks,
        ).archive_populator_inator()

        logging.info(
            f"[ARCHIVE] Archived note {self.note_id} with {len(archive_chunks)} chunks "
            f"(from registry, {len(self.note_chunks)} total)"
        )

    @staticmethod
    def _serialise_archived(data: dict) -> str:
        """
        Serialise archived_data to a JSON string regardless of whether values
        are ArchivedNote / Pydantic models / dataclasses / plain dicts.
        Returns an empty JSON object string when data is empty so the prompt
        placeholder is always replaced with valid JSON.
        """
        if not data:
            return "{}"

        def _to_dict(obj):
            if hasattr(obj, "model_dump"):  # Pydantic v2
                return obj.model_dump()
            if hasattr(obj, "dict"):  # Pydantic v1
                return obj.dict()
            if hasattr(obj, "__dataclass_fields__"):
                import dataclasses

                return dataclasses.asdict(obj)
            return obj  # already a plain dict / primitive

        return json.dumps(
            {k: _to_dict(v) for k, v in data.items()},
            ensure_ascii=False,
            default=str,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_markdown_artifact_path(self) -> Path:
        return (
            CerebrumPaths()
            .chunked_note_path(
                bubble_id=self.bubble_id,
                note_id=self.note_id,
            )
            .with_suffix(".md")
        )

    def _invoke_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        backoff_seconds: float = 5.0,
    ) -> str:
        import time

        from langchain_ollama.llms import OllamaLLM

        last_exc = None
        wait = backoff_seconds

        for attempt in range(1, max_retries + 1):
            try:
                return OllamaLLM(model=self.chat_model, temperature=0).invoke(prompt)
            except Exception as exc:
                last_exc = exc
                logging.warning(
                    f"[RETRY] Ollama request failed (attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {wait:.0f}s…"
                )
                time.sleep(wait)
                wait *= 2

        raise RuntimeError(f"Ollama failed after {max_retries} attempts") from last_exc

    def _translate_chunk(
        self, chunk_content: str, chunk_metadata: dict
    ) -> Optional[TranslatedQuery]:
        """Translate a single chunk's content into a KB query."""
        translation_prompt_template = RosePrompts().get_prompt("rose_note_to_query")
        if not translation_prompt_template:
            raise ValueError("Prompt 'rose_note_to_query' not found in RosePrompts")

        available_stores, _ = knowledgebase_index_inator(Path(self.kb_archives))
        logging.info(f"[TRANSLATE] available_stores: {available_stores}")

        raw_output = None
        try:
            filled_prompt = translation_prompt_template.format(
                user_note=chunk_content,
                available_stores=available_stores,
            )
            logging.info(
                f"[TRANSLATE] Chunk {chunk_metadata['chunk_index']} prompt preview:\n"
                f"{filled_prompt[:300]}"
            )

            raw_output = self._invoke_with_retry(prompt=filled_prompt)
            logging.info(
                f"[TRANSLATE] Chunk {chunk_metadata['chunk_index']} raw LLM output:\n{raw_output}"
            )

            parsed = self._parse_llm_json_output(raw_output)
            logging.info(
                f"[TRANSLATE] Chunk {chunk_metadata['chunk_index']} parsed keys: {list(parsed.keys())}"
            )
            logging.info(
                f"[TRANSLATE] Chunk {chunk_metadata['chunk_index']} rewritten: "
                f"{str(parsed.get('rewritten', ''))[:120]}"
            )
            logging.info(
                f"[TRANSLATE] Chunk {chunk_metadata['chunk_index']} subqueries: "
                f"{parsed.get('subqueries', [])}"
            )

            parsed.update(
                {
                    "chunk_id": chunk_metadata.get("chunk_index"),
                    "chunk_fingerprint": None,
                    "header": "",
                    "header_level": None,
                }
            )
            return TranslatedQuery(**parsed)

        except Exception as e:
            logging.error(
                f"[TRANSLATE] Chunk {chunk_metadata.get('chunk_index')}: failed — {e}. "
                f"Raw output: {raw_output[:500] if raw_output else 'None'}",
                exc_info=True,
            )
            return None

    def _constructor_inator(self) -> dict[str, Any]:
        """Build valid (domain, subject) routes from translated queries."""
        available_stores, _ = knowledgebase_index_inator(Path(self.kb_archives))

        valid_paths = set(
            zip(available_stores["domains"], available_stores["subjects"])
        )
        logging.info(f"[ROUTES] valid_paths: {valid_paths}")

        seen_collections: set[tuple] = set()

        for query in self.translation_results:
            for route in query.subqueries:
                logging.info(
                    f"[ROUTES] evaluating subquery — domain={route.domain} subject={route.subject}"
                )
                if not route.domain or not route.subject:
                    logging.warning("[ROUTES] skipping — missing domain or subject")
                    continue
                if (route.domain, route.subject) not in valid_paths:
                    logging.warning(
                        f"[ROUTES] skipping — ({route.domain}, {route.subject}) not in valid_paths"
                    )
                    continue
                collection_key = (route.domain, route.subject)
                if collection_key in seen_collections:
                    logging.info(
                        f"[ROUTES] skipping duplicate collection: {collection_key}"
                    )
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
                    f"[ROUTES] added route: {route.domain}/{route.subject} → {path}"
                )

        logging.info(
            f"[ROUTES] total constructed: {len(self.constructed_query['routes'])}"
        )
        return self.constructed_query

    def _retrieve_inator(self, k: int = 3) -> list[Document]:
        """Retrieve documents from FAISS stores for all constructed routes."""
        seen_content: set[str] = set()

        for route in self.constructed_query["routes"]:
            logging.info(
                f"[RETRIEVE] querying FAISS — collection={route['subject']} path={route['path']}"
            )
            logging.info(
                f"[RETRIEVE] subquery text: {getattr(route['subquery'], 'text', '')[:200]}"
            )
            try:
                store = get_or_create_store(
                    Path(route["path"]), get_embeddings(self.embedding_model)
                )
                retriever = store.as_retriever(
                    search_type="mmr", search_kwargs={"k": k, "fetch_k": 15}
                )
                results = retriever.invoke(route["subquery"].text)
                logging.info(
                    f"[RETRIEVE] {route['domain']}/{route['subject']}: "
                    f"{len(results)} results from FAISS"
                )

                new_docs = [d for d in results if d.page_content not in seen_content]
                for i, doc in enumerate(new_docs):
                    seen_content.add(doc.page_content)
                    self.retrieved_docs.append(doc)
                    source = (
                        doc.metadata.get("source")
                        or doc.metadata.get("title")
                        or doc.metadata.get("chunk_id")
                        or f"ref_{i}"
                    )
                    logging.info(
                        f"[RETRIEVE]   new doc[{i}] source={source} | {doc.page_content[:120]}"
                    )

                logging.info(
                    f"[RETRIEVE] {len(new_docs)} new docs added "
                    f"({len(results) - len(new_docs)} duplicates skipped)"
                )
            except Exception as e:
                logging.error(
                    f"[RETRIEVE] failed for {route['path']}: {e}", exc_info=True
                )
                continue

        logging.info(f"[RETRIEVE] total retrieved_docs: {len(self.retrieved_docs)}")
        return self.retrieved_docs

    def _build_context(self, top_k: int) -> str:
        """Deduplicate retrieved docs and pass raw content as context."""
        seen: set[str] = set()
        dedup: list[Document] = []
        for doc in self.retrieved_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                dedup.append(doc)

                logging.info(
                    f"[CONTEXT] {len(dedup)} unique docs after dedup (from {len(self.retrieved_docs)} total)"
                )

        parts = []
        for i, doc in enumerate(dedup[:top_k]):
            source = (
                doc.metadata.get("source")
                or doc.metadata.get("title")
                or doc.metadata.get("chunk_id")
                or f"ref_{i + 1}"
            )

            content = doc.page_content
            if "-->" in content:
                content = content[content.index("-->") + 3 :].strip()
            if not content:
                continue
            parts.append(f"[REF {i + 1} | {source}]\n{doc.page_content}")
            logging.info(
                f"[CONTEXT]   ref[{i + 1}] source={source} | {doc.page_content[:120]}"
            )

        context = "\n\n---\n\n".join(parts)
        logging.info(f"[CONTEXT] final context: {len(context)} chars")
        return context

    def _load_chunk_cache(self, chunk_index: int) -> Optional[dict]:
        if self.note is None:
            logging.warning("[CACHE] cannot load - no NoteStorage reference")
            return None
        cached_chunks = AnalysisCacheInator(
            bubble_id=self.bubble_id,
            note_id=self.note_id,
        ).get_cached_analysis(content_version=self.note.metadata.content_version)

        if cached_chunks is None:
            logging.info(f"[CACHE] miss — no valid cache for note {self.note_id}")
            return None

        try:
            result = cached_chunks[chunk_index]
            logging.info(f"[CACHE] hit — chunk {chunk_index} for note {self.note_id}")
            return result
        except IndexError:
            logging.info(f"[CACHE] miss — chunk {chunk_index} not in cache")
            return None

    def _store_chunk_cache(self, chunk_index: int, analysis_json: str) -> None:
        # Validate against schema before writing
        if self.note is None:
            logging.warning("[CACHE] cannot load - no NoteStorage reference")
            return None
        try:
            parsed = json.loads(analysis_json)
            jsonschema.validate(instance=parsed, schema=CHUNK_ANALYSIS_SCHEMA)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            logging.error(
                f"[CACHE] chunk {chunk_index} failed schema validation — not caching: {e}"
            )
            raise  # bubble up so chunk_stream_inator yields an error result
        AnalysisCacheInator(
            bubble_id=self.bubble_id, note_id=self.note_id
        ).cache_analysis(
            content_version=self.note.metadata.content_version,
            analysis=json.loads(analysis_json),
            chunk_index=chunk_index,
        )
        logging.info(f"[CACHE] stored chunk {chunk_index} → cache_file")

    def _reset_retrieval_state(self) -> None:
        logging.info(
            "[RESET] clearing retrieved_docs, constructed_query, translation_results"
        )
        self.retrieved_docs = []
        self.constructed_query = {"routes": []}
        self.translation_results = []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _attr(obj: Any, key: str) -> Any:
        """Unified attribute access for dataclass, namedtuple, or dict."""
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    def _parse_llm_json_output(self, output: str) -> dict:
        """Safely parse JSON from LLM output, with markdown fence fallback."""
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

            logging.error(f"[PARSE] Could not parse JSON. Raw output: {output[:500]}")
            raise ValueError(f"Could not parse JSON from: {output[:200]}...")

    def __repr__(self) -> str:
        return (
            f"ChunkAnalyserInator("
            f"note_id={self.note_id}, "
            f"bubble_id={self.bubble_id}, "
            f"chunks_analysed={len(self.chunk_analyses)})"
        )
