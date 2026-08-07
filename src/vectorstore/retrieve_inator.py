import json
import logging
import os
from pathlib import Path

from langchain_ollama import OllamaEmbeddings

from models.model_inator import TranslatedQuery
from vectorstore.embeddings_inator import get_embeddings
from vectorstore.faiss_store_inator import get_or_create_store
from common.file_util_inator import knowledgebase_index_inator
from common.ollama_compat.invoker_inator import ollama_local_call

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/cerebrum_debug.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cerebrum")


class RetrieverInator:
    """
    Generic RAG retriever. Accepts any filled *_to_query prompt,
    translates it into structured archive queries, and retrieves
    relevant chunks from FAISS vector stores.
    LLM generation is handled externally via ollama_local_call.
    """

    def __init__(self, archives_root: str, embedding_model: str) -> None:
        self.archives_root = archives_root
        self.embedding_model = get_embeddings(embedding_model)
        self.constructed_query = {}
        self.subqueries = []

    def translator_inator(self, filled_prompt: str) -> TranslatedQuery:
        """
        Generic translator — accepts any pre-filled *_to_query prompt
        and returns a structured TranslatedQuery.

        The caller is responsible for:
          - Choosing the right prompt (rose_query_translator,
            rose_analysis_to_query, rose_note_to_query, etc.)
          - Filling in all prompt variables before passing it in.
        """
        available_stores = knowledgebase_index_inator(Path(self.archives_root))

        # Inject available_stores if the prompt still has the placeholder
        if "{available_stores}" in filled_prompt:
            filled_prompt = filled_prompt.replace(
                "{available_stores}", str(available_stores)
            )

        raw = ollama_local_call(filled_prompt, TranslatedQuery.schema())
        logger.info(f"Raw translated query: {raw!r}")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"ollama_local_call did not return valid JSON: {raw}")

        return TranslatedQuery(**parsed)

    def constructor_inator(self, translated_query: TranslatedQuery) -> dict:
        """Construct valid archive paths from a translated query."""
        available_stores, _ = knowledgebase_index_inator(Path(self.archives_root))
        valid_paths = {
            (domain, subject)
            for domain in available_stores["domains"]
            for subject in available_stores["subjects"]
        }

        self.constructed_query = {"routes": []}

        for subquery in translated_query.subqueries:
            domain = subquery.domain
            subject = subquery.subject

            if not domain or not subject:
                logger.warning("Skipping subquery with missing domain/subject")
                continue

            if (domain, subject) not in valid_paths:
                logger.warning(
                    f"Invalid domain/subject pair: ({domain}, {subject}) - skipping"
                )
                continue

            path = Path(self.archives_root) / domain / subject
            self.constructed_query["routes"].append(
                {"subquery": subquery, "path": str(path)}
            )

        logger.info(f"Constructed {len(self.constructed_query['routes'])} valid routes")
        return self.constructed_query

    def retrieve_inator(self, k: int = 3) -> list[list]:
        """Query FAISS archives and return retrieved document chunks."""
        self.subqueries = []

        for route in self.constructed_query.get("routes", []):
            store = get_or_create_store(Path(route["path"]), self.embedding_model)
            retriever = store.as_retriever(
                search_type="mmr", search_kwargs={"k": k, "fetch_k": 15}
            )
            result = retriever.invoke(route["subquery"].text)
            self.subqueries.append(result)
            logger.info(
                f"Retrieved {len(result)} chunks from '{route['subquery'].subject}'"
            )

        return self.subqueries

    def context_inator(self, top_k: int = 3) -> list[str]:
        """
        Flatten, deduplicate, and return top-k chunk contents as plain strings.
        Ready to be injected as context into an ollama_local_call prompt.
        """
        flat_docs = [doc for docs in self.subqueries for doc in docs]

        seen = set()
        dedup_contents = []
        for doc in flat_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                dedup_contents.append(doc.page_content)

        selected = dedup_contents[:top_k]
        logger.info(f"Returning {len(selected)} deduplicated chunks")
        return selected
