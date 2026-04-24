import json
import logging
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from agents.rose import RosePrompts
from cerebrum_core.model_inator import TranslatedQuery
from cerebrum_core.utils.file_util_inator import knowledgebase_index_inator

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
    Loads Chroma databases and retrieves relevant chunks for RAG queries.
    """

    def __init__(
        self, archives_root: str, embedding_model: str, chat_model: str
    ) -> None:
        self.archives_root = archives_root
        self.embedding_model = OllamaEmbeddings(model=embedding_model)

        if not chat_model:
            raise ValueError("Chat model must be configured")
        self.chat_model = OllamaLLM(model=chat_model)

        self.constructed_query = {}
        self.subqueries = []

    def translator_inator(self, user_query: str, translation_prompt: str):
        """Translate user query into structured archive queries."""
        if not translation_prompt:
            raise ValueError("Prompt 'rose_query_translator' not found in RosePrompts")

        available_stores = knowledgebase_index_inator(Path(self.archives_root))

        filled_prompt = translation_prompt.format(
            user_query=user_query, available_stores=available_stores
        )
        translated_query = self.chat_model.invoke(filled_prompt)
        logger.info(f"Raw translated query: {translated_query!r}")

        try:
            parsed_query = json.loads(translated_query)
        except json.JSONDecodeError:
            raise ValueError(f"LLM did not return valid JSON: {translated_query}")

        return TranslatedQuery(**parsed_query)

    def constructor_inator(self, translated_query: TranslatedQuery):
        """Construct archive paths from translated query."""
        available_stores, _ = knowledgebase_index_inator(Path(self.archives_root))
        valid_paths = set()

        for domain in available_stores["domains"]:
            for subject in available_stores["subjects"]:
                valid_paths.add((domain, subject))

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

        return self.constructed_query

    def retrieve_inator(self, k: int = 3):
        """Query archives and retrieve relevant chunks."""
        for route in self.constructed_query["routes"]:
            store = Chroma(
                collection_name=route["subquery"].subject,
                persist_directory=route["path"],
                embedding_function=self.embedding_model,
            )
            retriever = store.as_retriever(
                search_type="mmr", search_kwargs={"k": k, "fetch_k": 15}
            )
            result = retriever.invoke(route["subquery"].text)
            self.subqueries.append(result)

        return self.subqueries

    def generate_inator(self, user_query: str, top_k_chunks: int = 5):
        """Generate response using retrieved documents."""
        # Flatten and deduplicate
        flat_docs = [doc for docs in self.subqueries for doc in docs]

        seen = set()
        dedup_docs = []
        for doc in flat_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                dedup_docs.append(doc)

        selected_docs = dedup_docs[:top_k_chunks]

        # Summarize each chunk
        chunk_summaries = []
        for doc in selected_docs:
            summary_prompt = f"""
            Summarize the following text in 1–2 sentences, keeping only the key factual information:
            {doc.page_content}
            """
            summary = self.chat_model.invoke(summary_prompt)
            chunk_summaries.append(summary.strip())

        context_text = "\n\n".join(chunk_summaries)

        # Generate answer
        base_prompt = RosePrompts.get_prompt("rose_answer")
        if not base_prompt:
            raise ValueError("Prompt 'rose_answer' not found in RosePrompts")

        final_prompt = (
            base_prompt + "\n\nAdditional Instructions:\n"
            "- First give a 1–2 sentence summary answer.\n"
            "- Then, if relevant, provide a more detailed explanation under 'Further Explanation:'.\n"
            "- Condense overlapping info and avoid repeating facts.\n"
            "- Only use the provided context; do not hallucinate."
        )

        final_prompt = final_prompt.format(question=user_query, context=context_text)
        response = self.chat_model.invoke(final_prompt)

        return response
