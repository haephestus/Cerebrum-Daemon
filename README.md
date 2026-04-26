# Cerebrum-Daemon

[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-purple.svg)](https://opensource.org/licenses/AGPL-3.0)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-latest-1C3C3C?logo=langchain)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-local-black)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()

> The backend engine powering Cerebrum — a RAG-based learning assistant built to run locally on consumer hardware.

This repository contains the Python/FastAPI backend that drives all AI functionality in the Cerebrum system. If you're looking for the full project overview, start with the [Cerebrum frontend repo](https://github.com/haephestus/Cerebrum).

---

## 1. What This Is

Cerebrum-Daemon is the backend service that handles everything the AI needs to do: ingesting documents, building and querying vector stores, running the RAG pipeline, and exposing it all through a clean HTTP API.

It runs as a **local daemon** — persistent, background, always available to the frontend without a network call leaving your machine. The design is intentionally similar to how Ollama itself works: start it once, let it run, let the frontend talk to it.

The name is literal. It's a daemon. It runs quietly in the background and does the hard work so the frontend doesn't have to.

---

## 2. Responsibilities

The daemon owns the entire AI layer:

- **Document ingestion** — accepts uploads, converts to Markdown, chunks content, generates embeddings, and indexes into domain-specific vector stores
- **Retrieval** — executes similarity search against embedded documents to retrieve relevant context for any given query
- **RAG orchestration** — assembles retrieved context and routes it through Ollama-managed local models via LangChain
- **Analysis** — produces structured feedback on user notes grounded strictly in retrieved source material
- **Caching** — caches retrieval results by domain and versions analysis outputs to avoid redundant computation
- **CRUD operations** — manages notes, study bubbles, and the knowledge base index

---

## 3. Architecture

```
Frontend (Flutter)
       │
       │ HTTP
       ▼
  cerebrum_inator.py   ← entry point / API server (FastAPI)
       │
       ├── ingest_inator.py          ← document ingestion pipeline
       ├── knowledgebase_index_inator.py  ← vector store indexing
       ├── retrieve_inator.py        ← similarity search / retrieval
       ├── markdown_converter.py     ← rich text → Markdown
       ├── markdown_chunker.py       ← chunking + metadata
       └── chunk_registry_inator.py  ← SQLite-backed embedding registry
              │
              ▼
         Ollama (local models)
         ChromaDB (vector stores)
         SQLite (registries + lightweight persistence)
```

---

## 4. Key Components

### `cerebrum_inator.py`
Main entry point. Initialises the FastAPI server and wires together ingestion, retrieval, and analysis components. Start here.

### `ingest_inator.py`
Handles the full ingestion pipeline: accepts documents, converts them to Markdown, chunks the content, and hands it off to the embedding layer. Supports AppFlowy's JSON rich-text format and standard documents.

### `knowledgebase_index_inator.py`
Manages registration and indexing of embedded documents into domain-specific vector stores. Domains are inferred from user input — biology, history, whatever the learner is working on.

### `retrieve_inator.py`
Executes similarity search against the relevant vector store during RAG queries. Returns ranked document chunks as context for LLM grounding.

### `markdown_converter.py` / `markdown_chunker.py`
Handle conversion from source format to Markdown, metadata handling, and chunk preparation before embedding.

### `chunk_registry_inator.py`
SQLite-backed registry that tracks which chunks have already been embedded. Prevents redundant re-embedding on re-analysis and lays the groundwork for versioning.

---

## 5. API Design

Routes are organised around **learning intent**, not pure CRUD semantics:

| Route group | Purpose |
|---|---|
| `/ingest` | Document ingestion and knowledge base management |
| `/analyse` | Note analysis against the knowledge base |
| `/retrieve` | Retrieval and AI chat |
| `/notes` | Note and study bubble CRUD |

Query parameters like `bubble_id`, `note_id`, and `version` preserve learning context and support result caching across sessions.

---

## 6. How to Run

**Requirements**
- Python 3.12
- Ollama (latest) running locally
- Dependencies in `requirements.txt`

**Setup**

```bash
git clone https://github.com/haephestus/Cerebrum-Daemon.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 cerebrum_inator.py
```

The daemon will start and expose its API on localhost. Point the Cerebrum frontend at it and you're running.

---

## 7. Why a Separate Repo?

The backend and frontend are separated deliberately. The daemon is designed to be frontend-agnostic — it exposes an HTTP API, so any client can talk to it. As Cerebrum matures, this separation makes it possible to swap or extend the frontend without touching the AI layer, and vice versa.

It also makes the codebase easier to reason about. The daemon does one thing: power the AI. The frontend does one thing: present it.

---

## 8. Current State

The daemon is functional for core use cases: ingestion, retrieval, analysis, and note management. Engram generation (flashcards, quizzes, mock exams) is planned but not yet implemented.

This is an actively developed project. It will keep improving.

---

## 9. Related

- [Cerebrum (frontend)](https://github.com/haephestus/Cerebrum) — The Flutter application this daemon powers
