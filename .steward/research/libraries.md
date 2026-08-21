# Libraries — dependency decisions

> Grounded against `src/requirements.txt` (fully pinned). Not a full audit — notable choices and anomalies only.

## Core stack
| Lib | Version | Why |
|---|---|---|
| fastapi + uvicorn (+uvloop) | 0.116.0 / 0.35.0 | HTTP daemon |
| faiss-cpu | 1.14.3 | vector search, replaced ChromaDB (ADR-0003) |
| langchain / langchain-core / langchain-community / langchain-text-splitters | v1.x line | RAG orchestration |
| langchain-ollama | 1.1.0 | local model binding |
| langgraph (+checkpoint/prebuilt/sdk) | 1.2.9 | agent graphing |
| ollama | 0.6.1 | native client for manifest/model ops |
| SQLAlchemy | 2.0.43 | present alongside hand-rolled SQLite registries — usage split unverified |
| pydantic(-settings/-yaml) | 2.13.4 | wire shapes |

## Ingestion & content
PyMuPDF + pymupdf4llm + pdfminer.six + pypdfium2 + ocrmypdf + pikepdf (PDF pipeline); pillow/img2pdf/fpdf2 (images/PDF out); trafilatura + beautifulsoup4 + lxml + jusText (content fetch/cleaning — feeds suggested reading's future online tiers); tiktoken + tokenizers + transformers + onnxruntime (tokenization/local embedding support).

## Observability & misc
sentry-sdk, opentelemetry-* suite (incl. anthropic/openai instrumentations — cloud-call tracing), coloredlogs. Also present: cachetools, backoff, tenacity, ulid/uuid_utils, mmh3 (hashing — likely bubble_id md5-adjacent helpers).

## Anomalies worth a pass in P3 ([[plan/hardening]])
- `Flask==3.1.0` + Werkzeug — no Flask app exists in the tree; probable vestigial transitive pin promoted to top level
- `kubernetes==33.1.0` — nothing in-tree references k8s; same suspicion
- Two venvs live inside `src/` (`.env`, `.direnv`) because `.gitignore` is empty — environment hygiene, not a lib decision
- `fastar==0.11.0` — obscure; verify still used
