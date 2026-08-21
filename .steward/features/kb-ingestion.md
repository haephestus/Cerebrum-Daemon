# Feature: KB ingestion pipeline

**Status**: DONE (P1)

## Goals
Turn arbitrary uploaded documents into a queryable, domain-organized knowledge base.

## Scope
- Upload → Markdown conversion (AppFlowy JSON rich-text + PDF via PyMuPDF/pymupdf4llm/ocrmypdf) → chunking → embedding → FAISS vector stores organized as `domain/subject/collection`
- File fingerprint identity throughout (`process-file/{fingerprint}`, batch processing, queue status, stream-progress)
- Derived artifacts: figure registry + per-figure serving, concept index, chunk locate/explain-text endpoints
- Access control per fingerprint (`access/{fingerprint}` GET/POST/DELETE)
- Deletion: by fingerprint, by metadata; collections CRUD + counts; search by query and by fingerprint; stats

## Dependencies
[[research/architecture]] storage trio: SQLite registries + FAISS stores; embedding model from [[features/model-management]].

## Notes
- Chunking pipeline shared with notes but KB path is file-centric; note path is block-aligned ([[features/bubbles-notes]], ADR-0005).
