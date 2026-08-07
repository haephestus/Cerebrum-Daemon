"""
notes — the note ingestion / processing pipeline.

AppFlowy delta -> markdown flatten, chunking, chunk analysis, chunk-queue.
Top-level package (sibling of api/, cerebrum_core/). Depends on infra
(vectorstore, database, storage); must NOT be imported *by* those infra layers.
"""
