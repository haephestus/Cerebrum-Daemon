"""
common — cross-cutting utilities used across every layer.

Cache, filesystem helpers, archiving, progress, ollama compatibility, resync,
deploy config. Top-level package (sibling of api/, cerebrum_core/). Must stay
dependency-light and must NOT import from cerebrum_core.
"""
