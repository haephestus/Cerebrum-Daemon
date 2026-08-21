# Feature: RAG chat (Rose)

**Status**: DONE (P1)

## Goals
Answer questions about the learner's material strictly from their knowledge base.

## Scope
- `agents/rose.py`: prompt set enforcing context-only answering — no fabrication; explicit fallback line when context is insufficient; special handling when KB contains exam-style questions but user asks for explanation
- Exposed at bubble chat route (`POST /{bubble_id}/chat`); retrieval via [[features/kb-ingestion]] stores; served through Ollama local or configured cloud model

## Dependencies
FAISS domain stores; model management; bubble/note context.

## Notes
- Rose's grounding policy is also the prompt-injection boundary of convenience: anything embedded into the KB becomes trusted context — see threat-model note before online ingestion tiers ship ([[features/suggested-reading]], [[research/threat-model|threat-model]]).
