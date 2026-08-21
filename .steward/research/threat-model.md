# Threat model — initial pass

> Scope: observations grounded in code read on 2026-08-21. This is a starting list feeding [[plan/hardening]]'s security pass — not a completed model.

## Trust boundaries
1. Client ↔ daemon (local): protected by shared daemon key minted at first boot, printed to stdout, stored plaintext in `<config>/daemon_api_key.txt`.
2. Client ↔ daemon (cloud): bearer user tokens; middleware branches on deployment mode (`is_local()`).
3. Uploads ↔ filesystem: documents, archives, note images.
4. LLM outputs ↔ user: analysis/grading/chat content rendered client-side.

## Observed exposures (to triage in P3)
- **CORS wildcard + credentials** (`allow_origins=["*"], allow_credentials=True`): browsers reject this combo for credentialed requests; today it mostly "works" because the key rides a custom header, but it's misconfigured rather than safe. Pick explicit origins.
- **Plaintext daemon key** on disk + printed to stdout: acceptable local posture, wrong cloud posture — verify it never ships in cloud mode logs.
- **Image serving path**: `/images/{image_name}` keyed by filename — check traversal/encoding on `{image_name}` and `{note_id}` path joins.
- **Upload surface**: file-type enforcement on upload/archive paths (`archive_inator` decompression bombs, PDF processing via pikepdf/ocrmypdf).
- **Password reset flow**: request/verify/update triple — token entropy, expiry, single-use, rate limiting unverified.
- **Prompt-injection into ingestion**: fetched/embedded third-party content (suggested reading ingest loop, future) becomes retrieval context that Rose treats as ground truth; poisoned sources could steer answers. License gate (ADR-0006) covers legality, not integrity — consider source-trust tiering before online tiers ship.
- **`routes_test.py` debug harnesses** (markdowninator/embeddinator/chunkinator/cache): confirm these are auth-gated and absent from production deploys.

## Assets ranked
1. User notes + knowledge bases (privacy-critical, local)
2. Credentials (daemon key, bearer tokens, password hashes — bcrypt present)
3. Grading/mastery integrity (tamper = corrupted learning signal)

## Out of scope here
Client-side threats (Flutter app), Ollama runtime hardening, host OS.
