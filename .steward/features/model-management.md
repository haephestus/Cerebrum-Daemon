# Feature: Model management

**Status**: DONE (P0)

## Goals
Let the user pick and provision local/cloud models without leaving the app.

## Scope
- Boot-time scrape of Ollama catalog → `models_manifest.json` baked once, skipped on later boots (deferred background task so startup isn't blocked)
- Endpoints: installed chat/embedding models, online models, cloud tags/details, download trigger, ollama status
- Config: chat / embedding / cloud model selection persisted via ConfigManager

## Dependencies
Ollama running locally (`common/ollama_compat/invoker_inator`); `huggingface-hub` present for hub interactions.

## Notes
- Manifest bake failure degrades gracefully (logged, server still starts).
